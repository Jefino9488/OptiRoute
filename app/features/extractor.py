"""Feature extractor — regex / keyword-based prompt analysis.

Detects task type, boolean feature flags, and complexity from raw
prompt text.  No ML models are used — everything is regex and keyword
matching so extraction is effectively instantaneous.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FeatureVector:
    """Extracted feature vector for a single prompt.

    Attributes:
        task_type: Dominant task category.
        contains_code: Whether the prompt involves code.
        contains_math: Whether the prompt involves mathematics.
        json_required: Whether the expected output should be JSON.
        is_creative: Whether the task is creative writing.
        is_translation: Whether the task involves translation.
        requires_reasoning: Whether multi-step reasoning is needed.
        requires_retrieval: Whether fact retrieval is the main goal.
        input_length: Word count of the prompt.
        expected_output_length: 'short', 'medium', or 'long'.
        question_count: Number of question marks / question indicators.
        complexity: Overall complexity score in [0, 1].
    """

    task_type: str = "general_qa"
    contains_code: bool = False
    contains_math: bool = False
    json_required: bool = False
    is_creative: bool = False
    is_translation: bool = False
    requires_reasoning: bool = False
    requires_retrieval: bool = False
    is_extraction: bool = False
    is_classification: bool = False
    is_summarization: bool = False
    is_sentiment: bool = False
    is_ner: bool = False
    has_strict_constraint: bool = False
    is_counting: bool = False
    has_numeric_literals: bool = False  # True if prompt contains standalone numbers/percentages
    input_length: int = 0
    expected_output_length: str = "medium"
    question_count: int = 0
    complexity: float = 0.0


# ---------------------------------------------------------------------------
# Pre-compiled detection patterns
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")

_CODE_KEYWORDS_RE = re.compile(
    r"\b(?:"
    r"function|def\s|class\s|import\s|variable|algorithm|implement|debug|"
    r"compile|syntax|refactor|codebase|subroutine|api|endpoint|"
    r"python|javascript|typescript|java\b|c\+\+|ruby|golang|go\b|rust|"
    r"kotlin|swift|scala|haskell|perl|php|sql|html|css|bash|shell|"
    r"write\s+(?:a\s+)?(?:python|javascript|typescript|java|c\+\+|ruby|"
    r"golang|go|rust|kotlin|swift|code|script|program|function|class|method|module)"
    r")\b",
    re.IGNORECASE,
)

_MATH_KEYWORDS_RE = re.compile(
    r"\b(?:"
    r"calculate|solve|equation|sum|product|integral|derivative|"
    r"probability|factorial|logarithm|sqrt|square\s+root|"
    r"multiply|divide|subtract|add|compute|arithmetic|"
    r"algebra|geometry|trigonometry|calculus|statistics|"
    r"matrix|vector|determinant|eigenvalue|"
    r"percentage|fraction|ratio|proportion"
    r")\b",
    re.IGNORECASE,
)

# Matches patterns like "3 + 4", "15 * 23", "100 / 5", "2^8"
_MATH_EXPR_RE = re.compile(
    r"\d+\s*[+\-*/^%]\s*\d+",
)

# Matches math word problem phrasing — covers arithmetic story problems
_MATH_WORD_PROBLEM_RE = re.compile(
    r"\b(?:"
    r"how\s+many\s+(?:remain|total|left|more|less|units|items|cookies|cups)|"
    r"what\s+is\s+the\s+(?:probability|chance|total|average|mean|median|mode|value|cost|speed|distance|rate)|"
    r"what\s+is\s+its\s+value|"
    r"calculate\s+the\s+(?:cost|price|value|amount|time|distance|speed|average|probability)|"
    r"(?:sells?|loses?|gains?|depreciates?|restocks?|removes?|adds?)\s+\d+(?:[,.]\d+)?\s*(?:%|percent|units?|items?|kg|m|km|mph|kmh)|"
    r"(?:per\s+(?:year|month|day|hour|unit))|"
    r"(?:\d+(?:[,.]\d+)?\s*%\s*(?:of|off|per))|"
    r"final\s+(?:count|total|amount|balance|value|stock)|"
    r"how\s+much\s+(?:sugar|flour|water|cost|money|does|will|would)"
    r")\b",
    re.IGNORECASE,
)

# Matches counting tasks — requires precise enumeration, NOT simple retrieval.
_COUNTING_KEYWORDS_RE = re.compile(
    r"\b(?:"
    r"count\s+(?:exactly|how\s+many|all|the\s+number\s+of)|"
    r"how\s+many\s+times\s+(?:does|is|the|a)|"
    r"number\s+of\s+(?:times|occurrences|instances|letters|words|characters|appearances)"
    r")\b",
    re.IGNORECASE,
)

_JSON_KEYWORDS_RE = re.compile(
    r"\bjson\b|"
    r"\bJSON\b|"
    r"output\s+as\s+json|"
    r"return\s+json|"
    r"json\s+format|"
    r"json\s+schema|"
    r"json\s+object|"
    r"json\s+array|"
    r"structured\s+output",
    re.IGNORECASE,
)

_CREATIVE_KEYWORDS_RE = re.compile(
    r"\b(?:"
    r"write\s+a\s+story|poem|essay|creative|imagine|fiction|narrative|"
    r"compose|haiku|sonnet|limerick|screenplay|dialogue|monologue|"
    r"short\s+story|fairy\s+tale|write\s+a\s+poem|write\s+an\s+essay|"
    r"brainstorm|invent|create\s+a\s+story"
    r")\b",
    re.IGNORECASE,
)

_TRANSLATION_KEYWORDS_RE = re.compile(
    r"\b(?:"
    r"translate|translation"
    r")\b|"
    r"\b(?:in|to|from)\s+(?:"
    r"french|spanish|german|italian|portuguese|chinese|japanese|korean|"
    r"arabic|russian|hindi|dutch|swedish|norwegian|danish|finnish|polish|"
    r"turkish|greek|hebrew|thai|vietnamese|indonesian|malay|tagalog|"
    r"english|latin|swahili"
    r")\b",
    re.IGNORECASE,
)

_REASONING_KEYWORDS_RE = re.compile(
    r"\b(?:"
    r"why|explain|compare|analyze|analyse|evaluate|"
    r"pros\s+and\s+cons|difference\s+between|"
    r"how\s+does|how\s+do|how\s+would|"
    r"step\s+by\s+step|think\s+through|reason\s+about|"
    r"what\s+are\s+the\s+implications|"
    r"cause\s+and\s+effect|argue\s+for|argue\s+against|"
    r"critically|justify|assessment|trade-?off"
    r")\b",
    re.IGNORECASE,
)

_RETRIEVAL_KEYWORDS_RE = re.compile(
    r"\b(?:"
    r"what\s+is|who\s+is|when\s+did|where\s+is|where\s+did|"
    r"define|list|name\s+the|tell\s+me\s+about|"
    r"what\s+are|who\s+was|when\s+was|how\s+many|"
    r"what\s+does|what\s+year|which\s+country|capital\s+of"
    r")\b",
    re.IGNORECASE,
)

_EXTRACTION_KEYWORDS_RE = re.compile(
    r"\b(?:extract|parse|identify|find\s+all|list\s+all|domains?|named\s+entities?|ingredients?)\b",
    re.IGNORECASE,
)

_CLASSIFICATION_KEYWORDS_RE = re.compile(
    r"\b(?:classify|sentiment|positive|negative|neutral|category|review|label)\b",
    re.IGNORECASE,
)

_SUMMARIZATION_KEYWORDS_RE = re.compile(
    r"\b(?:"
    r"summarize|summarise|summary|tldr|tl;dr|key\s+points|main\s+points|"
    r"gist|brief\s+overview|recap|abstract|condense|shorten|"
    r"give\s+(?:me\s+)?a\s+summary|sum\s+up|in\s+summary"
    r")\b",
    re.IGNORECASE,
)

_SENTIMENT_KEYWORDS_RE = re.compile(
    r"\b(?:"
    r"sentiment|tone|mood|feeling|opinion|attitude|"
    r"positive\s+or\s+negative|optimistic|pessimistic|"
    r"favorable|unfavorable|what\s+(?:is|are)\s+the\s+(?:tone|mood|sentiment)|"
    r"how\s+does\s+(?:the\s+)?(?:author|writer|speaker)\s+feel"
    r")\b",
    re.IGNORECASE,
)

_NER_KEYWORDS_RE = re.compile(
    r"(?:"
    r"named\s+entit|"   # "named entities" — prefix match, no \b needed
    r"\bNER\b|\bner\b|"
    r"\bidentify\s+(?:the\s+)?(?:people|persons|names|places|locations|organizations|companies|entities)\b|"
    r"\bextract\s+(?:the\s+)?(?:names|people|locations|organizations|entities)\b|"
    r"\bcategorize\s+(?:them|the\s+entities)|"
    r"\bwho\s+(?:is|are)\s+mentioned\b|"
    r"\bwhat\s+(?:places|people|organizations)\s+are\s+mentioned\b"
    r")",
    re.IGNORECASE,
)

_STRICT_CONSTRAINT_RE = re.compile(
    r"\b(?:exactly|strictly|must|only|all\s+unique|carefully|rigorously|without\s+fail)\b",
    re.IGNORECASE,
)

_QUESTION_MARK_RE = re.compile(r"\?")

# Multi-step indicators (numbered lists, "first … then …", etc.)
_MULTI_STEP_RE = re.compile(
    r"(?:\b(?:first|second|third|then|next|finally|step\s+\d)\b)|"
    r"(?:^\s*\d+[.)]\s)",
    re.IGNORECASE | re.MULTILINE,
)

# Explicit instruction patterns:
# These regex patterns will be used estimate output length

_BREVITY_PATTERNS = re.compile(
    r"\b(briefly|be brief|keep it brief|in brief|in short|one word|"
    r"one sentence|tldr|tl;dr|just the answer|no explanation|concise|"
    r"quick answer|yes or no|in \d+ words? or less|short answer)\b",
    re.IGNORECASE,
)

_VERBOSITY_PATTERNS = re.compile(
    r"\b(explain\w* in detail|step by step|show your work|elaborate\w*|"
    r"comprehensive|in-depth|walk me (through|thorough\w*)|"
    r"with examples|detailed explanation|justify)\b",
    re.IGNORECASE,
)

_FORMAT_LONG_PATTERNS = re.compile(
    r"\b(write a? ?(essay|article|story|report|script)|"
    r"generate .*(code|function|class)|full implementation)\b",
    re.IGNORECASE,
)

_FORMAT_SHORT_PATTERNS = re.compile(
    r"\b(what is|what's|how many|which one|true or false|"
    r"is it|does it|can you confirm)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helper: count pattern matches
# ---------------------------------------------------------------------------

def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text))


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Regex / keyword-based feature extractor.

    All detection is deterministic and runs in microseconds.  No ML
    models are loaded.
    """

    def extract(self, prompt: str) -> FeatureVector:
        """Analyse *prompt* and return a :class:`FeatureVector`.

        Args:
            prompt: Raw user prompt string.

        Returns:
            Populated feature vector with detected task type, flags,
            and complexity estimate.
        """
        self.prompt = prompt
        fv = FeatureVector()

        # -- word count --
        words = prompt.split()
        fv.input_length = len(words)

        # -- question count --
        fv.question_count = _count(_QUESTION_MARK_RE, prompt)

        # -- boolean feature flags --
        has_code_block = bool(_CODE_BLOCK_RE.search(prompt))
        has_code_kw = bool(_CODE_KEYWORDS_RE.search(prompt))
        fv.contains_code = has_code_block or has_code_kw

        has_math_kw = bool(_MATH_KEYWORDS_RE.search(prompt))
        has_math_expr = bool(_MATH_EXPR_RE.search(prompt))
        has_math_word = bool(_MATH_WORD_PROBLEM_RE.search(prompt))
        fv.contains_math = has_math_kw or has_math_expr or has_math_word

        fv.json_required = bool(_JSON_KEYWORDS_RE.search(prompt))
        fv.is_creative = bool(_CREATIVE_KEYWORDS_RE.search(prompt))
        fv.is_translation = bool(_TRANSLATION_KEYWORDS_RE.search(prompt))

        has_reasoning_kw = bool(_REASONING_KEYWORDS_RE.search(prompt))
        has_multi_step = bool(_MULTI_STEP_RE.search(prompt))
        fv.requires_reasoning = has_reasoning_kw or has_multi_step

        fv.requires_retrieval = bool(_RETRIEVAL_KEYWORDS_RE.search(prompt))
        fv.is_extraction = bool(_EXTRACTION_KEYWORDS_RE.search(prompt)) or fv.json_required
        fv.is_classification = bool(_CLASSIFICATION_KEYWORDS_RE.search(prompt))
        fv.is_summarization = bool(_SUMMARIZATION_KEYWORDS_RE.search(prompt))
        fv.is_sentiment = bool(_SENTIMENT_KEYWORDS_RE.search(prompt))
        fv.is_ner = bool(_NER_KEYWORDS_RE.search(prompt))
        fv.has_strict_constraint = bool(_STRICT_CONSTRAINT_RE.search(prompt))
        fv.is_counting = bool(_COUNTING_KEYWORDS_RE.search(prompt))
        # has_numeric_literals: True when prompt contains numbers/percentages.
        # Used to distinguish arithmetic word problems from conceptual comparisons.
        fv.has_numeric_literals = bool(re.search(r'\b\d+(?:[,.]\d+)?\s*(?:%|percent)?\b', prompt))

        # -- task type (dominant) --
        fv.task_type = self._classify_task_type(fv)

        # -- expected output length --
        fv.expected_output_length = self._estimate_output_length(fv, prompt)

        # -- complexity --
        fv.complexity = self._estimate_complexity(fv)

        return fv

    # -- private helpers ---------------------------------------------------

    @staticmethod
    def _classify_task_type(fv: FeatureVector) -> str:
        """Pick the dominant task type from detected feature flags.

        Priority order resolves ties (more specific types win).
        Now uses intent-aware scoring to prevent keyword collisions.
        """
        # Score each type — higher = stronger signal.
        scores: dict[str, float] = {
            "math": 0.0,
            "code": 0.0,
            "reasoning": 0.0,
            "creative": 0.0,
            "translation": 0.0,
            "extraction": 0.0,
            "retrieval": 0.0,
            "general_qa": 0.1,  # small default so it's the fallback
            "summarization": 0.0,
            "sentiment": 0.0,
            "ner": 0.0,
        }

        # Base keyword matches
        if fv.contains_math:
            scores["math"] += 1.0
        if fv.contains_code:
            scores["code"] += 1.0
        if fv.requires_reasoning:
            scores["reasoning"] += 0.8
        if fv.is_creative:
            scores["creative"] += 1.0
        if fv.is_translation:
            scores["translation"] += 1.0
        if fv.is_extraction:
            scores["extraction"] += 1.0
        if fv.requires_retrieval:
            scores["retrieval"] += 0.7
        if fv.is_classification:
            scores["reasoning"] += 0.8
            scores["general_qa"] += 0.5
        if fv.is_summarization:
            scores["summarization"] += 1.2
        if fv.is_sentiment:
            scores["sentiment"] += 1.2
            scores["reasoning"] += 0.3  # sentiment analysis involves reasoning
        if fv.is_ner:
            scores["ner"] += 1.2
            scores["extraction"] += 0.3  # NER is a form of extraction
        if fv.is_counting:
            # Counting tasks require precise enumeration — treat as math, suppress retrieval
            scores["math"] += 1.5
            scores["retrieval"] *= 0.1

        # Intent Resolution (resolving keyword collisions)

        # 0. Classification overrides incidental math (e.g. rating "5/5")
        if fv.is_classification and fv.contains_math:
            scores["math"] *= 0.1
            scores["reasoning"] += 1.0

        # 1. Creative intent overrides incidental math (e.g. "haiku 5-7-5")
        if fv.is_creative and fv.contains_math:
            scores["creative"] += 1.5

        # 2. Translation intent overrides general topics
        if fv.is_translation:
            scores["translation"] += 1.5

        # 3. Code intent with math (e.g. "implement fibonacci") -> Code wins
        if fv.contains_code and fv.contains_math:
            scores["code"] += 0.5

        # 4. JSON intent boosts extraction unless code is dominant
        if fv.json_required and not fv.contains_code:
            scores["extraction"] += 0.5

        # 5. Sentiment with retrieval keywords → sentiment wins
        # (e.g. "What is the sentiment of this review?")
        if fv.is_sentiment and fv.requires_retrieval:
            scores["retrieval"] *= 0.3
            scores["sentiment"] += 0.5

        # 5b. Sentiment overrides generic classification
        # ("Classify the sentiment..." -> sentiment, not generic classification/reasoning)
        if fv.is_sentiment and fv.is_classification:
            scores["reasoning"] *= 0.5
            scores["sentiment"] += 1.0

        # 6. Summarization overrides retrieval ("summarize" is not "what is")
        if fv.is_summarization and fv.requires_retrieval:
            scores["retrieval"] *= 0.3

        # 7. NER with extraction → NER is more specific, wins
        if fv.is_ner and fv.is_extraction:
            scores["extraction"] *= 0.5
            scores["ner"] += 0.5

        # 8. Factual comparison ("difference between X and Y") → reasoning, not math
        # Fixes: T01b "difference between machine learning and deep learning" → was math
        # IMPORTANT: Only suppress math for conceptual comparisons with NO numeric literals.
        # Arithmetic word problems (st02, T02, st07) always have numbers like 240, 15%, $20,000.
        is_conceptual_comparison = (
            fv.requires_reasoning and fv.contains_math
            and not fv.is_counting
            and not fv.has_numeric_literals  # arithmetic prompts always have numbers
        )
        if is_conceptual_comparison:
            scores["math"] *= 0.4
            scores["reasoning"] += 0.5

        return max(scores, key=scores.get)  # type: ignore[arg-type]

    @staticmethod
    def _estimate_output_length(fv: FeatureVector, prompt) -> str:
        """Heuristic estimate of expected output length."""

        if _BREVITY_PATTERNS.search(prompt):
            return "short"
        if _VERBOSITY_PATTERNS.search(prompt):
            return "long"
        if _FORMAT_LONG_PATTERNS.search(prompt):
            return "long"
        if _FORMAT_SHORT_PATTERNS.search(prompt) and not fv.requires_reasoning:
            return "short"

        # rest executes when prompt doesn't explicitly mention about length

        if fv.is_sentiment:
            return "short"  # sentiment = label + brief explanation
        if fv.is_ner:
            return "medium" if fv.input_length > 50 else "short"
        if fv.is_summarization:
            return "medium"  # summaries compress, not expand
        if fv.is_creative:
            return "long"
        if fv.contains_code:
            return "long" if fv.input_length > 50 else "medium"
        if fv.requires_reasoning:
            return "medium" if fv.input_length < 100 else "long"
        if fv.requires_retrieval:
            return "short"
        if fv.input_length < 15:
            return "short"
        if fv.input_length > 80:
            return "long"
        return "medium"

    @staticmethod
    def _estimate_complexity(fv: FeatureVector) -> float:
        """Estimate overall task complexity in [0, 1].

        Combines input length, question count, and number of detected
        feature dimensions.
        """
        # Length factor: long prompts → more complex (sigmoid-ish).
        length_score = min(fv.input_length / 200.0, 1.0)

        # Feature density: more detected features → more complex.
        feature_count = sum([
            fv.contains_code,
            fv.contains_math,
            fv.json_required,
            fv.is_creative,
            fv.is_translation,
            fv.requires_reasoning,
            fv.requires_retrieval,
            fv.is_summarization,
            fv.is_sentiment,
            fv.is_ner,
            fv.is_counting,
        ])
        feature_score = min(feature_count / 4.0, 1.0)

        # Question count factor.
        question_score = min(fv.question_count / 3.0, 1.0)

        # Strict constraints dramatically bump complexity
        strictness_penalty = 0.3 if fv.has_strict_constraint else 0.0

        # Weighted combination.
        complexity = (
            0.35 * length_score
            + 0.40 * feature_score
            + 0.25 * question_score
            + strictness_penalty
        )

        return round(min(max(complexity, 0.0), 1.0), 4)
