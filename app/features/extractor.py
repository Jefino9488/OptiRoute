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
        fv.contains_math = has_math_kw or has_math_expr

        fv.json_required = bool(_JSON_KEYWORDS_RE.search(prompt))
        fv.is_creative = bool(_CREATIVE_KEYWORDS_RE.search(prompt))
        fv.is_translation = bool(_TRANSLATION_KEYWORDS_RE.search(prompt))

        has_reasoning_kw = bool(_REASONING_KEYWORDS_RE.search(prompt))
        has_multi_step = bool(_MULTI_STEP_RE.search(prompt))
        fv.requires_reasoning = has_reasoning_kw or has_multi_step

        fv.requires_retrieval = bool(_RETRIEVAL_KEYWORDS_RE.search(prompt))

        # -- task type (dominant) --
        fv.task_type = self._classify_task_type(fv)

        # -- expected output length --
        fv.expected_output_length = self._estimate_output_length(fv)

        # -- complexity --
        fv.complexity = self._estimate_complexity(fv)

        return fv

    # -- private helpers ---------------------------------------------------

    @staticmethod
    def _classify_task_type(fv: FeatureVector) -> str:
        """Pick the dominant task type from detected feature flags.

        Priority order resolves ties (more specific types win).
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
        }

        if fv.contains_math:
            scores["math"] += 1.0
        if fv.contains_code:
            scores["code"] += 1.0
        if fv.requires_reasoning:
            scores["reasoning"] += 0.8
        if fv.is_creative:
            scores["creative"] += 0.9
        if fv.is_translation:
            scores["translation"] += 0.95
        if fv.json_required:
            scores["extraction"] += 0.6
        if fv.requires_retrieval:
            scores["retrieval"] += 0.7

        # If both code *and* math, code wins (e.g. "implement fibonacci").
        if fv.contains_code and fv.contains_math:
            scores["code"] += 0.2

        return max(scores, key=scores.get)  # type: ignore[arg-type]

    @staticmethod
    def _estimate_output_length(fv: FeatureVector) -> str:
        """Heuristic estimate of expected output length."""
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
        ])
        feature_score = min(feature_count / 4.0, 1.0)

        # Question count factor.
        question_score = min(fv.question_count / 3.0, 1.0)

        # Weighted combination.
        complexity = (
            0.35 * length_score
            + 0.40 * feature_score
            + 0.25 * question_score
        )

        return round(min(max(complexity, 0.0), 1.0), 4)
