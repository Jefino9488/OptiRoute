"""Confidence validator — checks output quality and triggers escalation.

Uses lightweight heuristics (no ML models) to estimate response confidence.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of confidence validation.

    Attributes
    ----------
    confidence : float
        Composite confidence score in [0, 1].
    issues : list[str]
        Human-readable list of detected problems.
    """

    confidence: float = 1.0
    issues: list[str] = field(default_factory=list)


class ConfidenceValidator:
    """Validate model output and produce a confidence score.

    Checks performed:
    * Empty / truncated response
    * JSON validity (when JSON was expected)
    * Code syntax (when code was expected)
    * Length ratio (vs expected output length)
    * Repetition detection
    """

    def validate(
        self,
        response: str,
        task_type: str = "general_qa",
        expected_json: bool = False,
        expected_code: bool = False,
        expected_length: str = "medium",
    ) -> ValidationResult:
        """Validate *response* and return a confidence score.

        Parameters
        ----------
        response : str
            The model-generated response text.
        task_type : str
            Dominant task type.
        expected_json : bool
            Whether JSON output was expected.
        expected_code : bool
            Whether code output was expected.
        expected_length : str
            Expected output length category.

        Returns
        -------
        ValidationResult
        """
        issues: list[str] = []
        penalties: list[float] = []

        # 1. Empty check
        if not response or not response.strip():
            return ValidationResult(confidence=0.0, issues=["Empty response"])

        # 2. Very short response
        word_count = len(response.split())
        if word_count < 3 and task_type not in ("math",):
            issues.append(f"Suspiciously short response ({word_count} words)")
            penalties.append(0.4)

        # 3. Error indicators
        error_patterns = ["[ERROR]", "I cannot", "I'm sorry, I can't", "as an ai"]
        lower_resp = response.lower()
        for pattern in error_patterns:
            if pattern.lower() in lower_resp:
                issues.append(f"Response contains error indicator: '{pattern}'")
                penalties.append(0.3)
                break

        # 4. JSON validation
        if expected_json:
            try:
                json.loads(response)
            except json.JSONDecodeError:
                # Try to find JSON within the response
                json_match = re.search(r'(\{[^}]+\}|\[[^\]]+\])', response, re.DOTALL)
                if json_match:
                    try:
                        json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        issues.append("Expected JSON but response is not valid JSON")
                        penalties.append(0.4)
                else:
                    issues.append("Expected JSON but no JSON found in response")
                    penalties.append(0.4)

        # 5. Code syntax check
        if expected_code:
            # Try to find Python code blocks
            code_blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
            if code_blocks:
                for block in code_blocks:
                    try:
                        ast.parse(block)
                    except SyntaxError:
                        issues.append("Code block has syntax errors")
                        penalties.append(0.2)
                        break
            # If no code blocks, check if the whole response is code
            elif task_type == "code":
                try:
                    ast.parse(response)
                except SyntaxError:
                    # Not necessarily an issue — could be pseudocode or non-Python
                    pass

        # 6. Length ratio check
        expected_words = {"short": 20, "medium": 100, "long": 300}
        expected = expected_words.get(expected_length, 100)
        if word_count < expected * 0.1 and task_type not in ("math", "retrieval"):
            issues.append(f"Response much shorter than expected ({word_count} vs ~{expected} words)")
            penalties.append(0.2)

        # 7. Structural completion checks
        completion_issues = self._check_structural_completion(response)
        if completion_issues:
            issues.extend(completion_issues)
            penalties.extend([0.6] * len(completion_issues))  # Heavy penalty for incomplete output

        # 8. Repetition detection
        if self._has_excessive_repetition(response):
            issues.append("Excessive repetition detected (possible degenerate output)")
            penalties.append(0.5)

        # 9. Task-type-specific validation
        task_issues, task_penalties = self._validate_task_specific(
            response, task_type, word_count
        )
        issues.extend(task_issues)
        penalties.extend(task_penalties)

        # Compute composite confidence
        confidence = 1.0
        for penalty in penalties:
            confidence *= (1.0 - penalty)
        confidence = max(0.0, min(1.0, confidence))

        if issues:
            logger.info(
                "confidence.issues_found",
                confidence=round(confidence, 4),
                issue_count=len(issues),
            )

        return ValidationResult(confidence=round(confidence, 4), issues=issues)

    @staticmethod
    def _has_excessive_repetition(text: str, threshold: int = 5) -> bool:
        """Check if any 3+ word phrase repeats more than *threshold* times."""
        words = text.split()
        if len(words) < 10:
            return False

        # Check trigram repetition
        trigrams: dict[str, int] = {}
        for i in range(len(words) - 2):
            trigram = " ".join(words[i:i + 3]).lower()
            trigrams[trigram] = trigrams.get(trigram, 0) + 1

        return any(count > threshold for count in trigrams.values())

    @staticmethod
    def _check_structural_completion(text: str) -> list[str]:
        """Check if the text is structurally complete.

        Returns a list of issue descriptions if incomplete.
        """
        issues = []
        
        # 1. Check for unbalanced markdown code fences
        fences = len(re.findall(r'^```', text, re.MULTILINE))
        if fences % 2 != 0:
            issues.append("Unbalanced markdown code fences")

        # 2. Check for unbalanced brackets — only for structured output
        #    (JSON, code) to avoid false positives on prose mentioning code.
        braces = text.count('{') - text.count('}')
        brackets = text.count('[') - text.count(']')
        parens = text.count('(') - text.count(')')

        # Only flag if the text likely contains structured data
        has_code_fence = bool(re.search(r'^```', text, re.MULTILINE))
        has_json_start = text.lstrip().startswith('{') or text.lstrip().startswith('[')

        if has_code_fence or has_json_start:
            if braces > 0:
                issues.append(f"Unbalanced braces: {braces} unclosed '{{'")
            if brackets > 0:
                issues.append(f"Unbalanced brackets: {brackets} unclosed '['")
            if parens > 0:
                issues.append(f"Unbalanced parentheses: {parens} unclosed '('")

        # 3. Check for obvious continuation patterns at the very end
        trimmed = text.rstrip()
        lower_trimmed = trimmed.lower()
        continuation_patterns = [
            "let's", "first", "here is the", "the answer is", "and", "or", "but", "so",
            "for example", "such as", "as follows:"
        ]
        
        if lower_trimmed.endswith(","):
            issues.append("Response ends abruptly with a comma")
        else:
            for pattern in continuation_patterns:
                if lower_trimmed.endswith(pattern):
                    issues.append(f"Response ends with continuation pattern: '{pattern}'")
                    break
                    
        return issues

    @staticmethod
    def _validate_task_specific(
        response: str,
        task_type: str,
        word_count: int,
    ) -> tuple[list[str], list[float]]:
        """Run task-type-specific quality checks.

        Returns
        -------
        tuple[list[str], list[float]]
            Lists of issues and corresponding penalties.
        """
        issues: list[str] = []
        penalties: list[float] = []
        lower_resp = response.lower()
        stripped = response.strip()

        if task_type == "math":
            # Math answers should contain at least one number.
            # NOTE: Do NOT penalize brief answers — thinking-mode models
            # (minimax/kimi with reasoning_effort=low) do their step-by-step
            # work inside hidden thinking tokens and emit a concise final answer.
            # A word_count check would incorrectly flag correct brief answers.
            has_number = bool(re.search(r'\d+', response))
            if not has_number:
                issues.append("Math response contains no numeric answer")
                penalties.append(0.5)

        elif task_type == "sentiment":
            # Response MUST start with a valid sentiment label.
            # Using a label-first format is enforced by the compiler; validate it here.
            valid_labels = ("positive", "negative", "neutral", "mixed")
            first_line = stripped.split("\n")[0].strip().lower().rstrip(".").rstrip(":")
            starts_with_label = any(first_line == label or first_line.startswith(label) for label in valid_labels)
            has_label_anywhere = any(label in lower_resp for label in valid_labels)
            if not starts_with_label:
                if has_label_anywhere:
                    # Label present but not at start — mild penalty
                    issues.append("Sentiment label not at start of response")
                    penalties.append(0.3)
                else:
                    # No label at all — strong penalty, will trigger escalation
                    issues.append("Sentiment response missing classification label (Positive/Negative/Neutral)")
                    penalties.append(0.55)

        elif task_type == "ner":
            # NER responses should contain identifiable entities or a list
            has_entities = bool(re.search(r'[A-Z][a-z]+', response))
            has_list = bool(
                re.search(r'[-•*]\s|^\d+[.)]', response, re.MULTILINE)
            )
            has_category_header = bool(
                re.search(r'(People|Organizations?|Locations?|Dates?)\s*:', response, re.IGNORECASE)
            )
            if not has_entities and not has_list and not has_category_header:
                issues.append("NER response lacks identifiable entities")
                penalties.append(0.3)

        elif task_type == "summarization":
            # Summaries that are very short might be poor
            if word_count < 10:
                issues.append("Summary is suspiciously short")
                penalties.append(0.2)

        elif task_type == "reasoning":
            # Multi-step reasoning tasks need substance.
            # NOTE: threshold kept deliberately low (15 words) to avoid penalizing
            # concise but correct answers from thinking-mode models.
            if word_count < 15:
                issues.append("Reasoning response too brief — likely incomplete")
                penalties.append(0.4)

        elif task_type == "extraction":
            # Extraction of JSON should produce an object, not prose
            import json as _json
            if stripped.startswith('{') or stripped.startswith('['):
                try:
                    parsed = _json.loads(stripped)
                    # A list of {key: count} items is wrong for domain-count extraction
                    # The expected format is {"domain": count, ...}
                    if isinstance(parsed, list):
                        # List is acceptable for some extractions (NER-like),
                        # but if each item is a dict with email keys, it's wrong
                        first = parsed[0] if parsed else {}
                        if isinstance(first, dict) and any('@' in str(k) for k in first.keys()):
                            issues.append("Extraction returned per-user records instead of aggregated values")
                            penalties.append(0.45)
                except Exception:
                    pass

        return issues, penalties
