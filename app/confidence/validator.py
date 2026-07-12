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
        _SHORT_ALLOWED = {"math", "retrieval", "extraction", "translation", "verification", "general_qa"}
        if word_count < 3 and task_type not in _SHORT_ALLOWED:
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
        if word_count < expected * 0.1 and task_type not in _SHORT_ALLOWED:
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

        # 2. Check for unbalanced brackets (only simple heuristics, as they might appear in code/strings)
        # We'll just check if there's a gross mismatch to avoid false positives.
        braces = text.count('{') - text.count('}')
        brackets = text.count('[') - text.count(']')
        parens = text.count('(') - text.count(')')
        
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
