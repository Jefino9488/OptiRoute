"""Benchmark evaluator — scores model outputs per category.

Scoring strategies are lightweight and category-specific:
- Math: exact numeric match
- Code: syntax validity
- Reasoning/QA: keyword overlap
- Creative: non-empty + length threshold
- Translation: simplified n-gram overlap
"""

from __future__ import annotations

import ast
import re
from typing import Any

import structlog

from app.benchmark.runner import BenchmarkResult

logger = structlog.get_logger(__name__)


class BenchmarkEvaluator:
    """Score model outputs against expected answers."""

    def evaluate(self, results: list[BenchmarkResult]) -> list[BenchmarkResult]:
        """Score all results in-place and return them.

        Parameters
        ----------
        results : list[BenchmarkResult]
            Raw results from the benchmark runner.

        Returns
        -------
        list[BenchmarkResult]
            Same list with ``score`` field populated (0.0–1.0).
        """
        for r in results:
            r.score = self._score(r)
        return results

    def _score(self, result: BenchmarkResult) -> float:
        """Score a single result based on its category."""
        category = result.category.lower()
        actual = result.actual_output.strip()
        expected = result.expected_output.strip()

        if not actual:
            return 0.0

        if category == "math":
            return self._score_math(actual, expected)
        elif category == "code":
            return self._score_code(actual, expected)
        elif category in ("reasoning", "general_qa", "qa"):
            return self._score_qa(actual, expected)
        elif category == "creative":
            return self._score_creative(actual)
        elif category == "translation":
            return self._score_translation(actual, expected)
        elif category == "extraction":
            return self._score_qa(actual, expected)
        else:
            return self._score_qa(actual, expected)

    @staticmethod
    def _score_math(actual: str, expected: str) -> float:
        """Score math responses via exact numeric match."""
        # Extract numbers from both strings
        actual_nums = re.findall(r'-?\d+\.?\d*', actual)
        expected_nums = re.findall(r'-?\d+\.?\d*', expected)

        if not expected_nums:
            return 1.0 if actual.strip() else 0.0

        if not actual_nums:
            return 0.0

        # Check if the expected number appears anywhere in the output
        for expected_num in expected_nums:
            try:
                e_val = float(expected_num)
                for actual_num in actual_nums:
                    try:
                        a_val = float(actual_num)
                        if abs(a_val - e_val) < 1e-6:
                            return 1.0
                    except ValueError:
                        continue
            except ValueError:
                continue

        return 0.0

    @staticmethod
    def _score_code(actual: str, expected: str) -> float:
        """Score code responses via syntax validity and optional match."""
        score = 0.0

        # Extract code blocks if present
        code_blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)```', actual, re.DOTALL)
        code_to_check = code_blocks[0] if code_blocks else actual

        # Check Python syntax validity
        try:
            ast.parse(code_to_check)
            score = 0.7  # Valid syntax = 70%
        except SyntaxError:
            score = 0.3  # Still some points for attempt

        # Bonus for non-empty, substantial response
        if len(actual.split()) > 10:
            score += 0.2

        # Bonus for matching expected keywords
        if expected:
            expected_words = set(expected.lower().split())
            actual_words = set(actual.lower().split())
            overlap = len(expected_words & actual_words)
            if expected_words:
                score += 0.1 * min(overlap / len(expected_words), 1.0)

        return min(score, 1.0)

    @staticmethod
    def _score_qa(actual: str, expected: str) -> float:
        """Score QA/reasoning responses via keyword overlap."""
        if not expected:
            # No expected output — score based on non-emptiness
            return 0.8 if len(actual.split()) > 5 else 0.3

        expected_words = set(
            w.lower().strip(".,!?;:'\"") for w in expected.split() if len(w) > 2
        )
        actual_words = set(
            w.lower().strip(".,!?;:'\"") for w in actual.split() if len(w) > 2
        )

        if not expected_words:
            return 0.8

        overlap = len(expected_words & actual_words)
        ratio = overlap / len(expected_words)

        # Exact match bonus
        if expected.lower().strip() == actual.lower().strip():
            return 1.0

        return min(ratio + 0.2, 1.0)  # Base 0.2 for attempt

    @staticmethod
    def _score_creative(actual: str) -> float:
        """Score creative responses on non-emptiness and length."""
        word_count = len(actual.split())
        if word_count < 10:
            return 0.2
        if word_count < 50:
            return 0.5
        if word_count < 150:
            return 0.7
        return 0.9

    @staticmethod
    def _score_translation(actual: str, expected: str) -> float:
        """Score translation with simplified n-gram overlap."""
        if not expected:
            return 0.8 if actual.strip() else 0.0

        # Simple unigram overlap (BLEU-like)
        expected_tokens = expected.lower().split()
        actual_tokens = actual.lower().split()

        if not expected_tokens:
            return 0.8

        matches = sum(1 for t in expected_tokens if t in actual_tokens)
        precision = matches / len(actual_tokens) if actual_tokens else 0.0
        recall = matches / len(expected_tokens)

        if precision + recall == 0:
            return 0.1

        # F1-like score
        f1 = 2 * precision * recall / (precision + recall)
        return min(f1 + 0.1, 1.0)  # Base 0.1 for attempt

    def aggregate_scores(
        self,
        results: list[BenchmarkResult],
    ) -> dict[str, dict[str, float]]:
        """Aggregate scores per model per category.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping of ``model_id → {category: avg_score}``.
        """
        # Group by model then category
        groups: dict[str, dict[str, list[float]]] = {}
        for r in results:
            if r.model_id not in groups:
                groups[r.model_id] = {}
            if r.category not in groups[r.model_id]:
                groups[r.model_id][r.category] = []
            groups[r.model_id][r.category].append(r.score)

        # Average per group
        aggregated: dict[str, dict[str, float]] = {}
        for model_id, categories in groups.items():
            aggregated[model_id] = {}
            for category, scores in categories.items():
                avg = sum(scores) / len(scores) if scores else 0.0
                aggregated[model_id][category] = round(avg, 4)

        return aggregated
