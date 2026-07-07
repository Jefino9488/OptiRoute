"""Tests for the benchmark evaluator scoring logic."""

from __future__ import annotations

import pytest

from app.benchmark.evaluator import BenchmarkEvaluator
from app.benchmark.runner import BenchmarkResult


@pytest.fixture()
def evaluator() -> BenchmarkEvaluator:
    return BenchmarkEvaluator()


class TestMathScoring:
    def test_exact_match(self, evaluator: BenchmarkEvaluator) -> None:
        r = BenchmarkResult(
            prompt="What is 15 * 23?",
            expected_output="345",
            actual_output="The answer is 345.",
            category="math",
            model_id="test",
        )
        evaluator.evaluate([r])
        assert r.score == 1.0

    def test_wrong_answer(self, evaluator: BenchmarkEvaluator) -> None:
        r = BenchmarkResult(
            prompt="What is 15 * 23?",
            expected_output="345",
            actual_output="The answer is 400.",
            category="math",
            model_id="test",
        )
        evaluator.evaluate([r])
        assert r.score == 0.0

    def test_empty_output(self, evaluator: BenchmarkEvaluator) -> None:
        r = BenchmarkResult(
            prompt="What is 5+3?",
            expected_output="8",
            actual_output="",
            category="math",
            model_id="test",
        )
        evaluator.evaluate([r])
        assert r.score == 0.0


class TestCodeScoring:
    def test_valid_python(self, evaluator: BenchmarkEvaluator) -> None:
        r = BenchmarkResult(
            prompt="Write a function",
            expected_output="def fibonacci",
            actual_output="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            category="code",
            model_id="test",
        )
        evaluator.evaluate([r])
        assert r.score >= 0.7

    def test_code_in_markdown(self, evaluator: BenchmarkEvaluator) -> None:
        r = BenchmarkResult(
            prompt="Write a function",
            expected_output="def sort",
            actual_output="```python\ndef sort_list(arr):\n    return sorted(arr)\n```\n\nThis function sorts a list.",
            category="code",
            model_id="test",
        )
        evaluator.evaluate([r])
        assert r.score > 0.5


class TestQAScoring:
    def test_keyword_overlap(self, evaluator: BenchmarkEvaluator) -> None:
        r = BenchmarkResult(
            prompt="What is photosynthesis?",
            expected_output="process plants use to convert sunlight into energy",
            actual_output="Photosynthesis is the process by which plants convert sunlight and carbon dioxide into energy and oxygen.",
            category="general_qa",
            model_id="test",
        )
        evaluator.evaluate([r])
        assert r.score > 0.5

    def test_no_overlap(self, evaluator: BenchmarkEvaluator) -> None:
        r = BenchmarkResult(
            prompt="What is the capital?",
            expected_output="Paris",
            actual_output="I am a large language model.",
            category="general_qa",
            model_id="test",
        )
        evaluator.evaluate([r])
        assert r.score < 0.5


class TestCreativeScoring:
    def test_long_creative(self, evaluator: BenchmarkEvaluator) -> None:
        r = BenchmarkResult(
            prompt="Write a poem",
            expected_output="",
            actual_output=" ".join(["word"] * 200),
            category="creative",
            model_id="test",
        )
        evaluator.evaluate([r])
        assert r.score >= 0.9

    def test_short_creative(self, evaluator: BenchmarkEvaluator) -> None:
        r = BenchmarkResult(
            prompt="Write a story",
            expected_output="",
            actual_output="Once upon a time.",
            category="creative",
            model_id="test",
        )
        evaluator.evaluate([r])
        assert r.score < 0.5


class TestAggregation:
    def test_aggregate_scores(self, evaluator: BenchmarkEvaluator) -> None:
        results = [
            BenchmarkResult(
                prompt="p1", expected_output="345", actual_output="345",
                category="math", model_id="model_a", score=1.0,
            ),
            BenchmarkResult(
                prompt="p2", expected_output="10", actual_output="10",
                category="math", model_id="model_a", score=1.0,
            ),
            BenchmarkResult(
                prompt="p3", expected_output="code", actual_output="code",
                category="code", model_id="model_a", score=0.5,
            ),
        ]
        agg = evaluator.aggregate_scores(results)
        assert agg["model_a"]["math"] == 1.0
        assert agg["model_a"]["code"] == 0.5
