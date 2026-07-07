"""Tests for app.features.extractor — FeatureExtractor."""

from __future__ import annotations

import pytest

from app.features.extractor import FeatureExtractor, FeatureVector


@pytest.fixture()
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


class TestCodeDetection:
    """Code-related prompts should set contains_code=True."""

    def test_python_function(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Write a Python function to sort a list")
        assert fv.contains_code is True
        assert fv.task_type == "code"

    def test_code_block_detection(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Fix this code:\n```python\nprint('hello')\n```")
        assert fv.contains_code is True

    def test_implement_algorithm(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Implement a binary search algorithm in Java")
        assert fv.contains_code is True
        assert fv.task_type == "code"

    def test_debug_keyword(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Debug this code and find the syntax error")
        assert fv.contains_code is True


class TestMathDetection:
    """Math-related prompts should set contains_math=True."""

    def test_calculate_expression(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Calculate 15 * 23")
        assert fv.contains_math is True
        assert fv.task_type == "math"

    def test_solve_equation(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Solve the equation 2x + 5 = 15")
        assert fv.contains_math is True

    def test_math_expression_pattern(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("What is 100 / 4?")
        assert fv.contains_math is True

    def test_probability_keyword(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("What is the probability of rolling a 6?")
        assert fv.contains_math is True


class TestJsonDetection:
    """JSON output requirement detection."""

    def test_return_as_json(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Return output as JSON")
        assert fv.json_required is True

    def test_json_format(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Provide the answer in json format")
        assert fv.json_required is True

    def test_json_schema(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Generate a JSON schema for the user model")
        assert fv.json_required is True


class TestCreativeDetection:
    """Creative writing detection."""

    def test_write_a_story(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Write a story about a dragon")
        assert fv.is_creative is True
        assert fv.task_type == "creative"

    def test_poem(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Compose a poem about the ocean")
        assert fv.is_creative is True


class TestTranslationDetection:
    """Translation task detection."""

    def test_translate_keyword(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Translate 'hello' to French")
        assert fv.is_translation is True
        assert fv.task_type == "translation"

    def test_in_spanish(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Say 'good morning' in Spanish")
        assert fv.is_translation is True


class TestReasoningDetection:
    """Reasoning task detection."""

    def test_explain(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Explain how photosynthesis works")
        assert fv.requires_reasoning is True

    def test_compare(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Compare Python and JavaScript")
        assert fv.requires_reasoning is True

    def test_pros_and_cons(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("What are the pros and cons of remote work?")
        assert fv.requires_reasoning is True


class TestRetrievalDetection:
    """Fact-retrieval task detection."""

    def test_what_is(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("What is the capital of France?")
        assert fv.requires_retrieval is True

    def test_who_is(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Who is Albert Einstein?")
        assert fv.requires_retrieval is True


class TestTaskTypeClassification:
    """Dominant task_type should match the primary intent."""

    def test_general_qa_fallback(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Tell me something interesting")
        assert fv.task_type == "general_qa"

    def test_mixed_code_and_math_favors_code(
        self, extractor: FeatureExtractor
    ) -> None:
        fv = extractor.extract(
            "Implement a function to calculate the factorial"
        )
        assert fv.task_type == "code"


class TestComplexity:
    """Complexity should scale with prompt length and feature count."""

    def test_short_simple_prompt_low_complexity(
        self, extractor: FeatureExtractor
    ) -> None:
        fv = extractor.extract("Hi")
        assert fv.complexity < 0.3

    def test_complex_prompt_higher_complexity(
        self, extractor: FeatureExtractor
    ) -> None:
        fv = extractor.extract(
            "Write a Python function that solves a system of linear equations "
            "step by step, then explain each step in detail and return the "
            "result as JSON. Compare the performance with numpy."
        )
        assert fv.complexity > 0.3


class TestOutputLength:
    """Expected output length heuristic."""

    def test_creative_is_long(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("Write a story about space exploration")
        assert fv.expected_output_length == "long"

    def test_retrieval_is_short(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("What is 42?")
        assert fv.expected_output_length == "short"


class TestInputLength:
    """Word count is captured correctly."""

    def test_word_count(self, extractor: FeatureExtractor) -> None:
        fv = extractor.extract("one two three four five")
        assert fv.input_length == 5
