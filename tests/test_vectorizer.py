"""Tests for app.features.vectorizer — TaskVectorGenerator."""

from __future__ import annotations

import pytest

from app.features.extractor import FeatureExtractor, FeatureVector
from app.features.vectorizer import (
    ResourceVector,
    RiskVector,
    TaskVector,
    TaskVectorGenerator,
)


@pytest.fixture()
def gen() -> TaskVectorGenerator:
    return TaskVectorGenerator()


@pytest.fixture()
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


def _extract_and_generate(
    prompt: str,
    extractor: FeatureExtractor,
    gen: TaskVectorGenerator,
) -> tuple[TaskVector, ResourceVector, RiskVector]:
    fv = extractor.extract(prompt)
    return gen.generate(fv)


class TestTaskVectorDominance:
    """The dominant dimension should be the highest value."""

    def test_math_prompt_high_math(
        self, gen: TaskVectorGenerator, extractor: FeatureExtractor
    ) -> None:
        tv, _, _ = _extract_and_generate("Calculate 15 * 23", extractor, gen)
        assert tv.math >= 0.7
        assert tv.math > tv.code
        assert tv.math > tv.creative

    def test_code_prompt_high_code(
        self, gen: TaskVectorGenerator, extractor: FeatureExtractor
    ) -> None:
        tv, _, _ = _extract_and_generate(
            "Write a Python function to sort a list", extractor, gen
        )
        assert tv.code >= 0.7
        assert tv.code > tv.math
        assert tv.code > tv.creative

    def test_creative_prompt_high_creative(
        self, gen: TaskVectorGenerator, extractor: FeatureExtractor
    ) -> None:
        tv, _, _ = _extract_and_generate(
            "Write a story about a dragon", extractor, gen
        )
        assert tv.creative >= 0.7
        assert tv.creative > tv.math

    def test_translation_prompt_high_translation(
        self, gen: TaskVectorGenerator, extractor: FeatureExtractor
    ) -> None:
        tv, _, _ = _extract_and_generate(
            "Translate this to French", extractor, gen
        )
        assert tv.translation >= 0.7

    def test_retrieval_prompt_high_retrieval(
        self, gen: TaskVectorGenerator, extractor: FeatureExtractor
    ) -> None:
        tv, _, _ = _extract_and_generate(
            "What is the capital of France?", extractor, gen
        )
        assert tv.retrieval >= 0.7


class TestVectorBounds:
    """All TaskVector dimensions must be in [0, 1]."""

    _PROMPTS = [
        "Calculate 15 * 23",
        "Write a Python function to sort a list",
        "Write a story about a dragon",
        "Translate this to French",
        "What is the capital of France?",
        "Explain the pros and cons of remote work step by step in JSON",
        "Hi",
        "",
    ]

    @pytest.mark.parametrize("prompt", _PROMPTS)
    def test_all_dimensions_in_range(
        self,
        prompt: str,
        gen: TaskVectorGenerator,
        extractor: FeatureExtractor,
    ) -> None:
        tv, rv, risk = _extract_and_generate(prompt, extractor, gen)
        for dim_name, value in tv.to_dict().items():
            assert 0.0 <= value <= 1.0, (
                f"TaskVector.{dim_name}={value} out of [0,1] for prompt={prompt!r}"
            )


class TestResourceVectorEstimates:
    """Resource vector values should be reasonable."""

    def test_input_tokens_proportional_to_word_count(
        self, gen: TaskVectorGenerator, extractor: FeatureExtractor
    ) -> None:
        _, rv, _ = _extract_and_generate("one two three four five", extractor, gen)
        # 5 words × 1.3 ≈ 6.5
        assert 5.0 <= rv.expected_input_tokens <= 10.0

    def test_code_gets_higher_output_estimate(
        self, gen: TaskVectorGenerator, extractor: FeatureExtractor
    ) -> None:
        _, rv_code, _ = _extract_and_generate(
            "Write a Python function to sort a list", extractor, gen
        )
        _, rv_qa, _ = _extract_and_generate(
            "What is the capital of France?", extractor, gen
        )
        _BUCKET_MAP = {"Small": 1, "Medium": 2, "Large": 3, "Very_Large": 4}
        assert _BUCKET_MAP[rv_code.output_budget_bucket] > _BUCKET_MAP[rv_qa.output_budget_bucket]

    def test_context_length_is_sum(
        self, gen: TaskVectorGenerator, extractor: FeatureExtractor
    ) -> None:
        _, rv, _ = _extract_and_generate("Explain recursion", extractor, gen)
        # Since output is bucketized, context length is input + bucket_size
        assert rv.expected_context_length > rv.expected_input_tokens

    def test_complexity_matches_features(
        self, gen: TaskVectorGenerator, extractor: FeatureExtractor
    ) -> None:
        fv = extractor.extract("Calculate 15 * 23")
        _, rv, _ = gen.generate(fv)
        assert rv.complexity == fv.complexity


class TestRiskVector:
    """Risk vector flags should reflect feature booleans."""

    def test_json_required_sets_flags(
        self, gen: TaskVectorGenerator, extractor: FeatureExtractor
    ) -> None:
        _, _, risk = _extract_and_generate(
            "Return the output as JSON", extractor, gen
        )
        assert risk.needs_json is True
        assert risk.strict_formatting is True

    def test_math_sets_high_accuracy(
        self, gen: TaskVectorGenerator, extractor: FeatureExtractor
    ) -> None:
        _, _, risk = _extract_and_generate("Calculate 15 * 23", extractor, gen)
        assert risk.needs_high_accuracy is True

    def test_code_sets_high_accuracy(
        self, gen: TaskVectorGenerator, extractor: FeatureExtractor
    ) -> None:
        _, _, risk = _extract_and_generate(
            "Write a Python function to sort a list", extractor, gen
        )
        assert risk.needs_high_accuracy is True

    def test_plain_qa_no_risk_flags(
        self, gen: TaskVectorGenerator, extractor: FeatureExtractor
    ) -> None:
        _, _, risk = _extract_and_generate(
            "Tell me something interesting", extractor, gen
        )
        assert risk.needs_json is False
        assert risk.needs_high_accuracy is False
        assert risk.strict_formatting is False


class TestToDict:
    """to_dict() should return plain dicts."""

    def test_task_vector_to_dict(self, gen: TaskVectorGenerator) -> None:
        fv = FeatureVector(task_type="math", contains_math=True, complexity=0.5)
        tv, _, _ = gen.generate(fv)
        d = tv.to_dict()
        assert isinstance(d, dict)
        assert set(d.keys()) == {
            "math", "reasoning", "code", "creative",
            "translation", "extraction", "retrieval", "general_qa",
            "summarization", "sentiment", "ner",
        }

    def test_resource_vector_to_dict(self, gen: TaskVectorGenerator) -> None:
        fv = FeatureVector()
        _, rv, _ = gen.generate(fv)
        d = rv.to_dict()
        assert isinstance(d, dict)
        assert "expected_input_tokens" in d

    def test_risk_vector_to_dict(self, gen: TaskVectorGenerator) -> None:
        fv = FeatureVector()
        _, _, risk = gen.generate(fv)
        d = risk.to_dict()
        assert isinstance(d, dict)
        assert "needs_json" in d
