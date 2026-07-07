"""Task vector generator — maps extracted features to numeric vectors.

Produces three vectors consumed by the decision engine:

* **TaskVector** — per-dimension task affinity scores in [0, 1].
* **ResourceVector** — estimated token / context requirements.
* **RiskVector** — boolean risk flags for deterministic / strict tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from app.features.extractor import FeatureVector


# ---------------------------------------------------------------------------
# Vector dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TaskVector:
    """Per-dimension task affinity in [0, 1].

    Each dimension corresponds to a capability column in the capability
    matrix.
    """

    math: float = 0.0
    reasoning: float = 0.0
    code: float = 0.0
    creative: float = 0.0
    translation: float = 0.0
    extraction: float = 0.0
    retrieval: float = 0.0
    general_qa: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialise to a plain dict."""
        return asdict(self)


@dataclass(slots=True)
class ResourceVector:
    """Estimated resource requirements for executing the task."""

    expected_input_tokens: float = 0.0
    expected_output_tokens: float = 0.0
    expected_context_length: float = 0.0
    complexity: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialise to a plain dict."""
        return asdict(self)


@dataclass(slots=True)
class RiskVector:
    """Boolean risk flags driving model filtering and validation."""

    needs_json: bool = False
    needs_deterministic: bool = False
    needs_high_accuracy: bool = False
    strict_formatting: bool = False

    def to_dict(self) -> dict[str, bool]:
        """Serialise to a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Dimension ↔ task-type mapping
# ---------------------------------------------------------------------------

_DIMENSIONS: list[str] = [
    "math", "reasoning", "code", "creative",
    "translation", "extraction", "retrieval", "general_qa",
]

# Which auxiliary dimensions get a moderate "related" boost for each
# primary task type.
_RELATED: dict[str, list[str]] = {
    "math": ["reasoning"],
    "code": ["reasoning", "extraction"],
    "reasoning": ["general_qa"],
    "creative": ["general_qa"],
    "translation": ["general_qa"],
    "extraction": ["reasoning"],
    "retrieval": ["general_qa"],
    "general_qa": ["reasoning", "retrieval"],
}

# Base output-token estimates per length category.
_OUTPUT_TOKENS: dict[str, float] = {
    "short": 50.0,
    "medium": 200.0,
    "long": 500.0,
}


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class TaskVectorGenerator:
    """Convert a :class:`FeatureVector` into the three routing vectors."""

    def generate(
        self,
        features: FeatureVector,
    ) -> tuple[TaskVector, ResourceVector, RiskVector]:
        """Build task, resource, and risk vectors from *features*.

        Args:
            features: The extracted feature vector for a prompt.

        Returns:
            A 3-tuple ``(task_vector, resource_vector, risk_vector)``.
        """
        task_vec = self._build_task_vector(features)
        resource_vec = self._build_resource_vector(features)
        risk_vec = self._build_risk_vector(features)
        return task_vec, resource_vec, risk_vec

    # -- private builders ---------------------------------------------------

    @staticmethod
    def _build_task_vector(features: FeatureVector) -> TaskVector:
        """Map feature flags to per-dimension affinity scores."""
        complexity = features.complexity
        dominant = features.task_type

        # Base values: dominant high, related moderate, others low.
        high = 0.7 + 0.25 * complexity   # 0.70 – 0.95
        moderate = 0.25 + 0.20 * complexity  # 0.25 – 0.45
        low = 0.05

        values: dict[str, float] = {}
        related_dims = _RELATED.get(dominant, [])

        for dim in _DIMENSIONS:
            if dim == dominant:
                values[dim] = high
            elif dim in related_dims:
                values[dim] = moderate
            else:
                values[dim] = low

        # Bonus bumps for secondary signals.
        if features.contains_code and dominant != "code":
            values["code"] = max(values["code"], moderate)
        if features.contains_math and dominant != "math":
            values["math"] = max(values["math"], moderate)
        if features.requires_reasoning and dominant != "reasoning":
            values["reasoning"] = max(values["reasoning"], moderate)
        if features.is_creative and dominant != "creative":
            values["creative"] = max(values["creative"], moderate)
        if features.is_translation and dominant != "translation":
            values["translation"] = max(values["translation"], moderate)
        if features.requires_retrieval and dominant != "retrieval":
            values["retrieval"] = max(values["retrieval"], moderate)
        if features.json_required and dominant != "extraction":
            values["extraction"] = max(values["extraction"], moderate)

        # Clamp all values to [0, 1].
        for dim in _DIMENSIONS:
            values[dim] = round(min(max(values[dim], 0.0), 1.0), 4)

        return TaskVector(**values)

    @staticmethod
    def _build_resource_vector(features: FeatureVector) -> ResourceVector:
        """Estimate token / context requirements."""
        input_tokens = features.input_length * 1.3
        base_output = _OUTPUT_TOKENS.get(features.expected_output_length, 200.0)

        # Code and creative tasks typically produce longer outputs.
        if features.contains_code:
            base_output *= 1.5
        if features.is_creative:
            base_output *= 1.3

        context_length = input_tokens + base_output

        return ResourceVector(
            expected_input_tokens=round(input_tokens, 1),
            expected_output_tokens=round(base_output, 1),
            expected_context_length=round(context_length, 1),
            complexity=features.complexity,
        )

    @staticmethod
    def _build_risk_vector(features: FeatureVector) -> RiskVector:
        """Derive boolean risk flags from feature booleans."""
        return RiskVector(
            needs_json=features.json_required,
            needs_deterministic=features.contains_math or features.json_required,
            needs_high_accuracy=features.contains_code or features.contains_math,
            strict_formatting=features.json_required,
        )
