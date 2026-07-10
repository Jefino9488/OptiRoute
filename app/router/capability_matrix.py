"""Capability Matrix — loads, queries, and updates model capability data.

The capability matrix is a JSON file describing each allowed model's per-dimension
accuracy scores, cost parameters, context limits, and known failure patterns. It is
the single source of truth for all routing decisions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# All recognised task dimensions — order does not matter for scoring,
# but we keep a canonical list for validation.
TASK_DIMENSIONS: list[str] = [
    "math",
    "reasoning",
    "code",
    "creative",
    "translation",
    "extraction",
    "retrieval",
    "general_qa",
    "summarization",
    "sentiment",
    "ner",
]


class CapabilityMatrix:
    """Load, query, and mutate the model capability matrix.

    Parameters
    ----------
    matrix_path : str
        Absolute or relative path to the ``capability_matrix.json`` file.
    """

    def __init__(self, matrix_path: str) -> None:
        self._path: Path = Path(matrix_path)
        self._data: dict[str, dict[str, Any]] = {}
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Read and parse the capability matrix JSON file.

        Raises
        ------
        FileNotFoundError
            If the matrix file does not exist at the configured path.
        json.JSONDecodeError
            If the file contains invalid JSON.
        """
        logger.info("capability_matrix.load", path=str(self._path))
        with self._path.open("r", encoding="utf-8") as fh:
            self._data = json.load(fh)
        logger.info(
            "capability_matrix.loaded",
            models=list(self._data.keys()),
            count=len(self._data),
        )

    def save(self) -> None:
        """Write the current in-memory matrix back to the JSON file."""
        logger.info("capability_matrix.save", path=str(self._path))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)
        logger.info("capability_matrix.saved", models=len(self._data))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_model_capabilities(self, model_id: str) -> dict[str, Any] | None:
        """Return the full entry for *model_id*, or ``None`` if unknown.

        Parameters
        ----------
        model_id : str
            Short model identifier (e.g. ``"gemma-4-31b-it"``).

        Returns
        -------
        dict | None
            Model entry including capabilities, costs, samples, etc.
        """
        entry = self._data.get(model_id)
        if entry is None:
            logger.warning("capability_matrix.model_not_found", model_id=model_id)
        return entry

    def get_or_create_model_capabilities(
        self,
        model_id: str,
    ) -> dict[str, Any]:
        """Return capabilities for *model_id*, generating defaults if unknown.

        If the model is not in the matrix, a conservative default profile
        is auto-generated based on naming heuristics (e.g. 'code' in name
        boosts code capability).

        Parameters
        ----------
        model_id : str
            Short model identifier.

        Returns
        -------
        dict
            Model entry (may be auto-generated).
        """
        entry = self._data.get(model_id)
        if entry is not None:
            return entry

        # Generate conservative defaults
        default = self._generate_default_entry(model_id)
        self._data[model_id] = default
        logger.info(
            "capability_matrix.generated_default",
            model_id=model_id,
            capabilities=default["capabilities"],
        )
        return default

    def _generate_default_entry(self, model_id: str) -> dict[str, Any]:
        """Create a conservative capability profile for an unknown model.

        Uses naming heuristics:
        - ``"code"`` in name → boost code capability
        - ``"mini"`` / small param indicators → lower general scores
        - Default cost: mid-range estimate between minimax and kimi
        """
        name_lower = model_id.lower()

        # Start with conservative baseline (below benchmarked models)
        base_score = 0.65
        caps: dict[str, float] = {dim: base_score for dim in TASK_DIMENSIONS}

        # Heuristic boosts from model name
        if "code" in name_lower:
            caps["code"] = 0.85
            caps["reasoning"] = 0.80
        if "math" in name_lower or "reason" in name_lower:
            caps["math"] = 0.80
            caps["reasoning"] = 0.82
        if "mini" in name_lower or "small" in name_lower:
            # Smaller models get slightly lower scores
            caps = {k: min(v, 0.60) for k, v in caps.items()}

        # Cost defaults — mid-range (between minimax and kimi)
        cost_input = 0.0005
        cost_output = 0.002
        if "nvfp4" in name_lower or "quant" in name_lower:
            cost_input *= 0.7  # quantized models are typically cheaper
            cost_output *= 0.7

        return {
            "capabilities": caps,
            "cost_per_1k_input": cost_input,
            "cost_per_1k_output": cost_output,
            "avg_output_multiplier": 1.0,
            "max_context": 128000,
            "fails_on": [],
            "supports_thinking": False,
            "thinking_cost_multiplier": 1.0,
            "samples": {},
            "avg_latency_ms": 3000.0,
            "_auto_generated": True,
        }

    def get_all_models(self) -> list[str]:
        """Return a list of all model IDs in the matrix."""
        return list(self._data.keys())

    def supports_thinking(self, model_id: str) -> bool:
        """Check if a model supports Fireworks reasoning_effort parameter."""
        entry = self._data.get(model_id)
        if entry is None:
            return False
        return entry.get("supports_thinking", False)

    def get_thinking_cost_multiplier(self, model_id: str) -> float:
        """Return the cost multiplier when thinking is enabled for a model."""
        entry = self._data.get(model_id)
        if entry is None:
            return 1.0
        return entry.get("thinking_cost_multiplier", 1.0)


    def get_capable_models(
        self,
        task_vector: dict[str, float],
        min_accuracy: float,
    ) -> list[dict[str, Any]]:
        """Return models whose weighted accuracy meets *min_accuracy*.

        Weighted accuracy is defined as::

            sum(task[d] * cap[d] for d in dims) / sum(task[d] for d in dims)

        This focuses scoring on the dimensions the task actually needs, preventing
        models from being rewarded for strengths in irrelevant areas.

        Parameters
        ----------
        task_vector : dict[str, float]
            Mapping of dimension → weight (0–1) describing the task.
        min_accuracy : float
            Minimum predicted accuracy to include a model.

        Returns
        -------
        list[dict]
            Each dict has ``model_id``, ``predicted_accuracy``, ``cost_per_1k_input``,
            ``cost_per_1k_output``, ``avg_output_multiplier``.
        """
        results: list[dict[str, Any]] = []
        for model_id, entry in self._data.items():
            accuracy = self._compute_weighted_accuracy(
                task_vector, entry.get("capabilities", {})
            )
            if accuracy >= min_accuracy:
                results.append(
                    {
                        "model_id": model_id,
                        "predicted_accuracy": round(accuracy, 6),
                        "cost_per_1k_input": entry["cost_per_1k_input"],
                        "cost_per_1k_output": entry["cost_per_1k_output"],
                        "avg_output_multiplier": entry.get("avg_output_multiplier", 1.0),
                        "supports_thinking": entry.get("supports_thinking", False),
                        "thinking_cost_multiplier": entry.get("thinking_cost_multiplier", 1.0),
                    }
                )
        return results

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update_model(
        self,
        model_id: str,
        capabilities: dict[str, float],
        samples: dict[str, int],
    ) -> None:
        """Update (or create) a model entry with new capability scores.

        Only the ``capabilities`` and ``samples`` fields are overwritten; cost
        and context fields are preserved if the model already exists.

        Parameters
        ----------
        model_id : str
            Short model identifier.
        capabilities : dict[str, float]
            Per-dimension accuracy scores.
        samples : dict[str, int]
            Number of benchmark samples per dimension.
        """
        existing = self._data.get(model_id, {})
        existing["capabilities"] = capabilities
        existing["samples"] = samples
        self._data[model_id] = existing
        logger.info(
            "capability_matrix.model_updated",
            model_id=model_id,
            dims=list(capabilities.keys()),
            samples=samples,
        )

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        thinking_enabled: bool = False,
    ) -> float | None:
        """Estimate inference cost in USD for the given token counts.

        The formula accounts for Kimi's mandatory thinking overhead via
        ``avg_output_multiplier`` and optional thinking cost via
        ``thinking_cost_multiplier``::

            cost = input_tokens * cost_per_1k_input / 1000
                 + output_tokens * avg_output_multiplier * cost_per_1k_output / 1000
                 * (thinking_cost_multiplier if thinking_enabled else 1.0)

        Parameters
        ----------
        model_id : str
            Short model identifier.
        input_tokens : int
            Estimated number of input tokens.
        output_tokens : int
            Estimated number of output tokens (before multiplier).
        thinking_enabled : bool
            Whether thinking/reasoning is enabled for this request.

        Returns
        -------
        float | None
            Estimated cost in USD, or ``None`` if the model is not found.
        """
        entry = self._data.get(model_id)
        if entry is None:
            logger.warning(
                "capability_matrix.estimate_cost.model_not_found",
                model_id=model_id,
            )
            return None

        multiplier = entry.get("avg_output_multiplier", 1.0)
        thinking_mult = entry.get("thinking_cost_multiplier", 1.0) if thinking_enabled else 1.0
        cost = (
            input_tokens * entry["cost_per_1k_input"] / 1000
            + output_tokens * multiplier * thinking_mult * entry["cost_per_1k_output"] / 1000
        )
        return cost

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_weighted_accuracy(
        task_vector: dict[str, float],
        model_capabilities: dict[str, float],
    ) -> float:
        """Compute weighted average accuracy.

        Parameters
        ----------
        task_vector : dict[str, float]
            Dimension → weight for the current task.
        model_capabilities : dict[str, float]
            Dimension → capability score for the model.

        Returns
        -------
        float
            Weighted accuracy in [0, 1]. Returns 0.0 if all weights are zero.
        """
        numerator = 0.0
        denominator = 0.0
        for dim, weight in task_vector.items():
            if weight <= 0:
                continue
            cap = model_capabilities.get(dim, 0.0)
            numerator += weight * cap
            denominator += weight
        if denominator == 0.0:
            return 0.0
        return numerator / denominator
