"""Decision Engine — core routing algorithm for OptiRoute.

Answers: "Among the allowed Fireworks models, which one is the cheapest that
can still solve this task accurately?"

The algorithm proceeds in seven steps:
1. Deterministic bypass (math/json tools at $0 cost)
2. Failure filtering (exclude models known to fail on dominant task)
3. Weighted-accuracy scoring per candidate
4. Accuracy threshold filtering
5. Cost estimation (with Kimi's 1.3× output multiplier)
6. Sort by cost, pick cheapest
7. Fallback to frontier (minimax-m3) if nothing qualifies
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import structlog

from app.router.capability_matrix import CapabilityMatrix

logger = structlog.get_logger(__name__)

# Default token estimates when resource_vector doesn't specify them.
_DEFAULT_INPUT_TOKENS: int = 500
_DEFAULT_OUTPUT_TOKENS: int = 300

# Frontier fallback model
_FALLBACK_MODEL: str = "minimax-m3"


@dataclass
class RoutingDecision:
    """Immutable result of a routing decision.

    Attributes
    ----------
    model_selected : str
        Model ID (or ``"deterministic:<tool>"`` for tool bypass).
    estimated_cost : float
        Estimated USD cost for this inference call.
    predicted_accuracy : float
        Weighted accuracy prediction for the selected model on this task.
    reasoning : str
        Human-readable explanation of the routing choice.
    alternatives_considered : list[dict]
        Other eligible models that were considered, sorted by cost.
    is_deterministic : bool
        ``True`` when a deterministic tool handles the request ($0 cost).
    """

    model_selected: str
    estimated_cost: float
    predicted_accuracy: float
    reasoning: str
    alternatives_considered: list[dict[str, Any]] = field(default_factory=list)
    is_deterministic: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the decision to a plain dict."""
        return asdict(self)


class DecisionEngine:
    """Core routing algorithm — selects the cheapest capable model.

    Parameters
    ----------
    capability_matrix : CapabilityMatrix
        Loaded capability matrix instance.
    """

    def __init__(self, capability_matrix: CapabilityMatrix) -> None:
        self._matrix: CapabilityMatrix = capability_matrix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_model(
        self,
        task_vector: dict[str, float],
        resource_vector: dict[str, Any],
        risk_vector: dict[str, Any],
        required_accuracy: float = 0.8,
    ) -> RoutingDecision:
        """Run the 7-step routing algorithm and return a decision.

        Parameters
        ----------
        task_vector : dict[str, float]
            Dimension → weight describing what the task needs.
        resource_vector : dict[str, Any]
            Token estimates and complexity info.  Expected keys:
            ``input_tokens``, ``output_tokens``, ``complexity``.
        risk_vector : dict[str, Any]
            Risk flags (e.g. ``needs_json``, ``needs_code_execution``).
        required_accuracy : float
            Minimum predicted accuracy to accept a model (default 0.8).

        Returns
        -------
        RoutingDecision
        """
        # Step 1: Deterministic bypass
        deterministic = self._check_deterministic(task_vector, resource_vector, risk_vector)
        if deterministic is not None:
            logger.info("decision_engine.deterministic_bypass", tool=deterministic)
            return RoutingDecision(
                model_selected=deterministic,
                estimated_cost=0.0,
                predicted_accuracy=1.0,
                reasoning=f"Task handled by deterministic tool ({deterministic}). No LLM needed — $0 cost.",
                is_deterministic=True,
            )

        # Step 2: Identify dominant task type and filter out failure models
        dominant_task = self._get_dominant_task(task_vector)
        failure_models = self._matrix.get_failure_models(dominant_task)
        all_models = self._matrix.get_all_models()
        candidates = [m for m in all_models if m not in failure_models]

        logger.info(
            "decision_engine.filtering",
            dominant_task=dominant_task,
            excluded=failure_models,
            candidates=candidates,
        )

        # Step 3 + 4: Compute weighted accuracy and filter by threshold
        eligible: list[dict[str, Any]] = []
        for model_id in candidates:
            entry = self._matrix.get_model_capabilities(model_id)
            if entry is None:
                continue
            accuracy = self._compute_weighted_accuracy(
                task_vector, entry.get("capabilities", {})
            )
            if accuracy >= required_accuracy:
                eligible.append(
                    {
                        "model_id": model_id,
                        "predicted_accuracy": round(accuracy, 6),
                        "cost_per_1k_input": entry["cost_per_1k_input"],
                        "cost_per_1k_output": entry["cost_per_1k_output"],
                        "avg_output_multiplier": entry.get("avg_output_multiplier", 1.0),
                    }
                )

        # Step 5: Estimate cost for each eligible model
        input_tokens = resource_vector.get("input_tokens", _DEFAULT_INPUT_TOKENS)
        output_tokens = resource_vector.get("output_tokens", _DEFAULT_OUTPUT_TOKENS)

        for model in eligible:
            model["estimated_cost"] = (
                input_tokens * model["cost_per_1k_input"] / 1000
                + output_tokens
                * model["avg_output_multiplier"]
                * model["cost_per_1k_output"]
                / 1000
            )

        # Step 6: Sort by cost ascending, pick cheapest
        eligible.sort(key=lambda m: m["estimated_cost"])

        if eligible:
            winner = eligible[0]
            alternatives = [
                {
                    "model": m["model_id"],
                    "cost": round(m["estimated_cost"], 8),
                    "accuracy": m["predicted_accuracy"],
                }
                for m in eligible[1:]
            ]
            reasoning = self._build_explanation(
                winner=winner,
                dominant_task=dominant_task,
                required_accuracy=required_accuracy,
                excluded=failure_models,
                alternatives=alternatives,
            )
            decision = RoutingDecision(
                model_selected=winner["model_id"],
                estimated_cost=round(winner["estimated_cost"], 8),
                predicted_accuracy=winner["predicted_accuracy"],
                reasoning=reasoning,
                alternatives_considered=alternatives,
            )
            logger.info(
                "decision_engine.selected",
                model=decision.model_selected,
                cost=decision.estimated_cost,
                accuracy=decision.predicted_accuracy,
            )
            return decision

        # Step 7: Fallback — no model met the threshold
        logger.warning(
            "decision_engine.fallback",
            required_accuracy=required_accuracy,
            dominant_task=dominant_task,
        )
        fallback_entry = self._matrix.get_model_capabilities(_FALLBACK_MODEL)
        fallback_accuracy = 0.0
        if fallback_entry:
            fallback_accuracy = self._compute_weighted_accuracy(
                task_vector, fallback_entry.get("capabilities", {})
            )
        fallback_cost = self._matrix.estimate_cost(
            _FALLBACK_MODEL, input_tokens, output_tokens
        ) or 0.0

        return RoutingDecision(
            model_selected=_FALLBACK_MODEL,
            estimated_cost=round(fallback_cost, 8),
            predicted_accuracy=round(fallback_accuracy, 6),
            reasoning=(
                f"No model met accuracy threshold ({required_accuracy}) for "
                f"dominant task '{dominant_task}'. Falling back to frontier model "
                f"{_FALLBACK_MODEL}."
            ),
            alternatives_considered=[],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_deterministic(
        task_vector: dict[str, float],
        resource_vector: dict[str, Any],
        risk_vector: dict[str, Any],
    ) -> str | None:
        """Return a deterministic tool name if the task qualifies, else None.

        Conditions:
        * Simple math: ``task_vector['math'] > 0.9`` **and**
          ``resource_vector['complexity'] < 0.3``.
        * JSON extraction: ``risk_vector['needs_json']`` is truthy **and**
          ``task_vector['extraction'] > 0.8``.
        """
        math_weight = task_vector.get("math", 0.0)
        complexity = resource_vector.get("complexity", 1.0)
        if math_weight > 0.9 and complexity < 0.3:
            return "deterministic:calculator"

        if risk_vector.get("needs_json") and task_vector.get("extraction", 0.0) > 0.8:
            return "deterministic:json"

        return None

    @staticmethod
    def _compute_weighted_accuracy(
        task_vector: dict[str, float],
        model_capabilities: dict[str, float],
    ) -> float:
        """Weighted average accuracy: ``Σ(w_i * c_i) / Σ(w_i)``.

        Dimensions with weight ≤ 0 are ignored. Returns 0.0 if all weights
        are zero.
        """
        numerator = 0.0
        denominator = 0.0
        for dim, weight in task_vector.items():
            if weight <= 0:
                continue
            cap = model_capabilities.get(dim, 0.0)
            numerator += weight * cap
            denominator += weight
        return numerator / denominator if denominator > 0 else 0.0

    @staticmethod
    def _get_dominant_task(task_vector: dict[str, float]) -> str:
        """Return the dimension name with the highest weight.

        Falls back to ``"general_qa"`` if the vector is empty.
        """
        if not task_vector:
            return "general_qa"
        return max(task_vector, key=task_vector.get)  # type: ignore[arg-type]

    @staticmethod
    def _build_explanation(
        *,
        winner: dict[str, Any],
        dominant_task: str,
        required_accuracy: float,
        excluded: list[str],
        alternatives: list[dict[str, Any]],
    ) -> str:
        """Build a human-readable routing explanation string."""
        parts: list[str] = [
            f"Task is primarily {dominant_task} ({winner.get('predicted_accuracy', 0):.2f} predicted accuracy).",
            f"{winner['model_id']} meets accuracy threshold "
            f"({winner.get('predicted_accuracy', 0):.2f} >= {required_accuracy:.2f}) "
            f"at lowest cost (${winner.get('estimated_cost', 0):.8f}).",
        ]
        if excluded:
            parts.append(
                f"Excluded {', '.join(excluded)} (fails_on includes {dominant_task})."
            )
        if alternatives:
            alt_strs = [
                f"{a['model']} (${a['cost']:.8f})" for a in alternatives[:3]
            ]
            parts.append(f"Preferred over {', '.join(alt_strs)}.")
        return " ".join(parts)
