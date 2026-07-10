"""Decision Engine — core routing algorithm for OptiRoute.

Answers: "Among the allowed Fireworks models, which one is the cheapest that
can still solve this task accurately?"

The algorithm proceeds in six steps:
1. Identify dominant task type and filter out failure models
2. Weighted-accuracy scoring per candidate
3. Accuracy threshold filtering
4. Cost estimation (with Kimi's 1.3× output multiplier)
5. Sort by cost, pick cheapest
6. Fallback to frontier (minimax-m3) if nothing qualifies
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import structlog

from app.config import get_settings
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
    """

    model_selected: str
    estimated_cost: float
    predicted_accuracy: float
    reasoning: str
    alternatives_considered: list[dict[str, Any]] = field(default_factory=list)

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
        # Set of model IDs confirmed callable at runtime (empty = no filtering)
        self._available_models: set[str] = set()

    def set_available_models(self, available: set[str]) -> None:
        """Update the set of models confirmed callable at runtime.

        Call this after probing the Fireworks /models endpoint at startup.
        Models not in this set are excluded from routing decisions.
        Local models (prefix ``local:``) are always included regardless.

        Parameters
        ----------
        available : set[str]
            Model IDs confirmed available (short IDs, e.g. ``"minimax-m3"``).
        """
        self._available_models = available
        logger.info(
            "decision_engine.available_models_updated",
            count=len(available),
            models=sorted(available),
        )

    @property
    def fallback_model(self) -> str:
        """Return the best available Fireworks model for use as fallback.

        Selects the model with the highest ``general_qa`` capability score
        from the currently available (probed) set. Falls back to
        ``minimax-m3`` if nothing is available.
        """
        fireworks_available = {
            m for m in self._available_models if not m.startswith("local:")
        }
        if not fireworks_available:
            return _FALLBACK_MODEL

        best: str = _FALLBACK_MODEL
        best_score: float = -1.0
        for model_id in fireworks_available:
            entry = self._matrix.get_or_create_model_capabilities(model_id)
            score = entry.get("capabilities", {}).get("general_qa", 0.0)
            if score > best_score:
                best_score = score
                best = model_id
        return best

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_model(
        self,
        task_vector: dict[str, float],
        resource_vector: dict[str, Any],
        risk_vector: dict[str, Any],
        required_accuracy: float = 0.8,
        prompt: str = "",
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
        dominant_task = self._get_dominant_task(task_vector)
        settings = get_settings()
        allowed_models = set(settings.allowed_models.keys())

        # Combine: matrix models + any models probed at runtime that are allowed
        matrix_models = set(self._matrix.get_all_models())
        # Add any allowed models not yet in matrix (generate defaults on access)
        all_candidate_ids = (allowed_models | matrix_models) | {
            m for m in self._available_models if not m.startswith("local:")
        }
        # Always include local models
        local_models = {m for m in matrix_models if m.startswith("local:")}
        all_candidate_ids |= local_models

        # Filter by allowed + runtime-available (local always passes)
        candidates: list[str] = []
        for model_id in all_candidate_ids:
            is_local = model_id.startswith("local:")
            in_allowed = model_id in allowed_models
            in_available = model_id in self._available_models
            # Accept if: local model, OR (in allowed_models AND (available or not yet probed))
            if is_local or (in_allowed and (in_available or not self._available_models)):
                candidates.append(model_id)

        logger.info(
            "decision_engine.filtering",
            dominant_task=dominant_task,
            candidates=candidates,
            available_probed=len(self._available_models),
        )

        # Step 1: Filter out models that are known to fail on the dominant task
        filtered_candidates: list[str] = []
        for model_id in candidates:
            entry = self._matrix.get_or_create_model_capabilities(model_id)
            fails_on: list[str] = entry.get("fails_on", [])
            if dominant_task in fails_on:
                logger.info(
                    "decision_engine.skipped_fails_on",
                    model=model_id,
                    dominant_task=dominant_task,
                )
                continue
            filtered_candidates.append(model_id)

        # Step 2 + 3: Compute weighted accuracy and filter by threshold
        complexity = resource_vector.get("complexity", 0.0)
        eligible: list[dict[str, Any]] = []
        for model_id in filtered_candidates:
            entry = self._matrix.get_or_create_model_capabilities(model_id)
            accuracy = self._compute_weighted_accuracy(
                task_vector, entry.get("capabilities", {})
            )
            # Apply a complexity cliff penalty for local 3B models.
            # At complexity > 0.5 these models saturate their working memory
            # and fail on multi-hop arithmetic and compositional reasoning.
            if model_id.startswith("local:") and complexity > 0.5:
                penalty = 0.75 - 0.5 * max(0.0, complexity - 0.5)  # 0.75x at 0.5, 0.5x at 1.0
                accuracy *= max(penalty, 0.50)
                logger.info(
                    "decision_engine.local_complexity_penalty",
                    model=model_id,
                    complexity=complexity,
                    penalty_factor=round(max(penalty, 0.50), 3),
                    adjusted_accuracy=round(accuracy, 4),
                )
            # Apply cost-accuracy tradeoff:
            # For free models (cost=0), if they are reliable at the dominant task (>0.75),
            # tolerate a slightly lower weighted accuracy (up to 15% drop due to secondary tasks).
            is_free = (entry.get("cost_per_1k_input", 1.0) == 0.0 and entry.get("cost_per_1k_output", 1.0) == 0.0)
            dominant_cap = entry.get("capabilities", {}).get(dominant_task, 0.0)
            
            is_eligible = accuracy >= required_accuracy
            if not is_eligible and is_free and dominant_cap > 0.75:
                if accuracy >= required_accuracy * 0.85:
                    is_eligible = True
                    logger.info(
                        "decision_engine.cost_accuracy_tradeoff",
                        model=model_id,
                        dominant_task=dominant_task,
                        dominant_cap=dominant_cap,
                        weighted_accuracy=round(accuracy, 4),
                        required=round(required_accuracy, 4)
                    )

            if is_eligible:
                eligible.append(
                    {
                        "model_id": model_id,
                        "predicted_accuracy": round(accuracy, 6),
                        "cost_per_1k_input": entry["cost_per_1k_input"],
                        "cost_per_1k_output": entry["cost_per_1k_output"],
                        "avg_output_multiplier": entry.get("avg_output_multiplier", 1.0),
                    }
                )

        # Step 4: Estimate cost for each eligible model
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

        # Step 5: Sort by cost ascending, pick cheapest
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

        # Step 6: Fallback — no model met the threshold
        # Use the dynamically selected best available model, not a hardcoded one
        fb_model = self.fallback_model
        logger.warning(
            "decision_engine.fallback",
            required_accuracy=required_accuracy,
            dominant_task=dominant_task,
            fallback_model=fb_model,
        )
        fallback_entry = self._matrix.get_or_create_model_capabilities(fb_model)
        fallback_accuracy = self._compute_weighted_accuracy(
            task_vector, fallback_entry.get("capabilities", {})
        )
        fallback_cost = self._matrix.estimate_cost(
            fb_model, input_tokens, output_tokens
        ) or 0.0

        return RoutingDecision(
            model_selected=fb_model,
            estimated_cost=round(fallback_cost, 8),
            predicted_accuracy=round(fallback_accuracy, 6),
            reasoning=(
                f"No model met accuracy threshold ({required_accuracy}) for "
                f"dominant task '{dominant_task}'. Falling back to best available "
                f"model {fb_model}."
            ),
            alternatives_considered=[],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        alternatives: list[dict[str, Any]],
    ) -> str:
        """Build a human-readable routing explanation string."""
        parts: list[str] = [
            f"Task is primarily {dominant_task} ({winner.get('predicted_accuracy', 0):.2f} predicted accuracy).",
            f"{winner['model_id']} meets accuracy threshold "
            f"({winner.get('predicted_accuracy', 0):.2f} >= {required_accuracy:.2f}) "
            f"at lowest cost (${winner.get('estimated_cost', 0):.8f}).",
        ]
        if alternatives:
            alt_strs = [
                f"{a['model']} (${a['cost']:.8f})" for a in alternatives[:3]
            ]
            parts.append(f"Preferred over {', '.join(alt_strs)}.")
        return " ".join(parts)
