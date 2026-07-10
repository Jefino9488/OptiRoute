"""Escalation Policy — governs when and how to retry with a stronger model.

When the Confidence Validator judges a response as low-confidence, the
escalation policy decides:
1. Whether to escalate at all (depth budget).
2. Which model to try next (next-cheapest eligible model).
3. When to give up and return the best response seen so far.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Frontier fallback — used when all else fails.
_FALLBACK_MODEL: str = "minimax-m3"


class EscalationPolicy:
    """Manage escalation decisions during the routing lifecycle.

    Parameters
    ----------
    max_depth : int
        Maximum number of escalation steps allowed per request (default 2).
    confidence_threshold : float
        Minimum confidence score for a response to be accepted without
        escalation (default 0.7).
    """

    def __init__(
        self,
        max_depth: int = 2,
        confidence_threshold: float = 0.7,
    ) -> None:
        self.max_depth: int = max_depth
        self.confidence_threshold: float = confidence_threshold

    def should_escalate(self, confidence: float, current_depth: int) -> bool:
        """Decide whether to escalate to a more capable (costlier) model.

        Escalation is triggered when:
        * ``confidence`` is below ``confidence_threshold``, **and**
        * ``current_depth`` is strictly less than ``max_depth``.

        Parameters
        ----------
        confidence : float
            Confidence score of the current response (0–1).
        current_depth : int
            How many escalation steps have already been performed (0-based).

        Returns
        -------
        bool
            ``True`` if the system should try a stronger model.
        """
        if current_depth >= self.max_depth:
            logger.info(
                "escalation.depth_exhausted",
                current_depth=current_depth,
                max_depth=self.max_depth,
            )
            return False

        should = confidence < self.confidence_threshold
        logger.info(
            "escalation.decision",
            confidence=confidence,
            threshold=self.confidence_threshold,
            current_depth=current_depth,
            should_escalate=should,
        )
        return should

    def get_next_model(
        self,
        current_model: str,
        eligible_models: list[dict[str, Any]],
        current_depth: int,
    ) -> str | None:
        """Select the next model to try after *current_model*.

        The *eligible_models* list is expected to be sorted by estimated cost
        (ascending).  This method finds *current_model* in that list and returns
        the next entry.  If *current_model* is not found (e.g. it was the
        fallback), the first eligible model is returned.

        Parameters
        ----------
        current_model : str
            The model that just produced a low-confidence response.
        eligible_models : list[dict]
            Models that passed accuracy filtering, sorted cheapest-first.
            Each dict must have at least a ``"model_id"`` key.
        current_depth : int
            Current escalation depth (0-based).

        Returns
        -------
        str | None
            Next model ID to try, or ``None`` if the depth budget is exhausted
            or there is no stronger model available.
        """
        if current_depth >= self.max_depth:
            logger.info(
                "escalation.max_depth_reached",
                current_depth=current_depth,
                max_depth=self.max_depth,
            )
            return None

        # Locate current model in the cost-sorted list.
        model_ids = [m["model_id"] for m in eligible_models]
        try:
            idx = model_ids.index(current_model)
        except ValueError:
            # current_model not in eligible list (e.g. was a fallback);
            # start from the beginning.
            logger.warning(
                "escalation.current_model_not_in_eligible",
                current_model=current_model,
            )
            idx = -1

        next_idx = idx + 1
        if next_idx < len(model_ids):
            next_model = model_ids[next_idx]
            logger.info(
                "escalation.next_model",
                current_model=current_model,
                next_model=next_model,
                depth=current_depth,
            )
            return next_model

        logger.info(
            "escalation.no_stronger_model",
            current_model=current_model,
        )
        return None

    @staticmethod
    def get_fallback_model() -> str:
        """Return the frontier fallback model ID.

        This is always ``minimax-m3`` — the most capable (and most expensive)
        model, used as the last resort when no eligible model meets the
        accuracy threshold.

        Returns
        -------
        str
            ``"minimax-m3"``
        """
        return _FALLBACK_MODEL
