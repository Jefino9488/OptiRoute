"""Routing pipeline — orchestrates the full request lifecycle.

Flow:
    Normalise → Cache check → Extract features → Build vectors →
    Decision engine → Execute → Validate confidence → Escalate if needed →
    Cache result → Log metrics → Return response.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import structlog

from app.cache.manager import CacheManager, CachedResponse
from app.confidence.validator import ConfidenceValidator
from app.config import get_settings
from app.executors.base import ExecutionResult
from app.executors.fireworks import FireworksExecutor
from app.executors.tools import DeterministicExecutor
from app.features.extractor import FeatureExtractor
from app.features.normalizer import RequestNormalizer
from app.features.vectorizer import TaskVectorGenerator
from app.metrics.collector import MetricsCollector, RequestMetric
from app.router.capability_matrix import CapabilityMatrix
from app.router.decision_engine import DecisionEngine
from app.router.policy import EscalationPolicy

logger = structlog.get_logger(__name__)


class RoutingPipeline:
    """End-to-end routing pipeline.

    Creates and manages all components.  Designed to be instantiated
    once at application startup and reused for every request.
    """

    def __init__(self) -> None:
        settings = get_settings()

        # Feature extraction
        self._normalizer = RequestNormalizer()
        self._extractor = FeatureExtractor()
        self._vectorizer = TaskVectorGenerator()

        # Routing
        self._matrix = CapabilityMatrix(settings.capability_matrix_path)
        self._engine = DecisionEngine(self._matrix)
        self._policy = EscalationPolicy(
            max_depth=settings.max_escalation_depth,
            confidence_threshold=settings.confidence_threshold,
        )

        # Execution
        self._fireworks = FireworksExecutor()
        self._deterministic = DeterministicExecutor()

        # Support
        self._cache = CacheManager()
        self._validator = ConfidenceValidator()
        self._metrics = MetricsCollector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route(
        self,
        prompt: str,
        required_accuracy: float = 0.8,
        force_model: str | None = None,
    ) -> dict[str, Any]:
        """Route a prompt through the full pipeline.

        Parameters
        ----------
        prompt : str
            User prompt.
        required_accuracy : float
            Minimum accuracy threshold.
        force_model : str | None
            Bypass the decision engine and force a specific model.

        Returns
        -------
        dict
            Full response payload matching the RouteResponse schema.
        """
        pipeline_start = time.perf_counter()

        # Step 1: Normalise
        normalized = self._normalizer.normalize(prompt)
        raw_hash = hashlib.sha256(prompt.encode()).hexdigest()

        # Step 2: Cache check
        cached, cache_tier = self._cache.get(raw_hash, normalized.prompt_hash)
        if cached is not None:
            elapsed = (time.perf_counter() - pipeline_start) * 1000
            self._metrics.record(RequestMetric(
                prompt_hash=normalized.prompt_hash,
                model_used=cached.model_used,
                cache_hit=True,
                cache_tier=cache_tier,
                cost=0.0,
                latency_ms=round(elapsed, 1),
                confidence=cached.confidence,
            ))
            return {
                "response": cached.response,
                "model_used": cached.model_used,
                "cost": 0.0,
                "latency_ms": round(elapsed, 1),
                "confidence": cached.confidence,
                "cache_hit": True,
                "escalated": False,
                "escalation_depth": 0,
                "task_vector": cached.task_vector,
                "routing_explanation": f"Cache hit ({cache_tier}). {cached.routing_explanation}",
            }

        # Step 3: Extract features + build vectors
        features = self._extractor.extract(prompt)
        task_vec, resource_vec, risk_vec = self._vectorizer.generate(features)

        task_dict = task_vec.to_dict()
        resource_dict = {
            "input_tokens": resource_vec.expected_input_tokens,
            "output_tokens": resource_vec.expected_output_tokens,
            "context_length": resource_vec.expected_context_length,
            "complexity": resource_vec.complexity,
        }
        risk_dict = risk_vec.to_dict()

        # Step 4: Force model or run decision engine
        if force_model:
            from app.router.decision_engine import RoutingDecision
            decision = RoutingDecision(
                model_selected=force_model,
                estimated_cost=0.0,
                predicted_accuracy=1.0,
                reasoning=f"Model forced to {force_model} by user.",
            )
        else:
            decision = self._engine.select_model(
                task_vector=task_dict,
                resource_vector=resource_dict,
                risk_vector=risk_dict,
                required_accuracy=required_accuracy,
            )

        # Step 5: Execute
        escalation_depth = 0
        escalated = False
        best_result: ExecutionResult | None = None

        # Check deterministic tools first
        if decision.is_deterministic:
            det_result = self._deterministic.try_execute(prompt, decision.model_selected)
            if det_result and det_result.confidence > 0.5:
                best_result = det_result

        # LLM execution (with escalation loop)
        current_model = decision.model_selected
        while best_result is None or (
            best_result.confidence < self._policy.confidence_threshold
            and escalation_depth < self._policy.max_depth
        ):
            if best_result is not None:
                # We're escalating
                eligible = self._matrix.get_capable_models(task_dict, required_accuracy)
                eligible.sort(key=lambda m: m.get("estimated_cost", float("inf")))
                next_model = self._policy.get_next_model(
                    current_model, eligible, escalation_depth,
                )
                if next_model is None:
                    break  # No more models to try
                current_model = next_model
                escalation_depth += 1
                escalated = True
                logger.info(
                    "pipeline.escalating",
                    from_model=best_result.model_used,
                    to_model=current_model,
                    depth=escalation_depth,
                )

            # Skip deterministic models in LLM loop
            if current_model.startswith("deterministic:"):
                break

            result = await self._fireworks.execute(
                prompt=prompt,
                model_id=current_model,
                task_type=features.task_type,
            )

            # Step 6: Validate confidence
            validation = self._validator.validate(
                response=result.response,
                task_type=features.task_type,
                expected_json=features.json_required,
                expected_code=features.contains_code,
                expected_length=features.expected_output_length,
            )
            result.confidence = validation.confidence

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

            if result.confidence >= self._policy.confidence_threshold:
                break

        # Ensure we have a result
        if best_result is None:
            best_result = ExecutionResult(
                response="Failed to generate a response.",
                model_used=current_model,
                confidence=0.0,
            )

        elapsed = (time.perf_counter() - pipeline_start) * 1000

        # Step 7: Cache the result
        self._cache.set(
            raw_hash=raw_hash,
            normalized_hash=normalized.prompt_hash,
            response=CachedResponse(
                response=best_result.response,
                model_used=best_result.model_used,
                cost=best_result.cost,
                confidence=best_result.confidence,
                task_vector=task_dict,
                routing_explanation=decision.reasoning,
            ),
        )

        # Step 8: Log metrics
        self._metrics.record(RequestMetric(
            prompt_hash=normalized.prompt_hash,
            model_used=best_result.model_used,
            task_type=features.task_type,
            tokens_input=best_result.tokens_input,
            tokens_output=best_result.tokens_output,
            latency_ms=round(elapsed, 1),
            cost=best_result.cost,
            confidence=best_result.confidence,
            cache_hit=False,
            escalated=escalated,
            escalation_depth=escalation_depth,
        ))

        return {
            "response": best_result.response,
            "model_used": best_result.model_used,
            "cost": best_result.cost,
            "latency_ms": round(elapsed, 1),
            "confidence": best_result.confidence,
            "cache_hit": False,
            "escalated": escalated,
            "escalation_depth": escalation_depth,
            "task_vector": task_dict,
            "routing_explanation": decision.reasoning,
        }

    # ------------------------------------------------------------------
    # Accessors for the API layer
    # ------------------------------------------------------------------

    @property
    def capability_matrix(self) -> CapabilityMatrix:
        """Return the capability matrix instance."""
        return self._matrix

    @property
    def metrics(self) -> MetricsCollector:
        """Return the metrics collector."""
        return self._metrics

    @property
    def cache(self) -> CacheManager:
        """Return the cache manager."""
        return self._cache
