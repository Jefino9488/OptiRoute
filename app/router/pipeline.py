"""Routing pipeline — orchestrates the full request lifecycle.

Flow:
    Normalise → Cache check → Extract features → Build vectors →
    Decision (local LLM router | heuristic engine) →
    Execute (local GGUF | Fireworks API) →
    Validate confidence → Escalate if needed →
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
from app.features.extractor import FeatureExtractor
from app.features.normalizer import RequestNormalizer
from app.features.preprocessor import PromptPreprocessor
from app.features.vectorizer import TaskVectorGenerator
from app.metrics.collector import MetricsCollector, RequestMetric
from app.router.capability_matrix import CapabilityMatrix
from app.router.decision_engine import DecisionEngine, RoutingDecision
from app.router.policy import EscalationPolicy

logger = structlog.get_logger(__name__)


class RoutingPipeline:
    """End-to-end routing pipeline.

    Creates and manages all components.  Designed to be instantiated
    once at application startup and reused for every request.

    Local model loading is performed synchronously during __init__ —
    this blocks briefly (~5-20s) but is acceptable at startup time.
    """

    def __init__(self) -> None:
        settings = get_settings()

        # Feature extraction
        self._normalizer = RequestNormalizer()
        self._extractor = FeatureExtractor()
        self._vectorizer = TaskVectorGenerator()

        # Routing (heuristic engine — kept as fallback)
        self._matrix = CapabilityMatrix(settings.capability_matrix_path)
        self._engine = DecisionEngine(self._matrix)
        self._policy = EscalationPolicy(
            max_depth=settings.max_escalation_depth,
            confidence_threshold=settings.confidence_threshold,
        )

        # Execution backends
        self._fireworks = FireworksExecutor()

        # Local model (optional — gracefully absent if GGUF missing)
        self._local = None
        if settings.local_model_enabled:
            try:
                from app.executors.local import LocalExecutor
                local = LocalExecutor(
                    model_path=settings.local_model_path,
                    context_length=settings.local_model_context_length,
                    n_threads=settings.local_model_threads,
                )
                if local.load():
                    self._local = local
                    logger.info(
                        "pipeline.local_model_ready",
                        path=settings.local_model_path,
                    )
                else:
                    logger.warning(
                        "pipeline.local_model_unavailable",
                        path=settings.local_model_path,
                    )
            except ImportError:
                logger.warning(
                    "pipeline.llama_cpp_not_installed",
                    hint="Install llama-cpp-python to enable local model",
                )

        # Support
        self._cache = CacheManager()
        self._validator = ConfidenceValidator()
        self._metrics = MetricsCollector()
        self._preprocessor = PromptPreprocessor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route(
        self,
        prompt: str,
        required_accuracy: float = 0.75,
        force_model: str | None = None,
    ) -> dict[str, Any]:
        """Route a prompt through the full pipeline.

        Parameters
        ----------
        prompt : str
            User prompt.
        required_accuracy : float
            Minimum accuracy threshold (0.75 allows local model).
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

        # Step 4: Decide model
        decision = await self._make_routing_decision(
            prompt=prompt,
            task_dict=task_dict,
            resource_dict=resource_dict,
            risk_dict=risk_dict,
            required_accuracy=required_accuracy,
            force_model=force_model,
        )

        # Step 5: Execute
        escalation_depth = 0
        escalated = False
        best_result: ExecutionResult | None = None

        # 5a: Local model dispatch (when model_selected starts with "local:")
        if best_result is None and decision.model_selected.startswith("local:") and self._local:
            local_result = await self._try_local_execute(
                prompt=prompt,
                model_id=decision.model_selected,
                features=features,
                resource_dict=resource_dict,
            )
            if local_result is not None:
                best_result = local_result
                if local_result.confidence < self._policy.confidence_threshold:
                    # Low confidence — escalate to cheapest Fireworks model
                    logger.info(
                        "pipeline.local_low_confidence_escalating",
                        confidence=local_result.confidence,
                        threshold=self._policy.confidence_threshold,
                    )
                    escalated = True
                    escalation_depth += 1
                    # Override decision to use Fireworks fallback
                    decision = RoutingDecision(
                        model_selected="minimax-m3",
                        estimated_cost=0.0,
                        predicted_accuracy=0.9,
                        reasoning="Escalated from local due to low confidence",
                    )
                    best_result = None  # will be set by Fireworks loop below

        # 5b: Fireworks execution (with escalation loop)
        # Preprocess prompt once before the loop: compress tokens + inject
        # anti-hallucination system prompt lines (false_memory, stale_knowledge, injection).
        # Local model execution is never preprocessed — it always received the original.
        preprocessed = await self._preprocessor.process(
            prompt=prompt,
            local_executor=self._local,
        )
        if preprocessed.risk_flags:
            logger.info(
                "pipeline.preprocessor_flags",
                flags=list(preprocessed.risk_flags),
                token_savings=preprocessed.token_savings,
            )
        fireworks_system_prompt = "\n".join(preprocessed.system_addons) or None

        current_model = decision.model_selected
        while best_result is None or (
            best_result.confidence < self._policy.confidence_threshold
            and escalation_depth < self._policy.max_depth
        ):
            if best_result is not None:
                # We're escalating — find the next model
                eligible = self._matrix.get_capable_models(task_dict, required_accuracy)
                for m in eligible:
                    m["estimated_cost"] = self._matrix.estimate_cost(
                        m["model_id"],
                        resource_dict.get("input_tokens", 0),
                        resource_dict.get("output_tokens", 0),
                    )
                # Filter out local models from Fireworks escalation list
                # Also restrict to models actually in ALLOWED_MODELS —
                # prevents ValueError from get_model_path() if the matrix
                # has models the harness hasn't permitted for this run.
                allowed = set(get_settings().allowed_models.keys())
                eligible = [
                    m for m in eligible
                    if not m["model_id"].startswith("local:")
                    and m["model_id"] in allowed
                ]
                eligible.sort(key=lambda m: m.get("estimated_cost", float("inf")))
                next_model = self._policy.get_next_model(
                    current_model, eligible, escalation_depth,
                )
                if next_model is None:
                    break
                current_model = next_model
                escalation_depth += 1
                escalated = True
                logger.info(
                    "pipeline.escalating",
                    from_model=best_result.model_used,
                    to_model=current_model,
                    depth=escalation_depth,
                )

            # Skip local models in Fireworks loop
            if current_model.startswith("local:"):
                break

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    result = await self._fireworks.execute(
                        prompt=preprocessed.forwarded,
                        model_id=current_model,
                        task_type=features.task_type,
                        system_prompt=fireworks_system_prompt,
                        max_tokens=resource_dict.get("output_tokens"),
                    )
                    
                    result.cost = self._matrix.estimate_cost(
                        current_model,
                        result.tokens_input,
                        result.tokens_output,
                    ) or 0.0

                    finish_reason = result.raw_metadata.get("finish_reason")
                    is_empty = not result.response.strip()
                    is_error = "[ERROR]" in result.response

                    # Only retry on transient failures
                    is_transient = is_empty or is_error or (finish_reason == "length")
                    
                    if is_transient and attempt < max_retries - 1:
                        logger.warning(
                            "pipeline.transient_failure_retry",
                            model=current_model,
                            attempt=attempt + 1,
                            reason=finish_reason or ("empty" if is_empty else "error")
                        )
                        continue # Try same model again
                        
                    break # Success or non-transient, exit retry loop
                    
                except ValueError as exc:
                    # model_id not in ALLOWED_MODELS — treat as zero-confidence failure
                    # so the escalation loop can try the next model
                    logger.error(
                        "pipeline.model_not_in_allowed_models",
                        model=current_model,
                        error=str(exc),
                    )
                    result = ExecutionResult(
                        response="",
                        model_used=current_model,
                        confidence=0.0,
                        raw_metadata={"error": str(exc)},
                    )
                    break

            # Step 6: Validate confidence
            validation = self._validator.validate(
                response=result.response,
                task_type=features.task_type,
                expected_json=features.json_required,
                expected_code=features.contains_code,
                expected_length=features.expected_output_length,
            )
            result.confidence = min(result.confidence, validation.confidence)

            if best_result is None or result.confidence > best_result.confidence:
                best_result = result

            if result.confidence >= self._policy.confidence_threshold:
                break

        # Ensure we always have a result
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
    # Internal routing decision
    # ------------------------------------------------------------------

    async def _make_routing_decision(
        self,
        prompt: str,
        task_dict: dict[str, float],
        resource_dict: dict[str, Any],
        risk_dict: dict[str, Any],
        required_accuracy: float,
        force_model: str | None,
    ) -> RoutingDecision:
        """Select the model using a 3-level priority chain.

        1. force_model override (testing / debugging)
        2. Local LLM router (Qwen2.5-3B, max_tokens=15)
        3. Heuristic decision engine (fallback — always works)
        """
        settings = get_settings()

        # Level 1: Forced model
        if force_model:
            return RoutingDecision(
                model_selected=force_model,
                estimated_cost=0.0,
                predicted_accuracy=1.0,
                reasoning=f"Model forced by caller: {force_model}",
            )

        # Level 2: Local LLM router
        if self._local and settings.local_router_enabled:
            routed = await self._local.route(prompt)
            if routed is not None:
                if routed == "local":
                    model_id = settings.local_model_name
                    return RoutingDecision(
                        model_selected=model_id,
                        estimated_cost=0.0,
                        predicted_accuracy=0.75,
                        reasoning="Local LLM router → local model ($0 Fireworks tokens)",
                    )
                else:
                    # Fireworks model selected by local router
                    cost = self._matrix.estimate_cost(
                        routed,
                        resource_dict.get("input_tokens", 500),
                        resource_dict.get("output_tokens", 300),
                    ) or 0.0
                    return RoutingDecision(
                        model_selected=routed,
                        estimated_cost=round(cost, 8),
                        predicted_accuracy=0.9,
                        reasoning=f"Local LLM router → {routed}",
                    )

        # Level 3: Heuristic decision engine (fallback)
        logger.info("pipeline.using_heuristic_engine")
        return self._engine.select_model(
            task_vector=task_dict,
            resource_vector=resource_dict,
            risk_vector=risk_dict,
            required_accuracy=required_accuracy,
            prompt=prompt,
        )

    async def _try_local_execute(
        self,
        prompt: str,
        model_id: str,
        features: Any,
        resource_dict: dict[str, Any],
    ) -> ExecutionResult | None:
        """Try executing on the local model with context-length pre-check.

        Returns None if the prompt is too long for the local context window.
        """
        assert self._local is not None

        # Context length pre-check
        model_entry = self._matrix.get_model_capabilities(model_id) or {}
        max_ctx = model_entry.get("max_context", 4096)
        est_tokens = resource_dict.get("input_tokens", 0)

        if est_tokens > max_ctx * 0.9:
            logger.info(
                "pipeline.local_context_overflow",
                est_tokens=est_tokens,
                max_context=max_ctx,
                model=model_id,
            )
            return None  # Caller will fall through to Fireworks

        result = await self._local.execute(
            prompt=prompt,
            task_type=features.task_type,
        )

        # Validate confidence
        validation = self._validator.validate(
            response=result.response,
            task_type=features.task_type,
            expected_json=features.json_required,
            expected_code=features.contains_code,
            expected_length=features.expected_output_length,
        )
        result.confidence = min(result.confidence, validation.confidence)

        return result

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
