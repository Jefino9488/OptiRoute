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
from app.features.compiler import InferencePolicyCompiler
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
        self._compiler = InferencePolicyCompiler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route(
        self,
        prompt: str,
        required_accuracy: float = 0.75,
        force_model: str | None = None,
        enable_thinking: bool | None = None,
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
        enable_thinking : bool | None
            None = auto-detect, True = force thinking ON, False = force OFF.

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
        _BUCKET_MAP = {
            "Small": 512,
            "Medium": 1024,
            "Large": 2048,
            "Very_Large": 4096,
        }
        output_tokens = _BUCKET_MAP.get(resource_vec.output_budget_bucket, 1024)

        resource_dict = {
            "input_tokens": resource_vec.expected_input_tokens,
            "output_tokens": output_tokens,
            "output_budget_bucket": resource_vec.output_budget_bucket,
            "context_length": resource_vec.expected_context_length,
            "complexity": resource_vec.complexity,
        }
        risk_dict = risk_vec.to_dict()

        # Step 3b: Determine reasoning_effort for thinking models
        reasoning_effort = self._compute_reasoning_effort(
            features=features,
            task_dict=task_dict,
            resource_dict=resource_dict,
            enable_thinking=enable_thinking,
            prompt=prompt,
        )
        logger.info(
            "pipeline.reasoning_effort_resolved",
            reasoning_effort=reasoning_effort,
            task_type=features.task_type,
            complexity=resource_dict.get("complexity", 0.0),
            requires_reasoning=features.requires_reasoning,
            contains_math=features.contains_math,
        )

        # Step 4: Decide model
        decision = await self._make_routing_decision(
            prompt=prompt,
            features=features,
            task_dict=task_dict,
            resource_dict=resource_dict,
            risk_dict=risk_dict,
            required_accuracy=required_accuracy,
            force_model=force_model,
        )

        # Step 4b: Compile System Prompt
        # Preprocessor still runs for anti-hallucination guards (before any model executes)
        preprocessed = await self._preprocessor.process(prompt=prompt)
        base_system = "\n\n".join(preprocessed.system_addons)
        
        compiled_policy = self._compiler.compile(
            prompt=prompt,
            task_vector=task_dict,
            resource_vector=resource_dict,
            risk_vector=risk_dict,
            model=decision.model_selected,
            base_system_prompt=base_system,
        )
        
        system_prompt = compiled_policy.system_prompt
        # compiled_policy.user_prompt is strictly untouched, so we continue passing `prompt`.

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
                system_prompt=system_prompt,
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
            else:
                # Local skipped (context/output overflow) — fall through to Fireworks
                logger.info(
                    "pipeline.local_skipped_fallback_fireworks",
                    model=decision.model_selected,
                )
                decision = RoutingDecision(
                    model_selected="minimax-m3",
                    estimated_cost=0.0,
                    predicted_accuracy=0.9,
                    reasoning="Local skipped due to output overflow, routing to Fireworks",
                )

        # 5b: Fireworks execution (with escalation loop)
        # Preprocessor has already run in Step 4b. We just use the compiled system_prompt.
        # But wait, preprocessed.forwarded has the injection clauses stripped (if any).
        # We must use the stripped prompt for Fireworks.
        if preprocessed.risk_flags:
            logger.info(
                "pipeline.preprocessor_flags",
                flags=list(preprocessed.risk_flags),
            )
            
        fireworks_system_prompt = system_prompt or None

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
                    # If eligible models were empty because of threshold/allowed filters,
                    # force fallback to the frontier model as a last resort,
                    # provided we haven't exceeded depth and haven't tried it yet.
                    fallback_model = self._policy.get_fallback_model()
                    if current_model != fallback_model and fallback_model in allowed and escalation_depth < self._policy.max_depth:
                        next_model = fallback_model
                        logger.warning("pipeline.forcing_fallback", model=fallback_model)
                    else:
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
                
                # Upgrade reasoning effort on escalation if appropriate
                if escalation_depth > 0 and reasoning_effort == "none" and features.task_type in ("reasoning", "creative", "extraction"):
                    reasoning_effort = "low"
                    logger.info("pipeline.upgrading_reasoning", model=current_model, task_type=features.task_type)

            # Skip local models in Fireworks loop
            if current_model.startswith("local:"):
                break

            # Adaptive output budget based on task characteristics
            base_output = max(resource_dict.get("output_tokens", 512), 256)
            current_max_tokens = int(base_output)
                
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = await self._fireworks.execute(
                        prompt=preprocessed.forwarded,
                        model_id=current_model,
                        task_type=features.task_type,
                        system_prompt=fireworks_system_prompt,
                        max_tokens=current_max_tokens,
                        reasoning_effort=reasoning_effort,
                    )
                    
                    result.cost = self._matrix.estimate_cost(
                        current_model,
                        result.tokens_input,
                        result.tokens_output,
                        thinking_enabled=(reasoning_effort is not None and reasoning_effort != "none"),
                    ) or 0.0

                    finish_reason = result.raw_metadata.get("finish_reason")
                    is_empty = not result.response.strip()
                    is_error = "[ERROR]" in result.response
                    is_not_found = "[NOT_FOUND]" in result.response
                    
                    if is_not_found:
                        logger.warning(
                            "pipeline.model_not_found_bypassing",
                            model=current_model,
                        )
                        break  # Break retry loop instantly to escalate

                    # Smarter retry policy
                    if finish_reason == "length":
                        # If it hit the length limit because it's caught in an infinite loop,
                        # don't waste tokens giving it a larger budget. Break and escalate.
                        if self._validator._has_excessive_repetition(result.response):
                            logger.warning(
                                "pipeline.infinite_loop_detected",
                                model=current_model,
                                reason="length with repetition"
                            )
                            break
                        
                        logger.warning(
                            "pipeline.transient_failure_retry",
                            model=current_model,
                            attempt=attempt + 1,
                            reason="length (increasing budget)"
                        )
                        if attempt == 0:
                            current_max_tokens = int(base_output * 1.5)
                        elif attempt == 1:
                            current_max_tokens = int(base_output * 2.0)
                        else:
                            break # Exceeded allowed budget growth, force escalation
                        continue

                    if is_empty or is_error:
                        logger.warning(
                            "pipeline.transient_failure_retry",
                            model=current_model,
                            attempt=attempt + 1,
                            reason="empty" if is_empty else "error"
                        )
                        continue # Retry same model
                        
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
            "compiler_metadata": compiled_policy.metadata,
        }

    # ------------------------------------------------------------------
    # Internal routing decision
    # ------------------------------------------------------------------

    def _compute_reasoning_effort(
        self,
        features: Any,
        task_dict: dict[str, float],
        resource_dict: dict[str, Any],
        enable_thinking: bool | None,
        prompt: str = "",
    ) -> str | None:
        """Determine the Fireworks reasoning_effort parameter.

        Logic:
        1. If enable_thinking is explicitly True → "high"
        2. If enable_thinking is explicitly False → "none"
        3. If enable_thinking is None (auto-detect):
           - High reasoning dimension OR complex task → "high"
           - Moderate reasoning → "low"
           - Simple tasks → "none"

        Parameters
        ----------
        features : FeatureVector
            Extracted features from the prompt.
        task_dict : dict
            Task vector with dimension weights.
        resource_dict : dict
            Resource estimates including complexity.
        enable_thinking : bool | None
            User override (None = auto-detect).

        Returns
        -------
        str | None
            reasoning_effort value, or None to omit the parameter.
        """
        if enable_thinking is True:
            logger.info(
                "pipeline.thinking_forced_on",
                prompt_preview=prompt[:80],
            )
            return "high"
        if enable_thinking is False:
            logger.info(
                "pipeline.thinking_forced_off",
                prompt_preview=prompt[:80],
            )
            return "none"

        # Auto-detect: check if task benefits from reasoning
        reasoning_weight = task_dict.get("reasoning", 0.0)
        math_weight = task_dict.get("math", 0.0)
        code_weight = task_dict.get("code", 0.0)
        complexity = resource_dict.get("complexity", 0.0)

        # As per optimization plan, we restrict reasoning to Math and Code tasks natively.
        # General reasoning tasks get "none" initially and escalate if they fail.
        if math_weight > 0.5 or code_weight > 0.5:
            logger.info(
                "pipeline.auto_thinking_low",
                trigger=f"math={math_weight:.2f} code={code_weight:.2f}",
                reasoning=reasoning_weight,
                math=math_weight,
                complexity=complexity,
                prompt_preview=prompt[:80],
            )
            return "low"

        logger.info(
            "pipeline.auto_thinking_none",
            reasoning=reasoning_weight,
            math=math_weight,
            code=code_weight,
            complexity=complexity,
            requires_reasoning=features.requires_reasoning,
            prompt_preview=prompt[:80],
        )
        return "none"

    async def _make_routing_decision(
        self,
        prompt: str,
        features: Any,
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

        # Dynamic required accuracy based on complexity and task type
        base_accuracy = 0.75
        complexity = resource_dict.get("complexity", 0.0)
        
        # Scale required accuracy based on complexity for ALL tasks (up to +0.20)
        # This pushes it out of reach of the local model for tricky edge cases
        base_accuracy += (0.20 * complexity)

        # Code tasks require extreme precision (syntax, edge cases)
        if features.task_type == "code":
            base_accuracy += 0.15
            
        # The local model has suspiciously high offline scores for extraction/retrieval 
        # (0.95+). We bump the requirement to ensure it only wins on simple prompts.
        if features.task_type in ("extraction", "retrieval"):
            base_accuracy += 0.15
            
        # If the risk vector detected strict constraints, bump requirement heavily
        if risk_dict.get("strict_formatting") or risk_dict.get("needs_high_accuracy"):
            base_accuracy += 0.15

        # Cap at 0.97 to ensure we can force a fallback for extreme complexity.
        # This prevents the local model (which has inflated 0.956+ scores for 
        # extraction/retrieval) from qualifying when constraints are strict.
        required_accuracy = min(base_accuracy, 0.97)

        try:
            decision = self._engine.select_model(
                task_vector=task_dict,
                resource_vector=resource_dict,
                risk_vector=risk_dict,
                required_accuracy=required_accuracy,
                prompt=prompt,
            )
            return decision
        except Exception:
            # Fallback in case heuristic engine fails
            return RoutingDecision(
                model_selected=settings.fallback_model,
                estimated_cost=0.0,
                predicted_accuracy=1.0,
                reasoning="Heuristic engine failure; defaulting to robust model",
            )

    async def _try_local_execute(
        self,
        prompt: str,
        model_id: str,
        features: Any,
        resource_dict: dict[str, Any],
        system_prompt: str | None = None,
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

        # Output generation limit pre-check is removed. 
        # Even if a reasoning task is estimated to be long, we want the local $0 model
        # to ATTEMPT it. If it fails or gets cut off, the confidence validator will catch
        # it and it will escalate naturally. Bypassing it prematurely burns Fireworks tokens.
        est_output = resource_dict.get("output_tokens", 256)

        result = await self._local.execute(
            prompt=prompt,
            task_type=features.task_type,
            system_prompt=system_prompt,
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
