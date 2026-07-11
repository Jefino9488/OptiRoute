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
from app.router.supra_router import SupraRouter

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

        # Supra-Router-51M (ML-based routing)
        self._supra_router: SupraRouter | None = None
        if settings.supra_router_enabled:
            try:
                self._supra_router = SupraRouter(
                    server_url=settings.supra_router_url,
                )
                logger.info(
                    "pipeline.supra_router_enabled",
                    url=settings.supra_router_url,
                )
            except Exception:
                logger.warning("pipeline.supra_router_init_failed")

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
                "tokens_input": 0,
                "tokens_output": 0,
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
        # Run SupraRouter first (ML-based); fall back to regex if unavailable.
        supra_result = None
        if self._supra_router is not None:
            try:
                supra_result = await self._supra_router.classify(prompt)
            except Exception as exc:
                logger.warning("pipeline.supra_router_error", error=str(exc))
                supra_result = None

        if supra_result is not None and supra_result.success:
            # Build FeatureVector from ML output (no regex needed)
            features = self._build_features_from_supra(prompt, supra_result)
            logger.info(
                "pipeline.supra_router_applied",
                domain=supra_result.domain,
                complexity=supra_result.complexity,
                route=supra_result.route,
                task_type=features.task_type,
            )
        else:
            # Regex fallback when SupraRouter unavailable
            features = self._extractor.extract(prompt)
            logger.info("pipeline.regex_fallback")

        task_vec, resource_vec, risk_vec = self._vectorizer.generate(features)

        # Step 3.5: Deterministic tool bypass (before any LLM call)
        deterministic_result = self._try_deterministic(prompt, features)
        if deterministic_result is not None:
            return deterministic_result

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
            supra_route=supra_result.route if supra_result else None,
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
        length_exhausted = False  # Track if escalation was triggered by length exhaustion

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

                # Reset length tracking for the new model's retry loop
                length_exhausted = False
                length_retry_count = 0

            # Skip local models in Fireworks loop
            if current_model.startswith("local:"):
                break

            # Adaptive output budget based on task characteristics
            # Task-type minimums: reasoning/code/math/general_qa need more tokens
            _TASK_MIN_TOKENS = {
                "math": 2048,
                "code": 2048,
                "reasoning": 2048,
                "general_qa": 2048,
                "creative": 1024,
                "extraction": 512,
                "verification": 512,
                "translation": 512,
                "retrieval": 512,
            }
            raw_output = resource_dict.get("output_tokens", 1024)
            task_min = _TASK_MIN_TOKENS.get(features.task_type, 512)
            base_output = max(raw_output, task_min)
            current_max_tokens = int(base_output)
                
            max_retries = 2
            length_retry_count = 0
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

                    # Smarter retry policy: aggressive budget growth to avoid escalation
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
                        
                        length_retry_count += 1
                        logger.warning(
                            "pipeline.transient_failure_retry",
                            model=current_model,
                            attempt=attempt + 1,
                            reason="length (increasing budget)"
                        )
                        if length_retry_count == 1:
                            current_max_tokens = int(base_output * 2)
                        elif length_retry_count == 2:
                            current_max_tokens = int(base_output * 4)
                        else:
                            length_exhausted = True
                            break # Exceeded allowed budget growth, force escalation
                        continue

                    if is_empty or is_error:
                        # If empty with reasoning enabled, disable thinking on retry
                        # (thinking tokens may consume budget without producing content)
                        if is_empty and reasoning_effort and reasoning_effort != "none":
                            logger.warning(
                                "pipeline.empty_with_thinking_disabling",
                                model=current_model,
                                reasoning_effort=reasoning_effort,
                            )
                            reasoning_effort = "none"
                        else:
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
            "tokens_input": best_result.tokens_input,
            "tokens_output": best_result.tokens_output,
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
    # Deterministic tool bypass
    # ------------------------------------------------------------------

    @staticmethod
    def _try_deterministic(prompt: str, features: Any) -> dict[str, Any] | None:
        """Try to answer purely computational tasks without an LLM call.

        Returns a pipeline-compatible result dict if the task is deterministic,
        or None if it should go through normal LLM routing.
        """
        import re

        # Pattern 1: "count the letter/character X in/within this sentence/text/..."
        # Extract the target character from "letter 'X'" or "character 'X'"
        target_match = re.search(
            r"(?:letter|character)s?\s+['\"](.+?)['\"]",
            prompt,
            re.IGNORECASE,
        )
        if target_match:
            target = target_match.group(1)
            # Find the text to count in — look for the longest quoted string
            quotes = re.findall(r"['\"](.+?)['\"]", prompt, re.IGNORECASE)
            if quotes:
                # Use the longest quoted string as the sentence
                sentence = max(quotes, key=len)
                count = sentence.count(target)
                return {
                    "response": str(count),
                    "model_used": "deterministic:count",
                    "tokens_input": 0,
                    "tokens_output": 0,
                    "cost": 0.0,
                    "latency_ms": 0.0,
                    "confidence": 1.0,
                    "cache_hit": False,
                    "escalated": False,
                    "escalation_depth": 0,
                    "task_vector": {},
                    "routing_explanation": f"Deterministic character count: '{target}' appears {count} times",
                }

        # Pattern 2: Simple arithmetic that can be evaluated
        # "What is X + Y * Z?" or "Calculate X * Y"
        if features.task_type == "math" and features.contains_math:
            # Try to extract a simple arithmetic expression
            # Look for patterns like "2 + 3", "10 * 5", "100 / 4"
            math_match = re.search(
                r"(?:what\s+is|calculate|compute|find)\s+([\d\s\+\-\*\/\.\(\)]+)",
                prompt,
                re.IGNORECASE,
            )
            if math_match:
                expr = math_match.group(1).strip()
                # Validate it's safe (only numbers and operators)
                if re.match(r"^[\d\s\+\-\*\/\.\(\)]+$", expr):
                    try:
                        result_val = eval(expr)  # noqa: S307 — validated safe
                        return {
                            "response": str(result_val),
                            "model_used": "deterministic:math",
                            "tokens_input": 0,
                            "tokens_output": 0,
                            "cost": 0.0,
                            "latency_ms": 0.0,
                            "confidence": 1.0,
                            "cache_hit": False,
                            "escalated": False,
                            "escalation_depth": 0,
                            "task_vector": {},
                            "routing_explanation": f"Deterministic math: {expr} = {result_val}",
                        }
                    except (ZeroDivisionError, ValueError, SyntaxError):
                        pass

        return None

    @staticmethod
    def _map_domain_to_task_type(domain: str, fallback: str) -> str:
        """Map Supra-Router domain labels to internal task types."""
        domain_lower = domain.lower()
        mapping = {
            # Math domains (unambiguous only)
            "math": "math",
            "probability": "math",
            "algebra": "math",
            "calculus": "math",
            "statistics": "math",
            "geometry": "math",
            "differential": "math",
            "arithmetic": "math",
            "equation": "math",
            # Code domains
            "code": "code",
            "programming": "code",
            "software": "code",
            "sql": "code",
            "python": "code",
            "debugging": "code",
            "web development": "code",
            "api": "code",
            # Reasoning domains
            "reasoning": "reasoning",
            "logic": "reasoning",
            "puzzle": "reasoning",
            "deduction": "reasoning",
            "comparison": "reasoning",
            "architecture": "reasoning",
            "operating system": "reasoning",
            "system design": "reasoning",
            "networking": "reasoning",
            "protocol": "reasoning",
            # Creative domains
            "creative": "creative",
            "writing": "creative",
            "poetry": "creative",
            "haiku": "creative",
            "sonnet": "creative",
            "literature": "creative",
            "content writing": "creative",
            # Translation
            "translation": "translation",
            "language": "translation",
            "english language": "translation",
            "french": "translation",
            "spanish": "translation",
            "german": "translation",
            "natural language processing": "translation",
            # Extraction
            "extraction": "extraction",
            "ner": "extraction",
            "entity": "extraction",
            "parsing": "extraction",
            "json": "extraction",
            "etymology": "extraction",
            # Retrieval
            "retrieval": "retrieval",
            "factual": "retrieval",
            "qa": "retrieval",
            "question": "retrieval",
            "lookup": "retrieval",
            # Classification / sentiment
            "sentiment": "retrieval",
            "classification": "retrieval",
            "rating": "retrieval",
            "review": "retrieval",
            # General (catch-all for ambiguous domains)
            "general": "general_qa",
            "machine learning": "general_qa",
            "ai": "general_qa",
            "computer": "general_qa",
            "computer science": "general_qa",
            "technology": "general_qa",
            "science": "general_qa",
            "medical": "general_qa",
            "healthcare": "general_qa",
            "communication": "general_qa",
            "e-commerce": "general_qa",
            "physics": "general_qa",
            "finance": "general_qa",
            "trading": "general_qa",
            "mixed structures": "general_qa",
            "finance/trading": "general_qa",
        }
        for key, task_type in mapping.items():
            if key in domain_lower:
                return task_type
        return fallback

    def _build_features_from_supra(self, prompt: str, supra: Any) -> Any:
        """Build a FeatureVector from SupraRouter output (replaces regex)."""
        from app.features.extractor import FeatureVector  # noqa: PLC0415

        fv = FeatureVector()
        fv.input_length = len(prompt.split())
        fv.question_count = prompt.count("?")
        fv.task_type = self._map_domain_to_task_type(supra.domain, "general_qa")
        fv.contains_math = supra.needs_math
        fv.contains_code = supra.needs_code
        fv.complexity = supra.complexity

        # Derive additional flags from domain
        domain_lower = supra.domain.lower()
        fv.requires_reasoning = "reason" in domain_lower or "logic" in domain_lower
        fv.requires_retrieval = "retriev" in domain_lower or "factual" in domain_lower
        fv.is_creative = "creat" in domain_lower or "writ" in domain_lower
        fv.is_translation = "translat" in domain_lower
        fv.is_extraction = "extract" in domain_lower

        # Output length heuristics
        if fv.is_creative or fv.contains_code:
            fv.expected_output_length = "long"
        elif fv.requires_retrieval or fv.is_extraction:
            fv.expected_output_length = "short"
        else:
            fv.expected_output_length = "medium"

        return fv

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
        supra_route: str | None = None,
    ) -> RoutingDecision:
        """Select the model using a 3-level priority chain.

        1. force_model override (testing / debugging)
        2. Local LLM router (Phi-4-mini, max_tokens=15)
        3. Heuristic decision engine (fallback — always works)

        If supra_route is "big model", required_accuracy is bumped to
        push the local model out of contention. If "small model", the
        threshold stays low to favour $0 local inference.
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

        # Code tasks require precision (syntax, edge cases) but don't over-bump
        # — kimi-k2p7-code has 0.9675 code score and should be eligible.
        if features.task_type == "code":
            base_accuracy += 0.10
            
        # The local model has suspiciously high offline scores for extraction/retrieval 
        # (0.95+). We bump the requirement to ensure it only wins on simple prompts.
        if features.task_type in ("extraction", "retrieval"):
            base_accuracy += 0.15
            
        # If the risk vector detected strict constraints, bump requirement heavily
        if risk_dict.get("strict_formatting") or risk_dict.get("needs_high_accuracy"):
            base_accuracy += 0.15

        # Supra-Router route override: bump threshold when ML says "big model"
        # This prevents the local $0 model from winning on complex prompts.
        # Keep the bump moderate (+0.10) so only genuinely complex tasks go to Fireworks.
        if supra_route == "big model":
            base_accuracy += 0.10
            logger.info(
                "pipeline.supra_route_big_model",
                base_accuracy=base_accuracy,
            )
        elif supra_route == "small model":
            # Keep threshold low — favour $0 local model
            base_accuracy = max(base_accuracy - 0.10, 0.70)
            logger.info(
                "pipeline.supra_route_small_model",
                base_accuracy=base_accuracy,
            )

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
