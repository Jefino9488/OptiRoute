"""Routing pipeline — orchestrates the full request lifecycle.

Flow:
    Normalise → Cache check → SupraRouter classify → Decide model →
    Execute (local GGUF | Fireworks API) → Validate confidence →
    Cache result → Log metrics → Return response.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any
import re

import structlog

from app.cache.manager import CacheManager, CachedResponse
from app.confidence.validator import ConfidenceValidator
from app.config import get_settings
from app.executors.base import ExecutionResult
from app.executors.fireworks import FireworksExecutor
from app.features.normalizer import RequestNormalizer
from app.features.preprocessor import PromptPreprocessor
from app.features.compiler import InferencePolicyCompiler
from app.metrics.collector import MetricsCollector, RequestMetric
from app.router.supra_router import SupraRouter

logger = structlog.get_logger(__name__)


class RoutingPipeline:
    """End-to-end routing pipeline."""

    def __init__(self) -> None:
        settings = get_settings()

        # Feature extraction & preprocessing
        self._normalizer = RequestNormalizer()
        self._preprocessor = PromptPreprocessor()
        self._compiler = InferencePolicyCompiler()

        # Execution backends
        self._fireworks = FireworksExecutor()

        # Local model executor
        self._local = None
        if settings.local_model_enabled:
            try:
                from app.executors.local import LocalExecutor
                local = LocalExecutor(
                    context_length=settings.local_model_context_length,
                    n_threads=settings.local_model_threads,
                )
                self._local = local
                logger.info("pipeline.local_executor_ready")
            except ImportError:
                logger.warning("pipeline.llama_cpp_not_installed")

        # Support
        self._cache = CacheManager()
        self._validator = ConfidenceValidator()
        self._metrics = MetricsCollector()

        # Supra-Router-51M (ML-based routing)
        self._supra_router: SupraRouter | None = None
        if settings.supra_router_enabled:
            try:
                self._supra_router = SupraRouter(
                    server_url=settings.supra_router_url,
                )
                logger.info("pipeline.supra_router_enabled", url=settings.supra_router_url)
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

        # Step 3: Classify with SupraRouter
        supra_result = None
        if self._supra_router is not None:
            try:
                supra_result = await self._supra_router.classify(prompt)
            except Exception as exc:
                logger.warning("pipeline.supra_router_error", error=str(exc))
                supra_result = None

        # Derive task_type
        if supra_result is not None and supra_result.success:
            domain = supra_result.domain
            route_decision = supra_result.route
            complexity = supra_result.complexity
            if supra_result.needs_code:
                task_type = "code"
            elif supra_result.needs_math:
                task_type = "math"
            else:
                task_type = self._map_domain_to_task_type(domain, "general_qa")
        else:
            domain = "unknown"
            route_decision = "big model"
            complexity = 1.0
            task_type = "general_qa"

        # Apply task type safeguards based on prompt contents
        features = self._detect_linguistic_features(prompt)
        if features["counting"]:
            task_type = "counting"
        elif features["sentiment"]:
            task_type = "extraction"
        elif features["logic"]:
            task_type = "reasoning"
        elif features["sql"]:
            task_type = "code"

        logger.info(
            "pipeline.classification",
            domain=domain,
            complexity=complexity,
            route=route_decision,
            task_type=task_type,
        )

        # Step 5: Route
        model_selected, routing_explanation = self._make_routing_decision(
            prompt, task_type, route_decision, complexity, force_model
        )

        logger.info("pipeline.routed", model=model_selected, reason=routing_explanation)

        # Preprocess
        preprocessed = await self._preprocessor.process(prompt=prompt)
        base_system = "\n\n".join(preprocessed.system_addons)

        # Populated task vector features to trigger compiler registry rules
        task_dict = {
            "task_type": task_type,
            "math": 1.0 if task_type in ("math", "counting") else 0.0,
            "code": 1.0 if task_type == "code" else 0.0,
            "reasoning": 1.0 if task_type == "reasoning" else 0.0,
            "retrieval": 1.0 if task_type == "retrieval" else 0.0,
            "creative": 1.0 if task_type == "creative" else 0.0,
            "extraction": 1.0 if task_type == "extraction" else 0.0,
            "translation": 1.0 if task_type == "translation" else 0.0,
            "general_qa": 1.0 if task_type == "general_qa" else 0.0,
        }

        # Decide budget bucket based on task type and complexity
        output_budget_bucket = "Small"
        if task_type in ("code", "creative") or complexity > 0.6:
            output_budget_bucket = "Large"

        resource_dict = {
            "input_tokens": len(prompt.split()) * 1.5,
            "complexity": complexity,
            "output_budget_bucket": output_budget_bucket,
            "prompt_text": prompt,
        }

        risk_dict = {
            "json_required": "json" in prompt.lower(),
            "needs_high_accuracy": required_accuracy >= 0.8,
        }

        # Settings
        settings = get_settings()
        confidence_threshold = settings.confidence_threshold  # default 0.8
        max_depth = settings.max_escalation_depth  # default 2

        # Override confidence threshold for logic puzzles, math, and general QA to 0.7
        current_threshold = confidence_threshold
        if task_type in ("reasoning", "math", "general_qa"):
            current_threshold = 0.7

        # Step 6: Execution Loop with Escalation
        current_model = model_selected
        escalation_depth = 0
        best_result: ExecutionResult | None = None

        while escalation_depth <= max_depth:
            compiled_policy = self._compiler.compile(
                prompt=prompt,
                task_vector=task_dict,
                resource_vector=resource_dict,
                risk_vector=risk_dict,
                model=current_model,
                base_system_prompt=base_system,
            )

            system_prompt = compiled_policy.system_prompt

            result: ExecutionResult | None = None

            if current_model.startswith("local:") and self._local:
                # Local Executor
                result = await self._local.execute(
                    prompt=preprocessed.forwarded,
                    model_id=current_model,
                    task_type=task_type,
                    system_prompt=system_prompt,
                )
                
                is_broken = (
                    not result.response.strip()
                    or result.response.startswith("[LOCAL_ERROR]")
                    or result.response.startswith("[LOCAL_UNAVAILABLE]")
                )
                if is_broken:
                    logger.info("pipeline.local_broken", model=current_model)
                    result = None
            else:
                # Fireworks Executor
                _TASK_MAX_TOKENS = {
                    "code": 2048,
                    "math": 2048,
                    "reasoning": 2048,
                    "creative": 2048,
                    "general_qa": 1024,
                    "extraction": 1024,
                    "verification": 512,
                    "translation": 1024,
                    "retrieval": 512,
                }
                current_max_tokens = _TASK_MAX_TOKENS.get(task_type, 1024)
                _forwarded = preprocessed.forwarded

                if task_type == "code":
                    if "sql" in _forwarded.lower() or "select" in _forwarded.lower():
                        _forwarded += "\n\nReply with ONLY the SQL query."
                    else:
                        _forwarded += "\n\nReply with ONLY the code. No explanation, no test cases, no commentary."
                elif task_type == "extraction":
                    _forwarded += "\n\nReply with ONLY the exact requested extracted data. If JSON is requested, output ONLY valid JSON. No explanation, no commentary, no markdown wrapping like ```json."
                elif task_type in ["translation", "retrieval", "verification"]:
                    _forwarded += "\n\nReply with ONLY the requested output. No explanation, no commentary, no preamble."
                elif task_type == "math":
                    _forwarded += "\n\nShow exact calculation steps with precise numbers."

                try:
                    result = await self._fireworks.execute(
                        prompt=_forwarded,
                        model_id=current_model,
                        task_type=task_type,
                        system_prompt=system_prompt,
                        max_tokens=current_max_tokens,
                        reasoning_effort="none",
                    )

                    result.cost = (
                        result.tokens_input * 0.0003 / 1000
                        + result.tokens_output * 0.0012 / 1000
                    )
                except Exception as exc:
                    logger.error("pipeline.fireworks_failed", model=current_model, error=str(exc))
                    result = None

            if result is not None:
                # Perform confidence validation
                validation = self._validator.validate(
                    response=result.response,
                    task_type=task_type,
                    expected_json=False,  
                    expected_code=(task_type == "code"),
                    expected_length="medium",
                )
                result.confidence = min(result.confidence, validation.confidence)

                # Keep the first/best result so far
                if best_result is None or result.confidence > best_result.confidence:
                    best_result = result

                # If confidence meets threshold, stop
                if result.confidence >= current_threshold:
                    logger.info("pipeline.confidence_satisfied", model=current_model, confidence=result.confidence)
                    break

            # Attempt escalation if not satisfied
            if escalation_depth < max_depth:
                next_model = self._get_escalation_model(current_model, task_type)
                if next_model:
                    logger.info(
                        "pipeline.escalating",
                        from_model=current_model,
                        to_model=next_model,
                        depth=escalation_depth + 1,
                        confidence=result.confidence if result else 0.0
                    )
                    current_model = next_model
                    escalation_depth += 1
                    continue

            # If no escalation model or reached max depth, stop
            break

        if best_result is None:
            best_result = ExecutionResult(
                response="Failed to generate a response.",
                model_used=model_selected,
                confidence=0.0,
            )

        elapsed = (time.perf_counter() - pipeline_start) * 1000

        # Cache
        self._cache.set(
            raw_hash=raw_hash,
            normalized_hash=normalized.prompt_hash,
            response=CachedResponse(
                response=best_result.response,
                model_used=best_result.model_used,
                cost=best_result.cost,
                confidence=best_result.confidence,
                task_vector=task_dict,
                routing_explanation=routing_explanation,
            ),
        )

        self._metrics.record(RequestMetric(
            prompt_hash=normalized.prompt_hash,
            model_used=best_result.model_used,
            task_type=task_type,
            tokens_input=best_result.tokens_input,
            tokens_output=best_result.tokens_output,
            latency_ms=round(elapsed, 1),
            cost=best_result.cost,
            confidence=best_result.confidence,
            cache_hit=False,
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
            "escalated": (escalation_depth > 0),
            "escalation_depth": escalation_depth,
            "task_vector": task_dict,
            "routing_explanation": routing_explanation,
            "compiler_metadata": compiled_policy.metadata,
        }

    def _make_routing_decision(
        self, prompt: str, task_type: str, supra_route: str, complexity: float, force_model: str | None
    ) -> tuple[str, str]:
        """Route to local or cloud models based on SupraRouter classification."""
        if force_model:
            return force_model, f"Forced by caller: {force_model}"

        input_length = len(prompt.split())
        
        # If context is very long, use minimax-m3 directly
        if input_length > 3000:
            return "minimax-m3", f"Large context ({input_length} words) -> minimax-m3"

        # Counting and logic puzzles route directly to cloud minimax-m3
        if task_type in ("counting", "reasoning"):
            return "minimax-m3", f"Task type {task_type} -> minimax-m3 directly for high accuracy"

        # Code tasks support direct cloud routing on high complexity
        if task_type == "code":
            if supra_route == "big model" or complexity > 0.6:
                return "kimi-k2p7-code", "High complexity Code task -> kimi-k2p7-code"
            return "local:phi-4-mini", "Code task -> local:phi-4-mini"

        # General Math tasks start on local:phi-4-mini
        if task_type == "math":
            return "local:phi-4-mini", "Math task -> local:phi-4-mini first"

        # All other categories (sentiment, NER, translation, summarization, creative, Q&A)
        # start on local:ministral-3b
        return "local:ministral-3b", f"Task type {task_type} -> local:ministral-3b"

    def _get_escalation_model(self, current_model: str, task_type: str) -> str | None:
        """Decide the next model to try when the current model has low confidence."""
        if current_model.startswith("local:"):
            # Local model failed, escalate to cloud
            if task_type == "code":
                return "kimi-k2p7-code"
            else:
                return "minimax-m3"
        elif current_model == "kimi-k2p7-code":
            # Kimi failed on code, try minimax-m3 as final resort
            return "minimax-m3"
        return None

    @staticmethod
    def _detect_linguistic_features(prompt: str) -> dict[str, bool]:
        """Detect generic linguistic task features from prompt text."""
        p_lower = prompt.lower()
        return {
            "counting": any(w in p_lower for w in ("count ", "how many ", "frequency of", "occurrences of", "syllable")),
            "sentiment": any(w in p_lower for w in ("sentiment", "classify", "positive, negative", "positive or negative")),
            "logic": any(w in p_lower for w in ("deduction", "logic puzzle", "clue", "who owns", "each own")),
            "sql": any(w in p_lower for w in ("sql", "query", "select ", "database")),
        }

    @staticmethod
    def _map_domain_to_task_type(domain: str, fallback: str) -> str:
        """Map Supra-Router domain labels to internal task types."""
        domain_lower = domain.lower()
        mapping = {
            "math": "math", "probability": "math", "algebra": "math",
            "code": "code", "programming": "code", "sql": "code", "python": "code", "debug": "code",
            "reasoning": "reasoning", "logic": "reasoning", "puzzle": "reasoning",
            "creative": "creative", "writing": "creative", "poetry": "creative",
            "translation": "translation", "language": "translation",
            "extraction": "extraction", "ner": "extraction", "json": "extraction",
            "sentiment": "extraction", "classification": "extraction", "summar": "extraction",
            "retrieval": "retrieval", "factual": "retrieval", "qa": "retrieval",
            "etymology": "math", "count": "math", "counting": "math",
        }
        for key, task_type in mapping.items():
            if key in domain_lower:
                return task_type
        return fallback


    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def cache(self) -> CacheManager:
        return self._cache
