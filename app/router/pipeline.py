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
            task_type = self._map_domain_to_task_type(domain, "general_qa")
        else:
            domain = "unknown"
            route_decision = "big model"
            complexity = 1.0
            task_type = "general_qa"

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

        # Dummy vectors since we removed them
        task_dict = {"task_type": task_type}
        resource_dict = {"input_tokens": len(prompt.split()) * 1.5, "complexity": complexity}
        risk_dict = {}

        compiled_policy = self._compiler.compile(
            prompt=prompt,
            task_vector=task_dict,
            resource_vector=resource_dict,
            risk_vector=risk_dict,
            model=model_selected,
            base_system_prompt=base_system,
        )

        system_prompt = compiled_policy.system_prompt
        _NO_SYSTEM = {"extraction", "general_qa", "translation", "retrieval", "verification"}
        if task_type in _NO_SYSTEM or model_selected.startswith("local:"):
            system_prompt = None

        # Execute
        best_result: ExecutionResult | None = None

        if model_selected.startswith("local:") and self._local:
            best_result = await self._local.execute(
                prompt=preprocessed.forwarded,
                model_id=model_selected,
                task_type=task_type,
                system_prompt=system_prompt,
            )
            is_broken = (
                not best_result.response.strip()
                or best_result.response.startswith("[LOCAL_ERROR]")
                or best_result.response.startswith("[LOCAL_UNAVAILABLE]")
            )
            if is_broken:
                logger.info("pipeline.local_broken_escalating", model=model_selected)
                model_selected = "minimax-m3"
                best_result = None

        if best_result is None:
            # Fireworks Execution
            _TASK_MAX_TOKENS = {
                "code": 1500,
                "math": 1000,
                "reasoning": 1200,
                "creative": 1500,
                "general_qa": 600,
                "extraction": 300,
                "verification": 150,
                "translation": 500,
                "retrieval": 200,
            }
            current_max_tokens = _TASK_MAX_TOKENS.get(task_type, 400)
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
                    model_id=model_selected,
                    task_type=task_type,
                    system_prompt=system_prompt,
                    max_tokens=current_max_tokens,
                    reasoning_effort="none",
                )

                result.cost = (
                    result.tokens_input * 0.0003 / 1000
                    + result.tokens_output * 0.0012 / 1000
                )
                
                # Confidence Validation
                validation = self._validator.validate(
                    response=result.response,
                    task_type=task_type,
                    expected_json=False,  
                    expected_code=(task_type == "code"),
                    expected_length="medium",
                )
                result.confidence = min(result.confidence, validation.confidence)
                best_result = result
            except Exception as exc:
                logger.error("pipeline.fireworks_failed", error=str(exc))
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
            "escalated": False,
            "escalation_depth": 0,
            "task_vector": task_dict,
            "routing_explanation": routing_explanation,
            "compiler_metadata": compiled_policy.metadata,
        }

    def _make_routing_decision(
        self, prompt: str, task_type: str, supra_route: str, complexity: float, force_model: str | None
    ) -> tuple[str, str]:
        """Route to Ministral-3B or Phi-4-Mini if small, else Minimax."""
        if force_model:
            return force_model, f"Forced by caller: {force_model}"

        input_length = len(prompt.split())
        
        # If big model required by Supra or context overflow -> Remote
        if supra_route == "big model" or input_length > 3500:
            return "minimax-m3", f"Supra route '{supra_route}' or large context -> minimax-m3"

        # It's a small task, use local model based on category
        ministral_categories = {"extraction", "translation", "reasoning", "general_qa", "retrieval"}
        phi_categories = {"math", "code"}

        # Keyword overrides for hybrid or specific tasks
        prompt_lower = prompt.lower()
        if "sentiment" in prompt_lower or "summarize" in prompt_lower or "count" in prompt_lower:
            task_type = "extraction"

        if task_type in phi_categories:
            return "local:phi-4-mini", f"Task type {task_type} -> local:phi-4-mini"
        elif task_type in ministral_categories:
            return "local:ministral-3b", f"Task type {task_type} -> local:ministral-3b"
            
        # Default fallback for small models
        return "local:ministral-3b", f"Default small model fallback -> local:ministral-3b"

    @staticmethod
    def _map_domain_to_task_type(domain: str, fallback: str) -> str:
        """Map Supra-Router domain labels to internal task types."""
        domain_lower = domain.lower()
        mapping = {
            "math": "math", "probability": "math", "algebra": "math",
            "code": "code", "programming": "code", "sql": "code", "python": "code",
            "reasoning": "reasoning", "logic": "reasoning", "puzzle": "reasoning",
            "creative": "creative", "writing": "creative", "poetry": "creative",
            "translation": "translation", "language": "translation",
            "extraction": "extraction", "ner": "extraction", "json": "extraction",
            "sentiment": "extraction", "classification": "extraction",
            "retrieval": "retrieval", "factual": "retrieval", "qa": "retrieval",
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
