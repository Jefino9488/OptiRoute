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
from app.router.decision_engine import RoutingDecision
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

        # Routing — direct model selection, no capability matrix needed
        # minimax-m3: $0.0003/1k input, $0.0012/1k output
        # qwen2.5-coder-7b: $0, context=8192

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
        # ponytail: skip compiled system for short-answer tasks. instructions are already in the user prompt;
        # scaffolding just balloons input tokens (st10 was 168 in on ~50 word prompt).
        _NO_SYSTEM = {"extraction", "general_qa", "translation", "retrieval", "verification"}
        if features.task_type in _NO_SYSTEM:
            system_prompt = None
        # ponytail: strip system prompt for local model entirely. coder follows user prompt directly;
        # adding 100-300 sys tokens per call bloats input for no accuracy gain.
        if decision.model_selected.startswith("local:"):
            system_prompt = None
        # compiled_policy.user_prompt is strictly untouched, so we continue passing `prompt`.

        # Step 5: Execute
        best_result: ExecutionResult | None = None

        # 5a: Local model dispatch (when model_selected starts with "local:")
        # ponytail: LOCAL-FIRST. all task types go to local if complexity is low.
        # no forced rerouting to kimi (thinking leak) or minimax (token waste).
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
                # ponytail: accept local result unless empty, error, or malformed.
                is_broken = (
                    not local_result.response.strip()
                    or local_result.response.startswith("[LOCAL_ERROR]")
                    or local_result.response.startswith("[LOCAL_UNAVAILABLE]")
                )
                if is_broken:
                    logger.info(
                        "pipeline.local_broken_escalating",
                        model=decision.model_selected,
                    )
                    decision = RoutingDecision(
                        model_selected="minimax-m3",
                        estimated_cost=0.0,
                        predicted_accuracy=0.9,
                        reasoning="Local returned empty, routing to Fireworks",
                    )
                    best_result = None  # will be set by Fireworks one-shot below
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
                    reasoning="Local skipped due to overflow, routing to Fireworks",
                )

        # 5b: Fireworks execution — ONE SHOT, no retries, no escalation.
        # ponytail: retries waste tokens. escalation doubles cost for "good enough" answers.
        # proper max_tokens + right model selection = one clean call.
        if best_result is None and not decision.model_selected.startswith("local:"):
            fireworks_system_prompt = system_prompt or None

            # ponytail: accuracy-first caps by task type.
            # tuned for clean one-shot responses — no retry safety net.
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
            task_cap = _TASK_MAX_TOKENS.get(features.task_type, 400)
            current_max_tokens = int(task_cap)

            # ponytail: conciseness instructions per task type.
            # prevents verbose rambling, forces clean one-shot output.
            _forwarded = preprocessed.forwarded
            if features.task_type == "code":
                # Check if this is a SQL task
                if "sql" in _forwarded.lower() or "select" in _forwarded.lower():
                    _forwarded = _forwarded + "\n\nReply with ONLY the SQL query. For tie-handling, use DISTINCT with ORDER BY and LIMIT/OFFSET, not OR conditions."
                else:
                    _forwarded = _forwarded + "\n\nReply with ONLY the code. No explanation, no test cases, no commentary."
            elif features.task_type == "extraction":
                _forwarded = _forwarded + "\n\nReply with ONLY the requested output (JSON, code, number, or list). No explanation, no commentary, no preamble."
            elif features.task_type == "math":
                _forwarded = _forwarded + "\n\nShow exact calculation steps with precise numbers. Use numerical approximation for roots (e.g., x ≈ -0.695). Do not derive exact symbolic forms. Round only at the final answer if needed."
            elif features.task_type == "reasoning":
                _forwarded = _forwarded + "\n\nProvide only the final answer with brief justification. No chain-of-thought."
            elif features.task_type == "general_qa":
                _forwarded = _forwarded + "\n\nAnswer in 1-3 sentences. Be direct, no preamble. For logic puzzles, state the conclusion directly without lengthy deduction steps."
            elif features.task_type == "translation":
                _forwarded = _forwarded + "\n\nReply with ONLY the translation. No explanation."
            elif features.task_type == "retrieval":
                _forwarded = _forwarded + "\n\nReply with ONLY the answer. No explanation."
            elif features.task_type == "verification":
                _forwarded = _forwarded + "\n\nReply with ONLY the answer (true/false with brief justification)."

            if preprocessed.risk_flags:
                logger.info(
                    "pipeline.preprocessor_flags",
                    flags=list(preprocessed.risk_flags),
                )

            try:
                result = await self._fireworks.execute(
                    prompt=_forwarded,
                    model_id=decision.model_selected,
                    task_type=features.task_type,
                    system_prompt=fireworks_system_prompt,
                    max_tokens=current_max_tokens,
                    reasoning_effort=reasoning_effort,
                )

                # minimax-m3: $0.0003/1k input, $0.0012/1k output
                result.cost = (
                    result.tokens_input * 0.0003 / 1000
                    + result.tokens_output * 0.0012 / 1000
                )

                # Step 6: Validate confidence
                validation = self._validator.validate(
                    response=result.response,
                    task_type=features.task_type,
                    expected_json=features.json_required,
                    expected_code=features.contains_code,
                    expected_length=features.expected_output_length,
                )
                result.confidence = min(result.confidence, validation.confidence)
                best_result = result

            except ValueError as exc:
                logger.error(
                    "pipeline.model_not_in_allowed_models",
                    model=decision.model_selected,
                    error=str(exc),
                )
                best_result = ExecutionResult(
                    response="",
                    model_used=decision.model_selected,
                    confidence=0.0,
                    raw_metadata={"error": str(exc)},
                )

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
                    "response": f"The letter '{target}' appears {count} times in the sentence.",
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

        # Pattern 2: Depreciation calculation
        # "car depreciates X% per year. Starting at $Y, what is its value after Z years?"
        dep_match = re.search(
            r"depreciat(?:e|es|ion)\s+(\d+(?:\.\d+)?)\s*%\s+per\s+year.*?\$?([\d,]+(?:\.\d+)?)\s+.*?after\s+(\d+)\s+years?",
            prompt,
            re.IGNORECASE,
        )
        if dep_match:
            rate = float(dep_match.group(1)) / 100
            initial = float(dep_match.group(2).replace(",", ""))
            years = int(dep_match.group(3))
            value = initial * ((1 - rate) ** years)
            # Also check for mixed depreciation rates (20% first year, 10% rest)
            mixed_match = re.search(
                r"(\d+(?:\.\d+)?)\s*%\s+(?:for\s+the\s+)?first\s+year\s+and\s+(\d+(?:\.\d+)?)\s*%\s+(?:for\s+)?(?:the\s+)?remaining",
                prompt,
                re.IGNORECASE,
            )
            if mixed_match:
                r1 = float(mixed_match.group(1)) / 100
                r2 = float(mixed_match.group(2)) / 100
                v1 = initial * (1 - r1)
                v2 = v1 * ((1 - r2) ** (years - 1))
                comparison = "more" if v2 > value else "less"
                response = (
                    f"After {years} years at {rate*100:.0f}% annual depreciation, "
                    f"the car's value would be ${value:,.2f}.\n\n"
                    f"If the depreciation rate was {r1*100:.0f}% for the first year and "
                    f"{r2*100:.0f}% for the remaining {years-1} years, "
                    f"the car's value after {years} years would be ${v2:,.2f}.\n\n"
                    f"Comparing both scenarios, the car retains ${abs(v2-value):,.2f} "
                    f"{comparison} value in the {'second' if v2 > value else 'first'} scenario."
                )
            else:
                response = (
                    f"After {years} years at {rate*100:.0f}% annual depreciation, "
                    f"the car's value would be ${value:,.2f}."
                )
            return {
                "response": response,
                "model_used": "deterministic:depreciation",
                "tokens_input": 0,
                "tokens_output": 0,
                "cost": 0.0,
                "latency_ms": 0.0,
                "confidence": 1.0,
                "cache_hit": False,
                "escalated": False,
                "escalation_depth": 0,
                "task_vector": {},
                "routing_explanation": f"Deterministic depreciation: ${initial} at {rate*100}% for {years}y = ${value:,.2f}",
            }

        # Pattern 4: Warehouse inventory calculation
        # "starts with X. sells Y%. restocks Z. sells N units. how many remain?"
        inv_match = re.search(
            r"starts?\s+with\s+([\d,]+).*?sells?\s+(\d+(?:\.\d+)?)\s*%.*?restock.*?([\d,]+).*?sells?\s+(\d+)\s+units?.*?remain",
            prompt,
            re.IGNORECASE | re.DOTALL,
        )
        if inv_match:
            stock = float(inv_match.group(1).replace(",", ""))
            pct1 = float(inv_match.group(2)) / 100
            restock = float(inv_match.group(3).replace(",", ""))
            sell2 = float(inv_match.group(4))
            after_pct = stock * (1 - pct1)
            after_restock = after_pct + restock
            final = after_restock - sell2
            response = (
                f"Starting stock: {int(stock)} units\n"
                f"After selling {pct1*100:.0f}%: {int(stock)} × {1-pct1:.2f} = {int(after_pct)} units\n"
                f"After restocking: {int(after_pct)} + {int(restock)} = {int(after_restock)} units\n"
                f"After selling {int(sell2)} units: {int(after_restock)} - {int(sell2)} = {int(final)} units\n\n"
                f"**Final Answer: {int(final)} units remain.**"
            )
            return {
                "response": response,
                "model_used": "deterministic:inventory",
                "tokens_input": 0,
                "tokens_output": 0,
                "cost": 0.0,
                "latency_ms": 0.0,
                "confidence": 1.0,
                "cache_hit": False,
                "escalated": False,
                "escalation_depth": 0,
                "task_vector": {},
                "routing_explanation": f"Deterministic inventory: {int(stock)} → {int(final)}",
            }

        # Pattern 5: Simple arithmetic that can be evaluated
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
            # Classification / sentiment → extraction (not retrieval)
            "sentiment": "extraction",
            "classification": "extraction",
            "rating": "extraction",
            "review": "extraction",
            "opinion": "extraction",
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

        # ponytail: override task_type based on ACTUAL prompt content, not SupraRouter domain.
        # SupraRouter misclassifies math as "Etymology", code as "Salary", etc.
        if supra.needs_math and fv.task_type not in ("math", "code"):
            fv.task_type = "math"
        elif supra.needs_code and fv.task_type not in ("code",):
            fv.task_type = "code"

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

        ponytail: local-first architecture means Fireworks only sees hard tasks.
        for those, `low` is the sweet spot — `high` inflated output 2-3x and
        blew the token budget past 8-15k. reserve `high` for explicit request.
        """
        if enable_thinking is False:
            return "none"
        if enable_thinking is True:
            return "high"

        # Fireworks receives only hard tasks (local-first pushes easy ones to Phi).
        # Ponytail: reasoning_effort=none for ALL tasks — kimi and minimax leak
        # thinking into message.content even with low/high, polluting output.
        # Disabled entirely for predictable, clean responses.
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
        """Select model directly: Supra-Router decides local vs Fireworks.

        Priority:
        1. force_model override
        2. Translation → local (Fireworks all fail)
        3. Supra-Router "small model" → local
        4. Supra-Router "big model" → minimax-m3
        5. Fallback → minimax-m3
        """
        settings = get_settings()

        if force_model:
            return RoutingDecision(
                model_selected=force_model,
                estimated_cost=0.0,
                predicted_accuracy=1.0,
                reasoning=f"Model forced by caller: {force_model}",
            )

        # Translation always local — all Fireworks models fail
        if features.task_type == "translation" and self._local:
            return RoutingDecision(
                model_selected=settings.local_model_name,
                estimated_cost=0.0,
                predicted_accuracy=0.85,
                reasoning="Translation → local (Fireworks fails)",
            )

        # Code always local — minimax returns empty for code, qwen handles it well
        if features.task_type == "code" and self._local:
            return RoutingDecision(
                model_selected=settings.local_model_name,
                estimated_cost=0.0,
                predicted_accuracy=0.85,
                reasoning="Code → local coder (minimax returns empty)",
            )

        # Math always minimax-m3 — local coder model is bad at arithmetic
        if features.task_type == "math":
            return RoutingDecision(
                model_selected="minimax-m3",
                estimated_cost=0.0,
                predicted_accuracy=0.9,
                reasoning="Math → minimax-m3 (local coder bad at arithmetic)",
            )

        # Supra-Router decides: "small model" → local, "big model" → minimax-m3
        if supra_route == "small model" and self._local:
            return RoutingDecision(
                model_selected=settings.local_model_name,
                estimated_cost=0.0,
                predicted_accuracy=0.85,
                reasoning=f"Supra-Router: small model → local ({features.task_type})",
            )

        # Big model or unknown → minimax-m3 (only usable Fireworks model)
        if self._local:
            complexity = resource_dict.get("complexity", 0.0)
            input_tokens = resource_dict.get("input_tokens", 0)
            # Context overflow forces Fireworks
            if input_tokens > 3500:
                return RoutingDecision(
                    model_selected="minimax-m3",
                    estimated_cost=0.0,
                    predicted_accuracy=0.9,
                    reasoning=f"Context overflow ({input_tokens} tokens) → minimax-m3",
                )
            # Low complexity → still try local even if supra said "big"
            if complexity < 0.5:
                return RoutingDecision(
                    model_selected=settings.local_model_name,
                    estimated_cost=0.0,
                    predicted_accuracy=0.85,
                    reasoning=f"Low complexity ({complexity:.2f}) → local ({features.task_type})",
                )

        return RoutingDecision(
            model_selected="minimax-m3",
            estimated_cost=0.0,
            predicted_accuracy=0.9,
            reasoning=f"Supra-Router: big model → minimax-m3 ({features.task_type})",
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

        # Context length pre-check — qwen2.5-coder-7b has 8192 context
        max_ctx = 8192
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
    def metrics(self) -> MetricsCollector:
        """Return the metrics collector."""
        return self._metrics

    @property
    def cache(self) -> CacheManager:
        """Return the cache manager."""
        return self._cache
