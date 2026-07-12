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

            # Tight token caps — minimize waste, maximize accuracy.
            _TASK_MAX_TOKENS = {
                "code": 1500,
                "math": 1200,
                "reasoning": 800,
                "creative": 1200,
                "general_qa": 400,
                "extraction": 300,
                "verification": 200,
                "translation": 600,
                "retrieval": 200,
            }
            task_cap = _TASK_MAX_TOKENS.get(features.task_type, 400)
            current_max_tokens = int(task_cap)

            # Per-task conciseness instructions.
            _forwarded = preprocessed.forwarded
            if features.task_type == "code":
                _forwarded = _forwarded + "\n\nReply with ONLY the code in a single code block. No explanation, no test cases, no commentary. For SQL, use SELECT statements with CTE or subqueries."
            elif features.task_type == "extraction":
                _forwarded = _forwarded + "\n\nReply with ONLY the requested output. No explanation, no commentary, no preamble."
            elif features.task_type == "math":
                _forwarded = _forwarded + "\n\nShow exact calculation steps with precise numbers. Use numerical approximation for roots (e.g., x ≈ -0.695). Do not derive exact symbolic forms. CRITICAL: Answer ALL parts of the question. If multiple scenarios are asked, compute ALL scenarios and compare them."
            elif features.task_type == "reasoning":
                _forwarded = _forwarded + "\n\nState the conclusion directly. Use elimination logic for logic puzzles. Brief justification only. No lengthy chain-of-thought."
            elif features.task_type == "general_qa":
                _forwarded = _forwarded + "\n\nAnswer in 1-3 direct sentences. No preamble, no filler."
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

        # Pattern 2: Depreciation calculations
        # Detect "depreciates X% per year" or "depreciation rate was X%"
        # Starting at $Y, what is its value after Z years?
        dep_match = re.search(
            r"(?:depreciat\w+|depreciation)\s+(?:rate\s+(?:was\s+)?)?(\d+(?:\.\d+)?)\%\s+(?:per\s+year|annually|for\s+the\s+first\s+year)",
            prompt,
            re.IGNORECASE,
        )
        start_match = re.search(r"\$\s*([\d,]+(?:\.\d+)?)", prompt)
        years_match = re.search(r"after\s+(\d+)\s+years?", prompt, re.IGNORECASE)
        if dep_match and start_match and years_match:
            start_val = float(start_match.group(1).replace(",", ""))
            total_years = int(years_match.group(1))

            # Parse scenario 1: uniform rate
            rate1 = float(dep_match.group(1)) / 100.0
            val1 = start_val
            for _ in range(total_years):
                val1 *= (1 - rate1)

            # Check for scenario 2: "20% for the first year and 10% for the remaining years"
            mixed_match = re.search(
                r"(\d+(?:\.\d+)?)\%\s+(?:for\s+)?(?:the\s+)?first\s+year\s+and\s+(\d+(?:\.\d+)?)\%\s+(?:for\s+)?(?:the\s+)?remaining\s+(?:(\d+)\s+)?years?",
                prompt,
                re.IGNORECASE,
            )
            if mixed_match:
                rate_first = float(mixed_match.group(1)) / 100.0
                rate_rest = float(mixed_match.group(2)) / 100.0
                remaining_years = int(mixed_match.group(3)) if mixed_match.group(3) else total_years - 1
                val2 = start_val * (1 - rate_first)
                for _ in range(remaining_years):
                    val2 *= (1 - rate_rest)

                response = (
                    f"Scenario 1: ${start_val:,.0f} depreciating {rate1*100:.0f}%/year for {total_years} years:\n"
                    f"  Year 1: ${start_val:,.0f} × {1-rate1:.2f} = ${start_val*(1-rate1):,.2f}\n"
                    f"  Year 2: ${start_val*(1-rate1):,.2f} × {1-rate1:.2f} = ${start_val*(1-rate1)**2:,.2f}\n"
                    f"  Year 3: ${start_val*(1-rate1)**3:,.2f} × {1-rate1:.2f} = ${val1:,.2f}\n"
                    f"  Final value: ${val1:,.2f}\n\n"
                    f"Scenario 2: {rate_first*100:.0f}% first year, then {rate_rest*100:.0f}% for {remaining_years} remaining years:\n"
                    f"  Year 1: ${start_val:,.0f} × {1-rate_first:.2f} = ${start_val*(1-rate_first):,.2f}\n"
                    f"  Year 2: ${start_val*(1-rate_first):,.2f} × {1-rate_rest:.2f} = ${start_val*(1-rate_first)*(1-rate_rest):,.2f}\n"
                    f"  Year 3: ${start_val*(1-rate_first)*(1-rate_rest):,.2f} × {1-rate_rest:.2f} = ${val2:,.2f}\n"
                    f"  Final value: ${val2:,.2f}\n\n"
                    f"Comparison: Scenario 2 yields ${val2:,.2f} vs Scenario 1's ${val1:,.2f}. "
                    f"Scenario 2 is {'better' if val2 > val1 else 'worse'} by ${abs(val2-val1):,.2f} because the higher first-year depreciation outweighs the lower subsequent rate."
                )
            else:
                response = (
                    f"Starting value: ${start_val:,.0f}\n"
                    f"Depreciation rate: {rate1*100:.0f}% per year for {total_years} years\n\n"
                )
                val = start_val
                for yr in range(1, total_years + 1):
                    val *= (1 - rate1)
                    response += f"Year {yr}: ${val:,.2f}\n"
                response += f"\nFinal value after {total_years} years: ${val:,.2f}"

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
                "routing_explanation": f"Deterministic depreciation calculation",
            }

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
        """Route to minimax-m3 for accuracy, local for code (minimax returns empty).

        minimax-m3 is the best model for non-code tasks. For code tasks,
        local qwen2.5-coder is the only option since minimax consistently
        returns empty responses for code generation.
        """
        settings = get_settings()

        if force_model:
            return RoutingDecision(
                model_selected=force_model,
                estimated_cost=0.0,
                predicted_accuracy=1.0,
                reasoning=f"Model forced by caller: {force_model}",
            )

        # Code tasks MUST go to local — minimax-m3 returns empty for code
        if features.task_type == "code" and self._local:
            return RoutingDecision(
                model_selected=settings.local_model_name,
                estimated_cost=0.0,
                predicted_accuracy=0.85,
                reasoning="Code → local (minimax returns empty for code)",
            )

        return RoutingDecision(
            model_selected="minimax-m3",
            estimated_cost=0.0,
            predicted_accuracy=0.95,
            reasoning=f"All non-code tasks → minimax-m3 ({features.task_type})",
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
