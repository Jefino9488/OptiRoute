"""Minimal routing pipeline — one model, one call, structured output.

Replaces the 849-line pipeline with ~120 lines.
All tasks go to minimax-m3 with JSON mode.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any

import structlog

from app.cache.manager import CacheManager, CachedResponse
from app.features.normalizer import RequestNormalizer
from app.executors.fireworks import FireworksExecutor

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = (
    "Solve the task. Output JSON: {\"answer\": \"your answer\"}. "
    "Be concise. No preamble."
)


def _try_deterministic_count(prompt: str) -> str | None:
    """Detect counting tasks and return the answer deterministically.

    Handles patterns like:
      - "count how many times the letter 'e' appears in this sentence: '...'"
      - "how many times does 'X' appear in '...'"
      - "count the occurrences of 'X' in '...'"

    Returns the count as a string, or None if not a counting task.
    """
    # Pattern 1: "count how many times the letter/word 'X' appears in ... sentence/text: '...'"
    m = re.search(
        r"count\s+exactly\s+how\s+many\s+times\s+the\s+(?:letter|word|character)\s+"
        r"['\"](.+?)['\"]\s+appears\s+in\s+.*?[:]\s*['\"](.+?)['\"]",
        prompt,
        re.IGNORECASE,
    )
    if m:
        target, source = m.group(1), m.group(2)
        return str(source.count(target))

    # Pattern 2: "how many times does 'X' appear in '...'"
    m = re.search(
        r"how\s+many\s+times\s+(?:does|do)\s+['\"](.+?)['\"]\s+appear\s+in\s+['\"](.+?)['\"]",
        prompt,
        re.IGNORECASE,
    )
    if m:
        target, source = m.group(1), m.group(2)
        return str(source.count(target))

    # Pattern 3: "count the occurrences of 'X' in '...'"
    m = re.search(
        r"count\s+(?:the\s+)?occurrences?\s+of\s+['\"](.+?)['\"]\s+in\s+['\"](.+?)['\"]",
        prompt,
        re.IGNORECASE,
    )
    if m:
        target, source = m.group(1), m.group(2)
        return str(source.count(target))

    return None


class _MetricsStub:
    """Minimal metrics stub — satisfies API router interface."""

    def summary(self) -> dict:
        return {"note": "metrics disabled in simplified pipeline"}


class RoutingPipeline:
    """Single-model pipeline: normalize → cache → call → respond."""

    def __init__(self) -> None:
        self.cache = CacheManager()
        self.normalizer = RequestNormalizer()
        self.fireworks = FireworksExecutor()
        self.metrics = _MetricsStub()
        logger.info("pipeline.init", model="minimax-m3")

    async def route(
        self,
        prompt: str,
        required_accuracy: float = 0.75,
        force_model: str | None = None,
        enable_thinking: bool = False,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Route a single prompt through minimax-m3.

        Returns dict with keys:
            response, model_used, cost, tokens_input, tokens_output,
            latency_ms, confidence, cache_hit, routing_explanation
        """
        t0 = time.perf_counter()

        # Step 1: Normalize
        norm = self.normalizer.normalize(prompt)

        # Step 2: Cache check
        cached, tier = self.cache.get(norm.prompt_hash, norm.prompt_hash)
        if cached is not None:
            latency = (time.perf_counter() - t0) * 1000
            logger.info("pipeline.cache_hit", tier=tier, latency_ms=latency)
            return {
                "response": cached.response,
                "model_used": cached.model_used,
                "cost": 0.0,
                "tokens_input": 0,
                "tokens_output": 0,
                "latency_ms": latency,
                "confidence": cached.confidence,
                "cache_hit": True,
                "routing_explanation": f"cache:{tier}",
            }

        # Step 2b: Try deterministic counter for counting tasks
        det_count = _try_deterministic_count(prompt)
        if det_count is not None:
            latency = (time.perf_counter() - t0) * 1000
            logger.info("pipeline.deterministic_count", answer=det_count, latency_ms=latency)
            cached_response = CachedResponse(
                response=det_count,
                model_used="deterministic:count",
                cost=0.0,
                confidence=1.0,
            )
            self.cache.set(norm.prompt_hash, norm.prompt_hash, cached_response)
            return {
                "response": det_count,
                "model_used": "deterministic:count",
                "cost": 0.0,
                "tokens_input": 0,
                "tokens_output": 0,
                "latency_ms": latency,
                "confidence": 1.0,
                "cache_hit": False,
                "routing_explanation": "deterministic:count",
            }

        # Step 3: Call minimax-m3 with JSON mode
        # Add unique suffix to bust server-side prompt caching
        cache_buster = f"\n\n[ref:{uuid.uuid4().hex[:8]}]"
        result = await self.fireworks.execute(
            prompt=prompt + cache_buster,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=2000,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        # Step 4: Parse JSON response
        raw = result.response.strip()
        try:
            data = json.loads(raw)
            answer = data.get("answer", raw)
            # Handle nested dict/list — convert to string
            if isinstance(answer, (dict, list)):
                answer = json.dumps(answer, indent=2)
        except (json.JSONDecodeError, TypeError):
            answer = raw

        # Step 5: Retry if empty
        if not answer.strip():
            logger.warning("pipeline.empty_response", retrying=True)
            result = await self.fireworks.execute(
                prompt=f"Complete this task and output JSON:\n{prompt}{cache_buster}",
                system_prompt="You must provide a response. Never return empty. Output JSON.",
                max_tokens=2000,
                temperature=0.3,
                response_format={"type": "json_object"},
            )
            raw = result.response.strip()
            try:
                data = json.loads(raw)
                answer = data.get("answer", raw)
                if isinstance(answer, (dict, list)):
                    answer = json.dumps(answer, indent=2)
            except (json.JSONDecodeError, TypeError):
                answer = raw

        latency = (time.perf_counter() - t0) * 1000

        # Step 6: Cache result
        cached_response = CachedResponse(
            response=answer,
            model_used=result.model_used,
            cost=result.cost,
            confidence=1.0,
        )
        self.cache.set(norm.prompt_hash, norm.prompt_hash, cached_response)

        logger.info(
            "pipeline.complete",
            model=result.model_used,
            tokens_in=result.tokens_input,
            tokens_out=result.tokens_output,
            cost=result.cost,
            latency_ms=latency,
        )

        return {
            "response": answer,
            "model_used": result.model_used,
            "cost": result.cost,
            "tokens_input": result.tokens_input,
            "tokens_output": result.tokens_output,
            "latency_ms": latency,
            "confidence": 1.0,
            "cache_hit": False,
            "routing_explanation": "minimax-m3 (single model)",
        }
