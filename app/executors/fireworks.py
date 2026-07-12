"""Fireworks AI executor — simplified for single-model pipeline."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError, NotFoundError

from app.config import get_settings
from app.executors.base import ExecutionResult

logger = structlog.get_logger(__name__)


class FireworksExecutor:
    """Execute prompts via the Fireworks AI inference API."""

    def __init__(self) -> None:
        settings = get_settings()
        import httpx
        self._client = AsyncOpenAI(
            api_key=settings.fireworks_api_key,
            base_url=settings.fireworks_base_url,
            timeout=httpx.Timeout(connect=5.0, read=90.0, write=5.0, pool=10.0),
            max_retries=0,
        )
        self._model_id = settings.get_model_path("minimax-m3")

    async def execute(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2000,
        temperature: float = 0.1,
        response_format: dict | None = None,
    ) -> ExecutionResult:
        """Send a prompt to minimax-m3 and return the result."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        logger.info(
            "fireworks.execute",
            model="minimax-m3",
            temperature=temperature,
            max_tokens=max_tokens,
            has_json_mode=response_format is not None,
        )

        start = time.perf_counter()

        # Retry on rate limit (2 attempts)
        for attempt in range(2):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                break
            except RateLimitError as exc:
                if attempt == 1:
                    logger.error("fireworks.rate_limit_exhausted", error=str(exc))
                    return ExecutionResult(
                        response="[ERROR] Rate limit exceeded.",
                        model_used="minimax-m3",
                        confidence=0.0,
                        raw_metadata={"error": str(exc)},
                    )
                delay = 2.0 * (2 ** attempt)
                logger.warning("fireworks.rate_limit_retry", attempt=attempt + 1, delay=delay)
                await asyncio.sleep(delay)
            except (NotFoundError, APITimeoutError, APIError) as exc:
                logger.error("fireworks.error", error=str(exc))
                return ExecutionResult(
                    response=f"[ERROR] {exc}",
                    model_used="minimax-m3",
                    confidence=0.0,
                    raw_metadata={"error": str(exc)},
                )

        elapsed_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        text = response.choices[0].message.content or "" if response.choices else ""

        logger.info(
            "fireworks.completed",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=round(elapsed_ms, 1),
        )

        return ExecutionResult(
            response=text,
            model_used="minimax-m3",
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            cost=0.0,
            latency_ms=round(elapsed_ms, 1),
            confidence=1.0 if text.strip() else 0.0,
            raw_metadata={
                "id": response.id,
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
            },
        )
