"""Fireworks AI executor — OpenAI-compatible API via the openai SDK.

All scored inference goes through this executor.  Uses async httpx under
the hood via ``openai.AsyncOpenAI``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError, NotFoundError

from app.config import get_settings
from app.executors.base import ExecutionResult

logger = structlog.get_logger(__name__)

# Per-task-type temperature presets.
_TEMP_MAP: dict[str, float] = {
    "code": 0.1,
    "math": 0.0,
    "extraction": 0.0,
    "translation": 0.3,
    "reasoning": 0.3,
    "retrieval": 0.0,
    "creative": 0.7,
    "general_qa": 0.4,
}



class FireworksExecutor:
    """Execute prompts via the Fireworks AI inference API.

    The executor is configured once and reused for all requests.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(
            api_key=settings.fireworks_api_key,
            base_url=settings.fireworks_base_url,
        )
        self._models = settings.allowed_models

    async def execute(
        self,
        prompt: str,
        model_id: str,
        task_type: str = "general_qa",
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> ExecutionResult:
        """Send a prompt to Fireworks and return the result.

        Parameters
        ----------
        prompt : str
            The user prompt.
        model_id : str
            Short model name (e.g. ``"gemma-4-26b-a4b-it"``).
        task_type : str
            Dominant task type — drives temperature and max_tokens defaults.
        system_prompt : str | None
            Optional system message.
        max_tokens : int | None
            Override for max output tokens.
        temperature : float | None
            Override for sampling temperature.
        reasoning_effort : str | None
            Fireworks reasoning_effort parameter.  Supported values:
            ``"none"`` (thinking off), ``"low"``, ``"medium"``, ``"high"``,
            ``"max"``.  ``None`` omits the parameter (model default).

        Returns
        -------
        ExecutionResult
        """
        settings = get_settings()
        full_model = settings.get_model_path(model_id)

        temp = temperature if temperature is not None else _TEMP_MAP.get(task_type, 0.4)
        max_tok = max_tokens or 4096

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.info(
            "fireworks.execute",
            model=model_id,
            task_type=task_type,
            temperature=temp,
            max_tokens=max_tok,
            reasoning_effort=reasoning_effort,
        )

        start = time.perf_counter()
        
        max_retries = 5
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": full_model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_tok,
                }
                if reasoning_effort is not None:
                    kwargs["reasoning_effort"] = reasoning_effort
                response = await self._client.chat.completions.create(**kwargs)
                break  # Success
            except RateLimitError as exc:
                if attempt == max_retries - 1:
                    logger.error("fireworks.rate_limit_exhausted", model=model_id, error=str(exc))
                    return ExecutionResult(
                        response="[ERROR] Rate limit exceeded after retries.",
                        model_used=model_id,
                        confidence=0.0,
                        raw_metadata={"error": str(exc)},
                    )
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "fireworks.rate_limit_retry", 
                    model=model_id, 
                    attempt=attempt + 1, 
                    delay=delay
                )
                await asyncio.sleep(delay)
            except NotFoundError as exc:
                logger.warning("fireworks.not_found", model=model_id, error=str(exc))
                return ExecutionResult(
                    response="[NOT_FOUND] Model not available (404).",
                    model_used=model_id,
                    confidence=0.0,
                    raw_metadata={"error": str(exc)},
                )
            except APITimeoutError as exc:
                logger.error("fireworks.timeout", model=model_id, error=str(exc))
                return ExecutionResult(
                    response="[ERROR] Request timed out.",
                    model_used=model_id,
                    confidence=0.0,
                    raw_metadata={"error": str(exc)},
                )
            except APIError as exc:
                logger.error("fireworks.api_error", model=model_id, error=str(exc))
                return ExecutionResult(
                    response=f"[ERROR] API error: {exc}",
                    model_used=model_id,
                    confidence=0.0,
                    raw_metadata={"error": str(exc)},
                )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract usage info.
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        # Cost calculation is handled by the RoutingPipeline using its in-memory matrix.
        cost = 0.0

        text = response.choices[0].message.content or "" if response.choices else ""

        logger.info(
            "fireworks.completed",
            model=model_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            latency_ms=round(elapsed_ms, 1),
        )

        return ExecutionResult(
            response=text,
            model_used=model_id,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            cost=cost,
            latency_ms=round(elapsed_ms, 1),
            confidence=1.0 if text.strip() else 0.0,
            raw_metadata={
                "id": response.id,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
            },
        )
