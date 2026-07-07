"""Fireworks AI executor — OpenAI-compatible API via the openai SDK.

All scored inference goes through this executor.  Uses async httpx under
the hood via ``openai.AsyncOpenAI``.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

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
    "retrieval": 0.2,
    "creative": 0.7,
    "general_qa": 0.4,
}

# Maximum output tokens per task type.
_MAX_TOKENS_MAP: dict[str, int] = {
    "code": 2048,
    "math": 512,
    "extraction": 1024,
    "translation": 1024,
    "reasoning": 1500,
    "retrieval": 512,
    "creative": 2048,
    "general_qa": 1024,
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

        Returns
        -------
        ExecutionResult
        """
        settings = get_settings()
        full_model = settings.get_model_path(model_id)

        temp = temperature if temperature is not None else _TEMP_MAP.get(task_type, 0.4)
        max_tok = max_tokens or _MAX_TOKENS_MAP.get(task_type, 1024)

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
        )

        start = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(
                model=full_model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
            )
        except RateLimitError as exc:
            logger.error("fireworks.rate_limit", model=model_id, error=str(exc))
            return ExecutionResult(
                response="[ERROR] Rate limit exceeded. Please retry.",
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

        # Compute cost.
        matrix_path = settings.capability_matrix_path
        # We import here to avoid circular imports at module level.
        from app.router.capability_matrix import CapabilityMatrix

        try:
            matrix = CapabilityMatrix(matrix_path)
            cost = matrix.estimate_cost(model_id, tokens_in, tokens_out) or 0.0
        except Exception:
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
