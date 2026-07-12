"""Local LLM executor — dynamically loads Llama instances using llama-cpp-python.

Design:
- Only ONE model is loaded in memory at a time to stay under 4GB RAM.
- Caches the current model instance. If a new model is requested, unloads the old one.
- Cost is always $0 — no Fireworks tokens consumed.
"""

import asyncio
import gc
import time
from typing import Any

import structlog

from app.config import get_settings
from app.executors.base import ExecutionResult

logger = structlog.get_logger(__name__)

_TEMP_MAP: dict[str, float] = {
    "code": 0.1,
    "math": 0.0,
    "extraction": 0.0,
    "translation": 0.3,
    "reasoning": 0.2,
    "retrieval": 0.0,
    "creative": 0.6,
    "general_qa": 0.3,
}

_MAX_TOKENS_MAP: dict[str, int] = {
    "code": 2048,
    "math": 2048,
    "extraction": 1024,
    "translation": 1024,
    "reasoning": 2048,
    "retrieval": 1024,
    "creative": 2048,
    "general_qa": 1024,
}


class LocalExecutor:
    """Execute prompts via llama-cpp-python directly in memory.

    Parameters
    ----------
    context_length : int
        Maximum context window in tokens.
    n_threads : int
        CPU thread count.
    """

    def __init__(
        self,
        context_length: int = 8192,
        n_threads: int = 2,
    ) -> None:
        self._context_length = context_length
        self._n_threads = n_threads
        self._current_model_id: str | None = None
        self._llm: Any | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        
    def load(self) -> bool:
        # Compatibility stub, we lazy load models on execute
        return True
        
    @property
    def is_available(self) -> bool:
        return True

    async def _ensure_model_loaded(self, model_id: str) -> bool:
        """Ensure the requested model is loaded, unloading the old one if needed."""
        if self._current_model_id == model_id and self._llm is not None:
            return True
            
        async with self._load_lock:
            # Double-check after acquiring lock
            if self._current_model_id == model_id and self._llm is not None:
                return True
                
            settings = get_settings()
            models = settings.allowed_models
            if model_id not in models:
                logger.error("local_executor.unknown_model", model_id=model_id)
                return False
                
            model_path = models[model_id]
            
            # Unload old model
            if self._llm is not None:
                logger.info("local_executor.unloading_model", model_id=self._current_model_id)
                del self._llm
                self._llm = None
                self._current_model_id = None
                gc.collect()
                
            # Load new model in a separate thread
            logger.info("local_executor.loading_model", model_id=model_id, path=model_path)
            try:
                def _load_sync():
                    from llama_cpp import Llama  # noqa: PLC0415
                    return Llama(
                        model_path=model_path,
                        n_ctx=self._context_length,
                        n_threads=self._n_threads,
                        verbose=False,
                    )
                    
                self._llm = await asyncio.to_thread(_load_sync)
                self._current_model_id = model_id
                logger.info("local_executor.model_loaded_successfully", model_id=model_id)
                return True
            except Exception as exc:
                logger.error("local_executor.load_failed", error=str(exc))
                return False

    async def execute(
        self,
        prompt: str,
        model_id: str,
        task_type: str = "general_qa",
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> ExecutionResult:
        """Run inference using the specified local model."""
        async with self._inference_lock:
            if not await self._ensure_model_loaded(model_id):
                return ExecutionResult(
                    response=f"[LOCAL_ERROR] Failed to load local model {model_id}",
                    model_used=model_id,
                    confidence=0.0,
                )

            temp = _TEMP_MAP.get(task_type, 0.3)
            max_tok = max_tokens or _MAX_TOKENS_MAP.get(task_type, 1024)

            # Merge instructions directly into the user message to prevent chat template parsing issues
            sys_msg = system_prompt or (
                "Be concise and accurate. Give direct answers without unnecessary preamble."
            )
            full_prompt = f"Instructions: {sys_msg}\n\nTask: {prompt}"
            
            messages = [
                {"role": "user", "content": full_prompt},
            ]

            logger.info(
                "local_executor.execute_inference",
                model_id=model_id,
                task_type=task_type,
                temperature=temp,
                max_tokens=max_tok,
            )

            start = time.perf_counter()
            try:
                response = await asyncio.to_thread(
                    self._llm.create_chat_completion,
                    messages=messages,
                    max_tokens=max_tok,
                    temperature=temp,
                )
            except Exception as exc:
                logger.error("local_executor.inference_failed", error=str(exc))
                return ExecutionResult(
                    response=f"[LOCAL_ERROR] Inference failed: {exc}",
                    model_used=model_id,
                    confidence=0.0,
                )

            elapsed_ms = (time.perf_counter() - start) * 1000

            choices = response.get("choices", [])
            text: str = choices[0].get("message", {}).get("content", "") if choices else ""
            
            usage: dict = response.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)

            logger.info(
                "local_executor.inference_completed",
                latency_ms=round(elapsed_ms, 1),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )

            return ExecutionResult(
                response=text,
                model_used=model_id,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                cost=0.0,
                latency_ms=round(elapsed_ms, 1),
                confidence=1.0 if text.strip() else 0.0,
            )
