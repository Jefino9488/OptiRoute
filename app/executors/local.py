"""Local LLM executor — Qwen2.5-Coder-7B-Instruct via llama-server HTTP API.

Design:
- llama-server runs as a background process in the container (started by entrypoint.sh)
- load() pings the health endpoint synchronously to check if server is up
- execute() calls /v1/chat/completions via httpx (async, non-blocking)
- thinking:false disables chain-of-thought tokens
- Cost is always $0 — no Fireworks tokens consumed
"""

from __future__ import annotations

import time

import httpx
import requests
import structlog

from app.config import get_settings
from app.executors.base import ExecutionResult

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Per-task temperature and token budgets (mirrors fireworks.py)
# ---------------------------------------------------------------------------

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
    "code": 1500,
    "math": 1000,
    "extraction": 300,
    "translation": 500,
    "reasoning": 1200,
    "retrieval": 200,
    "creative": 1500,
    "general_qa": 600,
}


class LocalExecutor:
    """Execute prompts via llama-server HTTP API running inside the container.

    Parameters
    ----------
    model_path : str
        Path to the GGUF file (used for logging only — server manages the model).
    context_length : int
        Maximum context window in tokens.
    n_threads : int
        CPU thread count (informational only — configured at server startup).
    """

    def __init__(
        self,
        model_path: str,
        context_length: int = 16384,
        n_threads: int = 2,
    ) -> None:
        settings = get_settings()
        self._server_url = settings.local_server_url
        self._model_path = model_path
        self._model_name = settings.local_model_name
        self._available: bool = False
        self._client = httpx.AsyncClient(timeout=300.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Ping llama-server health endpoint to confirm it is running.

        Uses requests (sync) so it can be called from synchronous startup code.
        Returns True only when the server responds OK — False marks the local
        model as unavailable so the pipeline falls through to Fireworks.
        """
        import requests  # noqa: PLC0415

        logger.info("local_executor.checking_server", url=self._server_url)
        base_url = self._server_url.rstrip("/v1").rstrip("/")
        try:
            resp = requests.get(f"{base_url}/health", timeout=5.0)
            data = (
                resp.json()
                if "application/json" in resp.headers.get("content-type", "")
                else {}
            )
            if resp.status_code == 200 or data.get("status") in ("ok", "loading"):
                self._available = True
                logger.info("local_executor.server_ready")
                return True
            logger.warning("local_executor.server_unhealthy", status=resp.status_code)
        except Exception as exc:
            logger.warning("local_executor.server_not_reachable", error=str(exc))

        self._available = False
        return False

    @property
    def is_available(self) -> bool:
        """True if llama-server is reachable."""
        return self._available

    # ------------------------------------------------------------------
    # Routing — disabled, heuristic engine handles routing
    # ------------------------------------------------------------------

    async def route(self, prompt: str) -> str | None:  # noqa: ARG002
        """Routing via local model is disabled — always returns None.

        The heuristic decision engine (capability matrix) handles all routing.
        This method is kept for interface compatibility with pipeline.py.
        """
        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        prompt: str,
        task_type: str = "general_qa",
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> ExecutionResult:
        """Run inference via llama-server /v1/chat/completions endpoint.

        Parameters
        ----------
        prompt : str
            The user prompt.
        task_type : str
            Task category — drives temperature and max_tokens defaults.
        system_prompt : str | None
            Optional system message override.
        max_tokens : int | None
            Override for max output tokens.

        Returns
        -------
        ExecutionResult
            cost is always 0.0 — no Fireworks tokens consumed.
        """
        if not self.is_available:
            return ExecutionResult(
                response="[LOCAL_UNAVAILABLE] llama-server is not running.",
                model_used=self._model_name,
                confidence=0.0,
            )

        temp = _TEMP_MAP.get(task_type, 0.3)
        max_tok = max_tokens or _MAX_TOKENS_MAP.get(task_type, 1024)

        sys_msg = system_prompt or (
            "You are a helpful assistant. Be concise and accurate. "
            "Give direct answers without unnecessary preamble."
        )

        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]

        logger.info(
            "local_executor.execute_api",
            task_type=task_type,
            temperature=temp,
            max_tokens=max_tok,
            url=self._server_url,
        )

        start = time.perf_counter()
        try:
            response = await self._client.post(
                f"{self._server_url}/chat/completions",
                json={
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_tok,
                    "thinking": False,      # disable chain-of-thought
                    "cache_prompt": False,  # no KV cache accumulation between requests
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException:
            logger.error("local_executor.timeout", timeout_s=300)
            return ExecutionResult(
                response="[LOCAL_ERROR] llama-server timed out",
                model_used=self._model_name,
                confidence=0.0,
            )
        except Exception as exc:
            # Connection error (server crashed) — wait and retry once
            logger.warning("local_executor.connection_error_retrying", error=str(exc))
            import asyncio
            await asyncio.sleep(3)
            try:
                response = await self._client.post(
                    f"{self._server_url}/chat/completions",
                    json={
                        "messages": messages,
                        "temperature": temp,
                        "max_tokens": max_tok,
                        "thinking": False,
                        "cache_prompt": False,
                    },
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                data = response.json()
            except Exception as exc2:
                logger.error("local_executor.api_failed", error=str(exc2))
                return ExecutionResult(
                    response=f"[LOCAL_ERROR] llama-server API call failed: {exc2}",
                    model_used=self._model_name,
                    confidence=0.0,
                )

        elapsed_ms = (time.perf_counter() - start) * 1000

        choices = data.get("choices", [])
        text: str = (
            choices[0].get("message", {}).get("content", "") if choices else ""
        )
        usage: dict = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        logger.info(
            "local_executor.api_completed",
            latency_ms=round(elapsed_ms, 1),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        return ExecutionResult(
            response=text,
            model_used=self._model_name,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            cost=0.0,
            latency_ms=round(elapsed_ms, 1),
            confidence=1.0 if text.strip() else 0.0,
        )
