"""Local LLM executor — Qwen2.5-3B-Instruct via llama-cpp-python.

Design:
- Model is loaded ONCE at startup via load() — expensive, ~5-15s
- asyncio.to_thread() wraps synchronous llama.cpp calls to avoid blocking the event loop
- route() uses max_tokens=15, temp=0.0 for fast, deterministic routing decisions
- execute() uses task-appropriate settings for actual answer generation
- Graceful degradation: returns low-confidence result if model is unavailable

The same model instance handles BOTH routing (cheap: max_tokens=15) and
execution (normal inference), keeping RAM footprint to a single load.

Thread safety: llama-cpp-python's Llama object is not thread-safe.
asyncio.to_thread() runs each call in the thread pool, but Python's GIL
plus FastAPI's single-event-loop default prevents concurrent access.
For the batch agent (sequential task processing), this is completely safe.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import structlog

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
    "retrieval": 0.1,
    "creative": 0.6,
    "general_qa": 0.3,
}

_MAX_TOKENS_MAP: dict[str, int] = {
    "code": 512,
    "math": 256,
    "extraction": 512,
    "translation": 512,
    "reasoning": 512,
    "retrieval": 256,
    "creative": 512,
    "general_qa": 512,
}

# ---------------------------------------------------------------------------
# Routing prompt + few-shot examples
# ---------------------------------------------------------------------------

# Valid routing outputs — must match keys in the routing examples + capability_matrix
_VALID_ROUTES: frozenset[str] = frozenset({"local", "minimax-m3", "kimi-k2p7-code"})

_ROUTING_SYSTEM = """\
You are an AI routing agent. Select the cheapest model that can correctly handle the task.

Models:
- local: Factual definitions, sentiment analysis, named entity recognition (NER), \
text summarization, simple translation. FREE — use ONLY when you are confident the answer \
is a well-known fact or a simple single-step classification.
- kimi-k2p7-code: Code writing, code debugging, programming tasks ONLY. EXPENSIVE.
- minimax-m3: Multi-step reasoning, sorting/filtering/ordering lists, arithmetic on large numbers, \
current events or recent facts you may not know reliably, any task with multiple transformation \
steps, or anything where you are not fully confident. EXPENSIVE.

Rules:
1. Use 'local' only for simple factual lookups and single-step classification — NOT for counting, sorting, reordering, or current events.
2. Use 'kimi-k2p7-code' ONLY for coding tasks (writing or debugging code).
3. Use 'minimax-m3' for everything else, including multi-step manipulation and when in doubt.

Respond with exactly one word: local, kimi-k2p7-code, or minimax-m3."""

_ROUTING_EXAMPLES: list[tuple[str, str]] = [
    # Factual knowledge → local
    ("What is the capital of France?", "local"),
    ("Who wrote Romeo and Juliet?", "local"),
    ("What is photosynthesis?", "local"),
    ("What is the speed of light?", "local"),
    # Sentiment classification → local
    ("Classify the sentiment: 'Great battery life but the screen scratches easily.'", "local"),
    ("What is the sentiment of: 'I loved the movie but the ending was disappointing.'", "local"),
    # Named entity recognition → local
    ("Extract all named entities from: Maria Sanchez joined Fireworks AI in Berlin last March.", "local"),
    ("Identify the entities in: Apple Inc. was founded by Steve Jobs in Cupertino.", "local"),
    # Text summarisation → local
    ("Summarize this paragraph in one sentence:", "local"),
    ("Condense the following text to two sentences:", "local"),
    # Code debugging → kimi
    ("This function has a bug: def get_max(nums): return nums[0]. Find and fix it.", "kimi-k2p7-code"),
    ("Debug this Python code: for i in range(10) print(i)", "kimi-k2p7-code"),
    # Code generation → kimi
    ("Write a Python function that returns the second-largest number in a list.", "kimi-k2p7-code"),
    ("Implement a binary search algorithm in Java.", "kimi-k2p7-code"),
    ("Write a function to check if a string is a palindrome.", "kimi-k2p7-code"),
    # Complex math word problems → minimax
    ("A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many remain?", "minimax-m3"),
    ("A car depreciates 15% per year. Starting at $20000, what is its value after 3 years?", "minimax-m3"),
    # Logical/deductive reasoning → minimax
    ("Three friends each own a different pet. Sam doesn't own the bird. Jo owns the dog. Who owns the cat?", "minimax-m3"),
    ("If all A are B, and some B are C, what can we conclude about A and C?", "minimax-m3"),
    # Multi-step list manipulation → minimax (NOT local)
    ("List the numbers from 1 to 50, but skip every multiple of 3, and reverse the final list.", "minimax-m3"),
    ("Sort these words alphabetically, then reverse the order: banana, apple, kiwi, fig.", "minimax-m3"),
    ("Filter the even numbers from this list, then sort them descending: 7, 2, 9, 4, 1, 6.", "minimax-m3"),
    # Current events / recent facts → minimax (local training data may be stale)
    ("Who won the most recent FIFA World Cup?", "minimax-m3"),
    ("What is the latest stable version of Python?", "minimax-m3"),
    ("Who is the current Formula 1 World Champion?", "minimax-m3"),
    # Exact character counting → local (intercepted by CharCounterTool before LLM execution)
    ("Count how many times the letter 'a' appears in 'banana'.", "local"),
    ("How many times does 'the' appear in the sentence 'the cat sat on the mat'?", "local"),
]


# ---------------------------------------------------------------------------
# LocalExecutor class
# ---------------------------------------------------------------------------


class LocalExecutor:
    """Execute and route prompts via a locally-loaded Qwen2.5-3B GGUF model.

    Parameters
    ----------
    model_path : str
        Absolute or relative path to the GGUF weights file.
    context_length : int
        Maximum context window in tokens (default 4096).
    n_threads : int
        CPU threads for inference (default 2, matches harness vCPU count).
    """

    def __init__(
        self,
        model_path: str,
        context_length: int = 4096,
        n_threads: int = 2,
    ) -> None:
        self._model_path = Path(model_path)
        self._context_length = context_length
        self._n_threads = n_threads
        self._llm: Any = None  # Set by load()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load the GGUF model into RAM.

        This is a blocking call — takes 5–20 seconds depending on model size
        and disk speed. Call once at application startup.

        Returns
        -------
        bool
            True on successful load, False if model file missing or load failed.
        """
        if not self._model_path.exists():
            logger.warning(
                "local_executor.model_not_found",
                path=str(self._model_path),
            )
            return False
        try:
            from llama_cpp import Llama  # type: ignore[import]

            logger.info("local_executor.loading", path=self._model_path.name)
            t0 = time.perf_counter()
            self._llm = Llama(
                model_path=str(self._model_path),
                n_ctx=self._context_length,
                n_threads=self._n_threads,
                verbose=False,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "local_executor.loaded",
                model=self._model_path.name,
                load_ms=round(elapsed, 0),
            )
            return True
        except Exception as exc:
            logger.error("local_executor.load_failed", error=str(exc))
            return False

    @property
    def is_available(self) -> bool:
        """True if the model was successfully loaded."""
        return self._llm is not None

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    async def route(self, prompt: str) -> str | None:
        """Ask the local model which backend to use for this prompt.

        Uses max_tokens=15 and temperature=0.0 — fast and deterministic.
        Few-shot examples ensure consistent single-token output.

        Parameters
        ----------
        prompt : str
            The user prompt to classify (truncated to 400 chars for speed).

        Returns
        -------
        str | None
            One of 'local', 'minimax-m3', 'kimi-k2p7-code', or None on failure.
        """
        if not self.is_available:
            return None

        # Build few-shot chat messages
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _ROUTING_SYSTEM},
        ]
        for task_example, decision in _ROUTING_EXAMPLES:
            messages.append({"role": "user", "content": task_example})
            messages.append({"role": "assistant", "content": decision})
        # Truncate long prompts — routing only needs the gist
        messages.append({"role": "user", "content": prompt[:400]})

        try:
            result = await asyncio.to_thread(
                self._llm.create_chat_completion,
                messages=messages,
                max_tokens=15,
                temperature=0.0,
            )
            raw: str = result["choices"][0]["message"]["content"].strip().lower()
            # Extract the first valid route token from the output
            for valid in ("local", "kimi-k2p7-code", "minimax-m3"):
                if valid in raw:
                    logger.info("local_executor.routing_decision", decision=valid, raw=raw[:30])
                    return valid
            logger.warning("local_executor.route_parse_failed", raw=raw[:60])
            return None
        except Exception as exc:
            logger.error("local_executor.route_error", error=str(exc))
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
        """Run local inference on the prompt.

        Parameters
        ----------
        prompt : str
            The user prompt.
        task_type : str
            Task category — drives temperature and max_tokens defaults.
        system_prompt : str | None
            Optional system message (uses concise default if None).
        max_tokens : int | None
            Override for max output tokens.

        Returns
        -------
        ExecutionResult
            cost is always 0.0 — no Fireworks tokens consumed.
        """
        if not self.is_available:
            return ExecutionResult(
                response="[LOCAL_UNAVAILABLE] Local model not loaded.",
                model_used="local",
                confidence=0.0,
            )

        temp = _TEMP_MAP.get(task_type, 0.3)
        max_tok = max_tokens or _MAX_TOKENS_MAP.get(task_type, 512)

        sys_msg = system_prompt or (
            "You are a helpful assistant. Be concise and accurate. "
            "Give direct answers without unnecessary preamble."
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ]

        logger.info(
            "local_executor.execute",
            task_type=task_type,
            temperature=temp,
            max_tokens=max_tok,
        )

        start = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                self._llm.create_chat_completion,
                messages=messages,
                max_tokens=max_tok,
                temperature=temp,
            )
        except Exception as exc:
            logger.error("local_executor.execute_failed", error=str(exc))
            return ExecutionResult(
                response=f"[LOCAL_ERROR] {exc}",
                model_used="local",
                confidence=0.0,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        text: str = result["choices"][0]["message"]["content"] or ""
        usage: dict[str, int] = result.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        logger.info(
            "local_executor.completed",
            task_type=task_type,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=round(elapsed_ms, 1),
        )

        return ExecutionResult(
            response=text,
            model_used="local:qwen-2.5-3b",
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            cost=0.0,  # Always $0 — no Fireworks tokens consumed
            latency_ms=round(elapsed_ms, 1),
            confidence=1.0 if text.strip() else 0.0,
        )
