"""OptiRoute Evaluation Agent.

This is the primary entry point for hackathon evaluation.

The evaluation harness:
  1. Mounts tasks at    /input/tasks.json   (read-only)
  2. Expects results at /output/results.json (written by this script)
  3. Checks exit code: 0 = success, non-zero = RUNTIME_ERROR

Input format:
    [{task_id: str, prompt: str}, ...]

Output format (MUST match exactly to avoid INVALID_RESULTS_SCHEMA):
    [{task_id: str, answer: str}, ...]

Environment variables injected by harness (do NOT hardcode):
    FIREWORKS_API_KEY    - API key for Fireworks AI
    FIREWORKS_BASE_URL   - Base URL for Fireworks API (ALL calls must go through this)
    ALLOWED_MODELS       - Comma-separated full model IDs (e.g. accounts/fireworks/models/minimax-m3)

Routing strategy:
    1. Deterministic tool ($0, instant)     — pure math / JSON
    2. Local LLM router ($0 tokens) →
        a. local model (Qwen2.5-3B GGUF)    — factual, sentiment, NER, summarisation
       b. kimi-k2p7-code                    — code debugging / generation
       c. minimax-m3                        — complex math, logical reasoning
    3. Heuristic engine (fallback)          — if local model unavailable

Run locally:
    INPUT_PATH=/tmp/input/tasks.json OUTPUT_PATH=/tmp/output/results.json \\
    FIREWORKS_API_KEY=key \\
    FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \\
    ALLOWED_MODELS=accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code \\
    uv run python agent.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Path configuration — can be overridden via env vars for local testing
# ---------------------------------------------------------------------------

INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/output/results.json"))

# Batch size for concurrent processing (configurable via env var)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5"))


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


async def _process_task(
    pipeline: Any,
    task: dict,
    task_index: int,
    total_tasks: int,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Process a single task with concurrency control.

    Returns a dict with task_id, answer, and token usage.
    """
    task_id: str = task.get("task_id", f"unknown-{task_index}")
    prompt: str = task.get("prompt", "")

    if not prompt:
        print(
            f"[agent] [{task_index+1}/{total_tasks}] {task_id}: empty prompt, skipping.",
            flush=True,
        )
        return {"task_id": task_id, "answer": "", "tokens_in": 0, "tokens_out": 0, "cost": 0.0}

    async with semaphore:
        print(
            f"[agent] [{task_index+1}/{total_tasks}] {task_id}: {prompt[:70].strip()}...",
            flush=True,
        )
        t_task = time.perf_counter()

        try:
            result = await pipeline.route(
                prompt=prompt,
                required_accuracy=0.75,
            )
            answer: str = result["response"]
            model_used: str = result["model_used"]
            task_cost: float = result["cost"]
            tokens_in: int = result.get("tokens_input", 0)
            tokens_out: int = result.get("tokens_output", 0)
            task_ms: float = (time.perf_counter() - t_task) * 1000

        except Exception as exc:
            print(
                f"[agent]   ERROR on {task_id}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            answer = ""
            model_used = "error"
            task_cost = 0.0
            tokens_in = 0
            tokens_out = 0
            task_ms = (time.perf_counter() - t_task) * 1000

        print(
            f"[agent]   → model={model_used}, cost=${task_cost:.6f}, "
            f"tokens={tokens_in}→{tokens_out}, latency={task_ms:.0f}ms",
            flush=True,
        )

        return {
            "task_id": task_id,
            "answer": answer,
            "model_used": model_used,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost": task_cost,
        }


async def main() -> None:
    """Process all tasks and write results.json.

    Uses batch concurrency: processes BATCH_SIZE tasks in parallel,
    then moves to the next batch. This reduces total latency by
    overlapping I/O-bound operations (API calls, model inference).
    """
    t_start = time.perf_counter()

    # --- Validate input ---
    if not INPUT_PATH.exists():
        print(
            f"[agent] FATAL: Input file not found: {INPUT_PATH}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    try:
        tasks: list[dict] = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"[agent] FATAL: Could not parse {INPUT_PATH}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    if not tasks:
        print("[agent] WARNING: No tasks found in input file.", flush=True)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text("[]", encoding="utf-8")
        return

    # --- Initialise routing pipeline (loads GGUF model into RAM) ---
    print(f"[agent] Initializing routing pipeline ({len(tasks)} tasks)...", flush=True)
    from app.router.pipeline import RoutingPipeline

    pipeline = RoutingPipeline()
    init_ms = (time.perf_counter() - t_start) * 1000
    print(f"[agent] Pipeline ready in {init_ms:.0f}ms", flush=True)

    # --- Process tasks with batch concurrency ---
    semaphore = asyncio.Semaphore(BATCH_SIZE)
    print(
        f"[agent] Processing {len(tasks)} tasks with batch_size={BATCH_SIZE}...",
        flush=True,
    )

    # Create coroutines for all tasks (semaphore limits actual concurrency)
    coroutines = [
        _process_task(pipeline, task, i, len(tasks), semaphore)
        for i, task in enumerate(tasks)
    ]

    # Run all tasks concurrently (semaphore controls parallelism)
    # Results maintain input order because asyncio.gather preserves order
    task_results: list[dict[str, Any]] = await asyncio.gather(*coroutines)

    # Extract only task_id + answer for output (per harness contract)
    results: list[dict[str, str]] = [
        {"task_id": r["task_id"], "answer": r["answer"]}
        for r in task_results
    ]

    # Aggregate token usage and cost
    total_tokens_in = sum(r.get("tokens_in", 0) for r in task_results)
    total_tokens_out = sum(r.get("tokens_out", 0) for r in task_results)
    total_cost = sum(r.get("cost", 0.0) for r in task_results)

    # Separate local vs Fireworks tokens
    local_tokens_in = 0
    local_tokens_out = 0
    fireworks_tokens_in = 0
    fireworks_tokens_out = 0
    local_count = 0
    fireworks_count = 0
    deterministic_count = 0
    for r in task_results:
        model = r.get("model_used", "")
        if model.startswith("local:"):
            local_tokens_in += r.get("tokens_in", 0)
            local_tokens_out += r.get("tokens_out", 0)
            local_count += 1
        elif model.startswith("deterministic:"):
            deterministic_count += 1
        else:
            fireworks_tokens_in += r.get("tokens_in", 0)
            fireworks_tokens_out += r.get("tokens_out", 0)
            fireworks_count += 1

    # --- Write output ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    total_s = time.perf_counter() - t_start
    cache_stats = pipeline.cache.stats
    print(
        f"\n[agent] ✓ Complete: {len(results)} tasks in {total_s:.1f}s\n"
        f"[agent]   Batch size:      {BATCH_SIZE}\n"
        f"[agent]   Tasks local:     {local_count} (Qwen2.5-3B, $0)\n"
        f"[agent]   Tasks fireworks:  {fireworks_count} (minimax-m3)\n"
        f"[agent]   Tasks deterministic: {deterministic_count}\n"
        f"[agent]   ─────────────────────────────────\n"
        f"[agent]   Local tokens:    {local_tokens_in:,} in → {local_tokens_out:,} out\n"
        f"[agent]   Fireworks tokens: {fireworks_tokens_in:,} in → {fireworks_tokens_out:,} out\n"
        f"[agent]   Total tokens:    {total_tokens_in:,} in → {total_tokens_out:,} out\n"
        f"[agent]   Total cost:      ${total_cost:.6f}\n"
        f"[agent]   Cache hits:      {cache_stats['hits']}/{cache_stats['hits'] + cache_stats['misses']} "
        f"({cache_stats['hit_rate']:.1%})\n"
        f"[agent]   Output:          {OUTPUT_PATH}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(main())
