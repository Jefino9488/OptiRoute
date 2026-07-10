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


# ---------------------------------------------------------------------------
# Path configuration — can be overridden via env vars for local testing
# ---------------------------------------------------------------------------

INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/output/results.json"))


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


async def main() -> None:
    """Process all tasks and write results.json."""
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

    # --- Process tasks sequentially ---
    results: list[dict[str, str]] = []
    cost_total = 0.0
    local_count = 0
    fireworks_count = 0
    total_fw_tokens = 0
    cache_count = 0

    for i, task in enumerate(tasks):
        task_id: str = task.get("task_id", f"unknown-{i}")
        prompt: str = task.get("prompt", "")

        if not prompt:
            print(
                f"[agent] [{i+1}/{len(tasks)}] {task_id}: empty prompt, skipping.",
                flush=True,
            )
            results.append({"task_id": task_id, "answer": ""})
            continue

        print(
            f"[agent] [{i+1}/{len(tasks)}] {task_id}: {prompt[:70].strip()}...",
            flush=True,
        )
        t_task = time.perf_counter()

        try:
            result = await pipeline.route(
                prompt=prompt,
                required_accuracy=0.75,  # allow local model (~0.72-0.76 accuracy)
            )
            answer: str = result["response"]
            model_used: str = result["model_used"]
            task_cost: float = result["cost"]
            task_ms: float = (time.perf_counter() - t_task) * 1000

        except Exception as exc:
            # Never let a single task failure crash the whole run
            print(
                f"[agent]   ERROR on {task_id}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            answer = ""
            model_used = "error"
            task_cost = 0.0
            task_ms = (time.perf_counter() - t_task) * 1000

        # --- Track stats ---
        cost_total += task_cost
        fw_tokens = result.get("fireworks_tokens", 0) if isinstance(result, dict) else 0
        total_fw_tokens += fw_tokens
        if model_used.startswith("local:") or model_used == "local":
            local_count += 1
        elif task_cost > 0:
            fireworks_count += 1
        if isinstance(result, dict) and result.get("cache_hit"):
            cache_count += 1

        print(
            f"[agent]   → model={model_used}, cost=${task_cost:.6f}, latency={task_ms:.0f}ms",
            flush=True,
        )

        results.append({"task_id": task_id, "answer": answer})

    # --- Write output ---
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    total_s = time.perf_counter() - t_start
    pct_local = (local_count / len(results) * 100) if results else 0
    pct_fireworks = (fireworks_count / len(results) * 100) if results else 0
    pct_cache = (cache_count / len(results) * 100) if results else 0
    print(
        f"\n[agent] ✓ Complete: {len(results)} tasks in {total_s:.1f}s\n"
        f"[agent]   Local (free):      {local_count}/{len(results)} ({pct_local:.0f}%)\n"
        f"[agent]   Fireworks:         {fireworks_count}/{len(results)} ({pct_fireworks:.0f}%)\n"
        f"[agent]   Cache hits:        {cache_count}/{len(results)} ({pct_cache:.0f}%)\n"
        f"[agent]   Fireworks tokens:  {total_fw_tokens:,} (scored)\n"
        f"[agent]   Total cost:        ${cost_total:.6f}\n"
        f"[agent]   Output:            {OUTPUT_PATH}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(main())
