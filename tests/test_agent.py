"""Tests for agent.py — the evaluation batch entrypoint.

These tests verify the I/O format contract:
  Input:  [{task_id, prompt}, ...]
  Output: [{task_id, answer}, ...]

The pipeline is mocked so no GGUF model or Fireworks API key is needed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_route_result(
    answer: str = "Test answer",
    model: str = "local:ministral-3b",
    cost: float = 0.0,
    tokens_in: int = 10,
    tokens_out: int = 20,
) -> dict:
    return {
        "response": answer,
        "model_used": model,
        "cost": cost,
        "latency_ms": 100.0,
        "confidence": 0.9,
        "cache_hit": False,
        "escalated": False,
        "escalation_depth": 0,
        "task_vector": {},
        "routing_explanation": "mocked",
        "tokens_input": tokens_in,
        "tokens_output": tokens_out,
    }


def _make_pipeline_mock(answer: str = "Test answer") -> MagicMock:
    """Return a mock RoutingPipeline whose route() is an async no-op."""
    mock = MagicMock()
    mock.route = AsyncMock(return_value=_make_route_result(answer=answer))
    mock.cache.stats = {"hits": 0, "misses": 0, "hit_rate": 0.0}
    return mock


async def _run_agent(input_file: Path, output_file: Path, mock_pipeline: MagicMock) -> None:
    """Run agent.main() with patched paths and pipeline."""
    # RoutingPipeline is imported lazily inside main(), so we patch at the source module
    with (
        patch("app.router.pipeline.RoutingPipeline", return_value=mock_pipeline),
        patch.dict(os.environ, {
            "INPUT_PATH": str(input_file),
            "OUTPUT_PATH": str(output_file),
            "FIREWORKS_API_KEY": "dummy",
        }),
    ):
        import importlib
        import agent
        # Reload to pick up the new env vars for INPUT_PATH / OUTPUT_PATH
        importlib.reload(agent)
        await agent.main()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_reads_input_and_writes_output(tmp_path):
    """Basic contract: input tasks → output results with task_id + answer."""
    input_file = tmp_path / "tasks.json"
    output_file = tmp_path / "results.json"

    tasks = [
        {"task_id": "t1", "prompt": "What is the capital of France?"},
        {"task_id": "t2", "prompt": "Write a Python sort function."},
    ]
    input_file.write_text(json.dumps(tasks), encoding="utf-8")

    await _run_agent(input_file, output_file, _make_pipeline_mock())

    assert output_file.exists()
    results = json.loads(output_file.read_text(encoding="utf-8"))
    assert isinstance(results, list)
    assert len(results) == 2
    for item in results:
        assert "task_id" in item
        assert "answer" in item


@pytest.mark.asyncio
async def test_agent_output_preserves_task_ids(tmp_path):
    """task_id in output must exactly match task_id from input, in order."""
    input_file = tmp_path / "tasks.json"
    output_file = tmp_path / "results.json"

    tasks = [
        {"task_id": "practice-01", "prompt": "What is the capital of Australia?"},
        {"task_id": "practice-06", "prompt": "Fix this bug: def get_max(nums): return nums[0]"},
    ]
    input_file.write_text(json.dumps(tasks), encoding="utf-8")

    await _run_agent(input_file, output_file, _make_pipeline_mock())

    results = json.loads(output_file.read_text(encoding="utf-8"))
    task_ids_out = [r["task_id"] for r in results]
    assert task_ids_out == ["practice-01", "practice-06"]


@pytest.mark.asyncio
async def test_agent_output_only_task_id_and_answer(tmp_path):
    """Output objects must have task_id and answer fields."""
    input_file = tmp_path / "tasks.json"
    output_file = tmp_path / "results.json"

    tasks = [{"task_id": "t1", "prompt": "Hello"}]
    input_file.write_text(json.dumps(tasks), encoding="utf-8")

    await _run_agent(input_file, output_file, _make_pipeline_mock(answer="Hello back!"))

    results = json.loads(output_file.read_text(encoding="utf-8"))
    # Must have at least task_id and answer (possibly only those two)
    assert "task_id" in results[0]
    assert "answer" in results[0]
    assert results[0]["task_id"] == "t1"
    assert results[0]["answer"] == "Hello back!"


@pytest.mark.asyncio
async def test_agent_handles_empty_task_list(tmp_path):
    """Empty task list should produce empty results and not crash."""
    input_file = tmp_path / "tasks.json"
    output_file = tmp_path / "results.json"
    input_file.write_text("[]", encoding="utf-8")

    # Empty list exits early — no pipeline needed
    with patch.dict(os.environ, {
        "INPUT_PATH": str(input_file),
        "OUTPUT_PATH": str(output_file),
        "FIREWORKS_API_KEY": "dummy",
    }):
        import importlib
        import agent
        importlib.reload(agent)
        await agent.main()

    assert output_file.exists()
    results = json.loads(output_file.read_text(encoding="utf-8"))
    assert results == []


@pytest.mark.asyncio
async def test_agent_recovers_from_single_task_error(tmp_path):
    """If one task throws, the run continues and all tasks produce output."""
    input_file = tmp_path / "tasks.json"
    output_file = tmp_path / "results.json"

    tasks = [
        {"task_id": "t1", "prompt": "Normal task"},
        {"task_id": "t2", "prompt": "Task that causes error"},
        {"task_id": "t3", "prompt": "Another normal task"},
    ]
    input_file.write_text(json.dumps(tasks), encoding="utf-8")

    call_count = 0

    async def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Simulated API failure")
        return _make_route_result()

    mock_pipeline = MagicMock()
    mock_pipeline.route = AsyncMock(side_effect=side_effect)
    mock_pipeline.cache.stats = {"hits": 0, "misses": 0, "hit_rate": 0.0}

    await _run_agent(input_file, output_file, mock_pipeline)

    results = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(results) == 3  # all 3 present
    task_ids = [r["task_id"] for r in results]
    assert "t1" in task_ids and "t2" in task_ids and "t3" in task_ids
    # Failed task has empty answer
    failed = next(r for r in results if r["task_id"] == "t2")
    assert failed["answer"] == ""


@pytest.mark.asyncio
async def test_agent_output_is_valid_json_with_unicode(tmp_path):
    """Output file must be valid JSON even with unicode/emoji in answers."""
    input_file = tmp_path / "tasks.json"
    output_file = tmp_path / "results.json"

    tasks = [{"task_id": "t1", "prompt": "Prompt with unicode: 日本語 and émoji 🎉"}]
    input_file.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")

    await _run_agent(input_file, output_file, _make_pipeline_mock(answer="日本語 answer 🎉"))

    content = output_file.read_text(encoding="utf-8")
    parsed = json.loads(content)  # must not raise
    assert parsed[0]["answer"] == "日本語 answer 🎉"
