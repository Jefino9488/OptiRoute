"""Tests for LocalExecutor — mock-based, no GGUF file required.

These tests inject a mock llama-cpp Llama instance directly into the
executor's _llm attribute to avoid requiring the real GGUF model.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.executors.local import LocalExecutor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_llm_response(content: str, prompt_tokens: int = 50, completion_tokens: int = 10):
    """Build a mock llama-cpp response dict."""
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


@pytest.fixture
def executor():
    """A LocalExecutor with a mock llama-cpp Llama injected."""
    exc = LocalExecutor(model_path="models/fake.gguf", context_length=4096, n_threads=2)
    exc._llm = MagicMock()  # bypass load()
    return exc


@pytest.fixture
def unavailable_executor():
    """A LocalExecutor with no model loaded (_llm is None)."""
    return LocalExecutor(model_path="models/nonexistent.gguf")


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


def test_is_available_false_when_not_loaded(unavailable_executor):
    assert unavailable_executor.is_available is False


def test_is_available_true_when_loaded(executor):
    assert executor.is_available is True


# ---------------------------------------------------------------------------
# load() — with mocked llama_cpp import
# ---------------------------------------------------------------------------


def test_load_returns_false_when_model_file_missing():
    exc = LocalExecutor(model_path="models/does_not_exist.gguf")
    result = exc.load()
    assert result is False
    assert exc.is_available is False


# ---------------------------------------------------------------------------
# route() — routing decisions for all 8 evaluation categories
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_factual_knowledge_returns_local(executor):
    executor._llm.create_chat_completion.return_value = _make_llm_response("local")
    result = await executor.route("What is the capital of France?")
    assert result == "local"


@pytest.mark.asyncio
async def test_route_sentiment_returns_local(executor):
    executor._llm.create_chat_completion.return_value = _make_llm_response("local")
    result = await executor.route(
        "Classify the sentiment: 'Great battery life but the screen scratches easily.'"
    )
    assert result == "local"


@pytest.mark.asyncio
async def test_route_ner_returns_local(executor):
    executor._llm.create_chat_completion.return_value = _make_llm_response("local")
    result = await executor.route(
        "Extract all named entities from: Maria Sanchez joined Fireworks AI in Berlin."
    )
    assert result == "local"


@pytest.mark.asyncio
async def test_route_summarisation_returns_local(executor):
    executor._llm.create_chat_completion.return_value = _make_llm_response("local")
    result = await executor.route("Summarize this paragraph in one sentence.")
    assert result == "local"


@pytest.mark.asyncio
async def test_route_code_debug_returns_kimi(executor):
    executor._llm.create_chat_completion.return_value = _make_llm_response("kimi-k2p7-code")
    result = await executor.route(
        "This function has a bug: def get_max(nums): return nums[0]. Find and fix it."
    )
    assert result == "kimi-k2p7-code"


@pytest.mark.asyncio
async def test_route_code_gen_returns_kimi(executor):
    executor._llm.create_chat_completion.return_value = _make_llm_response("kimi-k2p7-code")
    result = await executor.route(
        "Write a Python function that returns the second-largest number in a list."
    )
    assert result == "kimi-k2p7-code"


@pytest.mark.asyncio
async def test_route_complex_math_returns_minimax(executor):
    executor._llm.create_chat_completion.return_value = _make_llm_response("minimax-m3")
    result = await executor.route(
        "A store has 240 items. It sells 15% on Monday and 60 more on Tuesday. How many remain?"
    )
    assert result == "minimax-m3"


@pytest.mark.asyncio
async def test_route_logical_reasoning_returns_minimax(executor):
    executor._llm.create_chat_completion.return_value = _make_llm_response("minimax-m3")
    result = await executor.route(
        "Three friends each own a different pet. Sam doesn't own the bird. Jo owns the dog. Who owns the cat?"
    )
    assert result == "minimax-m3"


@pytest.mark.asyncio
async def test_route_returns_none_when_unavailable(unavailable_executor):
    result = await unavailable_executor.route("What is the capital of France?")
    assert result is None


@pytest.mark.asyncio
async def test_route_returns_none_on_unparseable_output(executor):
    executor._llm.create_chat_completion.return_value = _make_llm_response(
        "I cannot determine the best model for this task."
    )
    result = await executor.route("Some prompt")
    assert result is None


@pytest.mark.asyncio
async def test_route_returns_none_on_exception(executor):
    executor._llm.create_chat_completion.side_effect = RuntimeError("OOM")
    result = await executor.route("test prompt")
    assert result is None


# ---------------------------------------------------------------------------
# execute() — response generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_returns_response_text(executor):
    executor._llm.create_chat_completion.return_value = _make_llm_response(
        "Paris is the capital of France.", prompt_tokens=20, completion_tokens=8
    )
    result = await executor.execute("What is the capital of France?", "retrieval")
    assert result.response == "Paris is the capital of France."
    assert result.cost == 0.0  # always $0
    assert result.model_used == "local:qwen-2.5-3b"
    assert result.tokens_input == 20
    assert result.tokens_output == 8
    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_execute_cost_is_always_zero(executor):
    executor._llm.create_chat_completion.return_value = _make_llm_response("Some answer")
    result = await executor.execute("Any prompt", "general_qa")
    assert result.cost == 0.0


@pytest.mark.asyncio
async def test_execute_empty_response_gives_zero_confidence(executor):
    executor._llm.create_chat_completion.return_value = _make_llm_response("")
    result = await executor.execute("Some prompt")
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_execute_unavailable_returns_error_response(unavailable_executor):
    result = await unavailable_executor.execute("test prompt")
    assert result.confidence == 0.0
    assert "LOCAL_UNAVAILABLE" in result.response
    assert result.cost == 0.0


@pytest.mark.asyncio
async def test_execute_on_exception_returns_error_response(executor):
    executor._llm.create_chat_completion.side_effect = MemoryError("Out of memory")
    result = await executor.execute("test prompt")
    assert result.confidence == 0.0
    assert "LOCAL_ERROR" in result.response


@pytest.mark.asyncio
async def test_execute_uses_zero_temperature_for_math(executor):
    """Verify math tasks use temperature=0 for deterministic output."""
    executor._llm.create_chat_completion.return_value = _make_llm_response("345")
    await executor.execute("What is 15 * 23?", "math")
    call_kwargs = executor._llm.create_chat_completion.call_args[1]
    assert call_kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_execute_uses_low_temperature_for_code(executor):
    """Verify code tasks use low temperature for deterministic output."""
    executor._llm.create_chat_completion.return_value = _make_llm_response("def foo(): pass")
    await executor.execute("Write a function", "code")
    call_kwargs = executor._llm.create_chat_completion.call_args[1]
    assert call_kwargs["temperature"] == 0.1
