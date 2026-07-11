"""Tests for LocalExecutor — mock-based, no llama-server required.

Uses httpx.MockTransport / respx to mock the llama-server HTTP API
so no real server or GGUF file is needed during testing.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import httpx

from app.executors.local import LocalExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_api_response(content: str, prompt_tokens: int = 50, completion_tokens: int = 10):
    """Build a mock llama-server /v1/chat/completions JSON response."""
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


def _make_executor(available: bool = True) -> LocalExecutor:
    """Create a LocalExecutor with _available preset and a mocked httpx client."""
    exc = LocalExecutor(model_path="models/fake.gguf")
    exc._available = available
    return exc


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


def test_is_available_false_when_not_loaded():
    exc = _make_executor(available=False)
    assert exc.is_available is False


def test_is_available_true_when_loaded():
    exc = _make_executor(available=True)
    assert exc.is_available is True


# ---------------------------------------------------------------------------
# load() — health check ping
# ---------------------------------------------------------------------------


def test_load_returns_false_when_server_not_reachable():
    exc = LocalExecutor(model_path="models/does_not_exist.gguf")
    with patch("requests.get", side_effect=ConnectionError("refused")):
        result = exc.load()
    assert result is False
    assert exc.is_available is False


def test_load_returns_true_when_server_healthy():
    exc = LocalExecutor(model_path="models/fake.gguf")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "application/json"}
    mock_resp.json.return_value = {"status": "ok"}
    with patch("requests.get", return_value=mock_resp):
        result = exc.load()
    assert result is True
    assert exc.is_available is True


# ---------------------------------------------------------------------------
# route() — always returns None (routing handled by heuristic engine)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_returns_none():
    exc = _make_executor(available=True)
    result = await exc.route("What is the capital of France?")
    assert result is None


@pytest.mark.asyncio
async def test_route_returns_none_when_unavailable():
    exc = _make_executor(available=False)
    result = await exc.route("What is the capital of France?")
    assert result is None


# ---------------------------------------------------------------------------
# execute() — mocked HTTP responses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_returns_response_text():
    exc = _make_executor(available=True)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_api_response(
        "Paris is the capital of France.", prompt_tokens=20, completion_tokens=8
    )
    exc._client = MagicMock()
    exc._client.post = MagicMock(return_value=mock_response)

    import asyncio
    exc._client.post = AsyncMock(return_value=mock_response)

    result = await exc.execute("What is the capital of France?", "retrieval")
    assert result.response == "Paris is the capital of France."
    assert result.cost == 0.0
    assert result.model_used == "local:phi-4-mini"
    assert result.tokens_input == 20
    assert result.tokens_output == 8
    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_execute_cost_is_always_zero():
    exc = _make_executor(available=True)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_api_response("Some answer")

    from unittest.mock import AsyncMock
    exc._client.post = AsyncMock(return_value=mock_response)

    result = await exc.execute("Any prompt", "general_qa")
    assert result.cost == 0.0


@pytest.mark.asyncio
async def test_execute_empty_response_gives_zero_confidence():
    exc = _make_executor(available=True)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = _make_api_response("")

    from unittest.mock import AsyncMock
    exc._client.post = AsyncMock(return_value=mock_response)

    result = await exc.execute("Some prompt")
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_execute_unavailable_returns_error_response():
    exc = _make_executor(available=False)
    result = await exc.execute("test prompt")
    assert result.confidence == 0.0
    assert "LOCAL_UNAVAILABLE" in result.response
    assert result.cost == 0.0


@pytest.mark.asyncio
async def test_execute_on_exception_returns_error_response():
    exc = _make_executor(available=True)
    from unittest.mock import AsyncMock
    exc._client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    result = await exc.execute("test prompt")
    assert result.confidence == 0.0
    assert "LOCAL_ERROR" in result.response


# ---------------------------------------------------------------------------
# AsyncMock import helper for Python 3.8+ compatibility
# ---------------------------------------------------------------------------

try:
    from unittest.mock import AsyncMock
except ImportError:
    from unittest.mock import MagicMock as AsyncMock  # type: ignore
