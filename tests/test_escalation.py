import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.router.pipeline import RoutingPipeline
from app.executors.base import ExecutionResult
from app.router.supra_router import SupraRouterResult

@pytest.mark.asyncio
@patch.dict("os.environ", {"FIREWORKS_API_KEY": "dummy"})
async def test_routing_overrides_and_escalation():
    """Verify that routing decision maps tasks correctly and escalates on low confidence."""
    # Reset Settings cache to pick up the mocked API key
    from app.config import get_settings
    get_settings.cache_clear()
    pipeline = RoutingPipeline()
    
    # Mock preprocessor and compiler
    pipeline._preprocessor = AsyncMock()
    pipeline._preprocessor.process.return_value = MagicMock(system_addons=["addon"], forwarded="Clean prompt")
    
    pipeline._compiler = MagicMock()
    pipeline._compiler.compile.return_value = MagicMock(system_prompt="sys")
    
    # Mock normalizer
    pipeline._normalizer = MagicMock()
    pipeline._normalizer.normalize.return_value = MagicMock(prompt_hash="hash")
    
    # Mock cache
    pipeline._cache = MagicMock()
    pipeline._cache.get.return_value = (None, "miss")
    
    # Mock validators
    pipeline._validator = MagicMock()
    # Initial execution result has low confidence -> triggers escalation
    pipeline._validator.validate.side_effect = [
        MagicMock(confidence=0.2), # first validation (low confidence)
        MagicMock(confidence=0.9), # second validation (high confidence)
    ]
    
    # Mock executors
    pipeline._local = AsyncMock()
    pipeline._local.execute.return_value = ExecutionResult(
        response="Local answer",
        model_used="local:ministral-3b",
        confidence=1.0,
        tokens_input=10,
        tokens_output=10
    )
    
    pipeline._fireworks = AsyncMock()
    pipeline._fireworks.execute.return_value = ExecutionResult(
        response="Fireworks answer",
        model_used="minimax-m3",
        confidence=1.0,
        tokens_input=20,
        tokens_output=20
    )
    
    # 1. Test local model -> Fireworks minimax-m3 escalation
    pipeline._supra_router = AsyncMock()
    pipeline._supra_router.classify.return_value = SupraRouterResult(
        domain="translation",
        complexity=0.2,
        needs_math=False,
        needs_code=False,
        route="small model",
        justification="simple",
        raw_output="translation",
        success=True
    )
    
    result = await pipeline.route(prompt="Translate this", required_accuracy=0.75)
    
    # Verify it routed first to local:ministral-3b, then escalated to minimax-m3
    assert result["model_used"] == "minimax-m3"
    assert result["escalated"] is True
    assert result["escalation_depth"] == 1
    assert pipeline._local.execute.call_count == 1
    assert pipeline._fireworks.execute.call_count == 1
    
    # 2. Test local model -> kimi-k2p7-code escalation for coding tasks
    pipeline._local.execute.reset_mock()
    pipeline._fireworks.execute.reset_mock()
    pipeline._validator.validate.side_effect = [
        MagicMock(confidence=0.1), # first validation (low confidence)
        MagicMock(confidence=0.95), # second validation (high confidence)
    ]
    
    pipeline._supra_router.classify.return_value = SupraRouterResult(
        domain="programming",
        complexity=0.2,
        needs_math=False,
        needs_code=True,
        route="small model",
        justification="simple",
        raw_output="programming",
        success=True
    )
    
    pipeline._fireworks.execute.return_value = ExecutionResult(
        response="Fireworks code answer",
        model_used="kimi-k2p7-code",
        confidence=1.0,
        tokens_input=30,
        tokens_output=30
    )
    
    result = await pipeline.route(prompt="Write a Python script", required_accuracy=0.75)
    
    assert result["model_used"] == "kimi-k2p7-code"
    assert result["escalated"] is True
    assert result["escalation_depth"] == 1
    assert pipeline._local.execute.call_count == 1
    assert pipeline._fireworks.execute.call_count == 1
