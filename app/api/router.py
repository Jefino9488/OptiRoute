"""Placeholder API router for OptiRoute v1 endpoints.

These endpoints will be fully implemented in later phases.
For now they return mock responses to verify the server boots correctly.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import (
    ModelInfo,
    RouteRequest,
    RouteResponse,
)
from app.config import get_settings

router = APIRouter(prefix="/v1", tags=["routing"])


@router.post("/route", response_model=RouteResponse)
async def route_prompt(request: RouteRequest) -> RouteResponse:
    """Route a prompt to the optimal model.

    This is the primary endpoint. It accepts a prompt, analyses it,
    selects the cheapest capable model, executes inference, and returns
    the result with cost/latency metadata.

    **Phase 1 placeholder** — returns a mock response.
    """
    return RouteResponse(
        response=f"[placeholder] Echo: {request.prompt[:100]}",
        model_used="gemma-4-26b-a4b-it",
        cost=0.0,
        latency_ms=0.0,
        confidence=1.0,
        cache_hit=False,
        escalated=False,
        escalation_depth=0,
        task_vector={},
        routing_explanation="Phase 1 placeholder — no routing logic yet",
    )


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """List all available models and their metadata.

    **Phase 1 placeholder** — returns basic model info from config.
    """
    settings = get_settings()
    return [
        ModelInfo(
            model_id=short_name,
            capabilities={},
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            max_context=0,
            fails_on=[],
        )
        for short_name in settings.allowed_models
    ]
