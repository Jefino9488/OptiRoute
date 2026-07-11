"""API router — real routing endpoints backed by the full pipeline.

Replaces the Phase 1 placeholder endpoints with working implementations.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    ModelInfo,
    RouteRequest,
    RouteResponse,
)
from app.router.pipeline import RoutingPipeline

router = APIRouter(prefix="/v1", tags=["routing"])

# Module-level pipeline instance — initialised by the lifespan hook.
_pipeline: RoutingPipeline | None = None


def init_pipeline(pipeline: RoutingPipeline) -> None:
    """Set the module-level pipeline (called from main.py lifespan)."""
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> RoutingPipeline:
    """Return the active pipeline or raise if not initialised."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised")
    return _pipeline


@router.post("/route", response_model=RouteResponse)
async def route_prompt(request: RouteRequest) -> RouteResponse:
    """Route a prompt to the optimal model.

    This is the primary endpoint. It accepts a prompt, analyses it,
    selects the cheapest capable model, executes inference, and returns
    the result with cost/latency metadata.
    """
    pipeline = get_pipeline()

    try:
        result = await pipeline.route(
            prompt=request.prompt,
            required_accuracy=request.required_accuracy,
            force_model=request.force_model,
            enable_thinking=request.enable_thinking,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return RouteResponse(**result)


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """List available models."""
    return [
        ModelInfo(
            model_id="local:qwen2.5-coder-7b",
            capabilities={"math": 0.85, "code": 0.92, "translation": 0.85, "general_qa": 0.88},
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            max_context=8192,
            fails_on=[],
            supports_thinking=False,
        ),
        ModelInfo(
            model_id="minimax-m3",
            capabilities={"math": 0.90, "code": 0.97, "reasoning": 0.85, "general_qa": 0.74},
            cost_per_1k_input=0.0003,
            cost_per_1k_output=0.0012,
            max_context=512000,
            fails_on=["translation"],
            supports_thinking=False,
        ),
    ]


@router.get("/metrics")
async def get_metrics() -> dict:
    """Return aggregated routing metrics."""
    pipeline = get_pipeline()
    return pipeline.metrics.summary()


@router.get("/cache/stats")
async def cache_stats() -> dict:
    """Return cache statistics."""
    pipeline = get_pipeline()
    return pipeline.cache.stats
