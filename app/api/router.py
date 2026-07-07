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

    result = await pipeline.route(
        prompt=request.prompt,
        required_accuracy=request.required_accuracy,
        force_model=request.force_model,
    )

    return RouteResponse(**result)


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """List all available models with their capability scores."""
    pipeline = get_pipeline()
    matrix = pipeline.capability_matrix

    models: list[ModelInfo] = []
    for model_id in matrix.get_all_models():
        entry = matrix.get_model_capabilities(model_id)
        if entry is None:
            continue
        models.append(
            ModelInfo(
                model_id=model_id,
                capabilities=entry.get("capabilities", {}),
                cost_per_1k_input=entry.get("cost_per_1k_input", 0.0),
                cost_per_1k_output=entry.get("cost_per_1k_output", 0.0),
                max_context=entry.get("max_context", 0),
                fails_on=entry.get("fails_on", []),
            )
        )
    return models


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
