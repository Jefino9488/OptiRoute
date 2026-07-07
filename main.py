"""OptiRoute — Adaptive Capability-Based Hybrid AI Routing Framework.

FastAPI application entry point.
Run with: uvicorn main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router as api_router, init_pipeline
from app.api.schemas import HealthResponse
from app.config import get_settings
from app.logging import get_logger, setup_logging
from app.router.pipeline import RoutingPipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown hooks."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    logger = get_logger(__name__)
    logger.info(
        "optiroute_startup",
        version="0.1.0",
        models=list(settings.allowed_models.keys()),
        log_level=settings.log_level,
    )

    # Initialise the routing pipeline and inject into the API router.
    pipeline = RoutingPipeline()
    init_pipeline(pipeline)
    logger.info("pipeline_initialised")

    yield
    logger.info("optiroute_shutdown")


app = FastAPI(
    title="OptiRoute",
    description="Adaptive Capability-Based Hybrid AI Routing Framework",
    version="0.1.0",
    lifespan=lifespan,
)

# ─── CORS Middleware ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────
app.include_router(api_router)


# ─── Root Routes ─────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        models_available=list(settings.allowed_models.keys()),
    )


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {"service": "OptiRoute", "version": "0.1.0", "status": "running"}
