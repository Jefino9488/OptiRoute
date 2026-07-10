"""Pydantic v2 schemas for the OptiRoute API.

All request/response models are defined here for validation,
serialization, and OpenAPI documentation generation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ─── Request Models ───────────────────────────────────────────────


class RouteRequest(BaseModel):
    """Incoming routing request with a prompt and optional constraints."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="The user prompt to route to an appropriate model",
    )
    required_accuracy: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable accuracy for the response",
    )
    max_cost: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum acceptable cost in USD (None = no limit)",
    )
    force_model: str | None = Field(
        default=None,
        description="Force routing to a specific model (bypasses decision engine)",
    )
    enable_thinking: bool | None = Field(
        default=None,
        description=(
            "Enable extended reasoning/thinking for complex tasks. "
            "None = auto-detect based on task complexity, "
            "True = force thinking ON, False = force thinking OFF"
        ),
    )
    metadata: dict | None = Field(
        default=None,
        description="Optional metadata attached to the request",
    )


# ─── Vector Models ────────────────────────────────────────────────


class TaskVectorResponse(BaseModel):
    """Task-type probability vector extracted from the prompt.

    Each field represents the probability (0.0–1.0) that the prompt
    belongs to that task category.
    """

    math: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: float = Field(default=0.0, ge=0.0, le=1.0)
    code: float = Field(default=0.0, ge=0.0, le=1.0)
    creative: float = Field(default=0.0, ge=0.0, le=1.0)
    translation: float = Field(default=0.0, ge=0.0, le=1.0)
    extraction: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval: float = Field(default=0.0, ge=0.0, le=1.0)
    general_qa: float = Field(default=0.0, ge=0.0, le=1.0)


class ResourceVectorResponse(BaseModel):
    """Estimated resource requirements for the prompt."""

    expected_input_tokens: int = Field(default=0, ge=0)
    expected_output_tokens: int = Field(default=0, ge=0)
    expected_context_length: int = Field(default=0, ge=0)
    complexity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized complexity score (0 = trivial, 1 = extremely complex)",
    )


class RiskVectorResponse(BaseModel):
    """Risk/constraint flags for the prompt."""

    needs_json: bool = Field(
        default=False,
        description="Response must be valid JSON",
    )
    needs_high_accuracy: bool = Field(
        default=False,
        description="Task requires very high factual accuracy",
    )
    strict_formatting: bool = Field(
        default=False,
        description="Response must follow strict formatting rules",
    )


# ─── Routing Decision ────────────────────────────────────────────


class RoutingDecision(BaseModel):
    """The result of the decision engine's model selection."""

    model_selected: str = Field(
        ...,
        description="Short name of the selected model",
    )
    estimated_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated cost in USD",
    )
    predicted_accuracy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Predicted accuracy of the selected model for this task",
    )
    reasoning: str = Field(
        default="",
        description="Human-readable explanation of why this model was chosen",
    )
    alternatives_considered: list[str] = Field(
        default_factory=list,
        description="Other models that were evaluated but not chosen",
    )


# ─── Response Models ─────────────────────────────────────────────


class RouteResponse(BaseModel):
    """Full response returned to the client after routing + execution."""

    response: str = Field(
        ...,
        description="The model-generated response text",
    )
    model_used: str = Field(
        ...,
        description="Short name of the model that produced the response",
    )
    cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Actual cost in USD",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="End-to-end latency in milliseconds",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the response",
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether the response was served from cache",
    )
    escalated: bool = Field(
        default=False,
        description="Whether escalation occurred",
    )
    escalation_depth: int = Field(
        default=0,
        ge=0,
        description="Number of escalation levels used",
    )
    task_vector: dict = Field(
        default_factory=dict,
        description="Task vector extracted from the prompt",
    )
    routing_explanation: str = Field(
        default="",
        description="Human-readable explanation of the routing decision",
    )


# ─── Utility / Info Models ───────────────────────────────────────


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok")
    version: str = Field(default="0.1.0")
    models_available: list[str] = Field(
        default_factory=list,
        description="List of available model short names",
    )


class ModelInfo(BaseModel):
    """Detailed information about a single model."""

    model_id: str = Field(
        ...,
        description="Short model identifier",
    )
    capabilities: dict = Field(
        default_factory=dict,
        description="Model capability scores by task category",
    )
    cost_per_1k_input: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost per 1,000 input tokens in USD",
    )
    cost_per_1k_output: float = Field(
        default=0.0,
        ge=0.0,
        description="Cost per 1,000 output tokens in USD",
    )
    max_context: int = Field(
        default=0,
        ge=0,
        description="Maximum context window in tokens",
    )
    avg_output_multiplier: float = Field(
        1.0, description="Average output token multiplier vs input."
    )
    fails_on: list[str] = Field(
        default_factory=list,
        description="Task types this model is known to fail on",
    )
    supports_thinking: bool = Field(
        default=False,
        description="Whether this model supports Fireworks reasoning_effort parameter",
    )
    samples: dict[str, int] = Field(
        default_factory=dict,
        description="Optional tracking of how many samples were tested per task type.",
    )
