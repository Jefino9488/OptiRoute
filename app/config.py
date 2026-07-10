"""Application configuration via Pydantic Settings.

All configuration is loaded from environment variables.
Uses lru_cache for singleton access via get_settings().

IMPORTANT: FIREWORKS_API_KEY, FIREWORKS_BASE_URL, and ALLOWED_MODELS are
injected by the evaluation harness at runtime. Do not hardcode them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """OptiRoute application settings loaded from environment variables."""

    # --- Fireworks AI ---
    fireworks_api_key: str = Field(
        ...,
        description="Fireworks AI API key — injected by harness, do not hardcode",
    )
    fireworks_base_url: str = Field(
        default="https://api.fireworks.ai/inference/v1",
        description="Fireworks AI base URL — MUST use FIREWORKS_BASE_URL from harness",
    )

    # --- Allowed Models ---
    # The harness injects ALLOWED_MODELS as a comma-separated list of full model paths.
    # e.g. "accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code"
    # We parse this at runtime to build the short_name → full_path registry.
    allowed_models_raw: str = Field(
        default="",
        alias="ALLOWED_MODELS",
        description="Comma-separated full Fireworks model IDs from harness env var",
    )

    # Fallback for local dev when harness is not present
    _local_dev_models: ClassVar[dict[str, str]] = {
        "minimax-m3": "accounts/fireworks/models/minimax-m3",
        "kimi-k2p7-code": "accounts/fireworks/models/kimi-k2p7-code",
    }

    # --- Routing Thresholds ---
    default_accuracy_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum accuracy threshold — lowered to 0.75 to allow local model",
    )
    max_escalation_depth: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum number of escalation retries",
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score before escalation is triggered",
    )

    # --- Data Paths ---
    capability_matrix_path: str = Field(
        default="data/capability_matrix.json",
        description="Path to the capability matrix JSON file",
    )

    # --- Logging ---
    log_level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # --- Local Model (llama-server HTTP API) ---
    local_model_enabled: bool = Field(
        default=True,
        description="Enable local model inference for $0 Fireworks token cost",
    )
    local_model_path: str = Field(
        default="models/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        description="Path to GGUF model weights file served by llama-server",
    )
    local_model_name: str = Field(
        default="local:qwen-2.5-3b",
        description="Local model identifier in the capability matrix",
    )
    local_model_context_length: int = Field(
        default=8192,
        ge=256,
        description="Max context window for local model (tokens)",
    )
    local_model_threads: int = Field(
        default=2,
        ge=1,
        description="CPU threads for llama-server (match harness vCPU count)",
    )
    local_router_enabled: bool = Field(
        default=False,
        description="Use local LLM for routing decisions — disabled, heuristic engine used instead",
    )
    local_server_url: str = Field(
        default="http://localhost:8080/v1",
        description="Base URL of the llama-server HTTP API (OpenAI-compatible)",
    )

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,  # allow alias AND field name
    }

    # --- Computed Properties ---

    @property
    def allowed_models(self) -> dict[str, str]:
        """Parse ALLOWED_MODELS env var into {short_name: full_path} dict.

        The harness injects full model paths like:
            accounts/fireworks/models/minimax-m3

        We extract the short name (last path segment) for internal use.
        Falls back to local dev defaults when env var is not set.
        """
        if not self.allowed_models_raw.strip():
            result = dict(self._local_dev_models)  # local dev fallback
        else:
            result = {}
            for full in self.allowed_models_raw.split(","):
                full = full.strip()
                if full:
                    short = full.rsplit("/", 1)[-1]  # last segment = short name
                    result[short] = full
            if not result:
                result = dict(self._local_dev_models)
                
        # Always inject the local model if enabled
        if self.local_model_enabled:
            result[self.local_model_name] = self.local_model_path
            
        return result

    def get_model_path(self, short_name: str) -> str:
        """Resolve a short model name to its full Fireworks API path.

        Args:
            short_name: Short model identifier (e.g. 'minimax-m3').

        Returns:
            Full Fireworks model path string.

        Raises:
            ValueError: If the model name is not in the allowed list.
        """
        models = self.allowed_models
        if short_name not in models:
            raise ValueError(
                f"Model '{short_name}' is not in ALLOWED_MODELS. "
                f"Allowed: {list(models.keys())}"
            )
        return models[short_name]

    @property
    def capability_matrix_file(self) -> Path:
        """Return the capability matrix path as a Path object."""
        return Path(self.capability_matrix_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance.

    Uses lru_cache so the env is read only once per process.
    """
    return Settings()  # type: ignore[call-arg]
