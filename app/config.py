"""Application configuration via Pydantic Settings.

All configuration is loaded from environment variables.
Uses lru_cache for singleton access via get_settings().
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """OptiRoute application settings loaded from environment variables."""

    # --- Fireworks AI ---
    fireworks_api_key: str = Field(
        ...,
        description="Fireworks AI API key (required)",
    )
    fireworks_base_url: str = Field(
        default="https://api.fireworks.ai/inference/v1",
        description="Fireworks AI base URL for OpenAI-compatible API",
    )

    # --- Allowed Models ---
    # Mapping of short model names → full Fireworks model paths.
    # This is not loaded from env; it's a constant registry.
    allowed_models: dict[str, str] = Field(
        default={
            "minimax-m3": "accounts/fireworks/models/minimax-m3",
            "kimi-k2p7-code": "accounts/fireworks/models/kimi-k2p7-code",
            "gemma-4-31b-it": "accounts/fireworks/models/gemma-4-31b-it",
            "gemma-4-26b-a4b-it": "accounts/fireworks/models/gemma-4-26b-a4b-it",
            "gemma-4-31b-it-nvfp4": "accounts/fireworks/models/gemma-4-31b-it-nvfp4",
        },
        description="Registry of allowed Fireworks AI models",
    )

    # --- Routing Thresholds ---
    default_accuracy_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum accuracy threshold for model selection",
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

    # --- Local Model ---
    local_model_enabled: bool = Field(
        default=True,
        description="Enable local model inference for $0 Fireworks token cost",
    )
    local_model_path: str = Field(
        default="models/qwen2.5-3b-instruct-q4_k_m.gguf",
        description="Path to GGUF model weights file",
    )
    local_model_name: str = Field(
        default="local:qwen-2.5-3b",
        description="Local model identifier in the capability matrix",
    )
    local_model_context_length: int = Field(
        default=2048,
        ge=256,
        description="Max context window for local model (tokens)",
    )
    local_model_threads: int = Field(
        default=2,
        ge=1,
        description="CPU threads for local model inference",
    )

    model_config = {
        "env_prefix": "",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    # --- Helpers ---

    def get_model_path(self, short_name: str) -> str:
        """Resolve a short model name to its full Fireworks API path.

        Args:
            short_name: Short model identifier (e.g. 'minimax-m3').

        Returns:
            Full Fireworks model path string.

        Raises:
            ValueError: If the model name is not in the allowed list.
        """
        if short_name not in self.allowed_models:
            raise ValueError(
                f"Model '{short_name}' is not allowed. "
                f"Allowed: {list(self.allowed_models.keys())}"
            )
        return self.allowed_models[short_name]

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
