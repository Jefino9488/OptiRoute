"""Base executor types shared across all execution backends.

Defines the ExecutionResult dataclass returned by every executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    """Result of executing a prompt via any backend.

    Attributes
    ----------
    response : str
        The generated text.
    model_used : str
        Short model identifier that produced this response.
    tokens_input : int
        Number of input tokens consumed.
    tokens_output : int
        Number of output tokens generated.
    cost : float
        Estimated cost in USD.
    latency_ms : float
        Wall-clock latency in milliseconds.
    confidence : float
        Initial confidence estimate (may be refined by the validator).
    raw_metadata : dict
        Full API response for debugging.
    """

    response: str
    model_used: str
    tokens_input: int = 0
    tokens_output: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    confidence: float = 1.0
    raw_metadata: dict[str, Any] = field(default_factory=dict)
