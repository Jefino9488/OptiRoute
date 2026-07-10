"""Metrics collector — tracks per-request routing and cost data.

All data is kept in-memory with optional JSONL file dump.
No external dependencies (no Redis, no databases).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RequestMetric:
    """Metric record for a single routing request."""

    prompt_hash: str = ""
    model_used: str = ""
    task_type: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    confidence: float = 0.0
    cache_hit: bool = False
    cache_tier: str = "miss"
    escalated: bool = False
    escalation_depth: int = 0
    success: bool = True
    timestamp: float = 0.0
    fireworks_tokens: int = 0
    routing_path: str = ""

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class MetricsCollector:
    """Collect and aggregate per-request metrics.

    Parameters
    ----------
    output_path : str | None
        Optional path to a JSONL file for persistent logging.
    """

    def __init__(self, output_path: str | None = None) -> None:
        self._metrics: list[RequestMetric] = []
        self._output_path = Path(output_path) if output_path else None

    def record(self, metric: RequestMetric) -> None:
        """Record a single request metric.

        Parameters
        ----------
        metric : RequestMetric
            The metric to store.
        """
        self._metrics.append(metric)
        logger.debug(
            "metrics.recorded",
            model=metric.model_used,
            cost=metric.cost,
            cache_hit=metric.cache_hit,
        )

        # Append to JSONL file if configured.
        if self._output_path:
            try:
                self._output_path.parent.mkdir(parents=True, exist_ok=True)
                with self._output_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(asdict(metric)) + "\n")
            except OSError as exc:
                logger.warning("metrics.write_error", error=str(exc))

    # -- Aggregations -------------------------------------------------------

    @property
    def total_requests(self) -> int:
        """Total number of recorded requests."""
        return len(self._metrics)

    @property
    def total_cost(self) -> float:
        """Total cost across all requests."""
        return sum(m.cost for m in self._metrics)

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate (0–1)."""
        if not self._metrics:
            return 0.0
        hits = sum(1 for m in self._metrics if m.cache_hit)
        return hits / len(self._metrics)

    @property
    def escalation_rate(self) -> float:
        """Escalation rate (0–1)."""
        if not self._metrics:
            return 0.0
        escalated = sum(1 for m in self._metrics if m.escalated)
        return escalated / len(self._metrics)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if not self._metrics:
            return 0.0
        return sum(m.latency_ms for m in self._metrics) / len(self._metrics)

    def model_utilization(self) -> dict[str, int]:
        """Count of requests per model."""
        counts: dict[str, int] = {}
        for m in self._metrics:
            counts[m.model_used] = counts.get(m.model_used, 0) + 1
        return counts

    def cost_saved_vs_frontier(self) -> float:
        """Estimate cost savings vs. always using minimax-m3.

        Assumes minimax-m3 costs $0.0003/1k input + $0.0012/1k output.
        """
        frontier_cost = 0.0
        for m in self._metrics:
            frontier_cost += (
                m.tokens_input * 0.0003 / 1000
                + m.tokens_output * 0.0012 / 1000
            )
        return max(0.0, frontier_cost - self.total_cost)

    def summary(self) -> dict[str, Any]:
        """Return a full metrics summary dict."""
        local_count = sum(
            1 for m in self._metrics
            if m.model_used.startswith("local:") or m.model_used == "local"
        )
        fireworks_count = sum(
            1 for m in self._metrics
            if not (m.model_used.startswith("local:") or m.model_used == "local")
            and not m.cache_hit
        )
        cache_count = sum(1 for m in self._metrics if m.cache_hit)
        total_fw_tokens = sum(m.fireworks_tokens for m in self._metrics)

        return {
            "total_requests": self.total_requests,
            "total_cost": round(self.total_cost, 8),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "escalation_rate": round(self.escalation_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "model_utilization": self.model_utilization(),
            "cost_saved_vs_frontier": round(self.cost_saved_vs_frontier(), 8),
            "local_count": local_count,
            "fireworks_count": fireworks_count,
            "cache_count": cache_count,
            "total_fireworks_tokens": total_fw_tokens,
        }
