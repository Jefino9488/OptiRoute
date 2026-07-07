"""In-memory cache manager — exact + normalised prompt caching.

No external dependencies (no Redis, no FAISS).  The container runs for
at most 10 minutes, so simple Python dicts are sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CachedResponse:
    """A cached routing + execution result."""

    response: str
    model_used: str
    cost: float
    confidence: float
    task_vector: dict[str, Any] = field(default_factory=dict)
    routing_explanation: str = ""


class CacheManager:
    """Two-tier in-memory cache: exact hash → normalised hash.

    Tier 0 (exact):      SHA-256 of raw prompt.
    Tier 1 (normalised): SHA-256 of normalised prompt.
    """

    def __init__(self) -> None:
        self._exact: dict[str, CachedResponse] = {}
        self._normalized: dict[str, CachedResponse] = {}
        self._hits = 0
        self._misses = 0

    def get(
        self,
        raw_hash: str,
        normalized_hash: str,
    ) -> tuple[CachedResponse | None, str]:
        """Look up a cached response.

        Parameters
        ----------
        raw_hash : str
            SHA-256 of the raw (unmodified) prompt.
        normalized_hash : str
            SHA-256 of the normalised prompt.

        Returns
        -------
        tuple[CachedResponse | None, str]
            The cached response (or None) and the cache tier that matched
            (``"exact"``, ``"normalized"``, or ``"miss"``).
        """
        if raw_hash in self._exact:
            self._hits += 1
            logger.info("cache.hit", tier="exact")
            return self._exact[raw_hash], "exact"

        if normalized_hash in self._normalized:
            self._hits += 1
            logger.info("cache.hit", tier="normalized")
            return self._normalized[normalized_hash], "normalized"

        self._misses += 1
        return None, "miss"

    def set(
        self,
        raw_hash: str,
        normalized_hash: str,
        response: CachedResponse,
    ) -> None:
        """Store a response in both cache tiers.

        Parameters
        ----------
        raw_hash : str
            SHA-256 of the raw prompt.
        normalized_hash : str
            SHA-256 of the normalised prompt.
        response : CachedResponse
            The result to cache.
        """
        self._exact[raw_hash] = response
        self._normalized[normalized_hash] = response
        logger.debug(
            "cache.set",
            exact_size=len(self._exact),
            normalized_size=len(self._normalized),
        )

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate as a float in [0, 1]."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "exact_entries": len(self._exact),
            "normalized_entries": len(self._normalized),
        }
