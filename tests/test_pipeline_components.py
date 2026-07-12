"""Tests for the executor components — tools, cache, confidence, and metrics."""

from __future__ import annotations

import pytest

from app.cache.manager import CacheManager, CachedResponse
from app.confidence.validator import ConfidenceValidator
from app.metrics.collector import MetricsCollector, RequestMetric


# ─── Cache Manager ───────────────────────────────────────────────


class TestCacheManager:
    def setup_method(self) -> None:
        self.cache = CacheManager()

    def test_miss_on_empty_cache(self) -> None:
        result, tier = self.cache.get("hash1", "norm1")
        assert result is None
        assert tier == "miss"

    def test_exact_hit(self) -> None:
        resp = CachedResponse(
            response="test", model_used="gemma", cost=0.01, confidence=0.9
        )
        self.cache.set("hash1", "norm1", resp)
        result, tier = self.cache.get("hash1", "norm1")
        assert result is not None
        assert tier == "exact"
        assert result.response == "test"

    def test_normalized_hit(self) -> None:
        resp = CachedResponse(
            response="test", model_used="gemma", cost=0.01, confidence=0.9
        )
        self.cache.set("hash1", "norm1", resp)
        result, tier = self.cache.get("different_raw", "norm1")
        assert result is not None
        assert tier == "normalized"

    def test_hit_rate(self) -> None:
        resp = CachedResponse(
            response="test", model_used="gemma", cost=0.01, confidence=0.9
        )
        self.cache.set("h1", "n1", resp)
        self.cache.get("h1", "n1")  # hit
        self.cache.get("h2", "n2")  # miss
        assert self.cache.hit_rate == 0.5

    def test_stats(self) -> None:
        stats = self.cache.stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


# ─── Confidence Validator ────────────────────────────────────────


class TestConfidenceValidator:
    def setup_method(self) -> None:
        self.validator = ConfidenceValidator()

    def test_empty_response(self) -> None:
        result = self.validator.validate("")
        assert result.confidence == 0.0
        assert "Empty response" in result.issues

    def test_good_response(self) -> None:
        result = self.validator.validate(
            "The capital of France is Paris. It is located in northern France."
        )
        assert result.confidence > 0.7

    def test_error_indicator(self) -> None:
        result = self.validator.validate("[ERROR] Something went wrong")
        assert result.confidence < 0.8
        assert any("error" in i.lower() for i in result.issues)

    def test_repetition_detection(self) -> None:
        repeated = " ".join(["hello world foo"] * 20)
        result = self.validator.validate(repeated)
        assert any("repetition" in i.lower() for i in result.issues)

    def test_short_response_penalty(self) -> None:
        result = self.validator.validate("Yes", task_type="reasoning")
        assert result.confidence < 1.0


# ─── Metrics Collector ───────────────────────────────────────────


class TestMetricsCollector:
    def setup_method(self) -> None:
        self.collector = MetricsCollector()

    def test_empty_summary(self) -> None:
        summary = self.collector.summary()
        assert summary["total_requests"] == 0
        assert summary["total_cost"] == 0.0

    def test_record_and_summary(self) -> None:
        self.collector.record(RequestMetric(
            model_used="gemma-4-26b-a4b-it",
            cost=0.001,
            latency_ms=100.0,
            tokens_input=100,
            tokens_output=50,
        ))
        self.collector.record(RequestMetric(
            model_used="gemma-4-31b-it",
            cost=0.002,
            latency_ms=200.0,
            tokens_input=200,
            tokens_output=100,
        ))
        summary = self.collector.summary()
        assert summary["total_requests"] == 2
        assert summary["total_cost"] == 0.003
        assert summary["avg_latency_ms"] == 150.0

    def test_model_utilization(self) -> None:
        self.collector.record(RequestMetric(model_used="gemma-4-26b-a4b-it"))
        self.collector.record(RequestMetric(model_used="gemma-4-26b-a4b-it"))
        self.collector.record(RequestMetric(model_used="gemma-4-31b-it"))
        util = self.collector.model_utilization()
        assert util["gemma-4-26b-a4b-it"] == 2
        assert util["gemma-4-31b-it"] == 1

    def test_cache_hit_rate(self) -> None:
        self.collector.record(RequestMetric(cache_hit=True))
        self.collector.record(RequestMetric(cache_hit=False))
        assert self.collector.cache_hit_rate == 0.5

    def test_escalation_rate(self) -> None:
        self.collector.record(RequestMetric(escalated=True))
        self.collector.record(RequestMetric(escalated=False))
        self.collector.record(RequestMetric(escalated=False))
        assert abs(self.collector.escalation_rate - 1 / 3) < 0.01
