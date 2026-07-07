"""Tests for the Decision Engine — Phase 3 of OptiRoute.

Covers:
- Math task routing to cheapest model
- Code task routing to Kimi when accuracy demands it
- Creative task excluding Kimi (fails_on)
- Deterministic bypass for simple math
- Deterministic bypass for JSON extraction
- Fallback to minimax-m3 when no model meets threshold
- Cost estimation with Kimi's 1.3x multiplier
- Escalation policy basics
- Weighted accuracy computation
- RoutingDecision serialisation
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from app.router.capability_matrix import CapabilityMatrix
from app.router.decision_engine import DecisionEngine, RoutingDecision
from app.router.policy import EscalationPolicy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MATRIX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "capability_matrix.json"
)


@pytest.fixture()
def capability_matrix() -> CapabilityMatrix:
    """Load the real default capability matrix."""
    return CapabilityMatrix(_MATRIX_PATH)


@pytest.fixture()
def engine(capability_matrix: CapabilityMatrix) -> DecisionEngine:
    """Create a DecisionEngine backed by the default matrix."""
    return DecisionEngine(capability_matrix)


@pytest.fixture()
def policy() -> EscalationPolicy:
    """Create the default escalation policy."""
    return EscalationPolicy(max_depth=2, confidence_threshold=0.7)


# ---------------------------------------------------------------------------
# 1. Math task routes to cheapest model (gemma-4-26b-a4b-it)
# ---------------------------------------------------------------------------


def test_math_task_routes_to_cheapest(engine: DecisionEngine) -> None:
    """A general math task (not simple enough for deterministic) should pick
    the cheapest model whose weighted accuracy >= 0.8."""
    task_vector = {"math": 0.8, "reasoning": 0.2}
    resource_vector = {"input_tokens": 200, "output_tokens": 150, "complexity": 0.5}
    risk_vector: dict = {}

    decision = engine.select_model(task_vector, resource_vector, risk_vector)

    assert decision.model_selected == "gemma-4-26b-a4b-it"
    assert decision.predicted_accuracy >= 0.8
    assert decision.estimated_cost > 0
    assert not decision.is_deterministic


# ---------------------------------------------------------------------------
# 2. Code task routes to kimi-k2p7-code when accuracy demands it
# ---------------------------------------------------------------------------


def test_code_task_routes_to_kimi_when_accuracy_high(engine: DecisionEngine) -> None:
    """With required_accuracy=0.95 on a code task, only Kimi (0.98) qualifies."""
    task_vector = {"code": 0.9, "reasoning": 0.1}
    resource_vector = {"input_tokens": 500, "output_tokens": 300, "complexity": 0.7}
    risk_vector: dict = {}

    decision = engine.select_model(
        task_vector, resource_vector, risk_vector, required_accuracy=0.95
    )

    # Only kimi-k2p7-code has code capability >= 0.95
    assert decision.model_selected == "kimi-k2p7-code"
    assert decision.predicted_accuracy >= 0.95


# ---------------------------------------------------------------------------
# 3. Creative task excludes Kimi (fails_on)
# ---------------------------------------------------------------------------


def test_creative_task_excludes_kimi(engine: DecisionEngine) -> None:
    """Kimi has 'creative' in fails_on — it should never be selected for a
    creative-dominant task, even if accuracy would be acceptable."""
    task_vector = {"creative": 0.85, "general_qa": 0.15}
    resource_vector = {"input_tokens": 300, "output_tokens": 400, "complexity": 0.6}
    risk_vector: dict = {}

    decision = engine.select_model(task_vector, resource_vector, risk_vector)

    assert decision.model_selected != "kimi-k2p7-code"
    assert "kimi-k2p7-code" not in [
        a["model"] for a in decision.alternatives_considered
    ]
    assert not decision.is_deterministic


# ---------------------------------------------------------------------------
# 4. Deterministic bypass for simple math
# ---------------------------------------------------------------------------


def test_deterministic_bypass_calculator(engine: DecisionEngine) -> None:
    """A very simple math task (high math weight, low complexity) should be
    handled by the deterministic calculator at $0 cost."""
    task_vector = {"math": 0.95, "reasoning": 0.05}
    resource_vector = {"input_tokens": 20, "output_tokens": 10, "complexity": 0.1}
    risk_vector: dict = {}

    decision = engine.select_model(task_vector, resource_vector, risk_vector)

    assert decision.model_selected == "deterministic:calculator"
    assert decision.estimated_cost == 0.0
    assert decision.predicted_accuracy == 1.0
    assert decision.is_deterministic


# ---------------------------------------------------------------------------
# 5. Deterministic bypass for JSON extraction
# ---------------------------------------------------------------------------


def test_deterministic_bypass_json(engine: DecisionEngine) -> None:
    """A JSON extraction task with needs_json flag should bypass LLM."""
    task_vector = {"extraction": 0.9, "general_qa": 0.1}
    resource_vector = {"input_tokens": 100, "output_tokens": 50, "complexity": 0.4}
    risk_vector = {"needs_json": True}

    decision = engine.select_model(task_vector, resource_vector, risk_vector)

    assert decision.model_selected == "deterministic:json"
    assert decision.estimated_cost == 0.0
    assert decision.is_deterministic


# ---------------------------------------------------------------------------
# 6. Fallback to minimax-m3 when no model meets threshold
# ---------------------------------------------------------------------------


def test_fallback_to_minimax_when_threshold_impossible(
    engine: DecisionEngine,
) -> None:
    """With an impossibly high accuracy threshold (0.99), no model qualifies
    and the engine should fall back to minimax-m3."""
    task_vector = {"general_qa": 0.5, "math": 0.5}
    resource_vector = {"input_tokens": 500, "output_tokens": 300, "complexity": 0.5}
    risk_vector: dict = {}

    decision = engine.select_model(
        task_vector, resource_vector, risk_vector, required_accuracy=0.99
    )

    assert decision.model_selected == "minimax-m3"
    assert "fallback" in decision.reasoning.lower() or "falling back" in decision.reasoning.lower()


# ---------------------------------------------------------------------------
# 7. Cost estimation accounts for Kimi's 1.3x multiplier
# ---------------------------------------------------------------------------


def test_kimi_cost_includes_output_multiplier(
    capability_matrix: CapabilityMatrix,
) -> None:
    """Kimi's cost estimate must multiply output tokens by 1.3."""
    input_tokens = 1000
    output_tokens = 1000

    kimi_cost = capability_matrix.estimate_cost("kimi-k2p7-code", input_tokens, output_tokens)
    gemma_cost = capability_matrix.estimate_cost("gemma-4-31b-it", input_tokens, output_tokens)

    assert kimi_cost is not None
    assert gemma_cost is not None

    # Kimi: 1000 * 0.00095/1000 + 1000 * 1.3 * 0.004/1000
    #      = 0.00095 + 0.0052 = 0.00615
    expected_kimi = 1000 * 0.00095 / 1000 + 1000 * 1.3 * 0.004 / 1000
    assert abs(kimi_cost - expected_kimi) < 1e-9

    # Kimi should be significantly more expensive than Gemma 31B
    assert kimi_cost > gemma_cost * 2


# ---------------------------------------------------------------------------
# 8. Escalation policy respects depth limit
# ---------------------------------------------------------------------------


def test_escalation_respects_max_depth(policy: EscalationPolicy) -> None:
    """After max_depth escalations, should_escalate returns False."""
    # Should escalate at depth 0 and 1
    assert policy.should_escalate(confidence=0.3, current_depth=0) is True
    assert policy.should_escalate(confidence=0.3, current_depth=1) is True
    # Should NOT escalate at depth 2 (max_depth=2)
    assert policy.should_escalate(confidence=0.3, current_depth=2) is False


# ---------------------------------------------------------------------------
# 9. Escalation picks next model in cost-sorted order
# ---------------------------------------------------------------------------


def test_escalation_picks_next_cheapest(policy: EscalationPolicy) -> None:
    """get_next_model should return the next model after the current one
    in the cost-sorted eligible list."""
    eligible = [
        {"model_id": "gemma-4-26b-a4b-it", "estimated_cost": 0.0001},
        {"model_id": "gemma-4-31b-it-nvfp4", "estimated_cost": 0.0002},
        {"model_id": "gemma-4-31b-it", "estimated_cost": 0.0003},
    ]

    next_model = policy.get_next_model("gemma-4-26b-a4b-it", eligible, current_depth=0)
    assert next_model == "gemma-4-31b-it-nvfp4"

    next_model = policy.get_next_model("gemma-4-31b-it-nvfp4", eligible, current_depth=1)
    assert next_model == "gemma-4-31b-it"

    # No more models after the last one
    next_model = policy.get_next_model("gemma-4-31b-it", eligible, current_depth=1)
    assert next_model is None


# ---------------------------------------------------------------------------
# 10. RoutingDecision serialises correctly
# ---------------------------------------------------------------------------


def test_routing_decision_to_dict() -> None:
    """RoutingDecision.to_dict() should produce a JSON-serialisable dict."""
    decision = RoutingDecision(
        model_selected="gemma-4-31b-it",
        estimated_cost=0.00012,
        predicted_accuracy=0.87,
        reasoning="Test reasoning",
        alternatives_considered=[{"model": "gemma-4-26b-a4b-it", "cost": 0.00008, "accuracy": 0.83}],
    )
    d = decision.to_dict()
    assert d["model_selected"] == "gemma-4-31b-it"
    assert d["estimated_cost"] == 0.00012
    assert d["is_deterministic"] is False
    # Must be JSON-serialisable
    json.dumps(d)


# ---------------------------------------------------------------------------
# 11. Weighted accuracy computation correctness
# ---------------------------------------------------------------------------


def test_weighted_accuracy_computation(capability_matrix: CapabilityMatrix) -> None:
    """Verify weighted accuracy formula: Σ(w*c)/Σ(w)."""
    # gemma-4-26b-a4b-it: math=0.82, reasoning=0.79
    task_vector = {"math": 0.6, "reasoning": 0.4}
    expected = (0.6 * 0.82 + 0.4 * 0.79) / (0.6 + 0.4)
    # Use the static method directly
    caps = capability_matrix.get_model_capabilities("gemma-4-26b-a4b-it")
    assert caps is not None
    actual = CapabilityMatrix._compute_weighted_accuracy(task_vector, caps["capabilities"])
    assert abs(actual - expected) < 1e-9


# ---------------------------------------------------------------------------
# 12. Translation task excludes Kimi
# ---------------------------------------------------------------------------


def test_translation_task_excludes_kimi(engine: DecisionEngine) -> None:
    """Kimi has 'translation' in fails_on — never route translation tasks there."""
    task_vector = {"translation": 0.9, "general_qa": 0.1}
    resource_vector = {"input_tokens": 200, "output_tokens": 300, "complexity": 0.5}
    risk_vector: dict = {}

    decision = engine.select_model(task_vector, resource_vector, risk_vector)

    assert decision.model_selected != "kimi-k2p7-code"
    # Cheapest eligible should be picked
    assert decision.estimated_cost > 0


# ---------------------------------------------------------------------------
# 13. Escalation fallback model is always minimax-m3
# ---------------------------------------------------------------------------


def test_escalation_fallback_is_minimax(policy: EscalationPolicy) -> None:
    """get_fallback_model always returns minimax-m3."""
    assert policy.get_fallback_model() == "minimax-m3"


# ---------------------------------------------------------------------------
# 14. Missing model in capability matrix returns None
# ---------------------------------------------------------------------------


def test_missing_model_returns_none(capability_matrix: CapabilityMatrix) -> None:
    """Querying a non-existent model should return None, not crash."""
    assert capability_matrix.get_model_capabilities("nonexistent-model") is None
    assert capability_matrix.estimate_cost("nonexistent-model", 100, 100) is None


# ---------------------------------------------------------------------------
# 15. CapabilityMatrix.save round-trips correctly
# ---------------------------------------------------------------------------


def test_matrix_save_and_reload() -> None:
    """Saving and reloading the matrix should produce identical data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_matrix.json")
        # Write a minimal matrix
        data = {
            "test-model": {
                "capabilities": {"math": 0.9},
                "cost_per_1k_input": 0.001,
                "cost_per_1k_output": 0.002,
                "avg_output_multiplier": 1.0,
                "max_context": 100000,
                "fails_on": [],
                "avg_latency_ms": 500,
            }
        }
        with open(path, "w") as f:
            json.dump(data, f)

        matrix = CapabilityMatrix(path)
        assert matrix.get_all_models() == ["test-model"]

        # Update and save
        matrix.update_model("test-model", {"math": 0.95, "code": 0.8}, ["creative"])
        matrix.save()

        # Reload from disk
        matrix2 = CapabilityMatrix(path)
        entry = matrix2.get_model_capabilities("test-model")
        assert entry is not None
        assert entry["capabilities"]["math"] == 0.95
        assert entry["capabilities"]["code"] == 0.8
        assert entry["fails_on"] == ["creative"]
