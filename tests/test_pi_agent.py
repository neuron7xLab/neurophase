"""Tests for neurophase.agents.pi_agent."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.agents.pi_agent import (
    AgentEfficiency,
    MarketContext,
    PiAgent,
    PiRule,
    SemanticMemory,
)


def _rule() -> PiRule:
    return PiRule(conditions={"threshold": 0.65, "weight": 1.0}, action="enter_long")


def _context(R: float = 0.8) -> MarketContext:
    return MarketContext(R=R, dH=-0.08, kappa=-0.15, ism=1.0, regime_tag="trend")


def test_mutation_perturbs_conditions() -> None:
    rule = _rule()
    rng = np.random.default_rng(0)
    mutated = rule.generate_mutation(rng, std=0.1)
    assert mutated.conditions != rule.conditions
    assert mutated.action == rule.action


def test_mutation_rejects_negative_std() -> None:
    rule = _rule()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="non-negative"):
        rule.generate_mutation(rng, std=-0.1)


def test_clone_returns_independent_copy() -> None:
    agent = PiAgent(rule=_rule(), seed=1)
    cloned = agent.clone()
    agent.rule.conditions["threshold"] = 0.99
    assert cloned.rule.conditions["threshold"] == 0.65


def test_step_keeps_candidate_when_better() -> None:
    agent = PiAgent(rule=_rule(), seed=2)
    baseline = AgentEfficiency(sharpe=1.0, stability=0.5)
    candidate_rule = PiRule(conditions={"threshold": 0.70, "weight": 1.1}, action="enter_long")
    candidate_eff = AgentEfficiency(sharpe=2.0, stability=0.8)
    new_rule = agent.step(_context(), baseline, candidate_eff, candidate_rule)
    assert new_rule.conditions["threshold"] == 0.70
    assert len(agent.memory.store) == 1
    _, stored_rule, stored_eff = agent.memory.store[0]
    assert stored_eff.value == candidate_eff.value
    assert stored_rule.conditions["threshold"] == 0.70


def test_step_repairs_when_candidate_worse() -> None:
    agent = PiAgent(rule=_rule(), seed=3)
    baseline = AgentEfficiency(sharpe=2.0, stability=1.0)
    candidate_rule = PiRule(conditions={"threshold": 0.5, "weight": 0.5}, action="enter_short")
    candidate_eff = AgentEfficiency(sharpe=0.5, stability=0.1)
    new_rule = agent.step(_context(), baseline, candidate_eff, candidate_rule)
    # Repair restores the last stable rule — which is the initial one.
    assert new_rule.conditions["threshold"] == 0.65
    assert len(agent.memory.store) == 1


def test_efficiency_formula() -> None:
    eff = AgentEfficiency(sharpe=2.0, stability=1.0, weight=0.3)
    assert eff.value == pytest.approx(2.3)


def test_semantic_memory_retrieves_nearest() -> None:
    mem = SemanticMemory()
    ctx_a = _context(R=0.85)
    ctx_b = _context(R=0.70)
    rule = _rule()
    mem.add(ctx_a, rule, AgentEfficiency(sharpe=2.0, stability=0.8))
    mem.add(ctx_b, rule, AgentEfficiency(sharpe=1.2, stability=0.6))
    hits = mem.retrieve(_context(R=0.84), threshold=0.95, limit=1)
    assert len(hits) >= 1
    assert hits[0][0].R == 0.85


def test_semantic_memory_rejects_bad_threshold() -> None:
    mem = SemanticMemory()
    with pytest.raises(ValueError, match="threshold"):
        mem.retrieve(_context(), threshold=1.5)


def test_semantic_memory_rejects_bad_limit() -> None:
    mem = SemanticMemory()
    with pytest.raises(ValueError, match="limit"):
        mem.retrieve(_context(), limit=0)


def test_semantic_memory_zero_query_returns_empty() -> None:
    mem = SemanticMemory()
    zero_ctx = MarketContext(R=0.0, dH=0.0, kappa=0.0, ism=0.0)
    assert mem.retrieve(zero_ctx) == []
