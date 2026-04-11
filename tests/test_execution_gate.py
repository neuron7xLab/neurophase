"""Tests for neurophase.gate.execution_gate."""

from __future__ import annotations

import pytest

from neurophase.gate.execution_gate import ExecutionGate, GateDecision, GateState


def test_blocks_below_threshold() -> None:
    gate = ExecutionGate(threshold=0.65)
    decision = gate.evaluate(R=0.50)
    assert decision.state is GateState.BLOCKED
    assert not decision.execution_allowed


def test_permits_above_threshold() -> None:
    gate = ExecutionGate(threshold=0.65)
    decision = gate.evaluate(R=0.80)
    assert decision.state is GateState.READY
    assert decision.execution_allowed


def test_sensor_absent() -> None:
    gate = ExecutionGate()
    decision = gate.evaluate(R=0.99, sensor_present=False)
    assert decision.state is GateState.SENSOR_ABSENT
    assert not decision.execution_allowed


def test_degraded_on_nan() -> None:
    gate = ExecutionGate()
    decision = gate.evaluate(R=float("nan"))
    assert decision.state is GateState.DEGRADED
    assert not decision.execution_allowed


def test_degraded_on_none() -> None:
    gate = ExecutionGate()
    decision = gate.evaluate(R=None)
    assert decision.state is GateState.DEGRADED
    assert not decision.execution_allowed


def test_invariant_cannot_be_bypassed() -> None:
    with pytest.raises(ValueError, match="Invariant violated"):
        GateDecision(
            state=GateState.BLOCKED,
            execution_allowed=True,
            R=0.5,
            threshold=0.65,
            reason="bypass attempt",
        )


def test_rejects_bad_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be"):
        ExecutionGate(threshold=0.0)
    with pytest.raises(ValueError, match="threshold must be"):
        ExecutionGate(threshold=1.0)
