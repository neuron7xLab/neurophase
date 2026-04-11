"""Tests for ``neurophase.gate.execution_gate``.

Covers the four base invariants (``I₁``, ``I₂``, ``I₃``) and the new
fifth gate state ``UNNECESSARY`` introduced by invariant ``I₄``
(``StillnessDetector``).
"""

from __future__ import annotations

import pytest

from neurophase.gate.execution_gate import ExecutionGate, GateDecision, GateState
from neurophase.gate.stillness_detector import StillnessDetector, StillnessState

# ---------------------------------------------------------------------------
# Base 4-state behaviour (unchanged)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# I₄: UNNECESSARY state + stillness layer
# ---------------------------------------------------------------------------


def _ready_but_still_gate() -> ExecutionGate:
    """Build a gate whose stillness layer is primed and ready to fire."""
    det = StillnessDetector(window=3, eps_R=1.0, eps_F=1.0, delta_min=1.0)
    return ExecutionGate(threshold=0.65, stillness_detector=det)


def _ready_but_active_gate() -> ExecutionGate:
    """Build a gate whose stillness layer will classify as ACTIVE."""
    det = StillnessDetector(window=3, eps_R=1e-6, eps_F=1e-6, delta_min=1e-6)
    return ExecutionGate(threshold=0.65, stillness_detector=det)


def test_returns_unnecessary_when_ready_but_still() -> None:
    gate = _ready_but_still_gate()
    # Prime the stillness buffer (still returns READY during warmup).
    gate.evaluate(R=0.90, delta=0.01)
    gate.evaluate(R=0.90, delta=0.01)
    gate.evaluate(R=0.90, delta=0.01)
    decision = gate.evaluate(R=0.90, delta=0.01)
    assert decision.state is GateState.UNNECESSARY
    assert decision.execution_allowed is False
    assert decision.stillness_state is StillnessState.STILL
    assert "stillness layer rejects" in decision.reason


def test_still_is_not_blocked() -> None:
    """UNNECESSARY and BLOCKED are distinct enum members — both are
    non-permissive, but they carry different semantic meaning. The
    state-is-UNNECESSARY assertion below is sufficient at runtime;
    ``GateState`` being a stdlib ``Enum`` guarantees member uniqueness."""
    gate = _ready_but_still_gate()
    for _ in range(4):
        gate.evaluate(R=0.90, delta=0.01)
    out = gate.evaluate(R=0.90, delta=0.01)
    assert out.state is GateState.UNNECESSARY


def test_ready_when_stillness_detector_not_configured() -> None:
    """Backwards compatibility: no detector → behaves exactly as pre-I₄."""
    gate = ExecutionGate(threshold=0.65)
    decision = gate.evaluate(R=0.80, delta=0.01)
    assert decision.state is GateState.READY
    assert decision.execution_allowed is True
    assert decision.stillness_state is None


def test_ready_when_delta_missing_for_stillness_layer() -> None:
    """Missing δ must fall back to READY, NOT to DEGRADED or BLOCKED.

    Rationale: the stillness layer is optional and a missing δ is a
    caller-side choice (they did not provide it), not a hardware fault.
    """
    gate = _ready_but_still_gate()
    decision = gate.evaluate(R=0.80, delta=None)
    assert decision.state is GateState.READY
    assert decision.execution_allowed is True
    assert "stillness evaluation skipped" in decision.reason


@pytest.mark.parametrize("bad_delta", [float("nan"), float("inf"), -0.1, 10.0])
def test_ready_when_delta_invalid_for_stillness_layer(bad_delta: float) -> None:
    """Invalid δ (NaN / OOR) for the stillness layer → READY fallback,
    same rationale as missing δ."""
    gate = _ready_but_still_gate()
    decision = gate.evaluate(R=0.80, delta=bad_delta)
    assert decision.state is GateState.READY
    assert decision.execution_allowed is True
    assert "stillness evaluation skipped" in decision.reason


def test_five_gate_states_exhaustive() -> None:
    """Every state in ``GateState`` must be reachable via ``evaluate``."""
    reached: set[GateState] = set()

    # BLOCKED
    reached.add(ExecutionGate(threshold=0.65).evaluate(R=0.30).state)
    # READY
    reached.add(ExecutionGate(threshold=0.65).evaluate(R=0.80).state)
    # SENSOR_ABSENT
    reached.add(ExecutionGate().evaluate(R=0.99, sensor_present=False).state)
    # DEGRADED
    reached.add(ExecutionGate().evaluate(R=float("nan")).state)
    # UNNECESSARY
    gate = _ready_but_still_gate()
    for _ in range(4):
        gate.evaluate(R=0.90, delta=0.01)
    reached.add(gate.evaluate(R=0.90, delta=0.01).state)

    assert reached == set(GateState)


def test_active_stillness_returns_ready() -> None:
    """When the stillness layer says ACTIVE, the gate must emit READY,
    execution_allowed=True, and still record the stillness state."""
    gate = _ready_but_active_gate()
    decision = gate.evaluate(R=0.80, delta=0.10)
    assert decision.state is GateState.READY
    assert decision.execution_allowed is True
    assert decision.stillness_state is StillnessState.ACTIVE


# ---------------------------------------------------------------------------
# Priority tests — upstream invariants must dominate I₄
# ---------------------------------------------------------------------------


def test_sensor_absent_overrides_stillness_layer() -> None:
    gate = _ready_but_still_gate()
    for _ in range(4):
        gate.evaluate(R=0.90, delta=0.01)
    decision = gate.evaluate(R=0.99, sensor_present=False, delta=0.01)
    assert decision.state is GateState.SENSOR_ABSENT


def test_blocked_overrides_stillness_layer() -> None:
    gate = _ready_but_still_gate()
    for _ in range(4):
        gate.evaluate(R=0.90, delta=0.01)
    decision = gate.evaluate(R=0.30, delta=0.01)
    assert decision.state is GateState.BLOCKED


def test_degraded_overrides_stillness_layer() -> None:
    gate = _ready_but_still_gate()
    for _ in range(4):
        gate.evaluate(R=0.90, delta=0.01)
    decision = gate.evaluate(R=float("nan"), delta=0.01)
    assert decision.state is GateState.DEGRADED


# ---------------------------------------------------------------------------
# GateDecision invariant: execution_allowed=True requires state=READY
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [
        GateState.BLOCKED,
        GateState.SENSOR_ABSENT,
        GateState.DEGRADED,
        GateState.UNNECESSARY,
    ],
)
def test_invariant_holds_for_unnecessary_too(state: GateState) -> None:
    with pytest.raises(ValueError, match="Invariant violated"):
        GateDecision(
            state=state,
            execution_allowed=True,
            R=0.9,
            threshold=0.65,
            reason="bypass attempt",
        )
