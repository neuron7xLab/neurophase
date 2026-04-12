"""Tests verifying IEEE 754-derived epsilon tolerance for delta validation.

Covers both ``ExecutionGate`` and ``StillnessDetector`` boundary behaviour
around the ``_DELTA_UPPER`` constant.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from neurophase.gate.execution_gate import _DELTA_UPPER, ExecutionGate, GateState
from neurophase.gate.stillness_detector import StillnessDetector

# ---------------------------------------------------------------------------
# Unit tests on the constant itself
# ---------------------------------------------------------------------------


def test_delta_upper_is_above_pi() -> None:
    assert np.pi < _DELTA_UPPER


def test_delta_upper_rejects_clearly_invalid() -> None:
    # 0.01 rad slack is far beyond any float64 rounding — must be outside the tolerance.
    assert np.pi + 0.01 > _DELTA_UPPER


# ---------------------------------------------------------------------------
# Acceptance / rejection at the gate (stillness layer skip logic)
# ---------------------------------------------------------------------------


def _gate_with_detector() -> ExecutionGate:
    """Gate wired to a detector; R deliberately ≥ threshold so delta matters."""
    return ExecutionGate(threshold=0.65, stillness_detector=StillnessDetector())


def test_delta_upper_accepts_machine_epsilon_drift() -> None:
    """1 ULP drift (≈1e-16) above π must be accepted by both gate and detector."""
    delta_1ulp = np.pi + 1e-16

    # Gate: must not skip stillness due to invalid delta — it should route
    # to the detector path and return READY (warmup) rather than the
    # "skipped" READY.  We just verify no exception and the state is READY.
    gate = _gate_with_detector()
    decision = gate.evaluate(R=0.80, delta=delta_1ulp)
    assert decision.state is GateState.READY

    # Detector: must not raise ValueError.
    det = StillnessDetector()
    result = det.update(R=0.80, delta=delta_1ulp)
    assert result.delta == pytest.approx(delta_1ulp)


def test_delta_at_exact_pi_is_accepted() -> None:
    """Exact π must be accepted by both the gate and the detector."""
    gate = _gate_with_detector()
    decision = gate.evaluate(R=0.80, delta=np.pi)
    assert decision.state is GateState.READY

    det = StillnessDetector()
    result = det.update(R=0.80, delta=np.pi)
    assert result.delta == pytest.approx(np.pi)


def test_delta_above_tolerance_is_rejected_by_gate() -> None:
    """δ = π + 0.001 is clearly invalid; gate must skip stillness (READY, reason mentions skipped)."""
    gate = _gate_with_detector()
    decision = gate.evaluate(R=0.80, delta=np.pi + 0.001)
    assert decision.state is GateState.READY
    assert "skipped" in decision.reason


def test_delta_above_tolerance_is_rejected_by_detector() -> None:
    """δ = π + 0.001 must raise ValueError in the detector."""
    det = StillnessDetector()
    with pytest.raises(ValueError, match="delta must be in"):
        det.update(R=0.80, delta=np.pi + 0.001)


# ---------------------------------------------------------------------------
# Property-based boundary consistency
# ---------------------------------------------------------------------------


@given(
    delta=st.floats(
        min_value=np.pi - 1e-10,
        max_value=np.pi + 1e-10,
        allow_nan=False,
        allow_infinity=False,
    )
)
@settings(max_examples=500)
def test_boundary_behavior_consistent_between_gate_and_detector(delta: float) -> None:
    """Gate and detector must agree on accept/reject for deltas near π.

    The gate skips the stillness layer (returns READY with "skipped") when
    delta is outside [0, _DELTA_UPPER].  The detector raises ValueError for
    the same set.  Both must therefore have identical accept/reject decisions.
    """
    R = 0.80
    gate = _gate_with_detector()
    gate_decision = gate.evaluate(R=R, delta=delta)

    gate_skipped = "skipped" in gate_decision.reason

    det = StillnessDetector()
    detector_raised = False
    try:
        det.update(R=R, delta=delta)
    except ValueError:
        detector_raised = True

    # If the gate skipped (invalid delta), the detector must have raised,
    # and vice versa.
    assert gate_skipped == detector_raised, (
        f"Inconsistency at delta={delta!r}: "
        f"gate_skipped={gate_skipped}, detector_raised={detector_raised}"
    )
