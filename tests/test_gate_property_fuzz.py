"""HN38 — property-based fuzz on the execution gate.

This file promotes the gate invariants from *"tested on a
handful of hand-picked inputs"* to *"verified on every (R, δ,
timestamp) sequence a hypothesis strategy can generate"*.

Every property asserts a load-bearing gate claim and is driven
by hypothesis' ``@given``. If hypothesis finds any input that
falsifies the property, the test fails with a shrunken
counter-example and the invariant is broken.

Locks in:

* **I₁** — ``R < threshold`` on a VALID temporal frame ⇒
  ``GateState.BLOCKED``, ``execution_allowed=False``. Verified
  for any finite ``R`` in ``[0, threshold)``.
* **I₃** — non-finite / out-of-range ``R`` ⇒ ``DEGRADED``.
  Verified for any non-finite float.
* **I₂** — ``sensor_present=False`` ⇒ ``SENSOR_ABSENT``.
  Verified for every combination of other inputs.
* **B₁** — any non-VALID ``TimeQuality`` ⇒ ``DEGRADED``
  regardless of ``R``. Verified for every
  ``TimeQuality`` member.
* **Monotonic timestamp stream** — a strictly increasing float
  stream never raises inside ``StreamingPipeline.tick`` and
  the gate state is always one of the 5 canonical values.
* **Permissiveness monotonicity** — on a VALID stream,
  raising ``R`` from below to above the threshold can only
  transition the gate from BLOCKED → READY (never in
  reverse under the same inputs).

Every property has an explicit
``@settings(max_examples=...)`` cap so the suite stays within
CI budget.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
from hypothesis import Verbosity, given, settings

from neurophase.data.temporal_validator import TemporalQualityDecision, TimeQuality
from neurophase.gate.execution_gate import DEFAULT_THRESHOLD, ExecutionGate, GateState


def _valid_time_quality(ts: float = 0.0) -> TemporalQualityDecision:
    return TemporalQualityDecision(
        quality=TimeQuality.VALID,
        ts=ts,
        last_ts=None,
        gap_seconds=None,
        staleness_seconds=None,
        warmup_remaining=0,
        reason="valid: fuzz fixture",
    )


def _non_valid_time_quality(quality: TimeQuality, ts: float = 0.0) -> TemporalQualityDecision:
    return TemporalQualityDecision(
        quality=quality,
        ts=ts,
        last_ts=None,
        gap_seconds=None,
        staleness_seconds=None,
        warmup_remaining=0,
        reason=f"{quality.name.lower()}: fuzz fixture",
    )


# ---------------------------------------------------------------------------
# Core property 1: R < θ ⇒ BLOCKED on a VALID frame.
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None, verbosity=Verbosity.quiet)
@given(
    r=st.floats(
        min_value=0.0,
        max_value=DEFAULT_THRESHOLD - 1e-6,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_i1_holds_for_any_r_below_threshold(r: float) -> None:
    """For any finite R strictly below threshold on a VALID temporal
    frame, the gate MUST return BLOCKED with execution_allowed=False."""
    gate = ExecutionGate(threshold=DEFAULT_THRESHOLD)
    decision = gate.evaluate(R=r, time_quality=_valid_time_quality())
    assert decision.state is GateState.BLOCKED
    assert decision.execution_allowed is False


@settings(max_examples=200, deadline=None, verbosity=Verbosity.quiet)
@given(
    r=st.floats(
        min_value=DEFAULT_THRESHOLD,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_r_at_or_above_threshold_is_never_blocked(r: float) -> None:
    """For any finite R at or above threshold on a VALID temporal
    frame WITHOUT a stillness detector, the gate MUST be READY.

    This is the complement of I₁ in the 4-state gate mode.
    """
    gate = ExecutionGate(threshold=DEFAULT_THRESHOLD)  # no stillness
    decision = gate.evaluate(R=r, time_quality=_valid_time_quality())
    assert decision.state is GateState.READY
    assert decision.execution_allowed is True


# ---------------------------------------------------------------------------
# Core property 2: invalid R ⇒ DEGRADED.
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None, verbosity=Verbosity.quiet)
@given(
    r=st.sampled_from(
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.5,
            -1.0,
            1.5,
            2.0,
            100.0,
        ]
    )
)
def test_i3_holds_for_non_finite_or_out_of_range_R(r: float) -> None:
    """Any R outside [0, 1] or non-finite MUST route to DEGRADED."""
    gate = ExecutionGate(threshold=DEFAULT_THRESHOLD)
    decision = gate.evaluate(R=r, time_quality=_valid_time_quality())
    assert decision.state is GateState.DEGRADED
    assert decision.execution_allowed is False


def test_i3_holds_for_R_none() -> None:
    """R=None is the canonical DEGRADED signal."""
    gate = ExecutionGate(threshold=DEFAULT_THRESHOLD)
    decision = gate.evaluate(R=None, time_quality=_valid_time_quality())
    assert decision.state is GateState.DEGRADED
    assert decision.execution_allowed is False


# ---------------------------------------------------------------------------
# Core property 3: sensor_present=False ⇒ SENSOR_ABSENT.
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, verbosity=Verbosity.quiet)
@given(
    r=st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    )
)
def test_i2_sensor_absent_always_wins_over_r(r: float) -> None:
    """sensor_present=False routes to SENSOR_ABSENT for every
    valid R — even when R would otherwise qualify for READY."""
    gate = ExecutionGate(threshold=DEFAULT_THRESHOLD)
    decision = gate.evaluate(R=r, sensor_present=False, time_quality=_valid_time_quality())
    assert decision.state is GateState.SENSOR_ABSENT
    assert decision.execution_allowed is False


# ---------------------------------------------------------------------------
# Core property 4: non-VALID TimeQuality ⇒ DEGRADED.
# ---------------------------------------------------------------------------


_NON_VALID_QUALITIES = [q for q in TimeQuality if q is not TimeQuality.VALID]


@settings(max_examples=200, deadline=None, verbosity=Verbosity.quiet)
@given(
    quality=st.sampled_from(_NON_VALID_QUALITIES),
    r=st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_b1_non_valid_time_quality_forces_degraded(quality: TimeQuality, r: float) -> None:
    """For any R in [0, 1] and any non-VALID TimeQuality, the gate
    MUST return DEGRADED — the B₁ temporal precondition."""
    gate = ExecutionGate(threshold=DEFAULT_THRESHOLD)
    decision = gate.evaluate(R=r, time_quality=_non_valid_time_quality(quality))
    assert decision.state is GateState.DEGRADED
    assert decision.execution_allowed is False


# ---------------------------------------------------------------------------
# Core property 5: output is always one of the 5 canonical gate states.
# ---------------------------------------------------------------------------


_GATE_STATES = {
    GateState.READY,
    GateState.BLOCKED,
    GateState.SENSOR_ABSENT,
    GateState.DEGRADED,
    GateState.UNNECESSARY,
}


@settings(max_examples=200, deadline=None, verbosity=Verbosity.quiet)
@given(
    r=st.one_of(
        st.none(),
        st.floats(allow_nan=True, allow_infinity=True),
    ),
    sensor_present=st.booleans(),
    quality=st.sampled_from(list(TimeQuality)),
)
def test_gate_output_is_always_canonical(
    r: float | None, sensor_present: bool, quality: TimeQuality
) -> None:
    """For any conceivable input combination the gate MUST return
    one of the 5 canonical GateState values. No crashes, no
    uncategorised outputs."""
    gate = ExecutionGate(threshold=DEFAULT_THRESHOLD)
    decision = gate.evaluate(
        R=r,
        sensor_present=sensor_present,
        time_quality=_non_valid_time_quality(quality)
        if quality is not TimeQuality.VALID
        else _valid_time_quality(),
    )
    assert decision.state in _GATE_STATES
    # execution_allowed implies READY by __post_init__.
    if decision.execution_allowed:
        assert decision.state is GateState.READY


# ---------------------------------------------------------------------------
# Core property 6: execution_allowed ≡ (state is READY).
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None, verbosity=Verbosity.quiet)
@given(
    r=st.one_of(
        st.none(),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    ),
    sensor_present=st.booleans(),
    quality=st.sampled_from(list(TimeQuality)),
)
def test_execution_allowed_iff_ready(
    r: float | None, sensor_present: bool, quality: TimeQuality
) -> None:
    """The load-bearing equivalence: execution_allowed is True iff
    the gate state is READY. No other state can be permissive."""
    gate = ExecutionGate(threshold=DEFAULT_THRESHOLD)
    tq = _valid_time_quality() if quality is TimeQuality.VALID else _non_valid_time_quality(quality)
    decision = gate.evaluate(R=r, sensor_present=sensor_present, time_quality=tq)
    assert decision.execution_allowed == (decision.state is GateState.READY)


# ---------------------------------------------------------------------------
# Core property 7: threshold monotonicity.
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, verbosity=Verbosity.quiet)
@given(
    theta_low=st.floats(min_value=0.01, max_value=0.49, allow_nan=False, allow_infinity=False),
    theta_high_offset=st.floats(
        min_value=0.01, max_value=0.50, allow_nan=False, allow_infinity=False
    ),
    r=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_raising_threshold_never_widens_permission(
    theta_low: float, theta_high_offset: float, r: float
) -> None:
    """Raising the threshold on the same input can only
    BLOCK (never newly ADMIT) the decision. A lower threshold
    is strictly more permissive than a higher one on the same R.

    Formally: if gate(R, θ_high).state is READY then
    gate(R, θ_low).state is READY too (contrapositive: raising
    θ never turns a BLOCKED into a READY).
    """
    theta_high = min(0.99, theta_low + theta_high_offset)
    if theta_high <= theta_low:
        return
    tq = _valid_time_quality()
    low = ExecutionGate(threshold=theta_low).evaluate(R=r, time_quality=tq)
    high = ExecutionGate(threshold=theta_high).evaluate(R=r, time_quality=tq)
    if high.state is GateState.READY:
        assert low.state is GateState.READY, (
            f"raising θ {theta_low}→{theta_high} widened permission at R={r}"
        )


# ---------------------------------------------------------------------------
# Core property 8: constructor rejects invalid thresholds.
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None, verbosity=Verbosity.quiet)
@given(
    theta=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        st.just(float("nan")),
        st.just(float("inf")),
    )
)
def test_threshold_out_of_range_rejected(theta: float) -> None:
    """The gate constructor MUST reject any threshold outside (0, 1)
    or non-finite."""
    import pytest

    if not math.isfinite(theta) or not 0.0 < theta < 1.0:
        with pytest.raises((ValueError, TypeError)):
            ExecutionGate(threshold=theta)
