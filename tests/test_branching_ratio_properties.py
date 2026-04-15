"""Property-based fuzz on :mod:`neurophase.metrics.branching_ratio`.

Locks in:

* **Scale invariance** — ``σ(c · A) = σ(A)`` for every ``c > 0``; the
  branching ratio is a pure *rate* and must be unit-less.
* **Constant activity is exactly critical** — ``σ(const) = 1.0`` for
  every non-negative constant, including zero (honest-null).
* **Classifier monotonicity** — ``critical_phase`` is non-decreasing in
  σ across every admissible band.
* **EMA convergence to geometric growth** — repeated ``(a, c·a)``
  updates drive the EMA to exactly ``c``.
* **EMA boundedness** — the EMA never leaves the convex hull of the
  prior and the instantaneous ratios it has seen.
* **Symmetric-ε zero anchor** — any stream of ``(0, 0)`` updates leaves
  σ at its prior, independently of ``α``.
* **Rejection contracts** — invalid constructor arguments and invalid
  observations raise ``ValueError`` deterministically.

Each property carries an explicit ``@settings(max_examples=...)`` cap
so the suite stays within CI budget.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from neurophase.metrics.branching_ratio import (
    BranchingRatioEMA,
    CriticalPhase,
    branching_ratio,
    critical_phase,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def _non_negative_activity(min_size: int = 2, max_size: int = 500) -> st.SearchStrategy[np.ndarray]:
    """Finite, non-negative float64 activity-like series."""
    return hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=min_size, max_value=max_size),
        elements=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    )


_POSITIVE_SCALE = st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False)
_SIGMA = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
_ALPHA = st.floats(min_value=1e-3, max_value=1.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# branching_ratio (one-shot)
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(activity=_non_negative_activity(min_size=5), scale=_POSITIVE_SCALE)
def test_scale_invariance(activity: np.ndarray, scale: float) -> None:
    """σ(c · A) = σ(A) for every c > 0 — branching is a pure rate."""
    sigma_raw = branching_ratio(activity)
    sigma_scaled = branching_ratio(scale * activity)
    assert math.isclose(sigma_raw, sigma_scaled, rel_tol=1e-9, abs_tol=1e-12)


@settings(max_examples=50, deadline=None)
@given(
    constant=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    length=st.integers(min_value=2, max_value=500),
)
def test_constant_activity_is_exactly_critical(constant: float, length: int) -> None:
    """A constant stream has σ = 1.0 whether the constant is 0 or 10⁶."""
    arr = np.full(length, constant, dtype=np.float64)
    assert branching_ratio(arr) == 1.0


@settings(max_examples=75, deadline=None)
@given(activity=_non_negative_activity(min_size=5))
def test_branching_ratio_is_non_negative(activity: np.ndarray) -> None:
    """σ ≥ 0 — denominator is a mean of non-negative values, numerator likewise."""
    assert branching_ratio(activity) >= 0.0


# ---------------------------------------------------------------------------
# critical_phase — classifier monotonicity
# ---------------------------------------------------------------------------

_PHASE_ORDER = {
    CriticalPhase.SUBCRITICAL: 0,
    CriticalPhase.CRITICAL: 1,
    CriticalPhase.SUPERCRITICAL: 2,
}


@settings(max_examples=100, deadline=None)
@given(sigma_a=_SIGMA, sigma_b=_SIGMA)
def test_critical_phase_is_monotonic(sigma_a: float, sigma_b: float) -> None:
    """σ_a ≤ σ_b ⇒ phase(σ_a) ≤ phase(σ_b) in the (SUB, CRIT, SUPER) order."""
    if sigma_a > sigma_b:
        sigma_a, sigma_b = sigma_b, sigma_a
    assert _PHASE_ORDER[critical_phase(sigma_a)] <= _PHASE_ORDER[critical_phase(sigma_b)]


@settings(max_examples=50, deadline=None)
@given(sigma=_SIGMA)
def test_critical_phase_always_returns_enum(sigma: float) -> None:
    assert critical_phase(sigma) in set(CriticalPhase)


# ---------------------------------------------------------------------------
# BranchingRatioEMA — convergence and boundedness
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    scale=_POSITIVE_SCALE,
    a=st.floats(min_value=0.1, max_value=100.0, allow_nan=False),
    alpha=st.floats(min_value=0.2, max_value=1.0, allow_nan=False),
)
def test_ema_converges_to_geometric_scale(scale: float, a: float, alpha: float) -> None:
    """Repeated (a, scale·a) updates drive σ exactly to `scale`."""
    est = BranchingRatioEMA(alpha=alpha)
    for _ in range(500):
        est.update(a, scale * a)
    # Geometric series convergence: after N steps the gap shrinks as (1-α)^N
    # so on the order of 500 iterations at α ≥ 0.2 we are well below 1e-6.
    assert math.isclose(est.sigma, scale, rel_tol=1e-6, abs_tol=1e-9)


@settings(max_examples=50, deadline=None)
@given(
    alpha=_ALPHA,
    count=st.integers(min_value=1, max_value=500),
)
def test_ema_zero_activity_at_unit_prior_stays_at_unit(alpha: float, count: int) -> None:
    """(0, 0) updates keep σ at 1.0 when the prior is critical — the
    symmetric-ε guard maps zero-activity to the instant-ratio 1.0, and
    the EMA of 1.0 into a prior of 1.0 is identically 1.0."""
    est = BranchingRatioEMA(alpha=alpha, initial_sigma=1.0)
    for _ in range(count):
        est.update(0.0, 0.0)
    assert est.sigma == 1.0


@settings(max_examples=50, deadline=None)
@given(
    alpha=_ALPHA,
    prior=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
    count=st.integers(min_value=0, max_value=200),
)
def test_ema_zero_activity_contracts_geometrically_toward_unity(
    alpha: float, prior: float, count: int
) -> None:
    """With symmetric ε, (0, 0) updates feed the EMA a constant instant
    ratio of 1.0, so after ``N`` updates the gap from 1.0 shrinks
    exactly as ``|prior − 1| · (1 − α)^N`` — the geometric contraction
    of a first-order low-pass toward its driving value."""
    est = BranchingRatioEMA(alpha=alpha, initial_sigma=prior)
    for _ in range(count):
        est.update(0.0, 0.0)
    expected_gap = abs(prior - 1.0) * (1.0 - alpha) ** count
    assert math.isclose(
        est.sigma - 1.0,
        math.copysign(expected_gap, prior - 1.0),
        rel_tol=1e-9,
        abs_tol=1e-12,
    )


@settings(max_examples=50, deadline=None)
@given(
    alpha=_ALPHA,
    prior=st.floats(min_value=0.5, max_value=1.5, allow_nan=False),
    activities=hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=2, max_value=100),
        elements=st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
    ),
)
def test_ema_stays_in_convex_hull_of_ratios_and_prior(
    alpha: float, prior: float, activities: np.ndarray
) -> None:
    """σ_t after N steps lies between min(prior, ratios) and max(prior, ratios)."""
    eps = 1e-12
    ratios = (activities[1:] + eps) / (activities[:-1] + eps)
    est = BranchingRatioEMA(alpha=alpha, initial_sigma=prior)
    for i in range(activities.size - 1):
        est.update(float(activities[i]), float(activities[i + 1]))
    lo = min(float(ratios.min()), prior)
    hi = max(float(ratios.max()), prior)
    # Small epsilon tolerance for floating-point summation noise.
    tol = 1e-9 * max(hi, 1.0)
    assert lo - tol <= est.sigma <= hi + tol


@settings(max_examples=50, deadline=None)
@given(
    ratio=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    a=st.floats(min_value=1.0, max_value=100.0, allow_nan=False),
)
def test_ema_alpha_one_equals_instantaneous_ratio(ratio: float, a: float) -> None:
    """α = 1 collapses the EMA to the one-step instantaneous ratio."""
    est = BranchingRatioEMA(alpha=1.0)
    est.update(a, ratio * a)
    assert math.isclose(est.sigma, ratio, rel_tol=1e-9, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# Rejection contracts
# ---------------------------------------------------------------------------


@settings(max_examples=25, deadline=None)
@given(
    alpha=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0000001, max_value=10.0, allow_nan=False),
    )
)
def test_ema_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        BranchingRatioEMA(alpha=alpha)


@settings(max_examples=25, deadline=None)
@given(
    a_t=st.floats(max_value=-1e-9, min_value=-1e6, allow_nan=False),
    a_t1=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
)
def test_ema_rejects_negative_a_t(a_t: float, a_t1: float) -> None:
    est = BranchingRatioEMA()
    with pytest.raises(ValueError):
        est.update(a_t, a_t1)
