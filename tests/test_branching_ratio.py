"""Tests for neurophase.metrics.branching_ratio."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.metrics.branching_ratio import (
    BranchingRatioEMA,
    CriticalPhase,
    branching_ratio,
    critical_phase,
)

# ---------------------------------------------------------------------------
# branching_ratio — one-shot estimator
# ---------------------------------------------------------------------------


def test_constant_activity_is_critical() -> None:
    assert branching_ratio(np.full(100, 7.0)) == pytest.approx(1.0)


def test_geometric_decay_is_subcritical() -> None:
    """A_t = 2^(-t) ⇒ σ → 0.5 as N grows."""
    n = 200
    arr = np.power(2.0, -np.arange(n, dtype=np.float64))
    sigma = branching_ratio(arr)
    assert 0.45 < sigma < 0.55


def test_geometric_growth_is_supercritical() -> None:
    """A_t = 2^t ⇒ σ → 2.0 (bounded for numerical sanity)."""
    arr = np.power(2.0, np.arange(40, dtype=np.float64))
    sigma = branching_ratio(arr)
    assert 1.95 < sigma < 2.05


def test_degenerate_inputs_return_unit() -> None:
    assert branching_ratio([]) == 1.0
    assert branching_ratio([3.14]) == 1.0
    assert branching_ratio(np.zeros(50)) == 1.0  # zero activity → honest null


def test_negative_activity_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        branching_ratio([1.0, -0.5, 2.0])


# ---------------------------------------------------------------------------
# critical_phase — band classifier
# ---------------------------------------------------------------------------


def test_critical_phase_labels() -> None:
    assert critical_phase(0.5) is CriticalPhase.SUBCRITICAL
    assert critical_phase(1.0) is CriticalPhase.CRITICAL
    assert critical_phase(1.5) is CriticalPhase.SUPERCRITICAL


def test_critical_phase_band_edges_are_critical() -> None:
    # Boundaries are inside the critical band (closed interval).
    assert critical_phase(0.95) is CriticalPhase.CRITICAL
    assert critical_phase(1.05) is CriticalPhase.CRITICAL


def test_critical_phase_custom_band() -> None:
    assert (
        critical_phase(0.99, subcritical_max=0.90, supercritical_min=1.10) is CriticalPhase.CRITICAL
    )
    assert (
        critical_phase(0.85, subcritical_max=0.90, supercritical_min=1.10)
        is CriticalPhase.SUBCRITICAL
    )


def test_critical_phase_rejects_degenerate_band() -> None:
    with pytest.raises(ValueError):
        critical_phase(1.0, subcritical_max=1.10, supercritical_min=0.90)
    with pytest.raises(ValueError):
        critical_phase(1.0, subcritical_max=0.0, supercritical_min=1.10)


def test_critical_phase_str_roundtrip() -> None:
    assert str(CriticalPhase.CRITICAL) == "critical"
    assert str(CriticalPhase.SUBCRITICAL) == "subcritical"
    assert str(CriticalPhase.SUPERCRITICAL) == "supercritical"


# ---------------------------------------------------------------------------
# BranchingRatioEMA — streaming estimator
# ---------------------------------------------------------------------------


def test_ema_starts_at_prior() -> None:
    est = BranchingRatioEMA(initial_sigma=1.0)
    assert est.sigma == 1.0
    assert est.observations == 0
    assert est.phase is CriticalPhase.CRITICAL


def test_ema_constant_activity_stays_at_one() -> None:
    est = BranchingRatioEMA(alpha=0.5)
    for _ in range(20):
        est.update(5.0, 5.0)
    assert est.sigma == pytest.approx(1.0, abs=1e-9)
    assert est.phase is CriticalPhase.CRITICAL


def test_ema_converges_to_supercritical_on_doubling() -> None:
    est = BranchingRatioEMA(alpha=0.5)
    for _ in range(60):
        est.update(1.0, 2.0)
    assert est.sigma == pytest.approx(2.0, abs=1e-6)
    assert est.phase is CriticalPhase.SUPERCRITICAL


def test_ema_converges_to_subcritical_on_halving() -> None:
    est = BranchingRatioEMA(alpha=0.5)
    for _ in range(60):
        est.update(2.0, 1.0)
    assert est.sigma == pytest.approx(0.5, abs=1e-6)
    assert est.phase is CriticalPhase.SUBCRITICAL


def test_ema_symmetric_eps_anchors_zero_activity_at_prior() -> None:
    """(0, 0) updates must leave σ at its prior — the honest null."""
    est = BranchingRatioEMA(alpha=0.3, initial_sigma=1.0)
    for _ in range(50):
        est.update(0.0, 0.0)
    assert est.sigma == pytest.approx(1.0, abs=1e-9)


def test_ema_counts_observations() -> None:
    est = BranchingRatioEMA()
    for _ in range(7):
        est.update(1.0, 1.0)
    assert est.observations == 7


def test_ema_reset_restores_prior() -> None:
    est = BranchingRatioEMA(alpha=0.5, initial_sigma=1.0)
    for _ in range(10):
        est.update(1.0, 2.0)
    assert est.sigma > 1.5
    est.reset()
    assert est.sigma == 1.0
    assert est.observations == 0


def test_ema_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        BranchingRatioEMA(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        BranchingRatioEMA(alpha=1.5)


def test_ema_rejects_invalid_eps() -> None:
    with pytest.raises(ValueError, match="eps"):
        BranchingRatioEMA(eps=0.0)


def test_ema_rejects_negative_activity() -> None:
    est = BranchingRatioEMA()
    with pytest.raises(ValueError, match="non-negative"):
        est.update(-1.0, 2.0)
    with pytest.raises(ValueError, match="non-negative"):
        est.update(1.0, -2.0)


# ---------------------------------------------------------------------------
# Integration: one-shot ↔ streaming agree in the alpha = 1 limit
# ---------------------------------------------------------------------------


def test_streaming_alpha_one_matches_instantaneous_ratio() -> None:
    """With α = 1 the EMA collapses to the instantaneous ratio."""
    est = BranchingRatioEMA(alpha=1.0)
    est.update(1.0, 2.0)
    assert est.sigma == pytest.approx(2.0, abs=1e-9)
    est.update(4.0, 1.0)
    assert est.sigma == pytest.approx(0.25, abs=1e-9)
