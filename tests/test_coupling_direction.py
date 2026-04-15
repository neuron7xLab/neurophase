"""Tests for :mod:`neurophase.sync.coupling_direction`.

Covers:

* Output-type integrity — every :class:`CouplingDirection` field
  satisfies its declared invariant.
* Direction detection — when the brain side actually drives the market
  side (``ψ_market[t] = ψ_brain[t-1] + noise``) the verdict labels the
  brain as the leader and assigns a low forward p-value.
* Symmetry — independent phase pairs land on neither ``brain_leads``
  nor ``market_leads``.
* ``order_parameter`` handling — supplying ``R(t)`` populates ``σ_R``;
  omitting it returns the honest-null ``σ_R = 1.0``.
* Validation — shape and finiteness contracts at the boundary.
* Determinism under a shared seed.
* End-to-end integration with a real
  :class:`CoupledBrainMarketSystem` run.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.metrics.branching_ratio import CriticalPhase
from neurophase.sync.coupled_brain_market import CoupledBrainMarketSystem
from neurophase.sync.coupling_direction import (
    CouplingDirection,
    analyse_coupling,
)

# ---------------------------------------------------------------------------
# Fixtures: deterministic phase-pair generators
# ---------------------------------------------------------------------------


def _brain_drives_market(
    n: int = 1500, noise: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """ψ_market[t] = 0.9·ψ_brain[t-1] + noise — explicit causal coupling."""
    rng = np.random.default_rng(seed)
    psi_brain = rng.uniform(-np.pi, np.pi, n)
    psi_market = np.empty(n)
    psi_market[0] = rng.uniform(-np.pi, np.pi)
    psi_market[1:] = 0.9 * psi_brain[:-1] + noise * rng.standard_normal(n - 1)
    return psi_brain, psi_market


def _independent_phase_pair(n: int = 1500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, n), rng.uniform(-np.pi, np.pi, n)


# ---------------------------------------------------------------------------
# Output-type integrity
# ---------------------------------------------------------------------------


def test_returns_well_formed_coupling_direction() -> None:
    psi_b, psi_m = _independent_phase_pair()
    result = analyse_coupling(psi_b, psi_m, n_surrogates=30, seed=0)
    assert isinstance(result, CouplingDirection)
    assert result.te_brain_to_market >= 0.0
    assert result.te_market_to_brain >= 0.0
    assert result.net_flow_brain_to_market == pytest.approx(
        result.te_brain_to_market - result.te_market_to_brain
    )
    assert 0.0 < result.p_brain_to_market <= 1.0
    assert 0.0 < result.p_market_to_brain <= 1.0
    assert result.sigma_R >= 0.0
    assert result.phase_R in set(CriticalPhase)
    assert result.n_samples == psi_b.size
    assert result.n_surrogates == 30
    assert result.k == 1
    assert result.n_levels == 2


# ---------------------------------------------------------------------------
# Direction detection
# ---------------------------------------------------------------------------


def test_detects_brain_to_market_causation() -> None:
    """Synthetic ψ_market = f(ψ_brain[t-1]) + noise should fire brain_leads."""
    psi_b, psi_m = _brain_drives_market(n=2000, seed=11)
    result = analyse_coupling(psi_b, psi_m, n_surrogates=200, seed=13)
    assert result.net_flow_brain_to_market > 0.0
    assert result.p_brain_to_market < 0.05
    assert result.brain_leads is True
    assert result.market_leads is False


def test_independent_phases_yield_no_clear_leader() -> None:
    psi_b, psi_m = _independent_phase_pair(n=2000, seed=3)
    result = analyse_coupling(psi_b, psi_m, n_surrogates=200, seed=5)
    assert result.brain_leads is False
    assert result.market_leads is False


def test_summary_string_reports_dominant_direction() -> None:
    psi_b, psi_m = _brain_drives_market(n=2000, seed=17)
    result = analyse_coupling(psi_b, psi_m, n_surrogates=150, seed=19)
    summary = result.summary()
    assert "brain → market" in summary
    assert "sigma_R" in summary or "σ_R" in summary  # noqa: RUF001
    assert "[critical]" in summary or "[subcritical]" in summary or "[supercritical]" in summary


# ---------------------------------------------------------------------------
# Order-parameter handling
# ---------------------------------------------------------------------------


def test_order_parameter_populates_sigma_R() -> None:
    psi_b, psi_m = _independent_phase_pair(n=1000, seed=7)
    rng = np.random.default_rng(7)
    R = np.clip(rng.uniform(0.4, 0.95, size=psi_b.size), 0.0, 1.0)
    result = analyse_coupling(psi_b, psi_m, order_parameter=R, n_surrogates=20, seed=0)
    # σ for |ΔR| with i.i.d. samples is unlikely to land at exactly 1.0.
    assert result.sigma_R != 1.0
    assert result.phase_R in set(CriticalPhase)


def test_no_order_parameter_yields_honest_null() -> None:
    psi_b, psi_m = _independent_phase_pair(n=1000, seed=21)
    result = analyse_coupling(psi_b, psi_m, n_surrogates=20, seed=0)
    assert result.sigma_R == 1.0
    assert result.phase_R is CriticalPhase.CRITICAL


# ---------------------------------------------------------------------------
# Validation contracts
# ---------------------------------------------------------------------------


def test_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share shape"):
        analyse_coupling(np.zeros(100), np.zeros(101))


def test_rejects_non_finite_phases() -> None:
    bad = np.array([0.0, np.nan, 1.0] + [0.5] * 100)
    good = np.zeros_like(bad)
    with pytest.raises(ValueError, match="finite"):
        analyse_coupling(bad, good)


def test_rejects_order_parameter_length_mismatch() -> None:
    psi_b, psi_m = _independent_phase_pair(n=200, seed=0)
    R = np.zeros(psi_b.size + 5)
    with pytest.raises(ValueError, match="order_parameter must match phase length"):
        analyse_coupling(psi_b, psi_m, order_parameter=R)


def test_rejects_non_finite_order_parameter() -> None:
    psi_b, psi_m = _independent_phase_pair(n=200, seed=0)
    R = np.zeros_like(psi_b)
    R[10] = np.inf
    with pytest.raises(ValueError, match="order_parameter must be finite"):
        analyse_coupling(psi_b, psi_m, order_parameter=R)


# ---------------------------------------------------------------------------
# Determinism under shared seed
# ---------------------------------------------------------------------------


def test_deterministic_under_shared_seed() -> None:
    psi_b, psi_m = _brain_drives_market(n=1500, seed=2)
    a = analyse_coupling(psi_b, psi_m, n_surrogates=40, seed=99)
    b = analyse_coupling(psi_b, psi_m, n_surrogates=40, seed=99)
    assert a == b


# ---------------------------------------------------------------------------
# End-to-end with the real coupled-Kuramoto system
# ---------------------------------------------------------------------------


def test_end_to_end_with_coupled_brain_market_run() -> None:
    """Integration smoke: a real CoupledBrainMarketSystem trace flows
    through analyse_coupling without raising, and the verdict is well-formed."""
    sys = CoupledBrainMarketSystem(K=2.0, sigma=0.05, dt=0.01, seed=123)
    # Collect a few hundred steps of (R, ψ_brain, ψ_market).
    R_series, psi_b_series, psi_m_series = [], [], []
    for _ in range(800):
        R, psi_b, psi_m = sys.step()
        R_series.append(R)
        psi_b_series.append(psi_b)
        psi_m_series.append(psi_m)

    result = analyse_coupling(
        psi_b_series,
        psi_m_series,
        order_parameter=R_series,
        n_surrogates=50,
        seed=7,
    )
    assert isinstance(result, CouplingDirection)
    assert result.n_samples == 800
    assert result.te_brain_to_market >= 0.0
    assert result.te_market_to_brain >= 0.0
    # σ on |ΔR| of a coupled-Kuramoto trace should be a finite, classifiable scalar.
    assert np.isfinite(result.sigma_R)
    assert result.phase_R in set(CriticalPhase)
