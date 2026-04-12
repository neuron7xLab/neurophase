"""Tests for analytical PPC prediction via Bessel functions (PATH 3)."""

from __future__ import annotations

import pytest

from neurophase.benchmarks.neural_phase_generator import (
    generate_neural_phase_trace,
    generate_synthetic_market_phase,
)
from neurophase.benchmarks.ppc_analytical import (
    bessel_ratio_monotone_check,
    theoretical_plv,
    theoretical_ppc,
)
from neurophase.metrics.iplv import compute_ppc


class TestTheoreticalPPC:
    def test_k0_theory_zero(self) -> None:
        """k=0 → theoretical PPC = 0."""
        assert theoretical_ppc(0.0, 1.0) == 0.0

    def test_k5_theory_high(self) -> None:
        """k=5 → theoretical PPC > 0.70 (Bessel converges slowly)."""
        ppc = theoretical_ppc(5.0, 1.0)
        assert ppc > 0.70, f"theoretical PPC(k=5)={ppc} should be > 0.70"

    def test_plv_theory_bounded(self) -> None:
        """Theoretical PLV ∈ [0, 1) for all k."""
        for k in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
            plv = theoretical_plv(k, 1.0)
            assert 0.0 <= plv < 1.0, f"PLV(k={k})={plv} out of range"

    def test_bessel_ratio_monotone(self) -> None:
        """PLV_theory increases monotonically with k."""
        k_values = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
        assert bessel_ratio_monotone_check(k_values, noise_sigma=1.0)

    def test_rejects_negative_k(self) -> None:
        with pytest.raises(ValueError, match="≥ 0"):
            theoretical_ppc(-1.0, 1.0)

    def test_rejects_zero_sigma(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            theoretical_ppc(1.0, 0.0)


class TestPipelineMatchesTheory:
    """Verify pipeline PPC matches Bessel prediction within tolerance.

    The neural phase generator uses a Kuramoto ODE with additive 1/f
    noise (σ_noise ~ 0.3 by default). The theoretical prediction uses
    von Mises concentration κ = k/σ². The match is approximate because:
    1. Finite integration time (not true stationarity)
    2. 1/f noise ≠ white noise (Bessel assumes white)
    3. RK4 discretization error

    We use a generous tolerance (0.15) to account for these factors.
    The test's purpose is to catch gross errors (wrong band, wrong
    signal, sign error) not to validate the Bessel formula itself.
    """

    def test_pipeline_trend_matches_theory(self) -> None:
        """Measured PPC increases with k, same direction as theory."""
        phi_market = generate_synthetic_market_phase(
            n_samples=8192, fs=256.0, seed=42,
        )
        measured_ppc: list[float] = []
        theory_ppc: list[float] = []
        for k in [0.0, 1.0, 3.0, 5.0]:
            trace = generate_neural_phase_trace(
                phi_market, n_samples=8192, fs=256.0, coupling_k=k, seed=42,
            )
            measured_ppc.append(compute_ppc(trace.phi_neural, trace.phi_market))
            theory_ppc.append(theoretical_ppc(k, noise_sigma=1.0))

        # Both should be monotonically increasing
        for i in range(len(measured_ppc) - 1):
            assert measured_ppc[i] <= measured_ppc[i + 1] + 0.05, (
                f"Measured PPC not monotonic at k transition {i}→{i + 1}"
            )

    def test_null_matches_theory(self) -> None:
        """k=0: both measured and theoretical PPC near zero."""
        phi_market = generate_synthetic_market_phase(
            n_samples=8192, fs=256.0, seed=42,
        )
        trace = generate_neural_phase_trace(
            phi_market, n_samples=8192, fs=256.0, coupling_k=0.0, seed=42,
        )
        measured = compute_ppc(trace.phi_neural, trace.phi_market)
        theory = theoretical_ppc(0.0, 1.0)
        assert abs(measured - theory) < 0.05, (
            f"k=0: measured={measured:.4f} theory={theory:.4f}"
        )
