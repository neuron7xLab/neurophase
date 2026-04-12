"""Tests for analytical PPC prediction — Bessel + calibrated models (PATH 3)."""

from __future__ import annotations

import math

import pytest

from neurophase.benchmarks.neural_phase_generator import (
    generate_neural_phase_trace,
    generate_synthetic_market_phase,
)
from neurophase.benchmarks.ppc_analytical import (
    bessel_ratio_monotone_check,
    calibrated_ppc,
    theoretical_plv,
    theoretical_ppc,
)
from neurophase.metrics.iplv import compute_ppc


class TestTheoreticalPPC:
    def test_k0_theory_zero(self) -> None:
        """k=0 → theoretical PPC = 0."""
        assert theoretical_ppc(0.0, 1.0) == 0.0

    def test_k5_theory_high(self) -> None:
        """k=5, σ=1 → theoretical PPC > 0.70 (Bessel converges slowly)."""
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


class TestCalibratedPPC:
    def test_k0_returns_zero(self) -> None:
        assert calibrated_ppc(0.0) == 0.0

    def test_monotone_with_k(self) -> None:
        """Calibrated PPC increases monotonically."""
        ks = [0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 5.0]
        ppcs = [calibrated_ppc(k) for k in ks]
        for i in range(len(ppcs) - 1):
            assert ppcs[i] <= ppcs[i + 1], (
                f"Not monotone: k={ks[i]}→{ks[i + 1]}, PPC={ppcs[i]}→{ppcs[i + 1]}"
            )

    def test_high_k_near_one_empirical(self) -> None:
        """k=5 (well above Δω≈3.14) → empirical calibrated PPC > 0.95."""
        ppc = calibrated_ppc(5.0, model="empirical")
        assert ppc > 0.95, f"calibrated PPC(k=5, empirical)={ppc} should be > 0.95"

    def test_high_k_ott_antonsen(self) -> None:
        """k=8 (above K_c=2π≈6.28) → OA PPC > 0.
        Exact: R∞² = 1 − (K_c/K)² = 1 − (2π/8)² ≈ 0.383."""
        ppc = calibrated_ppc(8.0, model="ott_antonsen")
        expected = 1.0 - (2 * math.pi / 8.0) ** 2
        assert abs(ppc - expected) < 1e-10, f"OA PPC(k=8)={ppc} should be ≈ {expected:.6f}"

    def test_sub_critical_ott_antonsen_zero(self) -> None:
        """k=5 < K_c=2π → OA PPC = 0 (analytically exact)."""
        ppc = calibrated_ppc(5.0, model="ott_antonsen")
        assert ppc == 0.0, f"OA PPC(k=5)={ppc} should be exactly 0.0"

    def test_sub_critical_small(self) -> None:
        """k=1 (below Δω≈3.14) → calibrated PPC < 0.05."""
        ppc = calibrated_ppc(1.0)
        assert ppc < 0.05, f"calibrated PPC(k=1)={ppc} should be < 0.05"

    def test_matches_measured_at_endpoints(self) -> None:
        """Empirical calibrated prediction matches measured PPC at k=0 and k=5 within 0.05."""
        phi_market = generate_synthetic_market_phase(n_samples=20000, fs=256.0, seed=42)
        for k in [0.0, 5.0]:
            trace = generate_neural_phase_trace(
                phi_market,
                n_samples=20000,
                fs=256.0,
                coupling_k=k,
                seed=42,
            )
            ppc_meas = compute_ppc(trace.phi_neural, trace.phi_market)
            ppc_pred = calibrated_ppc(k, model="empirical")
            delta = abs(ppc_meas - ppc_pred)
            assert delta < 0.05, (
                f"k={k}: measured={ppc_meas:.4f} predicted={ppc_pred:.4f} δ={delta:.4f}"
            )


class TestPipelineMatchesTheory:
    def test_pipeline_trend_matches_theory(self) -> None:
        """Measured PPC increases with k, same direction as theory."""
        phi_market = generate_synthetic_market_phase(n_samples=8192, fs=256.0, seed=42)
        measured_ppc: list[float] = []
        for k in [0.0, 1.0, 3.0, 5.0]:
            trace = generate_neural_phase_trace(
                phi_market,
                n_samples=8192,
                fs=256.0,
                coupling_k=k,
                seed=42,
            )
            measured_ppc.append(compute_ppc(trace.phi_neural, trace.phi_market))

        for i in range(len(measured_ppc) - 1):
            assert measured_ppc[i] <= measured_ppc[i + 1] + 0.05

    def test_null_matches_theory(self) -> None:
        """k=0: both measured and theoretical PPC near zero."""
        phi_market = generate_synthetic_market_phase(n_samples=8192, fs=256.0, seed=42)
        trace = generate_neural_phase_trace(
            phi_market,
            n_samples=8192,
            fs=256.0,
            coupling_k=0.0,
            seed=42,
        )
        measured = compute_ppc(trace.phi_neural, trace.phi_market)
        theory = calibrated_ppc(0.0)
        assert abs(measured - theory) < 0.05
