"""Tests for the Rayleigh circular uniformity test (PATH 1)."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.metrics.rayleigh import R_MEDIUM, R_NEGLIGIBLE, rayleigh_test


class TestRayleigh:
    def test_uniform_phases_not_significant(self) -> None:
        """Uniform Δφ ∈ [−π, π], N=10000 → R < 0.05, effect negligible."""
        rng = np.random.default_rng(42)
        delta_phi = rng.uniform(-np.pi, np.pi, 10000)
        result = rayleigh_test(delta_phi)
        assert result.R < R_NEGLIGIBLE, f"R={result.R} should be < {R_NEGLIGIBLE}"
        assert result.effect_size == "negligible"
        assert not result.significant

    def test_locked_phases_significant(self) -> None:
        """Δφ ≈ constant + small noise → R large, significant."""
        rng = np.random.default_rng(42)
        delta_phi = 0.5 + rng.normal(0, 0.1, 1000)
        result = rayleigh_test(delta_phi)
        assert result.R > R_MEDIUM, f"R={result.R} should be > {R_MEDIUM}"
        assert result.effect_size == "large"
        assert result.significant

    def test_r_in_range(self) -> None:
        """R ∈ [0, 1] always."""
        rng = np.random.default_rng(42)
        delta_phi = rng.uniform(-np.pi, np.pi, 500)
        result = rayleigh_test(delta_phi)
        assert 0.0 <= result.R <= 1.0

    def test_z_equals_n_r_squared(self) -> None:
        """Z = N · R²."""
        rng = np.random.default_rng(42)
        delta_phi = rng.uniform(-np.pi, np.pi, 500)
        result = rayleigh_test(delta_phi)
        expected_z = result.n_samples * result.R**2
        np.testing.assert_allclose(result.Z, expected_z, atol=1e-10)

    def test_finite_sample_correction_applied(self) -> None:
        """Verify second-order correction term is present."""
        rng = np.random.default_rng(42)
        delta_phi = rng.uniform(-np.pi, np.pi, 50)
        result = rayleigh_test(delta_phi)
        naive_p = float(np.exp(-result.Z))
        assert result.p_value != pytest.approx(naive_p, abs=1e-15)

    def test_rejects_too_few_samples(self) -> None:
        with pytest.raises(ValueError, match="≥ 10"):
            rayleigh_test(np.array([0.1, 0.2, 0.3]))

    def test_rejects_non_finite(self) -> None:
        arr = np.ones(20)
        arr[5] = np.nan
        with pytest.raises(ValueError, match="finite"):
            rayleigh_test(arr)

    def test_p_value_in_range(self) -> None:
        """p ∈ [0, 1] always."""
        for s in range(10):
            rng = np.random.default_rng(s)
            delta_phi = rng.uniform(-np.pi, np.pi, 100)
            result = rayleigh_test(delta_phi)
            assert 0.0 <= result.p_value <= 1.0

    def test_r_stable_across_n(self) -> None:
        """k=0: R stays < 0.05 regardless of N (N-independent metric)."""
        from neurophase.benchmarks.neural_phase_generator import (
            generate_neural_phase_trace,
            generate_synthetic_market_phase,
        )

        phi_market = generate_synthetic_market_phase(n_samples=50000, fs=256.0, seed=42)
        for n in [1000, 5000, 10000, 50000]:
            trace = generate_neural_phase_trace(
                phi_market,
                n_samples=n,
                fs=256.0,
                coupling_k=0.0,
                seed=42,
            )
            delta_phi = trace.phi_neural - trace.phi_market
            result = rayleigh_test(delta_phi)
            assert result.R < R_NEGLIGIBLE, f"N={n}: R={result.R} should be < {R_NEGLIGIBLE}"

    def test_effect_size_classification(self) -> None:
        """Effect size labels match R thresholds."""
        from neurophase.metrics.rayleigh import _classify_effect_size

        assert _classify_effect_size(0.03) == "negligible"
        assert _classify_effect_size(0.07) == "small"
        assert _classify_effect_size(0.20) == "medium"
        assert _classify_effect_size(0.40) == "large"
