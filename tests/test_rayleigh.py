"""Tests for the Rayleigh circular uniformity test (PATH 1)."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.metrics.rayleigh import rayleigh_test


class TestRayleigh:
    def test_uniform_phases_not_significant(self) -> None:
        """Uniform Δφ ∈ [−π, π], N=10000 → p > 0.05."""
        rng = np.random.default_rng(42)
        delta_phi = rng.uniform(-np.pi, np.pi, 10000)
        result = rayleigh_test(delta_phi)
        assert result.p_value > 0.05, f"p={result.p_value} should be > 0.05 for uniform"
        assert not result.significant

    def test_locked_phases_significant(self) -> None:
        """Δφ ≈ constant + small noise → p < 0.001."""
        rng = np.random.default_rng(42)
        delta_phi = 0.5 + rng.normal(0, 0.1, 1000)
        result = rayleigh_test(delta_phi)
        assert result.p_value < 0.001, f"p={result.p_value} should be < 0.001 for locked"
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
        """Verify second-order correction term is present (not just exp(-Z))."""
        rng = np.random.default_rng(42)
        delta_phi = rng.uniform(-np.pi, np.pi, 50)
        result = rayleigh_test(delta_phi)
        # Naive p = exp(-Z). With correction, p ≠ exp(-Z) exactly.
        naive_p = float(np.exp(-result.Z))
        # They should differ (correction terms are non-zero)
        assert result.p_value != pytest.approx(naive_p, abs=1e-15), (
            "p-value equals naive exp(-Z) — correction not applied"
        )

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
