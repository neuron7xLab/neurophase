"""Tests for delta-band power envelope extraction."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.metrics.delta_power import extract_delta_power


class TestDeltaPower:
    def test_output_shape_matches_input(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5000)
        result = extract_delta_power(x, fs=500.0)
        assert result.envelope.shape == x.shape
        assert result.n_samples == x.size

    def test_zscore_applied(self) -> None:
        """DELTA-I1: envelope is z-scored (mean≈0, std≈1)."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10000)
        result = extract_delta_power(x, fs=500.0)
        assert abs(float(np.mean(result.envelope))) < 0.01
        assert abs(float(np.std(result.envelope)) - 1.0) < 0.01

    def test_smooth_reduces_variance(self) -> None:
        """Smoothing reduces high-frequency variance in envelope."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10000)
        result_smooth = extract_delta_power(x, fs=500.0, smooth_ms=500.0)
        result_less = extract_delta_power(x, fs=500.0, smooth_ms=100.0)
        # More smoothing → lower diff variance
        diff_smooth = float(np.var(np.diff(result_smooth.envelope)))
        diff_less = float(np.var(np.diff(result_less.envelope)))
        assert diff_smooth < diff_less

    def test_nan_raises(self) -> None:
        x = np.ones(1000)
        x[500] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            extract_delta_power(x, fs=500.0)

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="100"):
            extract_delta_power(np.ones(50), fs=500.0)

    def test_smooth_ms_minimum(self) -> None:
        with pytest.raises(ValueError, match="100"):
            extract_delta_power(np.ones(1000), fs=500.0, smooth_ms=50.0)

    def test_output_is_finite(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5000)
        result = extract_delta_power(x, fs=500.0)
        assert np.all(np.isfinite(result.envelope))
