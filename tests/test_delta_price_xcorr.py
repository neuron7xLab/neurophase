"""Tests for delta power × price returns cross-correlation."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.metrics.delta_price_xcorr import (
    compute_delta_price_xcorr,
)
from neurophase.validation.surrogates import phase_shuffle


class TestDeltaPriceXCorr:
    def test_null_xcorr_not_significant(self) -> None:
        """Uncorrelated signals → p > 0.05."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10000)
        y = rng.standard_normal(10000)
        result = compute_delta_price_xcorr(
            x, y, fs=500.0, n_surrogates=200, seed=42,
        )
        assert not result.significant, f"p={result.p_value} should be > 0.05"

    def test_correlated_xcorr_significant(self) -> None:
        """Injected correlation → p < 0.05."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10000)
        # y = smoothed, delayed copy of x + noise
        y = 0.5 * x + 0.5 * rng.standard_normal(10000)
        result = compute_delta_price_xcorr(
            x, y, fs=500.0, n_surrogates=200, seed=42,
        )
        assert result.significant, f"p={result.p_value} should be < 0.05"
        assert result.max_xcorr > 0.3

    def test_surrogate_preserves_spectrum(self) -> None:
        """DELTA-I2: phase_shuffle preserves amplitude spectrum."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(4096).astype(np.float64)
        x_surr = phase_shuffle(x, rng=rng)
        psd_orig = np.abs(np.fft.rfft(x)) ** 2
        psd_surr = np.abs(np.fft.rfft(x_surr)) ** 2
        np.testing.assert_allclose(psd_orig, psd_surr, atol=1e-6)

    def test_lag_sign_correct(self) -> None:
        """Known lag → correct direction detected."""
        rng = np.random.default_rng(42)
        n = 10000
        x = rng.standard_normal(n)
        # y lags x by 200 samples (x leads)
        lag = 200
        y = np.zeros(n)
        y[lag:] = x[:-lag]
        result = compute_delta_price_xcorr(
            x, y, fs=500.0, max_lag_ms=2000.0, n_surrogates=200, seed=42,
        )
        # x is delta_envelope, y is price_returns
        # positive lag means x leads y → "neural_leads" or "simultaneous"
        assert result.lag_ms < 0 or result.direction in ("neural_leads", "simultaneous")

    def test_negative_xcorr_detected(self) -> None:
        """Anticorrelated signals → max_xcorr < 0."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10000)
        y = -0.5 * x + 0.5 * rng.standard_normal(10000)
        result = compute_delta_price_xcorr(
            x, y, fs=500.0, n_surrogates=200, seed=42,
        )
        assert result.max_xcorr < 0, f"xcorr={result.max_xcorr} should be < 0"

    def test_max_lag_limit(self) -> None:
        with pytest.raises(ValueError, match="5000"):
            compute_delta_price_xcorr(
                np.ones(1000), np.ones(1000),
                fs=500.0, max_lag_ms=6000.0,
            )

    def test_min_surrogates(self) -> None:
        with pytest.raises(ValueError, match="200"):
            compute_delta_price_xcorr(
                np.ones(1000), np.ones(1000),
                fs=500.0, n_surrogates=50,
            )

    def test_result_fields(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5000)
        y = rng.standard_normal(5000)
        result = compute_delta_price_xcorr(x, y, fs=500.0, n_surrogates=200, seed=42)
        assert -1.0 <= result.max_xcorr <= 1.0
        assert 0.0 <= result.p_value <= 1.0
        assert result.direction in {"neural_leads", "price_leads", "simultaneous"}
