"""Tests for neurophase.metrics.hurst."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.metrics.hurst import hurst_dfa, hurst_rs


def _fractional_brownian_motion(n: int, H: float, seed: int) -> np.ndarray:
    """Generate an fBm path via the Davies–Harte circulant-embedding method.

    Small-n implementation sufficient for regression tests.
    """
    rng = np.random.default_rng(seed)
    # Covariance of fractional Gaussian noise.
    k = np.arange(n)
    r = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H))
    # Build circulant row and eigenvalues via FFT.
    row = np.concatenate([r, r[-2:0:-1]])
    eigenvalues = np.fft.fft(row).real
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    # Two independent Gaussians to form a complex vector.
    z = rng.standard_normal(row.size) + 1j * rng.standard_normal(row.size)
    w = np.sqrt(eigenvalues / row.size) * z
    fgn = np.fft.fft(w).real[:n]
    return np.cumsum(fgn).astype(np.float64)


def test_hurst_rs_white_noise_close_to_half() -> None:
    """R/S of stationary white-noise increments → H ≈ 0.5.

    R/S is the classic rescaled-range estimator applied to the series as-is;
    for i.i.d. zero-mean increments this recovers H ≈ 0.5 (Mandelbrot).
    """
    rng = np.random.default_rng(10)
    ts = rng.standard_normal(4096).astype(np.float64)
    H_est = hurst_rs(ts, min_lag=8, max_lag=200)
    assert 0.35 <= H_est <= 0.70


def test_hurst_rs_persistent_series() -> None:
    """Persistent fBm (target H = 0.75) → estimate > 0.5."""
    ts = _fractional_brownian_motion(2048, H=0.75, seed=11)
    H_est = hurst_rs(ts, min_lag=8, max_lag=200)
    assert H_est > 0.55


def test_hurst_dfa_random_walk_close_to_half() -> None:
    """Random walk has DFA exponent ≈ 1.5 (integrated white noise).

    Since DFA operates on the signal as-is and cumulates internally, a
    pre-cumulated random walk is equivalent to a smooth integrated signal.
    The test uses white noise directly so that DFA's internal cumsum
    recovers H ≈ 0.5.
    """
    rng = np.random.default_rng(12)
    ts = rng.standard_normal(4096)
    H_est = hurst_dfa(ts, min_lag=8, max_lag=200)
    assert 0.35 <= H_est <= 0.70


def test_hurst_rs_requires_enough_samples() -> None:
    with pytest.raises(ValueError, match="at least"):
        hurst_rs(np.zeros(50), min_lag=8, max_lag=100)


def test_hurst_dfa_requires_enough_samples() -> None:
    with pytest.raises(ValueError, match="at least"):
        hurst_dfa(np.zeros(50), min_lag=8, max_lag=100)


def test_hurst_rs_rejects_bad_lag_range() -> None:
    with pytest.raises(ValueError, match="max_lag must be"):
        hurst_rs(np.arange(1000, dtype=np.float64), min_lag=50, max_lag=40)


def test_hurst_dfa_rejects_min_lag_too_small() -> None:
    with pytest.raises(ValueError, match="min_lag must be >= 4"):
        hurst_dfa(np.arange(1000, dtype=np.float64), min_lag=2, max_lag=100)
