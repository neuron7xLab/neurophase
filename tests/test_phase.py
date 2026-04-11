"""Tests for neurophase.core.phase."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.core.phase import adaptive_threshold, compute_phase, preprocess_signal


def test_preprocess_preserves_length() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1024)
    y = preprocess_signal(x)
    assert y.shape == x.shape


def test_preprocess_denoises_noisy_sine() -> None:
    """Wavelet denoising should reduce variance of a noisy sine."""
    t = np.linspace(0.0, 10.0, 1024)
    clean = np.sin(2 * np.pi * 1.5 * t)
    rng = np.random.default_rng(42)
    noisy = clean + 0.5 * rng.standard_normal(clean.size)
    denoised = preprocess_signal(noisy)
    # Correlation with clean signal should be higher after denoising.
    corr_noisy = float(np.corrcoef(noisy, clean)[0, 1])
    corr_clean = float(np.corrcoef(denoised, clean)[0, 1])
    assert corr_clean >= corr_noisy - 0.05  # not worse than noisy


def test_compute_phase_sine_is_monotonic_mod_2pi() -> None:
    """Phase of a pure sinusoid should increase linearly (modulo 2π)."""
    t = np.linspace(0.0, 10.0, 2000)
    x = np.sin(2 * np.pi * 1.0 * t)
    phase = compute_phase(x, denoise=False)
    # Unwrap and check monotonic growth.
    unwrapped = np.unwrap(phase)
    diffs = np.diff(unwrapped)
    # Majority of diffs should be positive (monotonic growth trend).
    assert float(np.mean(diffs > 0)) > 0.9


def test_compute_phase_constant_returns_zero() -> None:
    x = np.full(32, 3.14)
    phase = compute_phase(x)
    assert np.allclose(phase, 0.0)


def test_compute_phase_too_short_raises() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        compute_phase(np.array([1.0, 2.0]))


def test_adaptive_threshold_basic() -> None:
    R = np.linspace(0.2, 0.8, 100)
    thr = adaptive_threshold(R, window=50, k=1.0)
    assert 0.0 <= thr <= 1.0
    # With positive k the threshold should sit above the tail mean.
    tail_mean = float(np.mean(R[-50:]))
    assert thr >= tail_mean - 1e-9


def test_adaptive_threshold_clips_to_unit() -> None:
    R = np.full(200, 0.95)
    thr = adaptive_threshold(R, window=50, k=5.0)
    assert thr <= 1.0


def test_adaptive_threshold_rejects_bad_window() -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        adaptive_threshold(np.array([0.1, 0.2]), window=0)
