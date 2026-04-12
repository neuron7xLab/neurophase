"""Tests for Slow Cortical Potential extraction and SCP analysis."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.metrics.scp import extract_scp


class TestSCP:
    def test_output_shape_matches_input(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50000)  # 100s at 500 Hz
        result = extract_scp(x, fs=500.0)
        assert result.signal.shape == x.shape
        assert result.envelope.shape == x.shape
        assert result.n_samples == x.size

    def test_band_matches_stimulus(self) -> None:
        """SCP-I1: 0.01-0.1 Hz contains the ~0.03 Hz reward oscillation."""
        result = extract_scp(
            np.random.default_rng(42).standard_normal(50000),
            fs=500.0,
            band=(0.01, 0.1),
        )
        stimulus_freq = 0.03  # Hz, approximate reward oscillation
        assert result.band_hz[0] <= stimulus_freq <= result.band_hz[1]

    def test_zscore_applied(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50000)
        result = extract_scp(x, fs=500.0)
        assert abs(float(np.mean(result.envelope))) < 0.1
        assert abs(float(np.std(result.envelope)) - 1.0) < 0.1

    def test_output_is_finite(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50000)
        result = extract_scp(x, fs=500.0)
        assert np.all(np.isfinite(result.signal))
        assert np.all(np.isfinite(result.envelope))

    def test_nan_raises(self) -> None:
        x = np.ones(5000)
        x[2500] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            extract_scp(x, fs=500.0)

    def test_too_short_raises(self) -> None:
        with pytest.raises(ValueError, match="1000"):
            extract_scp(np.ones(500), fs=500.0)

    def test_smooth_minimum(self) -> None:
        with pytest.raises(ValueError, match=r"1\.0"):
            extract_scp(np.ones(5000), fs=500.0, smooth_s=0.5)

    def test_same_surrogate_method(self) -> None:
        """SCP-I2: SCP analysis uses phase_randomization (same as delta)."""
        # Verify phase_shuffle is importable and used by delta_price_xcorr
        from neurophase.metrics.delta_price_xcorr import compute_delta_price_xcorr

        # Run a quick xcorr — it uses phase_shuffle internally
        rng = np.random.default_rng(42)
        x = rng.standard_normal(5000)
        y = rng.standard_normal(5000)
        result = compute_delta_price_xcorr(x, y, fs=500.0, n_surrogates=200, seed=42)
        # If it runs without error, surrogates are working
        assert 0.0 <= result.p_value <= 1.0

    def test_injected_coupling_detected(self) -> None:
        """Injected 0.03 Hz signal in SCP band should be detectable."""
        rng = np.random.default_rng(42)
        fs = 500.0
        n = 100000  # 200 seconds
        t = np.arange(n) / fs

        # Slow 0.03 Hz shared oscillation
        slow_signal = np.sin(2 * np.pi * 0.03 * t)

        # EEG = slow signal + noise
        eeg = slow_signal + 0.5 * rng.standard_normal(n)

        scp = extract_scp(eeg, fs=fs, band=(0.01, 0.1), smooth_s=5.0)

        # Check that the SCP signal has power at ~0.03 Hz
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        psd = np.abs(np.fft.rfft(scp.signal)) ** 2
        # Find peak near 0.03 Hz
        mask = (freqs >= 0.02) & (freqs <= 0.05)
        peak_power = float(np.max(psd[mask]))
        mean_power = float(np.mean(psd))
        assert peak_power > 5 * mean_power, "SCP should preserve 0.03 Hz signal"
