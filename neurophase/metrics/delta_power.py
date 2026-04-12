"""Delta-band power envelope extractor.

Extracts the instantaneous power envelope of the delta band (1–4 Hz)
from a single EEG channel. The envelope varies at the trial timescale,
matching reward probability oscillations — unlike the carrier phase
which is 4 orders of magnitude faster.

Pipeline:
    1. Bandpass 1–4 Hz (FIR via scipy)
    2. Hilbert → analytic signal
    3. Power = |analytic|²
    4. Smooth with Gaussian kernel (σ = smooth_ms)
    5. Z-score normalize (mean=0, std=1)

Reference: Toma & Miyakoshi (2021) — delta frontal EEG negatively
correlates with trial-by-trial stock prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, hilbert

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DeltaPowerTrace:
    """Delta-band power envelope from a single EEG channel.

    Attributes
    ----------
    envelope : FloatArray
        Z-scored delta-power envelope, shape (n_samples,).
    channel : str
        Channel name (e.g. "FC5").
    band_hz : tuple[float, float]
        Filter band (default (1.0, 4.0)).
    fs : float
        Sampling rate in Hz.
    n_samples : int
        Number of samples.
    """

    envelope: FloatArray
    channel: str
    band_hz: tuple[float, float]
    fs: float
    n_samples: int


def extract_delta_power(
    eeg_data: npt.ArrayLike,
    *,
    fs: float = 500.0,
    band: tuple[float, float] = (1.0, 4.0),
    smooth_ms: float = 500.0,
    channel_name: str = "FC5",
) -> DeltaPowerTrace:
    """Extract delta-band power envelope from a 1-D EEG channel.

    Parameters
    ----------
    eeg_data : array_like, shape (n_samples,)
        Single-channel EEG data in volts.
    fs : float
        Sampling rate in Hz.
    band : tuple[float, float]
        Delta band bounds [f_lo, f_hi] in Hz.
    smooth_ms : float
        Gaussian smoothing σ in milliseconds. Must be ≥ 100.
    channel_name : str
        Channel label for the output.

    Returns
    -------
    DeltaPowerTrace
    """
    x = np.asarray(eeg_data, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"eeg_data must be 1-D, got shape {x.shape}")
    if x.size < 100:
        raise ValueError(f"eeg_data must have ≥ 100 samples, got {x.size}")
    if not np.all(np.isfinite(x)):
        bad_idx = int(np.argmax(~np.isfinite(x)))
        raise ValueError(
            f"eeg_data contains non-finite value at sample {bad_idx} (channel {channel_name})"
        )
    if smooth_ms < 100.0:
        raise ValueError(f"smooth_ms must be ≥ 100, got {smooth_ms}")

    # Bandpass filter (Butterworth, order 4)
    nyq = fs / 2.0
    low = band[0] / nyq
    high = band[1] / nyq
    b, a = butter(4, [low, high], btype="bandpass")
    filtered = filtfilt(b, a, x).astype(np.float64)

    # Hilbert → power envelope
    analytic = hilbert(filtered)
    power = np.abs(analytic) ** 2

    # Gaussian smoothing
    sigma_samples = (smooth_ms / 1000.0) * fs
    smoothed = gaussian_filter1d(power, sigma=sigma_samples).astype(np.float64)

    # Z-score normalize
    mean = float(np.mean(smoothed))
    std = float(np.std(smoothed))
    envelope = (smoothed - mean) / std if std > 0 else smoothed - mean

    return DeltaPowerTrace(
        envelope=envelope.astype(np.float64),
        channel=channel_name,
        band_hz=band,
        fs=fs,
        n_samples=x.size,
    )
