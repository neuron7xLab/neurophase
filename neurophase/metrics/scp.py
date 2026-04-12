"""Slow Cortical Potential (SCP) extractor — 0.01-0.1 Hz.

SCPs match the timescale of trial-by-trial reward probability
oscillations (~0.03 Hz in ds003458). Classic delta (1-4 Hz) is
30-100x faster than the stimulus and cannot couple directly.

Pipeline:
    1. Bandpass 0.01-0.1 Hz (Butterworth, order 2 for stability)
    2. Z-score normalize
    3. Smooth with Gaussian kernel (σ = smooth_s seconds)

The low filter order (2) is deliberate: at 0.01 Hz with fs=500 Hz,
higher-order filters can become numerically unstable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SCPTrace:
    """Slow Cortical Potential from a single EEG channel.

    Attributes
    ----------
    signal : FloatArray
        Bandpass-filtered SCP signal.
    envelope : FloatArray
        Z-scored, smoothed SCP envelope.
    channel : str
        Channel name.
    band_hz : tuple[float, float]
        Filter band (default (0.01, 0.1)).
    fs : float
        Sampling rate in Hz.
    n_samples : int
        Number of samples.
    """

    signal: FloatArray
    envelope: FloatArray
    channel: str
    band_hz: tuple[float, float]
    fs: float
    n_samples: int


def extract_scp(
    eeg_data: npt.ArrayLike,
    *,
    fs: float = 500.0,
    band: tuple[float, float] = (0.01, 0.1),
    smooth_s: float = 10.0,
    channel_name: str = "FC5",
) -> SCPTrace:
    """Extract Slow Cortical Potential from a 1-D EEG channel.

    Parameters
    ----------
    eeg_data : array_like, shape (n_samples,)
        Single-channel EEG data.
    fs : float
        Sampling rate in Hz.
    band : tuple[float, float]
        SCP band [f_lo, f_hi] in Hz. Default (0.01, 0.1).
    smooth_s : float
        Gaussian smoothing sigma in seconds. Must be >= 1.0.
    channel_name : str
        Channel label.

    Returns
    -------
    SCPTrace
    """
    x = np.asarray(eeg_data, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"eeg_data must be 1-D, got shape {x.shape}")
    if x.size < 1000:
        raise ValueError(f"eeg_data must have >= 1000 samples, got {x.size}")
    if not np.all(np.isfinite(x)):
        bad_idx = int(np.argmax(~np.isfinite(x)))
        raise ValueError(
            f"eeg_data contains non-finite value at sample {bad_idx} "
            f"(channel {channel_name})"
        )
    if smooth_s < 1.0:
        raise ValueError(f"smooth_s must be >= 1.0, got {smooth_s}")

    nyq = fs / 2.0
    low = band[0] / nyq
    high = band[1] / nyq

    # Low order for numerical stability at very low frequencies
    order = 2
    b, a = butter(order, [low, high], btype="bandpass")
    filtered = filtfilt(b, a, x).astype(np.float64)

    # Z-score
    mean_val = float(np.mean(filtered))
    std_val = float(np.std(filtered))
    signal = (filtered - mean_val) / std_val if std_val > 0 else filtered - mean_val

    # Smooth envelope
    sigma_samples = smooth_s * fs
    envelope = gaussian_filter1d(np.abs(signal), sigma=sigma_samples).astype(np.float64)

    # Z-score envelope
    env_mean = float(np.mean(envelope))
    env_std = float(np.std(envelope))
    envelope = (envelope - env_mean) / env_std if env_std > 0 else envelope - env_mean

    return SCPTrace(
        signal=signal.astype(np.float64),
        envelope=envelope,
        channel=channel_name,
        band_hz=band,
        fs=fs,
        n_samples=x.size,
    )
