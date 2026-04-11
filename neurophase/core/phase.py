"""Phase extraction from raw time series.

Given a real-valued signal x(t) — price, volume, EEG, HRV, pupil — produce
the instantaneous phase φ(t) = arg(H(x_filt(t))), where H is the Hilbert
transform and x_filt is x after wavelet denoising.

The pipeline:
    1. Standardize: x' = (x − μ) / σ
    2. Denoise via Daubechies D4 wavelet (soft-threshold detail coefficients)
    3. Analytic signal via Hilbert transform
    4. Instantaneous phase = angle of analytic signal

Also provides adaptive_threshold(R, window, k) for dynamic gating:

    R_threshold(t) = mean(R[-window:]) + k · std(R[-window:])

which replaces a static θ with a context-aware threshold robust to
regime shifts.

Ported and type-annotated from the π-system reference (2025).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pywt
from scipy.signal import hilbert

FloatArray = npt.NDArray[np.float64]

_DEFAULT_WAVELET = "db4"


def preprocess_signal(
    x: npt.ArrayLike,
    wavelet: str = _DEFAULT_WAVELET,
    level: int | None = None,
) -> FloatArray:
    """Denoise a time series with a multilevel wavelet transform.

    Parameters
    ----------
    x : array_like, shape (T,)
        Raw signal.
    wavelet : str
        Wavelet family name (default ``"db4"`` — Daubechies 4).
    level : int | None
        Decomposition level. If None, chosen as ``min(4, pywt.dwt_max_level)``.

    Returns
    -------
    FloatArray
        Denoised signal of the same length as input.

    Raises
    ------
    ValueError
        If the input is not 1-D or has fewer than 2 samples.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x_arr.shape}")
    if x_arr.size < 2:
        raise ValueError(f"x must have at least 2 samples, got {x_arr.size}")

    max_level = pywt.dwt_max_level(x_arr.size, pywt.Wavelet(wavelet).dec_len)
    eff_level = min(4, max_level) if level is None else min(level, max_level)
    if eff_level <= 0:
        return x_arr.copy()

    coeffs = pywt.wavedec(x_arr, wavelet, level=eff_level)
    # Universal threshold (Donoho–Johnstone) using the finest detail level.
    detail = np.asarray(coeffs[-1], dtype=np.float64)
    sigma = float(np.median(np.abs(detail))) / 0.6745 if detail.size else 0.0
    threshold = sigma * np.sqrt(2.0 * np.log(x_arr.size)) if sigma > 0 else 0.0
    denoised_coeffs: list[Any] = [coeffs[0]]
    for detail_coeffs in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail_coeffs, threshold, mode="soft"))
    reconstructed = pywt.waverec(denoised_coeffs, wavelet)
    result: FloatArray = np.asarray(reconstructed, dtype=np.float64)[: x_arr.size]
    return result


def compute_phase(
    x: npt.ArrayLike,
    wavelet: str = _DEFAULT_WAVELET,
    denoise: bool = True,
) -> FloatArray:
    """Compute instantaneous phase φ(t) via analytic signal.

    The signal is standardized (zero mean, unit variance), optionally
    denoised with a wavelet transform, then passed through the Hilbert
    transform. The phase is ``angle`` of the resulting analytic signal.

    Parameters
    ----------
    x : array_like, shape (T,)
        Raw real-valued signal (price, volume, EEG, pupil…).
    wavelet : str
        Wavelet for denoising. Ignored when ``denoise=False``.
    denoise : bool
        Whether to apply wavelet denoising before the Hilbert transform.

    Returns
    -------
    FloatArray, shape (T,)
        Instantaneous phase in radians, in (-π, π].

    Raises
    ------
    ValueError
        If the input is not 1-D or is too short.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x_arr.shape}")
    if x_arr.size < 4:
        raise ValueError(f"x must have at least 4 samples, got {x_arr.size}")

    mean = float(np.mean(x_arr))
    std = float(np.std(x_arr))
    if std == 0.0:
        # Constant signal: phase is undefined. Return zeros — an honest null.
        return np.zeros_like(x_arr)

    x_norm = (x_arr - mean) / std
    x_clean = preprocess_signal(x_norm, wavelet=wavelet) if denoise else x_norm
    analytic = hilbert(x_clean)
    phase: FloatArray = np.angle(analytic).astype(np.float64)
    return phase


def adaptive_threshold(
    R: npt.ArrayLike,
    window: int = 100,
    k: float = 1.5,
) -> float:
    """Dynamic threshold for the order parameter.

    R_threshold(t) = mean(R[-window:]) + k · std(R[-window:])

    Lets the gate adapt to the current coupling regime instead of using
    a static cutoff. When ``len(R) < window``, uses all available samples.

    Parameters
    ----------
    R : array_like
        Recent history of order-parameter values in [0, 1].
    window : int
        Size of the trailing window. Must be positive.
    k : float
        Number of standard deviations above the rolling mean.

    Returns
    -------
    float
        Threshold clipped to [0, 1].

    Raises
    ------
    ValueError
        If ``window`` is not positive or ``R`` is empty.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    R_arr = np.asarray(R, dtype=np.float64)
    if R_arr.size == 0:
        raise ValueError("R must be non-empty")
    tail = R_arr[-window:] if R_arr.size > window else R_arr
    threshold = float(np.mean(tail) + k * np.std(tail))
    return float(np.clip(threshold, 0.0, 1.0))
