"""Market phase extractor — instantaneous phase from price time series.

Pipeline:
    1. Bandpass filter via neurodsp.filt.filter_signal
    2. Hilbert transform → analytic signal z(t)
    3. φ_mkt(t) = angle(z(t))  ∈ [−π, π]

The filter band defaults to [0.1, 2.0] Hz for intraday market cycles.
Configurable for different time-scales (e.g. [0.01, 0.1] Hz for daily).

Contracts:
    - Input must be finite (validated before processing)
    - Output shape == input shape
    - Uses neurodsp.filt.filter_signal (not raw scipy.signal)
    - NaN from filter edge effects are zero-filled (Hilbert-safe)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from neurodsp.filt import filter_signal
from scipy.signal import hilbert

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MarketPhaseResult:
    """Instantaneous phase extracted from a price time series.

    Attributes
    ----------
    phi : FloatArray
        Instantaneous phase in [−π, π].
    amplitude : FloatArray
        Instantaneous amplitude (envelope).
    band_hz : tuple[float, float]
        Filter band used [f_lo, f_hi] in Hz.
    n_samples : int
        Number of samples in the output.
    fs : float
        Sampling rate in Hz.
    """

    phi: FloatArray
    amplitude: FloatArray
    band_hz: tuple[float, float]
    n_samples: int
    fs: float


def extract_market_phase_from_price(
    price: npt.ArrayLike,
    fs: float,
    band_hz: tuple[float, float] = (0.1, 2.0),
    n_cycles: int = 3,
) -> MarketPhaseResult:
    """Extract instantaneous phase from a price time series.

    Parameters
    ----------
    price : array_like, shape (T,)
        Price time series (must be finite, non-constant).
    fs : float
        Sampling rate in Hz.
    band_hz : tuple[float, float]
        Bandpass filter bounds [f_lo, f_hi] in Hz.
    n_cycles : int
        Number of cycles for FIR filter design (passed to neurodsp).

    Returns
    -------
    MarketPhaseResult

    Raises
    ------
    ValueError
        If input is not 1-D, contains non-finite values, or is constant.
    """
    arr = np.asarray(price, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"price must be 1-D, got shape {arr.shape}")
    if arr.size < 4:
        raise ValueError(f"price must have at least 4 samples, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("price must contain only finite values")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if band_hz[0] >= band_hz[1]:
        raise ValueError(f"band_hz[0] must be < band_hz[1], got {band_hz}")
    if band_hz[1] >= fs / 2:
        raise ValueError(f"band_hz[1]={band_hz[1]} must be below Nyquist={fs / 2}")

    # Standardize before filtering for numerical stability
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std == 0.0:
        raise ValueError("price is constant — phase is undefined")
    x_norm = (arr - mean) / std

    # Bandpass via neurodsp (FIR filter)
    filtered = filter_signal(
        x_norm,
        fs,
        pass_type="bandpass",
        f_range=band_hz,
        n_cycles=n_cycles,
    )
    filtered = np.asarray(filtered, dtype=np.float64)

    # neurodsp FIR filters produce NaN at edges — zero-fill them.
    # This is expected behavior for FIR edge artifacts.
    nan_mask = ~np.isfinite(filtered)
    if np.any(nan_mask):
        filtered[nan_mask] = 0.0

    # Hilbert transform → analytic signal
    analytic = hilbert(filtered)
    phi: FloatArray = np.angle(analytic).astype(np.float64)
    amplitude: FloatArray = np.abs(analytic).astype(np.float64)

    return MarketPhaseResult(
        phi=phi,
        amplitude=amplitude,
        band_hz=band_hz,
        n_samples=arr.size,
        fs=fs,
    )
