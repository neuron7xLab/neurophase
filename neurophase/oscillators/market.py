"""Market-side oscillators → instantaneous phase.

Converts raw OHLCV data into a compact set of phase oscillators feeding
the joint Kuramoto network:

    price  → φ_price
    volume → φ_volume (log-volume to handle decadal scale)
    sigma  → φ_vol    (rolling realized volatility)

The signals are passed through the same Hilbert + wavelet pipeline used
for neural signals (``neurophase.core.phase.compute_phase``), so that
market and neural phases are directly comparable inside the order
parameter and PLV estimator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from neurophase.core.phase import compute_phase

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MarketOscillators:
    """Multi-channel market phase bundle.

    Attributes
    ----------
    phi_price : FloatArray, shape (T,)
        Instantaneous phase of the price signal.
    phi_volume : FloatArray, shape (T,)
        Instantaneous phase of log-volume.
    phi_volatility : FloatArray, shape (T,)
        Instantaneous phase of rolling realized volatility.
    """

    phi_price: FloatArray
    phi_volume: FloatArray
    phi_volatility: FloatArray

    def stack(self) -> FloatArray:
        """Stack channels into a (3, T) array for Kuramoto analysis."""
        return np.stack([self.phi_price, self.phi_volume, self.phi_volatility], axis=0).astype(
            np.float64
        )


def _rolling_volatility(
    prices: FloatArray,
    window: int,
) -> FloatArray:
    """Square-root of the mean squared log return over a trailing window."""
    returns = np.diff(np.log(np.clip(prices, 1e-12, None)))
    sq = returns**2
    if window <= 1:
        out_short: FloatArray = np.sqrt(sq).astype(np.float64)
        return out_short
    kernel = np.ones(window, dtype=np.float64) / float(window)
    rolling = np.convolve(sq, kernel, mode="same")
    # Re-align: prepend one zero to match original length.
    rolling_full = np.concatenate([[rolling[0]], rolling])
    out_long: FloatArray = np.sqrt(np.maximum(rolling_full, 0.0)).astype(np.float64)
    return out_long


def extract_market_phase(
    prices: npt.ArrayLike,
    volumes: npt.ArrayLike,
    volatility_window: int = 20,
    denoise: bool = True,
) -> MarketOscillators:
    """Extract a three-channel phase bundle from price and volume.

    Parameters
    ----------
    prices : array_like, shape (T,)
        Bar-close prices. Must be strictly positive.
    volumes : array_like, shape (T,)
        Bar volumes. Must be non-negative.
    volatility_window : int
        Window for the rolling realized volatility. Must be >= 1.
    denoise : bool
        Whether to apply wavelet denoising inside ``compute_phase``.

    Returns
    -------
    MarketOscillators

    Raises
    ------
    ValueError
        For shape mismatches, non-positive prices, or bad parameters.
    """
    p_arr = np.asarray(prices, dtype=np.float64)
    v_arr = np.asarray(volumes, dtype=np.float64)
    if p_arr.ndim != 1 or v_arr.ndim != 1:
        raise ValueError(f"prices and volumes must be 1-D; got {p_arr.shape} / {v_arr.shape}")
    if p_arr.shape != v_arr.shape:
        raise ValueError(
            f"prices and volumes must have the same shape, got {p_arr.shape} vs {v_arr.shape}"
        )
    if p_arr.size < 8:
        raise ValueError(f"need at least 8 samples, got {p_arr.size}")
    if np.any(p_arr <= 0):
        raise ValueError("prices must be strictly positive")
    if np.any(v_arr < 0):
        raise ValueError("volumes must be non-negative")
    if volatility_window < 1:
        raise ValueError(f"volatility_window must be >= 1, got {volatility_window}")

    log_volume = np.log1p(v_arr)
    realized_vol = _rolling_volatility(p_arr, window=volatility_window)
    phi_price = compute_phase(p_arr, denoise=denoise)
    phi_volume = compute_phase(log_volume, denoise=denoise)
    phi_vol = compute_phase(realized_vol, denoise=denoise)
    return MarketOscillators(
        phi_price=phi_price,
        phi_volume=phi_volume,
        phi_volatility=phi_vol,
    )
