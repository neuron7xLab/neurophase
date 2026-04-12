"""Cross-correlation between delta power envelope and price returns.

Measures amplitude coupling between frontal delta EEG power and
trial-by-trial reward probability changes. Significance via
phase-randomization surrogates (Theiler et al. 1992).

Lag interpretation:
    lag < -100 ms  → neural anticipates price (predictive)
    lag ≈ 0 ms     → simultaneous coupling
    lag > +100 ms  → neural responds to price (reactive)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from neurophase.validation.surrogates import phase_shuffle

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DeltaPriceXCorrResult:
    """Cross-correlation result between delta power and price returns.

    Attributes
    ----------
    max_xcorr : float
        Peak cross-correlation ∈ [-1, 1].
    lag_samples : int
        Lag at peak in samples.
    lag_ms : float
        Lag at peak in milliseconds.
    p_value : float
        From phase-randomization surrogate test.
    significant : bool
        p < alpha.
    direction : str
        "neural_leads" | "price_leads" | "simultaneous".
    n_surrogates : int
        Number of surrogates used.
    alpha : float
        Significance level.
    """

    max_xcorr: float
    lag_samples: int
    lag_ms: float
    p_value: float
    significant: bool
    direction: str
    n_surrogates: int
    alpha: float


def _normalized_xcorr(
    x: FloatArray,
    y: FloatArray,
    max_lag: int,
) -> tuple[FloatArray, FloatArray]:
    """Normalized cross-correlation for lags in [-max_lag, max_lag].

    Uses FFT-based correlation for O(N log N) instead of O(N * max_lag).
    Results are mathematically identical to the direct method.

    Returns (lags, xcorr_values).
    """
    from scipy.signal import fftconvolve

    n = min(len(x), len(y))
    x_z = x[:n].copy()
    y_z = y[:n].copy()

    # Z-score
    xm, xs = float(np.mean(x_z)), float(np.std(x_z))
    ym, ys = float(np.mean(y_z)), float(np.std(y_z))
    if xs > 0:
        x_z = (x_z - xm) / xs
    if ys > 0:
        y_z = (y_z - ym) / ys

    # FFT-based full cross-correlation: xcorr[k] = sum(x[m] * y[m+k]) / overlap
    full_xcorr = fftconvolve(x_z, y_z[::-1], mode="full")
    # full_xcorr has length 2*n - 1, centered at index n-1 (lag=0)
    center = n - 1

    lags = np.arange(-max_lag, max_lag + 1)
    xcorr = np.empty(len(lags), dtype=np.float64)

    for i, lag in enumerate(lags):
        idx = center + lag
        if 0 <= idx < len(full_xcorr):
            overlap = n - abs(lag)
            xcorr[i] = full_xcorr[idx] / overlap if overlap > 0 else 0.0
        else:
            xcorr[i] = 0.0

    return lags.astype(np.float64), xcorr


def compute_delta_price_xcorr(
    delta_envelope: npt.ArrayLike,
    price_returns: npt.ArrayLike,
    *,
    fs: float = 500.0,
    max_lag_ms: float = 2000.0,
    n_surrogates: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> DeltaPriceXCorrResult:
    """Cross-correlate delta power envelope with price returns.

    Parameters
    ----------
    delta_envelope : array_like, shape (T,)
        Z-scored delta-power envelope.
    price_returns : array_like, shape (T,)
        Z-scored reward probability returns, interpolated to EEG rate.
    fs : float
        Sampling rate.
    max_lag_ms : float
        Maximum lag in milliseconds (both directions).
    n_surrogates : int
        Phase-randomization surrogates.
    alpha : float
        Significance level.
    seed : int
        RNG seed.

    Returns
    -------
    DeltaPriceXCorrResult
    """
    x = np.asarray(delta_envelope, dtype=np.float64)
    y = np.asarray(price_returns, dtype=np.float64)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Both inputs must be 1-D")
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    if not np.all(np.isfinite(x)):
        raise ValueError("delta_envelope contains non-finite values")
    if not np.all(np.isfinite(y)):
        raise ValueError("price_returns contains non-finite values")
    if max_lag_ms > 5000.0:
        raise ValueError(f"max_lag_ms must be ≤ 5000, got {max_lag_ms}")
    if n_surrogates < 200:
        raise ValueError(f"n_surrogates must be ≥ 200, got {n_surrogates}")

    max_lag_samples = int(max_lag_ms / 1000.0 * fs)

    # Observed cross-correlation
    lags, xcorr = _normalized_xcorr(x, y, max_lag_samples)
    abs_xcorr = np.abs(xcorr)
    peak_idx = int(np.argmax(abs_xcorr))
    observed_max = float(xcorr[peak_idx])
    observed_abs = float(abs_xcorr[peak_idx])
    peak_lag = int(lags[peak_idx])
    peak_lag_ms = float(peak_lag) / fs * 1000.0

    # Surrogate test: phase-randomize delta envelope
    rng = np.random.default_rng(seed)
    surr_max = np.empty(n_surrogates, dtype=np.float64)
    for i in range(n_surrogates):
        x_surr = phase_shuffle(x, rng=rng)
        _, xcorr_surr = _normalized_xcorr(x_surr, y, max_lag_samples)
        surr_max[i] = float(np.max(np.abs(xcorr_surr)))

    # p-value: fraction of surrogates with |xcorr| ≥ observed
    p_value = float((1 + np.sum(surr_max >= observed_abs)) / (1 + n_surrogates))

    # Direction
    if abs(peak_lag_ms) < 100.0:
        direction = "simultaneous"
    elif peak_lag_ms < 0:
        direction = "neural_leads"
    else:
        direction = "price_leads"

    return DeltaPriceXCorrResult(
        max_xcorr=observed_max,
        lag_samples=peak_lag,
        lag_ms=peak_lag_ms,
        p_value=p_value,
        significant=p_value < alpha,
        direction=direction,
        n_surrogates=n_surrogates,
        alpha=alpha,
    )
