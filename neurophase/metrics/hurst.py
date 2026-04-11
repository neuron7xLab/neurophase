"""Hurst exponent via R/S analysis and Detrended Fluctuation Analysis.

Two independent estimators of long-range dependence in a time series:

    R/S analysis:
        for lags n, estimate ⟨R/S(n)⟩ ~ n^H
    Detrended Fluctuation Analysis (DFA):
        for window sizes n, estimate F(n) = sqrt(mean((y_k - trend_k)^2)) ~ n^H

Interpretation:
    H = 0.5  → random walk (uncorrelated increments)
    H > 0.5  → persistent trend (long memory)
    H < 0.5  → mean-reverting / anti-persistent

Both estimators perform a Huber-regression fit in log–log space to reduce
sensitivity to outliers at short lags.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import HuberRegressor

FloatArray = npt.NDArray[np.float64]


def _huber_slope(log_x: FloatArray, log_y: FloatArray) -> float:
    """Robust slope of a log–log regression using HuberRegressor.

    Falls back to ordinary least-squares if Huber fails to converge.
    """
    X = log_x.reshape(-1, 1)
    try:
        model = HuberRegressor(max_iter=200)
        model.fit(X, log_y)
        return float(model.coef_[0])
    except (ValueError, RuntimeError):
        # Fallback: OLS via polyfit.
        slope, _ = np.polyfit(log_x, log_y, 1)
        return float(slope)


def hurst_rs(ts: npt.ArrayLike, min_lag: int = 8, max_lag: int = 100) -> float:
    """Hurst exponent via classic R/S analysis.

    For each lag n in [min_lag, max_lag], splits the series into
    non-overlapping segments of length n, computes mean-centred
    cumulative deviations, and takes the rescaled-range statistic R/S.

    Parameters
    ----------
    ts : array_like
        Time series (must be 1-D).
    min_lag, max_lag : int
        Range of lags. ``max_lag`` must be < len(ts) // 2 for the statistic
        to be meaningful.

    Returns
    -------
    float
        Hurst exponent estimate.

    Raises
    ------
    ValueError
        For malformed inputs or too-short series.
    """
    arr = np.asarray(ts, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"ts must be 1-D, got shape {arr.shape}")
    if min_lag < 2:
        raise ValueError(f"min_lag must be >= 2, got {min_lag}")
    if max_lag <= min_lag:
        raise ValueError(f"max_lag must be > min_lag, got max_lag={max_lag}")
    if arr.size < 2 * max_lag:
        raise ValueError(f"need at least 2·max_lag={2 * max_lag} samples, got {arr.size}")

    lags = list(range(min_lag, max_lag + 1))
    rs_values: list[float] = []
    for lag in lags:
        n_segments = arr.size // lag
        rs_seg: list[float] = []
        for s in range(n_segments):
            seg = arr[s * lag : (s + 1) * lag]
            mean = float(np.mean(seg))
            y = np.cumsum(seg - mean)
            r = float(np.max(y) - np.min(y))
            std = float(np.std(seg))
            if std > 0 and r > 0:
                rs_seg.append(r / std)
        if rs_seg:
            rs_values.append(float(np.mean(rs_seg)))
    if len(rs_values) < 2:
        raise ValueError("insufficient valid R/S values — series may be degenerate")
    log_x = np.log(np.array(lags[: len(rs_values)], dtype=np.float64))
    log_y = np.log(np.array(rs_values, dtype=np.float64))
    return _huber_slope(log_x, log_y)


def hurst_dfa(ts: npt.ArrayLike, min_lag: int = 8, max_lag: int = 100) -> float:
    """Hurst exponent via Detrended Fluctuation Analysis.

    Constructs the integrated (mean-removed cumulative) series, splits it
    into windows of length n, fits a linear trend to each window, and
    takes the RMS detrended fluctuation F(n). Expected scaling: F(n) ~ n^H.

    Parameters
    ----------
    ts : array_like
    min_lag, max_lag : int
        Window size range. max_lag must be < len(ts) // 2.

    Returns
    -------
    float
        Hurst / scaling exponent estimate.
    """
    arr = np.asarray(ts, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"ts must be 1-D, got shape {arr.shape}")
    if min_lag < 4:
        raise ValueError(f"min_lag must be >= 4, got {min_lag}")
    if max_lag <= min_lag:
        raise ValueError(f"max_lag must be > min_lag, got max_lag={max_lag}")
    if arr.size < 2 * max_lag:
        raise ValueError(f"need at least 2·max_lag={2 * max_lag} samples, got {arr.size}")

    y = np.cumsum(arr - float(np.mean(arr)))
    lags = list(range(min_lag, max_lag + 1))
    F: list[float] = []
    for n in lags:
        n_segments = y.size // n
        if n_segments < 1:
            continue
        seg_rms: list[float] = []
        for s in range(n_segments):
            seg = y[s * n : (s + 1) * n]
            t = np.arange(seg.size, dtype=np.float64)
            slope, intercept = np.polyfit(t, seg, 1)
            trend = slope * t + intercept
            seg_rms.append(float(np.sqrt(np.mean((seg - trend) ** 2))))
        if seg_rms:
            F.append(float(np.mean(seg_rms)))
    if len(F) < 2:
        raise ValueError("insufficient DFA fluctuation values")
    log_x = np.log(np.array(lags[: len(F)], dtype=np.float64))
    log_y = np.log(np.array(F, dtype=np.float64))
    return _huber_slope(log_x, log_y)
