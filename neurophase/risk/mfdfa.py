"""Multifractal Detrended Fluctuation Analysis (MF-DFA).

Generalises single-exponent DFA to a *spectrum* of scaling exponents
h(q), exposing the multifractality of the underlying series. For q > 0
the metric emphasises large-amplitude fluctuations; for q < 0 it
emphasises small ones. A single-scaling series is monofractal; the
width of h(q) is a direct measure of market heterogeneity.

    F_q(n) = [ (1/N_s) · Σ_v F²(v, n)^{q/2} ]^{1/q}
    F_q(n) ∝ n^{h(q)}

The *multifractal instability index* is the range of h(q) over a fixed
q grid — wider spectrum ⇔ more heterogeneous regime ⇔ higher tail risk.

Implementation notes:
- Uses non-overlapping windows only (forward direction); the full
  MF-DFA-2 symmetrisation is overkill for risk sizing.
- Detrending is linear (polynomial order 1). Higher-order detrending is
  possible but not implemented here.
- Huber regression in log-log space for robustness to short-lag noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import HuberRegressor

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MFDFAResult:
    """Full MF-DFA spectrum result.

    Attributes
    ----------
    q_values : FloatArray, shape (Q,)
        Moment orders used.
    h_q : FloatArray, shape (Q,)
        Generalised Hurst exponents h(q).
    instability : float
        Width of the h(q) spectrum: max(h) − min(h). Zero for a
        monofractal series; grows with multifractality.
    """

    q_values: FloatArray
    h_q: FloatArray
    instability: float


def _huber_slope(log_x: FloatArray, log_y: FloatArray) -> float:
    """Robust log-log slope; OLS fallback on HuberRegressor failure."""
    try:
        model = HuberRegressor(max_iter=200)
        model.fit(log_x.reshape(-1, 1), log_y)
        return float(model.coef_[0])
    except (ValueError, RuntimeError):
        slope, _ = np.polyfit(log_x, log_y, 1)
        return float(slope)


def mfdfa(
    ts: npt.ArrayLike,
    q_values: npt.ArrayLike | None = None,
    min_lag: int = 8,
    max_lag: int = 100,
) -> MFDFAResult:
    """Run MF-DFA on a time series and return the h(q) spectrum.

    Parameters
    ----------
    ts : array_like, shape (N,)
        Time series (real-valued).
    q_values : array_like | None
        Moment orders. Defaults to ``[-4, -2, -1, 1, 2, 4]`` (skipping 0,
        which is a logarithmic limit).
    min_lag, max_lag : int
        Window size range (positive, min < max, max < N/2).

    Returns
    -------
    MFDFAResult

    Raises
    ------
    ValueError
        For malformed inputs or an insufficient series.
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

    q_arr = (
        np.asarray(q_values, dtype=np.float64)
        if q_values is not None
        else np.array([-4.0, -2.0, -1.0, 1.0, 2.0, 4.0])
    )
    if np.any(q_arr == 0):
        raise ValueError("q = 0 is a logarithmic limit; remove it from q_values")

    # Integrated, mean-removed cumulative process.
    y = np.cumsum(arr - float(np.mean(arr)))
    lags = np.arange(min_lag, max_lag + 1)

    # Detrended variances per scale: F²(v, n) for every window v of size n.
    f2_per_scale: list[FloatArray] = []
    for n in lags:
        n_segments = y.size // n
        if n_segments < 1:
            continue
        segment_variances: list[float] = []
        for s in range(n_segments):
            seg = y[s * n : (s + 1) * n]
            t = np.arange(seg.size, dtype=np.float64)
            slope, intercept = np.polyfit(t, seg, 1)
            trend = slope * t + intercept
            segment_variances.append(float(np.mean((seg - trend) ** 2)))
        f2_per_scale.append(np.asarray(segment_variances, dtype=np.float64))

    if len(f2_per_scale) < 2:
        raise ValueError("insufficient scales — widen [min_lag, max_lag] or extend ts")

    # Generalised fluctuation per q per scale.
    h_values: list[float] = []
    log_lags = np.log(lags[: len(f2_per_scale)].astype(np.float64))
    for q in q_arr:
        f_q = np.empty(len(f2_per_scale), dtype=np.float64)
        for i, f2_arr in enumerate(f2_per_scale):
            # Regularise zero variances.
            f2_reg = np.where(f2_arr > 0, f2_arr, np.finfo(np.float64).tiny)
            f_q[i] = float(np.mean(f2_reg ** (q / 2.0))) ** (1.0 / q)
        log_f = np.log(np.where(f_q > 0, f_q, np.finfo(np.float64).tiny))
        h_values.append(_huber_slope(log_lags, log_f))

    h_arr = np.array(h_values, dtype=np.float64)
    instability = float(np.max(h_arr) - np.min(h_arr))
    return MFDFAResult(q_values=q_arr, h_q=h_arr, instability=instability)


def multifractal_instability(
    ts: npt.ArrayLike,
    q_values: npt.ArrayLike | None = None,
    min_lag: int = 8,
    max_lag: int = 100,
) -> float:
    """Shortcut: return just the instability index width(h(q)).

    Zero for a monofractal series (single scaling exponent); larger
    values indicate broader heterogeneity / more fat-tailed behaviour.
    """
    return mfdfa(ts, q_values=q_values, min_lag=min_lag, max_lag=max_lag).instability
