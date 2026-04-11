"""Quantum Integrated Liquidity Metric (QILM).

QILM quantifies whether a price move is *underwritten* by new capital
flowing into the market or, conversely, is driven by positions being
closed.

    QILM_t = S_t · (|ΔOI_t| / ATR_14,t) · (|ΔV_t| + HV_t) / (V_t + HV_t)

where:
    ΔOI_t       — change in open interest between t and t-1
    ΔV_t        — delta volume (aggressive buys minus aggressive sells)
    HV_t        — hidden (iceberg) volume
    V_t         — total visible volume
    ATR_14,t    — 14-period average true range

The sign S_t follows the directional agreement between OI flow and
delta volume:

    S_t = +1  if ΔOI_t > 0 and sign(ΔV_t) = sign(ΔOI_t)
    S_t = -1  otherwise

Intuition: new positions aligned with the delta-volume direction support
the trend; any other configuration signals weakness or a trap.

Ported from Neuron7X technical reference (2025).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_EPSILON = 1e-9


def compute_qilm(
    open_interest: npt.ArrayLike,
    volume: npt.ArrayLike,
    delta_vol: npt.ArrayLike,
    hidden_vol: npt.ArrayLike,
    atr: npt.ArrayLike,
) -> FloatArray:
    """Compute the QILM indicator for a time series of bars.

    All inputs must have the same length.

    Parameters
    ----------
    open_interest : array_like, shape (N,)
        Open interest at each bar.
    volume : array_like, shape (N,)
        Total visible volume.
    delta_vol : array_like, shape (N,)
        Delta volume (signed aggressive flow).
    hidden_vol : array_like, shape (N,)
        Hidden / iceberg volume.
    atr : array_like, shape (N,)
        14-period ATR at each bar. Zero values are regularised with ε.

    Returns
    -------
    FloatArray, shape (N,)
        QILM values. Positive — bullish capital inflow aligned with
        price direction. Negative — weakening move / distribution.

    Raises
    ------
    ValueError
        If the inputs are not all the same length or are not 1-D.
    """
    oi_arr = np.asarray(open_interest, dtype=np.float64)
    v_arr = np.asarray(volume, dtype=np.float64)
    dv_arr = np.asarray(delta_vol, dtype=np.float64)
    hv_arr = np.asarray(hidden_vol, dtype=np.float64)
    atr_arr = np.asarray(atr, dtype=np.float64)
    arrays = [oi_arr, v_arr, dv_arr, hv_arr, atr_arr]
    for a, name in zip(
        arrays,
        ("open_interest", "volume", "delta_vol", "hidden_vol", "atr"),
        strict=True,
    ):
        if a.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    n = oi_arr.size
    if any(a.size != n for a in arrays):
        raise ValueError("all QILM inputs must have the same length")
    if n == 0:
        return np.array([], dtype=np.float64)

    d_oi = np.diff(oi_arr, prepend=oi_arr[0])
    eff_vol = v_arr + hv_arr + _EPSILON
    magnitude = (np.abs(d_oi) / (atr_arr + _EPSILON)) * ((np.abs(dv_arr) + hv_arr) / eff_vol)
    aligned = (d_oi > 0) & (np.sign(dv_arr) == np.sign(d_oi))
    sign = np.where(aligned, 1.0, -1.0)
    qilm: FloatArray = (sign * magnitude).astype(np.float64)
    return qilm
