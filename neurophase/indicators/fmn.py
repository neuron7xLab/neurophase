"""Flow Momentum Network (FMN).

FMN combines order-book imbalance with cumulative delta volume into a
bounded momentum signal:

    OB_imbalance_t = (B_t − A_t) / (B_t + A_t)         ∈ [-1, 1]
    CVD_t          = Σ_{i ≤ t} ΔV_i
    FMN_t          = tanh( w₁ · OB_imbalance_t  +  w₂ · CVD_t / N )   ∈ (-1, 1)

where B_t, A_t are cumulative bid/ask volumes (book depth), ΔV_i is the
per-bar delta volume, and N is a normalisation constant — total absolute
delta volume over the window — so the two components are comparable in
magnitude.

Ported from Neuron7X technical reference (2025).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_EPSILON = 1e-9


def compute_fmn(
    delta_vol: npt.ArrayLike,
    bid_vol: npt.ArrayLike,
    ask_vol: npt.ArrayLike,
    w_imbalance: float = 1.0,
    w_cvd: float = 1.0,
) -> FloatArray:
    """Compute the Flow Momentum Network indicator.

    Parameters
    ----------
    delta_vol : array_like, shape (N,)
        Signed per-bar delta volume.
    bid_vol, ask_vol : array_like, shape (N,)
        Aggregate bid/ask volumes at each bar.
    w_imbalance : float
        Weight on the order-book imbalance term. Must be non-negative.
    w_cvd : float
        Weight on the normalised cumulative delta term. Must be non-negative.

    Returns
    -------
    FloatArray, shape (N,)
        FMN values in (-1, 1).

    Raises
    ------
    ValueError
        For shape mismatches, non-1-D inputs, or negative weights.
    """
    if w_imbalance < 0 or w_cvd < 0:
        raise ValueError("FMN weights must be non-negative")
    dv_arr = np.asarray(delta_vol, dtype=np.float64)
    bid_arr = np.asarray(bid_vol, dtype=np.float64)
    ask_arr = np.asarray(ask_vol, dtype=np.float64)
    for a, name in zip(
        (dv_arr, bid_arr, ask_arr),
        ("delta_vol", "bid_vol", "ask_vol"),
        strict=True,
    ):
        if a.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    n = dv_arr.size
    if bid_arr.size != n or ask_arr.size != n:
        raise ValueError("FMN inputs must have the same length")
    if n == 0:
        return np.array([], dtype=np.float64)

    imbalance = (bid_arr - ask_arr) / (bid_arr + ask_arr + _EPSILON)
    cvd = np.cumsum(dv_arr)
    normaliser = float(np.sum(np.abs(dv_arr))) + _EPSILON
    linear = w_imbalance * imbalance + w_cvd * cvd / normaliser
    fmn: FloatArray = np.tanh(linear).astype(np.float64)
    return fmn
