"""Information-Structural Metric — balances entropy rate against topological energy.

    ISM(t) = η · H'(t) / E_topo(t)

where:
    H'(t)     = |dH_S / dt|           — instantaneous entropy rate (Shannon)
    E_topo(t) = ⟨κ̄²⟩_T                — moving average of squared mean curvature

Interpretation:
    ISM ≈ 1      → balanced regime
    ISM ≫ 1      → chaotic overload (information rate dominates structure)
    ISM ≪ 1      → structural inertia (topology frozen, entropy flat)

This metric is one of four conditions in the π-system emergent-phase
criterion (R > 0.75 ∧ ΔH_S < −0.05 ∧ κ̄ < −0.1 ∧ ISM ∈ [0.8, 1.2]).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

_EPSILON = 1e-12


def compute_topological_energy(
    ricci_series: npt.ArrayLike,
    window: int = 100,
) -> float:
    """Moving average of squared mean curvature over a trailing window.

        E_topo = ⟨κ̄²⟩_T

    Parameters
    ----------
    ricci_series : array_like
        History of mean-curvature values κ̄(t).
    window : int
        Trailing window size. Must be positive.

    Returns
    -------
    float
        Topological energy (always ≥ 0).
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    arr = np.asarray(ricci_series, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    tail = arr[-window:] if arr.size > window else arr
    return float(np.mean(tail**2))


def compute_ism(
    entropy_series: npt.ArrayLike,
    ricci_series: npt.ArrayLike,
    window: int = 100,
    eta: float = 1.0,
    dt: float = 1.0,
) -> float:
    """Information-Structural Metric ISM(t).

    Numerically differentiates the entropy series with ``np.gradient`` and
    divides |H'| by ⟨κ̄²⟩_T. Returns 0.0 when the topological energy is
    indistinguishable from zero (undefined ratio — honest null).

    Parameters
    ----------
    entropy_series : array_like
        Recent history of Shannon entropy values.
    ricci_series : array_like
        Recent history of mean-curvature values κ̄(t).
    window : int
        Trailing window for the topological energy average.
    eta : float
        Scaling constant (dimensional balance factor).
    dt : float
        Sampling interval for the entropy gradient.

    Returns
    -------
    float
        ISM value. Zero when the denominator is below ``_EPSILON``.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if eta < 0:
        raise ValueError(f"eta must be non-negative, got {eta}")
    entropy_arr = np.asarray(entropy_series, dtype=np.float64)
    if entropy_arr.size < 2:
        return 0.0
    h_prime = float(np.abs(np.gradient(entropy_arr, dt)[-1]))
    e_topo = compute_topological_energy(ricci_series, window=window)
    if e_topo < _EPSILON:
        return 0.0
    return float(eta * h_prime / e_topo)


def ism_derivative(ism_series: npt.ArrayLike, dt: float = 1.0) -> float:
    """Last-point derivative of the ISM series.

    Useful as a second-order trigger: rapid ISM acceleration signals
    imminent regime change even before ISM itself leaves the balanced band.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    arr = np.asarray(ism_series, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    return float(np.gradient(arr, dt)[-1])
