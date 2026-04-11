"""Kuramoto order parameter R(t) and mean phase Ψ(t).

The complex order parameter for N phase oscillators θ_k(t) is:

    R(t) · exp(i·Ψ(t)) = (1/N) · Σ_{k=1..N} exp(i·θ_k(t))

where:
    R(t) ∈ [0, 1]  —  magnitude of phase coherence
    Ψ(t) ∈ [-π, π] —  mean phase (angle of the resultant vector)

R ≈ 0  →  phases uniformly distributed (incoherent)
R ≈ 1  →  all phases aligned (fully synchronized)

Supports both single snapshots (shape (N,)) and trajectories (shape (T, N)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class OrderParameterResult:
    """Result of order parameter computation.

    For a single snapshot, R and psi are scalars (float).
    For a trajectory of length T, R and psi are arrays of shape (T,).
    """

    R: float | FloatArray
    psi: float | FloatArray


def order_parameter(theta: npt.ArrayLike) -> OrderParameterResult:
    """Compute the Kuramoto order parameter R(t) and mean phase Ψ(t).

    Parameters
    ----------
    theta : array_like
        Phase vector. Supported shapes:
            (N,)    — single snapshot of N oscillators
            (T, N)  — trajectory of N oscillators over T time steps

    Returns
    -------
    OrderParameterResult
        R in [0, 1] and psi in [-π, π]. Scalars for 1-D input,
        arrays of length T for 2-D input.

    Raises
    ------
    ValueError
        If ``theta`` is not 1-D or 2-D, or is empty.
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.size == 0:
        raise ValueError("theta must be non-empty")
    if theta_arr.ndim == 1:
        z = np.mean(np.exp(1j * theta_arr))
        return OrderParameterResult(R=float(np.abs(z)), psi=float(np.angle(z)))
    if theta_arr.ndim == 2:
        z = np.mean(np.exp(1j * theta_arr), axis=1)
        return OrderParameterResult(
            R=np.abs(z).astype(np.float64),
            psi=np.angle(z).astype(np.float64),
        )
    raise ValueError(f"theta must be 1-D or 2-D, got shape {theta_arr.shape}")
