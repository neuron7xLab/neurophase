"""Rayleigh test — analytical circular uniformity test.

Tests whether a set of phase differences Δφ(t) is non-uniformly
distributed on the circle. Purely analytical — no surrogates,
no permutations, no randomness.

The Rayleigh test statistic:
    R = |mean(exp(i·Δφ))|    (mean resultant length)
    Z = N · R²
    p ≈ exp(−Z) · (1 + (2Z − Z²)/(4N) − (24Z − 132Z² + 76Z³ − 9Z⁴)/(288N²))

This is the second-order correction from Mardia & Jupp (2000),
which is accurate for N ≥ 10.

Independence from surrogate-based tests:
    - No cyclic shift, no phase shuffle, no permutations
    - Deterministic closed-form p-value
    - Answers: "is the phase-difference distribution non-uniform?"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RayleighResult:
    """Result of the Rayleigh test for circular uniformity.

    Attributes
    ----------
    R : float
        Mean resultant length ∈ [0, 1].
    Z : float
        Rayleigh Z statistic = N · R².
    p_value : float
        Approximate p-value with second-order finite-sample correction.
    n_samples : int
        Number of phase-difference samples.
    significant : bool
        True when p_value < alpha.
    alpha : float
        Significance level used.
    """

    R: float
    Z: float
    p_value: float
    n_samples: int
    significant: bool
    alpha: float


def rayleigh_test(
    delta_phi: npt.ArrayLike,
    *,
    alpha: float = 0.05,
) -> RayleighResult:
    """Rayleigh test for non-uniformity of circular data.

    Parameters
    ----------
    delta_phi : array_like, shape (T,)
        Phase differences Δφ(t) = φ_x(t) − φ_y(t) in radians.
    alpha : float
        Significance level.

    Returns
    -------
    RayleighResult

    Raises
    ------
    ValueError
        If input has fewer than 10 samples or contains non-finite values.
    """
    arr = np.asarray(delta_phi, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"delta_phi must be 1-D, got shape {arr.shape}")
    if arr.size < 10:
        raise ValueError(
            f"Rayleigh test requires ≥ 10 samples for reliable p-value, got {arr.size}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("delta_phi must contain only finite values")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = arr.size

    # Mean resultant length
    mean_vec = np.mean(np.exp(1j * arr))
    r_val = float(np.abs(mean_vec))

    # Rayleigh Z statistic
    z_val = float(n * r_val**2)

    # p-value with second-order correction (Mardia & Jupp 2000)
    # p ≈ exp(−Z) · (1 + (2Z − Z²)/(4N) − (24Z − 132Z² + 76Z³ − 9Z⁴)/(288N²))
    term1 = (2.0 * z_val - z_val**2) / (4.0 * n)
    term2 = (
        24.0 * z_val - 132.0 * z_val**2 + 76.0 * z_val**3 - 9.0 * z_val**4
    ) / (288.0 * n**2)
    p_val = float(np.exp(-z_val) * (1.0 + term1 - term2))
    # Clamp to [0, 1]
    p_val = float(np.clip(p_val, 0.0, 1.0))

    return RayleighResult(
        R=r_val,
        Z=z_val,
        p_value=p_val,
        n_samples=n,
        significant=p_val < alpha,
        alpha=alpha,
    )
