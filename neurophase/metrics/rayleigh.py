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

Effect size thresholds (based on R, not p):
    R < 0.05  → negligible  (consistent with uniform)
    R < 0.10  → small
    R < 0.30  → medium
    R ≥ 0.30  → large

Using R rather than p for the gate decision avoids the N-sensitivity
problem: p-values shrink with sample size and can reject trivially
small effects at large N. R is sample-size invariant and measures
true phase concentration.

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

# Effect size thresholds for R (circular statistics convention)
R_NEGLIGIBLE: float = 0.05
R_SMALL: float = 0.10
R_MEDIUM: float = 0.30


def _classify_effect_size(r: float) -> str:
    """Classify R into effect size category."""
    if r < R_NEGLIGIBLE:
        return "negligible"
    if r < R_SMALL:
        return "small"
    if r < R_MEDIUM:
        return "medium"
    return "large"


@dataclass(frozen=True)
class RayleighResult:
    """Result of the Rayleigh test for circular uniformity.

    Attributes
    ----------
    R : float
        Mean resultant length ∈ [0, 1]. Primary metric for gate decision.
    Z : float
        Rayleigh Z statistic = N · R².
    p_value : float
        Approximate p-value with second-order finite-sample correction.
        Kept for reference; gate decisions use R thresholds.
    n_samples : int
        Number of phase-difference samples.
    effect_size : str
        Classification based on R: "negligible" | "small" | "medium" | "large".
    significant : bool
        True when R ≥ R_SMALL (0.10). Based on effect size, not p-value,
        to avoid N-sensitivity.
    alpha : float
        Significance level (used for p-value reference only).
    """

    R: float
    Z: float
    p_value: float
    n_samples: int
    effect_size: str
    significant: bool
    alpha: float


def rayleigh_test(
    delta_phi: npt.ArrayLike,
    *,
    alpha: float = 0.05,
    r_threshold: float = R_SMALL,
) -> RayleighResult:
    """Rayleigh test for non-uniformity of circular data.

    Parameters
    ----------
    delta_phi : array_like, shape (T,)
        Phase differences Δφ(t) = φ_x(t) − φ_y(t) in radians.
    alpha : float
        Significance level (for p-value reference).
    r_threshold : float
        R threshold for the ``significant`` flag. Default is R_SMALL (0.10).
        Using R instead of p avoids N-sensitivity.

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
    term1 = (2.0 * z_val - z_val**2) / (4.0 * n)
    term2 = (24.0 * z_val - 132.0 * z_val**2 + 76.0 * z_val**3 - 9.0 * z_val**4) / (288.0 * n**2)
    p_val = float(np.exp(-z_val) * (1.0 + term1 - term2))
    p_val = float(np.clip(p_val, 0.0, 1.0))

    effect = _classify_effect_size(r_val)

    return RayleighResult(
        R=r_val,
        Z=z_val,
        p_value=p_val,
        n_samples=n,
        effect_size=effect,
        significant=r_val >= r_threshold,
        alpha=alpha,
    )
