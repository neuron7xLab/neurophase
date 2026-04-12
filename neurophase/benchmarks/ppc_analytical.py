"""Analytical PPC prediction for Kuramoto oscillators.

For a single oscillator coupled to an external driver via:
    dφ/dt = ω + k·sin(φ_ext − φ)

at stationarity the phase-difference distribution concentrates
around zero with a von Mises–like shape. The theoretical PLV is:
    PLV_theory = I₁(κ) / I₀(κ)
where κ = k / σ² is the signal-to-noise ratio, and I₀, I₁ are
modified Bessel functions of the first kind.

The theoretical PPC is then:
    PPC_theory = PLV_theory²

This provides a closed-form ground truth for calibration — if the
pipeline's measured PPC deviates from the Bessel prediction by more
than a tolerance, something is wrong in the phase extraction or
coupling estimation.

Reference: Kuramoto (1984), Acebrón et al. (2005) Rev. Mod. Phys. 77.
"""

from __future__ import annotations

from scipy.special import i0, i1


def theoretical_plv(
    coupling_k: float,
    noise_sigma: float = 1.0,
) -> float:
    """Theoretical PLV for a Kuramoto oscillator at stationarity.

    Parameters
    ----------
    coupling_k : float
        Coupling strength k ≥ 0.
    noise_sigma : float
        Noise standard deviation σ > 0.

    Returns
    -------
    float
        Theoretical PLV = I₁(κ) / I₀(κ) where κ = k / σ².
    """
    if coupling_k < 0:
        raise ValueError(f"coupling_k must be ≥ 0, got {coupling_k}")
    if noise_sigma <= 0:
        raise ValueError(f"noise_sigma must be > 0, got {noise_sigma}")
    if coupling_k == 0.0:
        return 0.0
    kappa = coupling_k / (noise_sigma**2)
    return float(i1(kappa) / i0(kappa))


def theoretical_ppc(
    coupling_k: float,
    noise_sigma: float = 1.0,
) -> float:
    """Theoretical PPC = PLV² for a Kuramoto oscillator.

    Parameters
    ----------
    coupling_k : float
        Coupling strength k ≥ 0.
    noise_sigma : float
        Noise standard deviation σ > 0.

    Returns
    -------
    float
        Theoretical PPC ∈ [0, 1].
    """
    plv = theoretical_plv(coupling_k, noise_sigma)
    return float(plv**2)


def bessel_ratio_monotone_check(
    k_values: list[float],
    noise_sigma: float = 1.0,
) -> bool:
    """Verify that theoretical PLV increases monotonically with k.

    Parameters
    ----------
    k_values : list[float]
        Coupling strengths to check (must be sorted ascending).
    noise_sigma : float
        Noise σ.

    Returns
    -------
    bool
        True if PLV is strictly monotone increasing.
    """
    plv_values = [theoretical_plv(k, noise_sigma) for k in k_values]
    return all(plv_values[i] < plv_values[i + 1] for i in range(len(plv_values) - 1))
