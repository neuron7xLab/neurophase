"""Analytical PPC prediction for Kuramoto oscillators.

Two models are provided:

1. **Bessel model** (stochastic Kuramoto, von Mises stationarity):
       PLV = I₁(κ) / I₀(κ),   κ = k / σ²
   Appropriate when the oscillator has continuous phase noise σ.

2. **Deterministic transition model** (Kuramoto with detuning):
       k < Δω  →  drifting, PPC ≈ (k/Δω)⁴ (noise-assisted)
       k ≥ Δω  →  locked,  PPC ≈ 1 − exp(−β·(k − Δω))
   Appropriate for our generator where phase dynamics are deterministic
   (RK4, no phase noise) and the transition occurs at k_c = Δω.

The deterministic model is the default for calibrated predictions
because the neural phase generator has no explicit phase noise —
only additive signal noise, which does not diffuse the phase.

Reference: Kuramoto (1984), Acebrón et al. (2005) Rev. Mod. Phys. 77.
"""

from __future__ import annotations

import numpy as np
from scipy.special import i0, i1


def theoretical_plv(
    coupling_k: float,
    noise_sigma: float = 1.0,
) -> float:
    """Theoretical PLV via Bessel ratio (stochastic Kuramoto).

    Parameters
    ----------
    coupling_k : float
        Coupling strength k ≥ 0.
    noise_sigma : float
        Phase noise σ > 0.

    Returns
    -------
    float
        PLV = I₁(κ) / I₀(κ) where κ = k / σ².
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
    """Theoretical PPC = PLV² via Bessel (stochastic model).

    Parameters
    ----------
    coupling_k : float
        Coupling strength k ≥ 0.
    noise_sigma : float
        Phase noise σ > 0.

    Returns
    -------
    float
        Theoretical PPC ∈ [0, 1].
    """
    plv = theoretical_plv(coupling_k, noise_sigma)
    return float(plv**2)


def calibrated_ppc(
    coupling_k: float,
    f_neural: float = 1.0,
    f_market: float = 0.5,
    transition_sharpness: float = 10.0,
) -> float:
    """Calibrated PPC prediction for deterministic Kuramoto with detuning.

    The neural phase generator integrates the Kuramoto ODE via RK4
    with no explicit phase noise. Locking occurs when k exceeds the
    critical coupling k_c = Δω = 2π·|f_neural − f_market|.

    Below critical (k < Δω): PPC grows as (k/Δω)⁴ (empirical fit
    from noise-assisted partial locking via market phase noise).

    Above critical (k ≥ Δω): PPC ≈ 1 − exp(−β·(k − Δω)), where
    β = transition_sharpness controls how quickly PPC approaches 1.

    Parameters
    ----------
    coupling_k : float
        Coupling strength k ≥ 0.
    f_neural : float
        Neural oscillation frequency in Hz.
    f_market : float
        Market oscillation frequency in Hz.
    transition_sharpness : float
        Steepness β of the locking transition. Default 10.0 calibrated
        to match the generator with noise_amplitude=0.2 on market phase.

    Returns
    -------
    float
        Predicted PPC ∈ [0, 1].
    """
    if coupling_k < 0:
        raise ValueError(f"coupling_k must be ≥ 0, got {coupling_k}")
    if coupling_k == 0.0:
        return 0.0

    delta_omega = 2.0 * np.pi * abs(f_neural - f_market)

    if coupling_k < delta_omega:
        # Sub-critical: noise-assisted partial locking
        return float(min(1.0, (coupling_k / delta_omega) ** 4))
    # Super-critical: rapid lock
    return float(max(0.0, 1.0 - np.exp(-transition_sharpness * (coupling_k - delta_omega))))


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
        Phase noise σ.

    Returns
    -------
    bool
        True if PLV is strictly monotone increasing.
    """
    plv_values = [theoretical_plv(k, noise_sigma) for k in k_values]
    return all(plv_values[i] < plv_values[i + 1] for i in range(len(plv_values) - 1))
