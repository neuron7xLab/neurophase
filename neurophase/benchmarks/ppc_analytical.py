"""Analytical PPC prediction for Kuramoto oscillators.

Three models are provided:

1. **Bessel model** (stochastic Kuramoto, von Mises stationarity):
       PLV = I₁(κ) / I₀(κ),   κ = k / σ²
   Appropriate when the oscillator has continuous phase noise σ.

2. **Ott-Antonsen model** (deterministic Kuramoto, exact mean-field):
       R∞ = 0                          when K < K_c = 2·Δω  (sub-critical)
       R∞ = sqrt(1 − (K_c/K)²)        when K ≥ K_c          (super-critical)
       PPC = R∞²
   Derived analytically from the Ott-Antonsen manifold reduction (2008).
   This is the default for calibrated predictions.

3. **Empirical calibrated model** (legacy, HN_EMPIRICAL):
       K < Δω  →  PPC ≈ (K/Δω)⁴  (exponent 4 not derived — empirical)
       K ≥ Δω  →  PPC ≈ 1 − exp(−β·(K − Δω))  (β calibrated, not derived)
   Retained for back-compatibility; clearly marked as advisory-only.

References
----------
Ott, E. & Antonsen, T.M. (2008) "Low dimensional behavior of large systems of
globally coupled oscillators." *Chaos* **18**, 037113.
DOI: 10.1063/1.2930766

Kuramoto, Y. (1984) *Chemical Oscillations, Waves, and Turbulence.* Springer.

Acebrón, J.A. et al. (2005) "The Kuramoto model: A simple paradigm for
synchronization phenomena." *Rev. Mod. Phys.* **77**, 137.
"""

from __future__ import annotations

import math
from typing import Literal

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


def ott_antonsen_order_parameter(
    coupling_k: float,
    f_neural: float = 1.0,
    f_market: float = 0.5,
) -> float:
    """Asymptotic order parameter R∞ from the Ott-Antonsen manifold reduction.

    For a population of globally coupled identical oscillators with mean-field
    coupling K and frequency detuning Δω = 2π|f_neural − f_market|, the
    Ott-Antonsen ansatz (2008, Eq. 12) yields the exact long-time solution:

        R∞ = 0                       K < K_c  (incoherent)
        R∞ = sqrt(1 − (K_c/K)²)     K ≥ K_c  (partially/fully locked)

    where K_c = 2·Δω is the critical coupling.

    Parameters
    ----------
    coupling_k : float
        Mean-field coupling strength K ≥ 0.
    f_neural : float
        Neural oscillation frequency in Hz.
    f_market : float
        Market oscillation frequency in Hz.

    Returns
    -------
    float
        Asymptotic order parameter R∞ ∈ [0, 1].

    References
    ----------
    Ott, E. & Antonsen, T.M. (2008) Chaos 18, 037113.
    DOI: 10.1063/1.2930766
    """
    if coupling_k < 0:
        raise ValueError(f"coupling_k must be ≥ 0, got {coupling_k}")
    if coupling_k == 0.0:
        return 0.0

    delta_omega = 2.0 * math.pi * abs(f_neural - f_market)
    k_c = 2.0 * delta_omega

    if delta_omega == 0.0:
        # No detuning: any positive coupling fully locks
        return 1.0

    if coupling_k < k_c:
        return 0.0

    return float(math.sqrt(1.0 - (k_c / coupling_k) ** 2))


def ott_antonsen_ppc(
    coupling_k: float,
    f_neural: float = 1.0,
    f_market: float = 0.5,
) -> float:
    """Analytical PPC = R∞² from the Ott-Antonsen (2008) solution.

    PPC ≈ PLV² for large N (phase-locking value squared equals the squared
    order parameter in the mean-field limit).  The underlying R∞ is computed
    from ``ott_antonsen_order_parameter``.

    Parameters
    ----------
    coupling_k : float
        Mean-field coupling strength K ≥ 0.
    f_neural : float
        Neural oscillation frequency in Hz.
    f_market : float
        Market oscillation frequency in Hz.

    Returns
    -------
    float
        Predicted PPC ∈ [0, 1].

    References
    ----------
    Ott, E. & Antonsen, T.M. (2008) Chaos 18, 037113.
    DOI: 10.1063/1.2930766
    """
    r_inf = ott_antonsen_order_parameter(coupling_k, f_neural, f_market)
    return float(r_inf**2)


def calibrated_ppc(
    coupling_k: float,
    f_neural: float = 1.0,
    f_market: float = 0.5,
    transition_sharpness: float = 10.0,
    model: Literal["ott_antonsen", "empirical"] = "ott_antonsen",
) -> float:
    """Calibrated PPC prediction for deterministic Kuramoto with detuning.

    The neural phase generator integrates the Kuramoto ODE via RK4 with no
    explicit phase noise.  Locking occurs when K exceeds the critical coupling
    K_c = 2·Δω = 4π·|f_neural − f_market|  (Ott & Antonsen 2008, Eq. 12).

    **Ott-Antonsen model** (default, ``model="ott_antonsen"``):

        K < K_c  →  R∞ = 0  →  PPC = 0
            Analytically correct: no partial locking below K_c for identical
            oscillators on the Ott-Antonsen manifold.

        K ≥ K_c  →  R∞ = sqrt(1 − (K_c/K)²)  →  PPC = R∞²
            Near K_c, Taylor-expanding: R∞² ≈ 2·(K − K_c)/K_c, so the
            effective β = 2/K_c.  The exponential form 1 − exp(−β·ΔK) is
            a first-order approximation to R∞² valid close to threshold.

    **Empirical model** (``model="empirical"``):  # HN_EMPIRICAL: empirical fit,
    not derived — advisory only.

        K < K_c  →  PPC ≈ (K/Δω)⁴  (exponent 4 has no analytic derivation)
        K ≥ K_c  →  PPC ≈ 1 − exp(−β·(K − K_c))  (β = transition_sharpness,
                                                     calibrated, not derived)

    Parameters
    ----------
    coupling_k : float
        Coupling strength K ≥ 0.
    f_neural : float
        Neural oscillation frequency in Hz.
    f_market : float
        Market oscillation frequency in Hz.
    transition_sharpness : float
        Steepness β for the empirical model only.  Ignored when
        ``model="ott_antonsen"``.  Default 10.0 was calibrated against the
        generator with noise_amplitude=0.2; it has no analytic derivation.
    model : {"ott_antonsen", "empirical"}
        Which model to use.  Default is ``"ott_antonsen"``.

    Returns
    -------
    float
        Predicted PPC ∈ [0, 1].

    References
    ----------
    Ott, E. & Antonsen, T.M. (2008) Chaos 18, 037113.
    DOI: 10.1063/1.2930766
    """
    if coupling_k < 0:
        raise ValueError(f"coupling_k must be ≥ 0, got {coupling_k}")
    if coupling_k == 0.0:
        return 0.0

    if model == "ott_antonsen":
        return ott_antonsen_ppc(coupling_k, f_neural, f_market)

    # HN_EMPIRICAL: empirical fit, not derived — advisory only
    delta_omega = 2.0 * math.pi * abs(f_neural - f_market)
    k_c = 2.0 * delta_omega

    if coupling_k < k_c:
        # HN_EMPIRICAL: exponent 4 has no analytic derivation
        return float(min(1.0, (coupling_k / delta_omega) ** 4))
    # HN_EMPIRICAL: β = transition_sharpness calibrated, not derived
    return float(max(0.0, 1.0 - np.exp(-transition_sharpness * (coupling_k - k_c))))


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
