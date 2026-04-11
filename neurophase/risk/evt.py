"""Extreme Value Theory — POT / GPD fit with VaR and CVaR.

Crypto returns have *heavy tails*. Normal-distribution VaR systematically
under-reports the probability of large drawdowns. EVT models the tail
itself via the Pickands–Balkema–de Haan theorem: excesses over a high
threshold converge to a Generalised Pareto Distribution.

    X | (X > u) ~ GPD(ξ, σ)

    G_{ξ,σ}(y) = 1 - (1 + ξy/σ)^(-1/ξ),    y > 0

Risk metrics at confidence level p (e.g. 0.99):

    ζ = k / n                              (tail proportion at threshold u)
    VaR_p = u + (σ/ξ) · [(α/ζ)^(-ξ) - 1]   where α = 1 - p
    CVaR_p = (VaR_p + σ - ξ·u) / (1 - ξ)    (requires ξ < 1)

CVaR is the *Expected Shortfall*: average loss conditional on the loss
exceeding VaR. It is regulators' preferred spectral risk measure.

Convention here: positive values represent *losses*. Callers should pass
losses directly, or use the negative of returns if they want left-tail
risk. The convention is documented per-function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import genpareto

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class EVTFit:
    """Result of a Peaks-Over-Threshold Generalised Pareto fit.

    Attributes
    ----------
    xi : float
        Shape parameter of the GPD. ξ > 0 → heavy tail; ξ = 0 → exponential;
        ξ < 0 → bounded tail.
    sigma : float
        Scale parameter of the GPD (σ > 0).
    threshold : float
        The chosen threshold ``u``.
    zeta : float
        Fraction of observations exceeding ``u`` (k / n).
    n_exceedances : int
        Number of points above the threshold.
    n_total : int
        Total number of points in the fitted series.
    """

    xi: float
    sigma: float
    threshold: float
    zeta: float
    n_exceedances: int
    n_total: int


def fit_gpd_pot(losses: npt.ArrayLike, threshold_quantile: float = 0.95) -> EVTFit:
    """Fit a Generalised Pareto to the right tail via Peaks-Over-Threshold.

    Parameters
    ----------
    losses : array_like
        Loss series (positive values = larger losses). Pass -returns if
        your data is in return form and you want left-tail risk.
    threshold_quantile : float
        Quantile used as the threshold u. Typical values: 0.90–0.97.

    Returns
    -------
    EVTFit
        Fit parameters with threshold metadata.

    Raises
    ------
    ValueError
        For bad quantile, insufficient exceedances, or degenerate input.
    """
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError(f"threshold_quantile must be in (0, 1), got {threshold_quantile}")
    arr = np.asarray(losses, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"losses must be 1-D, got shape {arr.shape}")
    if arr.size < 50:
        raise ValueError(f"need at least 50 samples for a POT fit, got {arr.size}")

    u = float(np.quantile(arr, threshold_quantile))
    excesses = arr[arr > u] - u
    if excesses.size < 10:
        raise ValueError(
            f"too few exceedances above u={u:.4g} ({excesses.size}); lower the threshold quantile"
        )

    xi, _loc, sigma = genpareto.fit(excesses, floc=0.0)
    if not np.isfinite(xi) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"GPD fit produced invalid parameters: xi={xi}, sigma={sigma}")

    return EVTFit(
        xi=float(xi),
        sigma=float(sigma),
        threshold=u,
        zeta=float(excesses.size) / float(arr.size),
        n_exceedances=int(excesses.size),
        n_total=int(arr.size),
    )


def compute_var(fit: EVTFit, p: float) -> float:
    """VaR_p from a POT/GPD fit.

        VaR_p = u + (σ/ξ) · [(α/ζ)^(-ξ) - 1],   α = 1 - p

    The ξ → 0 limit recovers the exponential form:

        VaR_p = u + σ · log(ζ / α)

    Parameters
    ----------
    fit : EVTFit
        Fitted GPD parameters.
    p : float
        Confidence level in (0, 1). Must satisfy ``p > 1 - fit.zeta`` —
        otherwise the requested quantile lies *below* the threshold and
        EVT is the wrong tool.

    Returns
    -------
    float
        VaR at confidence level p (same units as the input losses).

    Raises
    ------
    ValueError
        If p is out of range or too shallow for the fitted threshold.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    alpha = 1.0 - p
    if alpha >= fit.zeta:
        raise ValueError(
            f"p={p:.4g} too shallow: 1-p={alpha:.4g} >= ζ={fit.zeta:.4g}. "
            "Choose a smaller threshold_quantile or a larger p."
        )
    if abs(fit.xi) < 1e-8:
        return float(fit.threshold + fit.sigma * np.log(fit.zeta / alpha))
    factor = (alpha / fit.zeta) ** (-fit.xi) - 1.0
    return float(fit.threshold + (fit.sigma / fit.xi) * factor)


def compute_cvar(fit: EVTFit, p: float) -> float:
    """CVaR_p (Expected Shortfall) from a POT/GPD fit.

        CVaR_p = (VaR_p + σ - ξ·u) / (1 - ξ),   ξ < 1

    Parameters
    ----------
    fit : EVTFit
    p : float
        Confidence level in (0, 1).

    Returns
    -------
    float
        CVaR at confidence level p. Always ≥ VaR_p.

    Raises
    ------
    ValueError
        If ξ ≥ 1 (infinite mean — CVaR undefined) or p is shallow.
    """
    if fit.xi >= 1.0:
        raise ValueError(
            f"CVaR undefined for ξ={fit.xi:.4g} ≥ 1 (infinite mean). "
            "Use VaR only or refit on a different regime."
        )
    var_p = compute_var(fit, p)
    return float((var_p + fit.sigma - fit.xi * fit.threshold) / (1.0 - fit.xi))
