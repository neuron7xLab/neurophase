"""Tests for neurophase.risk.evt."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.risk.evt import compute_cvar, compute_var, fit_gpd_pot


def _heavy_tail_losses(n: int, xi_true: float, sigma: float, seed: int) -> np.ndarray:
    """Generate losses: N(0, 1) body + GPD(ξ, σ) right tail appended."""
    rng = np.random.default_rng(seed)
    body = np.abs(rng.standard_normal(int(n * 0.9)))
    # Scipy genpareto tail samples.
    from scipy.stats import genpareto  # local import to keep top clean

    tail = genpareto.rvs(c=xi_true, loc=0.0, scale=sigma, size=n - body.size, random_state=seed)
    # Shift tail above the body's mean to ensure it defines the exceedance tail.
    tail_shifted = tail + float(np.quantile(body, 0.95))
    return np.concatenate([body, tail_shifted]).astype(np.float64)


def test_fit_recovers_positive_shape() -> None:
    losses = _heavy_tail_losses(5000, xi_true=0.35, sigma=1.0, seed=42)
    fit = fit_gpd_pot(losses, threshold_quantile=0.90)
    # Shape parameter should land in the heavy-tail regime.
    assert fit.xi > 0.0
    assert fit.sigma > 0.0
    assert fit.n_exceedances > 100


def test_var_monotone_in_confidence() -> None:
    losses = _heavy_tail_losses(5000, xi_true=0.25, sigma=1.0, seed=1)
    fit = fit_gpd_pot(losses, threshold_quantile=0.90)
    v95 = compute_var(fit, 0.95)
    v99 = compute_var(fit, 0.99)
    assert v99 > v95


def test_cvar_dominates_var() -> None:
    losses = _heavy_tail_losses(5000, xi_true=0.25, sigma=1.0, seed=2)
    fit = fit_gpd_pot(losses, threshold_quantile=0.90)
    v = compute_var(fit, 0.99)
    cv = compute_cvar(fit, 0.99)
    assert cv >= v


def test_cvar_undefined_for_xi_ge_one() -> None:
    from neurophase.risk.evt import EVTFit

    fit = EVTFit(xi=1.1, sigma=1.0, threshold=0.1, zeta=0.05, n_exceedances=50, n_total=1000)
    with pytest.raises(ValueError, match="CVaR undefined"):
        compute_cvar(fit, 0.99)


def test_var_rejects_shallow_p() -> None:
    losses = _heavy_tail_losses(2000, xi_true=0.25, sigma=1.0, seed=3)
    fit = fit_gpd_pot(losses, threshold_quantile=0.90)
    # Shallow p with large fraction above threshold → 1-p >= ζ → error.
    with pytest.raises(ValueError, match="too shallow"):
        compute_var(fit, 0.5)


def test_fit_rejects_small_sample() -> None:
    with pytest.raises(ValueError, match="at least 50"):
        fit_gpd_pot(np.random.default_rng(0).standard_normal(10))


def test_fit_rejects_bad_threshold_quantile() -> None:
    with pytest.raises(ValueError, match="threshold_quantile"):
        fit_gpd_pot(np.random.default_rng(0).standard_normal(200), threshold_quantile=1.5)


def test_var_exponential_limit() -> None:
    """Near-zero xi should activate the exponential fallback branch."""
    from neurophase.risk.evt import EVTFit

    fit = EVTFit(xi=0.0, sigma=1.0, threshold=1.0, zeta=0.1, n_exceedances=100, n_total=1000)
    v = compute_var(fit, 0.99)
    # Closed form: u + σ · log(ζ/α) = 1 + 1 · log(0.1/0.01) = 1 + log(10).
    assert v == pytest.approx(1.0 + float(np.log(10.0)), rel=1e-6)
