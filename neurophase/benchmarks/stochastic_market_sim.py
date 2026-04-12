"""Level 2 synthetic validation — stochastic market + tracking agent.

Generates a realistic test scenario:
    1. Market signal: geometric Brownian motion (random walk with drift)
       — unpredictable, requires continuous tracking
    2. Neural agent: tracks market via exponential moving average (EMA)
       — simulates prediction-error-driven neural response
    3. Coupling: neural agent's tracking error → modulates θ-power
       — stronger prediction error → higher θ-power (established FMθ)

This produces a signal pair where:
    - Neural θ-power covaries with market volatility (trial-level)
    - Neural θ-phase does NOT lock to market phase (different frequencies)
    - The coupling is NONLINEAR (via prediction error threshold)

If the pipeline detects the trial-level coupling but correctly rejects
phase coupling, it proves the pipeline can distinguish:
    - True coupling (nonlinear, trial-level) from
    - Null coupling (linear, continuous phase)

This is the scenario that ds003458 SHOULD have provided but didn't
because its reward probabilities are deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class StochasticMarketScenario:
    """A stochastic market + neural tracking scenario with known coupling.

    Attributes
    ----------
    market_price : FloatArray
        Geometric Brownian motion price series, shape (n_trials,).
    market_returns : FloatArray
        Trial-by-trial returns, shape (n_trials,).
    neural_prediction : FloatArray
        Agent's EMA prediction, shape (n_trials,).
    prediction_error : FloatArray
        |actual - predicted|, shape (n_trials,).
    theta_power : FloatArray
        Simulated θ-power = baseline + k * prediction_error + noise.
    coupling_strength : float
        k parameter controlling neural-market coupling.
    n_trials : int
        Number of trials.
    has_coupling : bool
        True when coupling_strength > 0.
    """

    market_price: FloatArray
    market_returns: FloatArray
    neural_prediction: FloatArray
    prediction_error: FloatArray
    theta_power: FloatArray
    coupling_strength: float
    n_trials: int
    has_coupling: bool


def generate_stochastic_market(
    n_trials: int = 480,
    initial_price: float = 100.0,
    drift: float = 0.0005,
    volatility: float = 0.02,
    seed: int = 42,
) -> tuple[FloatArray, FloatArray]:
    """Generate a GBM price series.

    Returns (prices, returns).
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, volatility, n_trials)
    prices = initial_price * np.exp(np.cumsum(log_returns))
    returns = np.diff(prices, prepend=prices[0]) / np.maximum(prices, 1e-10)
    return prices.astype(np.float64), returns.astype(np.float64)


def generate_stochastic_scenario(
    n_trials: int = 480,
    coupling_strength: float = 2.0,
    ema_alpha: float = 0.1,
    noise_level: float = 0.5,
    volatility: float = 0.02,
    seed: int = 42,
) -> StochasticMarketScenario:
    """Generate a full stochastic market + neural tracking scenario.

    Parameters
    ----------
    n_trials : int
        Number of trials.
    coupling_strength : float
        k ≥ 0. How strongly prediction error modulates θ-power.
        k=0 → no coupling (null). k>0 → coupled.
    ema_alpha : float
        EMA smoothing factor for neural prediction (0 < α ≤ 1).
    noise_level : float
        Noise on θ-power (std of additive Gaussian noise).
    volatility : float
        Market volatility (σ of GBM).
    seed : int
        For determinism.

    Returns
    -------
    StochasticMarketScenario
    """
    if coupling_strength < 0:
        raise ValueError(f"coupling_strength must be ≥ 0, got {coupling_strength}")

    rng = np.random.default_rng(seed)

    # Market
    prices, returns = generate_stochastic_market(
        n_trials=n_trials,
        volatility=volatility,
        seed=seed,
    )

    # Neural EMA prediction of price
    prediction = np.empty(n_trials, dtype=np.float64)
    prediction[0] = prices[0]
    for t in range(1, n_trials):
        prediction[t] = ema_alpha * prices[t - 1] + (1 - ema_alpha) * prediction[t - 1]

    # Prediction error
    pred_error = np.abs(prices - prediction)
    # Z-score
    pe_mean = float(np.mean(pred_error))
    pe_std = float(np.std(pred_error))
    pred_error_z = (pred_error - pe_mean) / pe_std if pe_std > 0 else pred_error - pe_mean

    # Theta power = baseline + coupling * prediction_error + noise
    baseline = 1.0
    noise = rng.normal(0, noise_level, n_trials)
    theta_power = baseline + coupling_strength * pred_error_z + noise
    # Ensure positive (power can't be negative)
    theta_power = np.maximum(theta_power, 0.01)

    return StochasticMarketScenario(
        market_price=prices,
        market_returns=returns,
        neural_prediction=prediction,
        prediction_error=pred_error_z.astype(np.float64),
        theta_power=theta_power.astype(np.float64),
        coupling_strength=coupling_strength,
        n_trials=n_trials,
        has_coupling=coupling_strength > 0,
    )
