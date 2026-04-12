"""Tests for Level 2 stochastic market synthetic validation."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr

from neurophase.benchmarks.stochastic_market_sim import (
    generate_stochastic_market,
    generate_stochastic_scenario,
)


class TestStochasticMarket:
    def test_gbm_positive_prices(self) -> None:
        prices, _ = generate_stochastic_market(n_trials=1000, seed=42)
        assert np.all(prices > 0)

    def test_returns_shape(self) -> None:
        prices, returns = generate_stochastic_market(n_trials=500, seed=42)
        assert prices.shape == (500,)
        assert returns.shape == (500,)

    def test_deterministic(self) -> None:
        p1, _ = generate_stochastic_market(seed=42)
        p2, _ = generate_stochastic_market(seed=42)
        np.testing.assert_array_equal(p1, p2)


class TestStochasticScenario:
    def test_null_coupling_no_correlation(self) -> None:
        """k=0 → θ-power independent of prediction error."""
        sc = generate_stochastic_scenario(coupling_strength=0.0, seed=42)
        r, p = pearsonr(sc.theta_power, sc.prediction_error)
        assert abs(r) < 0.15, f"k=0: r={r} should be near 0"
        assert not sc.has_coupling

    def test_strong_coupling_significant(self) -> None:
        """k=2 → θ-power correlates with prediction error."""
        sc = generate_stochastic_scenario(coupling_strength=2.0, seed=42)
        r, p = pearsonr(sc.theta_power, sc.prediction_error)
        assert r > 0.5, f"k=2: r={r} should be > 0.5"
        assert p < 0.001
        assert sc.has_coupling

    def test_trial_level_method_detects_coupling(self) -> None:
        """Trial-level correlation (Toma method) detects the coupling."""
        from neurophase.metrics.trial_theta_lme import (
            compute_trial_theta_reward_correlation,
        )

        sc = generate_stochastic_scenario(coupling_strength=2.0, seed=42)
        result = compute_trial_theta_reward_correlation(
            sc.theta_power, sc.prediction_error,
            subject_id="synthetic", channel="FC5",
        )
        assert result.r_pearson > 0.5
        assert result.p_pearson < 0.001

    def test_trial_correlation_distinguishes_null_from_coupled(self) -> None:
        """Trial-level r distinguishes k=0 (null) from k=2 (coupled)."""
        sc_null = generate_stochastic_scenario(coupling_strength=0.0, seed=42)
        sc_coupled = generate_stochastic_scenario(coupling_strength=2.0, seed=42)

        r_null, _ = pearsonr(sc_null.theta_power, sc_null.prediction_error)
        r_coupled, _ = pearsonr(sc_coupled.theta_power, sc_coupled.prediction_error)

        assert abs(r_null) < 0.15, f"k=0: r={r_null} should be near 0"
        assert r_coupled > 0.50, f"k=2: r={r_coupled} should be > 0.5"
        assert r_coupled > r_null + 0.3, "Coupling should be clearly stronger"

    def test_shapes(self) -> None:
        sc = generate_stochastic_scenario(n_trials=200, seed=42)
        assert sc.market_price.shape == (200,)
        assert sc.theta_power.shape == (200,)
        assert sc.prediction_error.shape == (200,)
        assert sc.n_trials == 200

    def test_coupling_monotonic(self) -> None:
        """Stronger coupling → higher correlation."""
        rs: list[float] = []
        for k in [0.0, 0.5, 1.0, 2.0, 5.0]:
            sc = generate_stochastic_scenario(coupling_strength=k, seed=42)
            r, _ = pearsonr(sc.theta_power, sc.prediction_error)
            rs.append(r)
        for i in range(len(rs) - 1):
            assert rs[i] <= rs[i + 1] + 0.05, f"Not monotonic at k={i}"
