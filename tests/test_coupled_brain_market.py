"""Tests for ``neurophase.sync.coupled_brain_market``.

Each required invariant from the R&D report is covered:

* shared ``R(t)`` over brain ∪ market
* synchronization onset at high coupling ``K``
* gate blocking semantics when ``R < threshold``
* numerical fidelity of the RK4 integrator to equation 8.1
* delay reduces synchronization at fixed ``K``
* noise amplitude keeps the output bounded
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from neurophase.sync.coupled_brain_market import (
    CoupledBrainMarketSystem,
    CoupledStep,
)

# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_constructor(self) -> None:
        sys = CoupledBrainMarketSystem()
        assert sys.n_brain == 3
        assert sys.n_market == 3
        assert sys.N == 6
        assert 0.0 <= sys.R <= 1.0
        assert sys.t == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_brain": 0},
            {"n_market": 0},
            {"K": -0.1},
            {"tau": -1.0},
            {"sigma": -0.01},
            {"dt": 0.0},
            {"dt": -1.0},
            {"threshold": 0.0},
            {"threshold": 1.0},
        ],
    )
    def test_invalid_params_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            CoupledBrainMarketSystem(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# REQUIRED: test_R_is_shared_between_brain_and_market
# ---------------------------------------------------------------------------


class TestSharedOrderParameter:
    def test_R_is_shared_between_brain_and_market(self) -> None:
        """The order parameter is computed over the joint brain ∪ market
        population — not per sub-population. The value returned by ``step``
        must match the direct joint computation.
        """
        sys = CoupledBrainMarketSystem(seed=1)
        R_step, _, _ = sys.step()
        R_joint = float(np.abs(np.mean(np.exp(1j * sys.theta))))
        assert math.isclose(R_step, R_joint, rel_tol=1e-12, abs_tol=1e-12)
        # And it is NOT equal to either sub-population R alone (with prob 1).
        R_brain_only = float(np.abs(np.mean(np.exp(1j * sys.theta[: sys.n_brain]))))
        R_market_only = float(np.abs(np.mean(np.exp(1j * sys.theta[sys.n_brain :]))))
        assert not math.isclose(R_step, R_brain_only, abs_tol=1e-9)
        assert not math.isclose(R_step, R_market_only, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# REQUIRED: test_synchronizes_at_high_K
# ---------------------------------------------------------------------------


class TestSynchronization:
    def test_synchronizes_at_high_K(self) -> None:
        """At large coupling and zero noise, the joint order parameter
        should climb close to 1.0 within a few hundred steps."""
        sys = CoupledBrainMarketSystem(K=10.0, sigma=0.0, dt=0.01, seed=3)
        df = sys.run(n_steps=500)
        final_R = float(df["R"].iloc[-500:].mean())
        assert final_R > 0.9

    def test_low_K_stays_desynchronized(self) -> None:
        """At zero coupling, the system cannot synchronize above a loose
        chance floor — the populations simply drift at their own frequencies."""
        sys = CoupledBrainMarketSystem(K=0.0, sigma=0.0, dt=0.01, seed=5)
        df = sys.run(n_steps=500)
        # The time-averaged R should be far from 1.0.
        mean_R = float(df["R"].mean())
        assert mean_R < 0.9


# ---------------------------------------------------------------------------
# REQUIRED: test_gate_blocks_when_R_below_threshold
# ---------------------------------------------------------------------------


class TestGateSemantics:
    def test_gate_blocks_when_R_below_threshold(self) -> None:
        sys = CoupledBrainMarketSystem(K=0.0, sigma=0.0, threshold=0.9, seed=7)
        df = sys.run(n_steps=300)
        # Whenever R < 0.9 the execution_allowed flag MUST be False.
        blocked = df[df["R"] < 0.9]
        assert not blocked.empty, "Need at least some sub-threshold samples"
        assert (blocked["execution_allowed"] == False).all()  # noqa: E712

    def test_gate_allows_when_R_at_or_above_threshold(self) -> None:
        sys = CoupledBrainMarketSystem(K=20.0, sigma=0.0, threshold=0.65, seed=9)
        df = sys.run(n_steps=500)
        ready = df[df["R"] >= 0.65]
        assert not ready.empty
        assert (ready["execution_allowed"] == True).all()  # noqa: E712


# ---------------------------------------------------------------------------
# REQUIRED: test_equations_match_8_1_numerically (RK4 verification)
# ---------------------------------------------------------------------------


class TestRK4Numerics:
    def test_equations_match_8_1_numerically(self) -> None:
        """One analytic RK4 step on the deterministic drift must match the
        implementation to machine precision.

        We disable noise to keep the step deterministic.
        """
        sys = CoupledBrainMarketSystem(
            n_brain=2,
            n_market=2,
            K=1.5,
            tau=0.0,
            sigma=0.0,
            dt=0.005,
            seed=11,
        )

        theta0 = sys.theta.copy()
        omega = sys.omega.copy()
        K = sys.K
        dt = sys.dt
        n_brain = sys.n_brain

        def drift(theta: np.ndarray) -> np.ndarray:
            R = float(np.abs(np.mean(np.exp(1j * theta))))
            psi_b = float(np.angle(np.mean(np.exp(1j * theta[:n_brain]))))
            psi_m = float(np.angle(np.mean(np.exp(1j * theta[n_brain:]))))
            d = np.empty_like(theta)
            d[:n_brain] = omega[:n_brain] + K * R * np.sin(psi_m - theta[:n_brain])
            d[n_brain:] = omega[n_brain:] + K * R * np.sin(psi_b - theta[n_brain:])
            return d

        k1 = drift(theta0)
        k2 = drift(theta0 + 0.5 * dt * k1)
        k3 = drift(theta0 + 0.5 * dt * k2)
        k4 = drift(theta0 + dt * k3)
        expected_next = theta0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        expected_next_wrapped = np.mod(expected_next + np.pi, 2 * np.pi) - np.pi

        sys.step()
        np.testing.assert_allclose(sys.theta, expected_next_wrapped, atol=1e-12)

    def test_zero_coupling_preserves_phase_linear_drift(self) -> None:
        """With K=0 and σ=0 each oscillator advances by exactly ω·dt per step,
        which is the defining property of the decoupled limit of eq. 8.1."""
        sys = CoupledBrainMarketSystem(n_brain=1, n_market=1, K=0.0, sigma=0.0, dt=0.01, seed=13)
        theta0 = sys.theta.copy()
        omega = sys.omega.copy()
        sys.step()
        # RK4 of a constant ODE θ' = ω is exact.
        expected = theta0 + omega * sys.dt
        expected_wrapped = np.mod(expected + np.pi, 2 * np.pi) - np.pi
        np.testing.assert_allclose(sys.theta, expected_wrapped, atol=1e-13)


# ---------------------------------------------------------------------------
# REQUIRED: test_delay_reduces_synchronization
# ---------------------------------------------------------------------------


class TestDelay:
    def test_delay_reduces_synchronization(self) -> None:
        """Adding a positive propagation delay must NOT increase the
        time-averaged joint synchronization at fixed ``K`` and noise."""
        no_delay = CoupledBrainMarketSystem(K=5.0, tau=0.0, sigma=0.0, dt=0.01, seed=17)
        delayed = CoupledBrainMarketSystem(K=5.0, tau=0.2, sigma=0.0, dt=0.01, seed=17)
        df_a = no_delay.run(n_steps=600)
        df_b = delayed.run(n_steps=600)

        mean_no_delay = float(df_a["R"].iloc[-200:].mean())
        mean_delayed = float(df_b["R"].iloc[-200:].mean())
        # Strict ≤ with a tiny slack for the noise-free case.
        assert mean_delayed <= mean_no_delay + 1e-3


# ---------------------------------------------------------------------------
# REQUIRED: test_noise_sigma_bounded_output
# ---------------------------------------------------------------------------


class TestNoiseBounds:
    def test_noise_sigma_bounded_output(self) -> None:
        """For any finite sigma, R(t) must stay in [0, 1] and phases stay in
        (-π, π]. No NaN, no overflow."""
        sys = CoupledBrainMarketSystem(K=2.0, sigma=0.5, dt=0.01, seed=19)
        df = sys.run(n_steps=400)
        assert df["R"].between(0.0, 1.0).all()
        assert (df["psi_brain"].abs() <= math.pi + 1e-9).all()
        assert (df["psi_market"].abs() <= math.pi + 1e-9).all()
        assert not df.isna().any().any()

    def test_zero_noise_is_deterministic(self) -> None:
        """Two instances with identical seed, zero noise, and identical
        params must produce identical trajectories."""
        a = CoupledBrainMarketSystem(K=3.0, sigma=0.0, seed=23).run(100)
        b = CoupledBrainMarketSystem(K=3.0, sigma=0.0, seed=23).run(100)
        pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class TestRunSchema:
    def test_run_schema(self) -> None:
        sys = CoupledBrainMarketSystem(seed=29)
        df = sys.run(n_steps=5)
        assert list(df.columns) == [
            "t",
            "R",
            "psi_brain",
            "psi_market",
            "execution_allowed",
        ]
        assert len(df) == 5
        assert df["execution_allowed"].dtype == bool

    def test_zero_steps_returns_empty_frame(self) -> None:
        sys = CoupledBrainMarketSystem(seed=31)
        df = sys.run(n_steps=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_coupled_step_dataclass_is_frozen(self) -> None:
        step = CoupledStep(t=0.1, R=0.5, psi_brain=0.0, psi_market=0.1, execution_allowed=False)
        with pytest.raises(dataclasses.FrozenInstanceError):
            step.R = 0.9  # type: ignore[misc]
