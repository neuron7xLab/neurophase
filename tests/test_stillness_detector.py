"""Tests for ``neurophase.gate.stillness_detector`` (invariant I₄).

The required five cases from the spec:

* ``test_still_when_all_criteria_met``
* ``test_active_when_R_changing``
* ``test_active_during_warmup``
* ``test_active_when_delta_too_large``
* ``test_rejects_invalid_inputs``

Plus substantial over-coverage:

* clause-wise isolation (each of the three clauses tested alone)
* boundary / ε-equality behavior
* window-wide vs last-sample differential test (the core design claim)
* dt-scaling correctness
* reset semantics
* frozen-dataclass invariant
* reason-string stability (first-token contract)
* hysteresis (hold_steps) residency lock
* parametrized window-size sweep
* long-horizon stability under noise
* ``free_energy_proxy`` helper validation
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from neurophase.gate.stillness_detector import (
    DEFAULT_DELTA_MIN,
    DEFAULT_DT,
    DEFAULT_EPS_F,
    DEFAULT_EPS_R,
    DEFAULT_WINDOW,
    StillnessDecision,
    StillnessDetector,
    StillnessState,
    free_energy_proxy,
)

# ---------------------------------------------------------------------------
# Required spec tests
# ---------------------------------------------------------------------------


class TestRequiredStillCriterion:
    def test_still_when_all_criteria_met(self) -> None:
        """Constant R above threshold and near-zero δ → STILL."""
        det = StillnessDetector(window=5, eps_R=1e-3, eps_F=1e-3, delta_min=0.05, dt=1.0)
        # Warmup with a clearly stationary signal.
        for _ in range(4):
            out = det.update(R=0.95, delta=0.01)
            assert out.state is StillnessState.ACTIVE  # warmup
        final = det.update(R=0.95, delta=0.01)
        assert final.state is StillnessState.STILL
        assert final.window_filled is True
        assert final.reason.startswith("still:")
        assert final.dR_dt_max is not None and final.dR_dt_max < 1e-3
        assert final.dF_proxy_dt_max is not None and final.dF_proxy_dt_max < 1e-3
        assert final.delta_max is not None and final.delta_max < 0.05


class TestRequiredActiveOnRChange:
    def test_active_when_R_changing(self) -> None:
        """Even tiny δ: if R oscillates, the state must be ACTIVE."""
        det = StillnessDetector(window=5, eps_R=1e-3, eps_F=1e-3, delta_min=0.05, dt=1.0)
        # Prime a bulky R ramp that clearly exceeds eps_R per step.
        for i in range(5):
            det.update(R=0.80 + 0.02 * i, delta=0.01)
        out = det.update(R=0.90, delta=0.01)
        assert out.state is StillnessState.ACTIVE
        assert "R dynamics" in out.reason


class TestRequiredWarmup:
    def test_active_during_warmup(self) -> None:
        det = StillnessDetector(window=8)
        for _i in range(7):
            out = det.update(R=0.95, delta=0.01)
            assert out.state is StillnessState.ACTIVE
            assert out.window_filled is False
            assert out.reason.startswith("warmup:")
            assert out.dR_dt_max is None
            assert out.dF_proxy_dt_max is None
            assert out.delta_max is None
            assert pytest.approx(0.95) == out.R
            assert out.delta == pytest.approx(0.01)
        # Eighth sample fills the buffer.
        eighth = det.update(R=0.95, delta=0.01)
        assert eighth.window_filled is True


class TestRequiredActiveOnLargeDelta:
    def test_active_when_delta_too_large(self) -> None:
        """δ stationary but above δ_min → ACTIVE (third clause)."""
        det = StillnessDetector(window=4, delta_min=0.05)
        for _ in range(3):
            det.update(R=0.95, delta=0.50)
        out = det.update(R=0.95, delta=0.50)
        assert out.state is StillnessState.ACTIVE
        assert "delta exceeds delta_min" in out.reason


class TestRequiredInputValidation:
    @pytest.mark.parametrize(
        "R,delta",
        [
            (float("nan"), 0.0),
            (float("inf"), 0.0),
            (-0.1, 0.0),
            (1.1, 0.0),
            (0.5, float("nan")),
            (0.5, float("-inf")),
            (0.5, -0.1),
            (0.5, math.pi + 1.0),
        ],
    )
    def test_rejects_invalid_inputs(self, R: float, delta: float) -> None:
        det = StillnessDetector()
        with pytest.raises(ValueError):
            det.update(R=R, delta=delta)


# ---------------------------------------------------------------------------
# Clause-wise isolation
# ---------------------------------------------------------------------------


class TestClauseIsolation:
    """Each of the three clauses must be able to veto STILL on its own."""

    def test_only_R_clause_fails(self) -> None:
        det = StillnessDetector(window=4, eps_R=1e-4, eps_F=1.0, delta_min=1.0)
        det.update(R=0.90, delta=0.0)
        det.update(R=0.90, delta=0.0)
        det.update(R=0.90, delta=0.0)
        out = det.update(R=0.91, delta=0.0)  # bumps |dR/dt| to 0.01
        assert out.state is StillnessState.ACTIVE
        assert "R dynamics" in out.reason

    def test_only_F_clause_fails(self) -> None:
        """F_proxy depends on δ·(dδ/dt). If δ moves *and* is nonzero, F_proxy_dt > 0."""
        det = StillnessDetector(window=4, eps_R=1.0, eps_F=1e-4, delta_min=1.0)
        det.update(R=0.90, delta=0.10)
        det.update(R=0.90, delta=0.10)
        det.update(R=0.90, delta=0.10)
        out = det.update(R=0.90, delta=0.15)  # dδ/dt = 0.05, F_dt = 0.15·0.05
        assert out.state is StillnessState.ACTIVE
        assert "free-energy proxy" in out.reason

    def test_only_delta_clause_fails(self) -> None:
        det = StillnessDetector(window=4, eps_R=1.0, eps_F=1.0, delta_min=0.05)
        det.update(R=0.90, delta=0.10)
        det.update(R=0.90, delta=0.10)
        det.update(R=0.90, delta=0.10)
        out = det.update(R=0.90, delta=0.10)  # all constant; only δ>0.05 fires
        assert out.state is StillnessState.ACTIVE
        assert "delta exceeds delta_min" in out.reason


# ---------------------------------------------------------------------------
# Boundary / ε-equality behavior
# ---------------------------------------------------------------------------


class TestBoundaries:
    def test_delta_exactly_at_delta_min_is_active(self) -> None:
        """The spec uses *strict* < in the criterion."""
        det = StillnessDetector(window=3, eps_R=1.0, eps_F=1.0, delta_min=0.05)
        for _ in range(3):
            det.update(R=0.90, delta=0.05)
        out = det.update(R=0.90, delta=0.05)
        assert out.state is StillnessState.ACTIVE

    def test_delta_just_below_delta_min_is_still(self) -> None:
        det = StillnessDetector(window=3, eps_R=1.0, eps_F=1.0, delta_min=0.05)
        for _ in range(3):
            det.update(R=0.90, delta=0.04)
        out = det.update(R=0.90, delta=0.04)
        assert out.state is StillnessState.STILL

    def test_dR_dt_exactly_at_eps_R_is_active(self) -> None:
        det = StillnessDetector(window=3, eps_R=0.01, eps_F=1.0, delta_min=1.0)
        det.update(R=0.90, delta=0.0)
        det.update(R=0.91, delta=0.0)
        out = det.update(R=0.92, delta=0.0)  # max |dR/dt| = 0.01
        assert out.state is StillnessState.ACTIVE


# ---------------------------------------------------------------------------
# The core design claim: window-wide beats last-sample
# ---------------------------------------------------------------------------


class TestWindowWideVsLastSample:
    def test_window_wide_beats_last_sample(self) -> None:
        """A sinusoidal R that crosses zero has |dR/dt|=0 at extrema.

        A naive last-sample criterion would emit STILL at those points;
        the window-wide max correctly rejects them.
        """
        det = StillnessDetector(window=6, eps_R=1e-4, eps_F=1e-4, delta_min=0.05)
        # Build a piecewise ramp that has a last-sample zero derivative
        # at the final step but nonzero derivatives earlier in the window.
        Rs = [0.80, 0.82, 0.84, 0.86, 0.88, 0.88]
        for R in Rs:
            out = det.update(R=R, delta=0.01)
        assert out.state is StillnessState.ACTIVE
        # The last-sample derivative is 0, but the window-wide max is 0.02/dt.
        assert out.dR_dt_max == pytest.approx(0.02, abs=1e-12)


# ---------------------------------------------------------------------------
# Window sliding behavior
# ---------------------------------------------------------------------------


class TestWindowSliding:
    def test_window_slides_out_old_noise(self) -> None:
        """After ``window`` calm samples the detector recovers STILL."""
        det = StillnessDetector(window=4, eps_R=1e-3, eps_F=1e-3, delta_min=0.05)
        # Noisy prime.
        det.update(R=0.60, delta=0.8)
        det.update(R=0.90, delta=0.0)
        det.update(R=0.30, delta=0.5)
        noisy = det.update(R=0.90, delta=0.0)
        assert noisy.state is StillnessState.ACTIVE
        # Now feed 4 calm samples — noisy ones slide out after `window` calls.
        for _ in range(4):
            det.update(R=0.95, delta=0.01)
        calm = det.update(R=0.95, delta=0.01)
        assert calm.state is StillnessState.STILL


# ---------------------------------------------------------------------------
# dt scaling
# ---------------------------------------------------------------------------


class TestDtScaling:
    def test_dt_affects_derivatives(self) -> None:
        """Doubling dt halves the finite-difference derivative estimate."""
        det_a = StillnessDetector(window=3, eps_R=1.0, eps_F=1.0, delta_min=1.0, dt=1.0)
        det_b = StillnessDetector(window=3, eps_R=1.0, eps_F=1.0, delta_min=1.0, dt=2.0)
        for R in (0.80, 0.82, 0.84):
            a = det_a.update(R=R, delta=0.0)
            b = det_b.update(R=R, delta=0.0)
        # Both returned ACTIVE on constant R differences, but b's |dR/dt|
        # is exactly half of a's.
        assert a.dR_dt_max is not None
        assert b.dR_dt_max is not None
        assert b.dR_dt_max == pytest.approx(a.dR_dt_max / 2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# F_proxy chain-rule correctness
# ---------------------------------------------------------------------------


class TestFreeEnergyProxy:
    def test_free_energy_proxy_matches_formula(self) -> None:
        for d in (0.0, 0.1, 0.5, math.pi / 2, math.pi):
            assert free_energy_proxy(d) == pytest.approx(0.5 * d * d)

    def test_free_energy_proxy_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            free_energy_proxy(-0.1)

    def test_free_energy_proxy_rejects_nonfinite(self) -> None:
        with pytest.raises(ValueError):
            free_energy_proxy(float("nan"))

    def test_internal_dF_dt_matches_chain_rule(self) -> None:
        """``dF_proxy/dt`` at the right endpoint equals ``δ · dδ/dt``."""
        det = StillnessDetector(window=3, eps_R=1.0, eps_F=1.0, delta_min=1.0)
        # Two known δ values → one diff = 0.1; at the right endpoint δ = 0.3.
        det.update(R=0.90, delta=0.10)
        det.update(R=0.90, delta=0.20)
        out = det.update(R=0.90, delta=0.30)
        # dδ/dt in the window: [0.10, 0.10] at right endpoints [0.20, 0.30].
        # |dF_proxy/dt| sampled right = max(|0.20*0.10|, |0.30*0.10|) = 0.030.
        assert out.dF_proxy_dt_max == pytest.approx(0.030, rel=1e-12)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"window": 1},
            {"window": 0},
            {"eps_R": 0.0},
            {"eps_R": -1e-3},
            {"eps_F": 0.0},
            {"eps_F": -1e-3},
            {"delta_min": 0.0},
            {"delta_min": -0.01},
            {"dt": 0.0},
            {"dt": -1.0},
            {"hold_steps": -1},
        ],
    )
    def test_rejects_bad_config(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            StillnessDetector(**kwargs)  # type: ignore[arg-type]

    def test_defaults_used(self) -> None:
        det = StillnessDetector()
        assert det.window == DEFAULT_WINDOW
        assert det.eps_R == DEFAULT_EPS_R
        assert det.eps_F == DEFAULT_EPS_F
        assert det.delta_min == DEFAULT_DELTA_MIN
        assert det.dt == DEFAULT_DT


# ---------------------------------------------------------------------------
# Reset + diagnostics
# ---------------------------------------------------------------------------


class TestResetAndDiagnostics:
    def test_reset_restores_warmup(self) -> None:
        det = StillnessDetector(window=4)
        for _ in range(4):
            det.update(R=0.90, delta=0.01)
        assert det.window_filled is True
        det.reset()
        assert det.window_filled is False
        assert det.n_updates == 0
        out = det.update(R=0.90, delta=0.01)
        assert out.reason.startswith("warmup:")

    def test_n_updates_counts_all_calls(self) -> None:
        det = StillnessDetector(window=4)
        for i in range(10):
            det.update(R=0.9, delta=0.0)
            assert det.n_updates == i + 1


# ---------------------------------------------------------------------------
# Reason-string contract
# ---------------------------------------------------------------------------


class TestReasonStrings:
    """First token of the reason string is a stable parseable tag."""

    @pytest.mark.parametrize("warmup_i", [0, 1, 2])
    def test_warmup_tag(self, warmup_i: int) -> None:
        det = StillnessDetector(window=4)
        for _ in range(warmup_i + 1):
            out = det.update(R=0.9, delta=0.0)
        assert out.reason.startswith("warmup:")

    def test_still_tag(self) -> None:
        det = StillnessDetector(window=3, eps_R=1.0, eps_F=1.0, delta_min=1.0)
        for _ in range(3):
            det.update(R=0.9, delta=0.0)
        out = det.update(R=0.9, delta=0.0)
        assert out.reason.startswith("still:")

    def test_active_tag(self) -> None:
        det = StillnessDetector(window=3, eps_R=1e-6, eps_F=1.0, delta_min=1.0)
        for R in (0.80, 0.82, 0.84):
            out = det.update(R=R, delta=0.0)
        assert out.reason.startswith("active:")


# ---------------------------------------------------------------------------
# Frozen-dataclass invariant
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_stillness_decision_frozen(self) -> None:
        d = StillnessDecision(
            state=StillnessState.STILL,
            R=0.9,
            delta=0.01,
            dR_dt_max=1e-4,
            dF_proxy_dt_max=1e-4,
            delta_max=0.01,
            window_filled=True,
            reason="still: test",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.state = StillnessState.ACTIVE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Hysteresis (optional hold_steps)
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_hold_prevents_immediate_flicker(self) -> None:
        """Once STILL is entered with ``hold_steps=3``, three consecutive
        attempts to leave must be suppressed — unless the criterion still
        holds, which obviously keeps the state anyway."""
        det = StillnessDetector(
            window=3,
            eps_R=1e-4,
            eps_F=1e-4,
            delta_min=0.05,
            dt=1.0,
            hold_steps=3,
        )
        # Warmup + STILL.
        for _ in range(3):
            det.update(R=0.90, delta=0.01)
        first = det.update(R=0.90, delta=0.01)
        assert first.state is StillnessState.STILL
        # Attempt to leave with a big R jump — hysteresis should veto.
        held = det.update(R=0.95, delta=0.01)
        assert held.state is StillnessState.STILL
        assert "held" in held.reason

    def test_hold_zero_is_no_op(self) -> None:
        det = StillnessDetector(window=2, hold_steps=0)
        det.update(R=0.90, delta=0.0)
        det.update(R=0.90, delta=0.0)  # window filled
        out = det.update(R=0.95, delta=0.0)
        assert "held" not in out.reason


# ---------------------------------------------------------------------------
# Parametrized window sweep + long-horizon stability
# ---------------------------------------------------------------------------


class TestStability:
    @pytest.mark.parametrize("window", [2, 4, 8, 16, 32])
    def test_constant_signal_converges_to_still(self, window: int) -> None:
        det = StillnessDetector(window=window, eps_R=1e-3, eps_F=1e-3, delta_min=0.05)
        for _ in range(window):
            det.update(R=0.95, delta=0.01)
        out = det.update(R=0.95, delta=0.01)
        assert out.state is StillnessState.STILL

    def test_long_horizon_with_noise_respects_eps(self) -> None:
        """500 updates with small Gaussian noise → STILL only when criterion holds."""
        rng = np.random.default_rng(13)
        det = StillnessDetector(window=16, eps_R=1e-2, eps_F=1e-2, delta_min=0.05)
        hits = 0
        total_after_warmup = 0
        for i in range(500):
            R = float(np.clip(0.95 + rng.normal(0, 1e-4), 0.0, 1.0))
            delta = float(np.clip(0.01 + abs(rng.normal(0, 1e-4)), 0.0, math.pi))
            out = det.update(R=R, delta=delta)
            if i >= 16:
                total_after_warmup += 1
                if out.state is StillnessState.STILL:
                    hits += 1
        # Under micro-noise the criterion should hold the vast majority of
        # the time — this is a sanity check on numerical robustness.
        assert hits / total_after_warmup > 0.95
