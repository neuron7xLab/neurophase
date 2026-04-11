"""Tests for kuramoto-trader core modules."""

from __future__ import annotations

import numpy as np
import pytest

from kuramoto_trader.sync.order_param import order_parameter
from kuramoto_trader.sync.plv import plv, plv_significance, rolling_plv
from kuramoto_trader.sync.kuramoto import KuramotoNetwork
from kuramoto_trader.gate.execution_gate import ExecutionGate, GateState
from kuramoto_trader.analysis.falsification import (
    FalsificationVerdict,
    run_falsification,
)


# ─────────────────────────────────────────────
# Order parameter
# ─────────────────────────────────────────────

def test_order_parameter_fully_synchronized() -> None:
    """All phases equal → R = 1."""
    theta = np.full(10, 0.5)
    result = order_parameter(theta)
    assert abs(result.R - 1.0) < 1e-10


def test_order_parameter_fully_incoherent() -> None:
    """Uniformly distributed phases → R ≈ 0."""
    theta = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
    result = order_parameter(theta)
    assert result.R < 0.05


def test_order_parameter_2d_trajectory() -> None:
    """2-D input returns arrays of length T."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, size=(50, 8))
    result = order_parameter(theta)
    assert len(result.R) == 50  # type: ignore[arg-type]
    assert np.all((result.R >= 0) & (result.R <= 1))  # type: ignore[operator]


# ─────────────────────────────────────────────
# PLV
# ─────────────────────────────────────────────

def test_plv_identical_phases() -> None:
    """Identical phases → PLV = 1."""
    phi = np.linspace(0, 4 * np.pi, 500)
    assert abs(plv(phi, phi) - 1.0) < 1e-10


def test_plv_orthogonal_phases() -> None:
    """Constant π/2 offset → PLV = 1 (fixed phase difference)."""
    phi = np.linspace(0, 4 * np.pi, 500)
    assert abs(plv(phi, phi + np.pi / 2) - 1.0) < 1e-10


def test_plv_random_phases_near_zero() -> None:
    """Independent random phases → PLV ≈ 0."""
    rng = np.random.default_rng(7)
    phi_x = rng.uniform(-np.pi, np.pi, 5000)
    phi_y = rng.uniform(-np.pi, np.pi, 5000)
    assert plv(phi_x, phi_y) < 0.05


def test_plv_significance_detects_locking() -> None:
    """Sinusoidal phase-locked signals → significant PLV."""
    t = np.linspace(0, 10, 2000)
    phi_x = 2 * np.pi * 1.0 * t
    phi_y = 2 * np.pi * 1.0 * t + 0.3
    result = plv_significance(phi_x, phi_y, n_surrogates=200, seed=42)
    assert result.significant
    assert result.plv > 0.9


def test_rolling_plv_shape() -> None:
    phi = np.linspace(0, 4 * np.pi, 200)
    window = 50
    result = rolling_plv(phi, phi, window=window)
    assert len(result) == 200 - window + 1


# ─────────────────────────────────────────────
# Kuramoto network
# ─────────────────────────────────────────────

def test_kuramoto_synchronizes_at_high_coupling() -> None:
    """High coupling → R → 1 after transient."""
    omega = np.linspace(-0.5, 0.5, 20)
    net = KuramotoNetwork(omega, coupling=5.0, dt=0.05, seed=0)
    trajectory = net.run(n_steps=500)
    R_final = order_parameter(trajectory[-1]).R
    assert R_final > 0.90


def test_kuramoto_stays_incoherent_at_zero_coupling() -> None:
    """Zero coupling → R stays low."""
    omega = np.linspace(-2.0, 2.0, 20)
    net = KuramotoNetwork(omega, coupling=0.0, dt=0.05, seed=0)
    trajectory = net.run(n_steps=200)
    R_final = order_parameter(trajectory[-1]).R
    assert R_final < 0.4


# ─────────────────────────────────────────────
# Execution gate
# ─────────────────────────────────────────────

def test_gate_blocks_below_threshold() -> None:
    gate = ExecutionGate(threshold=0.65)
    decision = gate.evaluate(R=0.50)
    assert decision.state == GateState.BLOCKED
    assert not decision.execution_allowed


def test_gate_permits_above_threshold() -> None:
    gate = ExecutionGate(threshold=0.65)
    decision = gate.evaluate(R=0.80)
    assert decision.state == GateState.READY
    assert decision.execution_allowed


def test_gate_sensor_absent() -> None:
    gate = ExecutionGate()
    decision = gate.evaluate(R=0.99, sensor_present=False)
    assert decision.state == GateState.SENSOR_ABSENT
    assert not decision.execution_allowed


def test_gate_degraded_on_nan() -> None:
    gate = ExecutionGate()
    decision = gate.evaluate(R=float("nan"))
    assert decision.state == GateState.DEGRADED
    assert not decision.execution_allowed


def test_gate_invariant_cannot_be_bypassed() -> None:
    """execution_allowed=True with non-READY state must raise."""
    from kuramoto_trader.gate.execution_gate import GateDecision
    with pytest.raises(ValueError, match="Invariant violated"):
        GateDecision(
            state=GateState.BLOCKED,
            execution_allowed=True,
            R=0.5,
            threshold=0.65,
            reason="bypass attempt",
        )


# ─────────────────────────────────────────────
# Falsification
# ─────────────────────────────────────────────

def test_falsification_confirms_locked_signal() -> None:
    """Phase-locked signal → CONFIRMED."""
    T = 2000
    t = np.linspace(0, 20, T)
    market = 2 * np.pi * 1.0 * t
    neural = 2 * np.pi * 1.0 * t + 0.5
    result = run_falsification(market, neural, n_surrogates=200, seed=42)
    assert result.verdict == FalsificationVerdict.CONFIRMED


def test_falsification_rejects_random_signal() -> None:
    """Independent random phases → REJECTED."""
    rng = np.random.default_rng(99)
    T = 2000
    market = rng.uniform(-np.pi, np.pi, T)
    neural = rng.uniform(-np.pi, np.pi, T)
    result = run_falsification(market, neural, n_surrogates=200, seed=42)
    assert result.verdict == FalsificationVerdict.REJECTED


def test_falsification_insufficient_data() -> None:
    market = np.zeros(50)
    neural = np.zeros(50)
    result = run_falsification(market, neural, min_test_samples=100)
    assert result.verdict == FalsificationVerdict.INSUFFICIENT_DATA
