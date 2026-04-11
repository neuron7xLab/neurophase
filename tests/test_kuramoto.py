"""Tests for neurophase.core.kuramoto."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.core.kuramoto import KuramotoNetwork
from neurophase.core.order_parameter import order_parameter


def test_synchronizes_at_high_coupling() -> None:
    """Strong coupling drives R → 1 after transient."""
    omega = np.linspace(-0.5, 0.5, 20)
    net = KuramotoNetwork(omega, coupling=5.0, dt=0.05, seed=0)
    trajectory = net.run(n_steps=500)
    final = order_parameter(trajectory[-1])
    assert isinstance(final.R, float)
    assert final.R > 0.85


def test_stays_incoherent_at_zero_coupling() -> None:
    """Zero coupling keeps R low for broadly distributed frequencies."""
    omega = np.linspace(-2.0, 2.0, 20)
    net = KuramotoNetwork(omega, coupling=0.0, dt=0.05, seed=0)
    trajectory = net.run(n_steps=300)
    R_vals = order_parameter(trajectory).R
    assert isinstance(R_vals, np.ndarray)
    assert float(np.mean(R_vals[-100:])) < 0.5


def test_trajectory_shape() -> None:
    omega = np.linspace(-0.2, 0.2, 8)
    net = KuramotoNetwork(omega, coupling=1.5, dt=0.05, seed=1)
    traj = net.run(n_steps=120)
    assert traj.shape == (120, 8)


def test_rejects_single_oscillator() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        KuramotoNetwork(np.array([0.1]))


def test_rejects_negative_coupling() -> None:
    with pytest.raises(ValueError, match="coupling must be non-negative"):
        KuramotoNetwork(np.array([0.1, 0.2]), coupling=-1.0)


def test_rejects_bad_delays_diagonal() -> None:
    delays = np.ones((4, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="diagonal must be zero"):
        KuramotoNetwork(np.linspace(-0.1, 0.1, 4), delays=delays)


def test_delayed_coupling_runs() -> None:
    """Network with small uniform delays still integrates and produces valid R(t)."""
    N = 8
    omega = np.linspace(-0.3, 0.3, N)
    delays = np.full((N, N), 2, dtype=np.int64)
    np.fill_diagonal(delays, 0)
    net = KuramotoNetwork(omega, coupling=3.0, dt=0.05, delays=delays, seed=2)
    traj = net.run(n_steps=200)
    assert traj.shape == (200, N)
    final = order_parameter(traj[-1])
    assert isinstance(final.R, float)
    assert 0.0 <= final.R <= 1.0


def test_noise_produces_distinct_runs() -> None:
    """Noisy integration with different seeds should produce distinct trajectories."""
    omega = np.linspace(-0.1, 0.1, 6)
    run_a = KuramotoNetwork(omega, coupling=1.2, dt=0.05, noise_sigma=0.3, seed=100).run(
        n_steps=100
    )
    run_b = KuramotoNetwork(omega, coupling=1.2, dt=0.05, noise_sigma=0.3, seed=200).run(
        n_steps=100
    )
    assert not np.allclose(run_a, run_b)


def test_liquidity_fn_rejects_negative_multiplier() -> None:
    omega = np.linspace(-0.1, 0.1, 4)
    net = KuramotoNetwork(omega, coupling=1.0, dt=0.05, seed=0)
    # Inject a negative liquidity factor directly via the frozen dataclass.
    object.__setattr__(net.params, "liquidity_fn", lambda step, state: -0.5)
    with pytest.raises(ValueError, match="non-negative multiplier"):
        net.run(n_steps=2)
