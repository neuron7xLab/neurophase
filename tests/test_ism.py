"""Tests for neurophase.metrics.ism."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.metrics.ism import compute_ism, compute_topological_energy, ism_derivative


def test_topological_energy_nonnegative() -> None:
    ricci = np.array([-0.3, 0.1, -0.05, 0.2, -0.1])
    assert compute_topological_energy(ricci, window=3) >= 0.0


def test_topological_energy_empty_is_zero() -> None:
    assert compute_topological_energy(np.array([]), window=10) == 0.0


def test_compute_ism_basic() -> None:
    entropy = np.array([1.0, 1.1, 1.3, 1.6, 2.0])
    ricci = np.array([-0.1, -0.12, -0.09, -0.1, -0.11])
    value = compute_ism(entropy, ricci, window=5, eta=1.0, dt=1.0)
    assert value > 0.0


def test_compute_ism_zero_energy_returns_zero() -> None:
    """If curvature is zero → ISM collapses to 0 (honest null, not NaN)."""
    entropy = np.array([1.0, 1.2, 1.5, 1.9])
    ricci = np.zeros(10)
    assert compute_ism(entropy, ricci) == 0.0


def test_compute_ism_too_short_entropy() -> None:
    assert compute_ism(np.array([1.0]), np.array([-0.1, -0.2])) == 0.0


def test_ism_derivative_monotonic_input() -> None:
    ism = np.linspace(0.5, 1.5, 10)
    d = ism_derivative(ism, dt=1.0)
    assert d > 0.0


def test_ism_derivative_short_input_zero() -> None:
    assert ism_derivative(np.array([0.5]), dt=1.0) == 0.0


def test_compute_ism_rejects_bad_params() -> None:
    entropy = np.arange(10, dtype=np.float64)
    ricci = np.arange(10, dtype=np.float64)
    with pytest.raises(ValueError, match="dt must be positive"):
        compute_ism(entropy, ricci, dt=0.0)
    with pytest.raises(ValueError, match="eta must be non-negative"):
        compute_ism(entropy, ricci, eta=-1.0)
