"""Tests for neurophase.core.order_parameter."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.core.order_parameter import order_parameter


def test_fully_synchronized() -> None:
    """All phases equal → R = 1."""
    theta = np.full(10, 0.5)
    result = order_parameter(theta)
    assert isinstance(result.R, float)
    assert abs(result.R - 1.0) < 1e-10


def test_fully_incoherent() -> None:
    """Uniform phases → R ≈ 0."""
    theta = np.linspace(0.0, 2 * np.pi, 1000, endpoint=False)
    result = order_parameter(theta)
    assert isinstance(result.R, float)
    assert result.R < 0.01


def test_two_phase_result_types() -> None:
    """2-D input returns arrays of length T."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, size=(50, 8))
    result = order_parameter(theta)
    assert isinstance(result.R, np.ndarray)
    assert isinstance(result.psi, np.ndarray)
    assert result.R.shape == (50,)
    assert np.all((result.R >= 0.0) & (result.R <= 1.0))


def test_mean_phase_direction() -> None:
    """Mean phase Ψ should match the common phase for aligned oscillators."""
    theta = np.full(16, 1.2345)
    result = order_parameter(theta)
    assert isinstance(result.psi, float)
    assert abs(result.psi - 1.2345) < 1e-10


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        order_parameter(np.array([]))


def test_3d_raises() -> None:
    with pytest.raises(ValueError, match="1-D or 2-D"):
        order_parameter(np.zeros((2, 3, 4)))
