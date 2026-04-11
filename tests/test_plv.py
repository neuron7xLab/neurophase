"""Tests for neurophase.metrics.plv."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.metrics.plv import plv, plv_significance, rolling_plv


def test_identical_phases() -> None:
    phi = np.linspace(0.0, 4 * np.pi, 500)
    assert abs(plv(phi, phi) - 1.0) < 1e-10


def test_constant_offset_is_fully_locked() -> None:
    """A fixed π/2 offset gives perfect phase locking."""
    phi = np.linspace(0.0, 4 * np.pi, 500)
    assert abs(plv(phi, phi + np.pi / 2) - 1.0) < 1e-10


def test_random_phases_near_zero() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(-np.pi, np.pi, 5000)
    y = rng.uniform(-np.pi, np.pi, 5000)
    assert plv(x, y) < 0.05


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same shape"):
        plv(np.zeros(10), np.zeros(11))


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        plv(np.array([]), np.array([]))


def test_significance_detects_locking() -> None:
    t = np.linspace(0.0, 10.0, 2000)
    phi_x = 2 * np.pi * 1.0 * t
    phi_y = 2 * np.pi * 1.0 * t + 0.3
    result = plv_significance(phi_x, phi_y, n_surrogates=200, seed=42)
    assert result.significant
    assert result.plv > 0.9


def test_significance_rejects_noise() -> None:
    rng = np.random.default_rng(99)
    phi_x = rng.uniform(-np.pi, np.pi, 2000)
    phi_y = rng.uniform(-np.pi, np.pi, 2000)
    result = plv_significance(phi_x, phi_y, n_surrogates=200, seed=42)
    assert not result.significant


def test_rolling_plv_shape() -> None:
    phi = np.linspace(0.0, 4 * np.pi, 200)
    out = rolling_plv(phi, phi, window=50)
    assert out.shape == (200 - 50 + 1,)
    assert np.allclose(out, 1.0)


def test_rolling_plv_window_too_large() -> None:
    phi = np.zeros(10)
    with pytest.raises(ValueError, match="larger than series"):
        rolling_plv(phi, phi, window=20)
