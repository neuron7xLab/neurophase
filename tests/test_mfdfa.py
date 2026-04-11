"""Tests for neurophase.risk.mfdfa."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.risk.mfdfa import mfdfa, multifractal_instability


def test_mfdfa_returns_h_for_each_q() -> None:
    rng = np.random.default_rng(0)
    ts = rng.standard_normal(2048)
    result = mfdfa(ts, min_lag=8, max_lag=150)
    assert result.q_values.shape == result.h_q.shape
    assert result.q_values.size >= 4


def test_monofractal_has_small_instability() -> None:
    """White noise is approximately monofractal → narrow h(q) spectrum."""
    rng = np.random.default_rng(1)
    ts = rng.standard_normal(4096)
    inst = multifractal_instability(ts, min_lag=8, max_lag=200)
    # Tight bound — sampling noise widens the spectrum a bit.
    assert 0.0 <= inst < 0.35


def test_multifractal_series_has_wider_spectrum() -> None:
    """A multiplicative cascade is strongly multifractal — wider than white noise."""
    rng = np.random.default_rng(2)
    n = 2**12
    # Simple binomial cascade: multiply random ±1 weights in a tree.
    series = np.ones(n, dtype=np.float64)
    depth = 10
    for _ in range(depth):
        weights = rng.choice([0.4, 0.6], size=n)
        series = series * weights
    # Integrate to make a visible scaling series.
    walk = np.cumsum(series - float(np.mean(series)))
    inst_wn = multifractal_instability(rng.standard_normal(n), min_lag=8, max_lag=200)
    inst_mf = multifractal_instability(walk, min_lag=8, max_lag=200)
    assert inst_mf > inst_wn


def test_rejects_q_zero() -> None:
    rng = np.random.default_rng(3)
    ts = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="q = 0"):
        mfdfa(ts, q_values=np.array([-1.0, 0.0, 1.0]), min_lag=8, max_lag=100)


def test_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="at least"):
        mfdfa(np.zeros(100), min_lag=8, max_lag=200)


def test_rejects_bad_lag_range() -> None:
    rng = np.random.default_rng(4)
    ts = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="max_lag must be"):
        mfdfa(ts, min_lag=50, max_lag=40)


def test_rejects_min_lag_too_small() -> None:
    rng = np.random.default_rng(5)
    ts = rng.standard_normal(1000)
    with pytest.raises(ValueError, match="min_lag must be >= 4"):
        mfdfa(ts, min_lag=2, max_lag=100)
