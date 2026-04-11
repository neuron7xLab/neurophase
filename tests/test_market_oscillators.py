"""Tests for neurophase.oscillators.market."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.oscillators.market import MarketOscillators, extract_market_phase


def _ramp(n: int = 256) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 10.0, n)
    # Positive ramp + sinusoidal oscillation for a rich phase signal.
    prices = 100.0 + 20.0 * t + 5.0 * np.sin(2 * np.pi * 0.5 * t)
    volumes = 1000.0 + 200.0 * np.abs(np.cos(2 * np.pi * 0.3 * t))
    return prices.astype(np.float64), volumes.astype(np.float64)


def test_extract_returns_three_channels() -> None:
    p, v = _ramp()
    out = extract_market_phase(p, v)
    assert isinstance(out, MarketOscillators)
    assert out.phi_price.shape == p.shape
    assert out.phi_volume.shape == p.shape
    assert out.phi_volatility.shape == p.shape


def test_stack_shape() -> None:
    p, v = _ramp(128)
    out = extract_market_phase(p, v)
    stacked = out.stack()
    assert stacked.shape == (3, 128)


def test_phases_in_range() -> None:
    p, v = _ramp()
    out = extract_market_phase(p, v)
    for arr in (out.phi_price, out.phi_volume, out.phi_volatility):
        assert np.all(arr >= -np.pi - 1e-6)
        assert np.all(arr <= np.pi + 1e-6)


def test_rejects_non_positive_prices() -> None:
    p = np.array([1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    v = np.ones_like(p)
    with pytest.raises(ValueError, match="prices must be strictly positive"):
        extract_market_phase(p, v)


def test_rejects_negative_volumes() -> None:
    p = np.linspace(100.0, 110.0, 16)
    v = np.ones_like(p)
    v[5] = -1.0
    with pytest.raises(ValueError, match="volumes must be non-negative"):
        extract_market_phase(p, v)


def test_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        extract_market_phase(np.ones(10), np.ones(12))


def test_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        extract_market_phase(np.ones(4), np.ones(4))


def test_rejects_bad_volatility_window() -> None:
    p, v = _ramp()
    with pytest.raises(ValueError, match="volatility_window"):
        extract_market_phase(p, v, volatility_window=0)
