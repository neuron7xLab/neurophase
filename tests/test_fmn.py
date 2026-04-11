"""Tests for neurophase.indicators.fmn."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.indicators.fmn import compute_fmn


def test_bounded_in_open_interval() -> None:
    """FMN is the output of tanh — strictly in (-1, 1)."""
    rng = np.random.default_rng(0)
    n = 200
    result = compute_fmn(
        delta_vol=rng.normal(0, 100, n),
        bid_vol=rng.uniform(500, 1500, n),
        ask_vol=rng.uniform(500, 1500, n),
    )
    assert np.all(np.abs(result) < 1.0)


def test_bullish_imbalance_gives_positive_fmn() -> None:
    """Bid-dominated book + positive delta-vol → FMN > 0."""
    dv = np.array([50.0, 80.0, 100.0])
    bid = np.array([1500.0, 1500.0, 1500.0])
    ask = np.array([500.0, 500.0, 500.0])
    result = compute_fmn(dv, bid, ask)
    assert np.all(result > 0)


def test_bearish_imbalance_gives_negative_fmn() -> None:
    dv = np.array([-50.0, -80.0, -100.0])
    bid = np.array([500.0, 500.0, 500.0])
    ask = np.array([1500.0, 1500.0, 1500.0])
    result = compute_fmn(dv, bid, ask)
    assert np.all(result < 0)


def test_balanced_book_near_zero() -> None:
    """Balanced book with zero delta → FMN ≈ 0."""
    dv = np.zeros(5)
    bid = np.full(5, 1000.0)
    ask = np.full(5, 1000.0)
    result = compute_fmn(dv, bid, ask)
    assert np.allclose(result, 0.0, atol=1e-6)


def test_mismatched_length_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_fmn(
            delta_vol=np.zeros(5),
            bid_vol=np.zeros(4),
            ask_vol=np.zeros(5),
        )


def test_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_fmn(
            delta_vol=np.zeros(5),
            bid_vol=np.zeros(5),
            ask_vol=np.zeros(5),
            w_imbalance=-0.1,
        )


def test_non_1d_raises() -> None:
    with pytest.raises(ValueError, match="1-D"):
        compute_fmn(
            delta_vol=np.zeros((2, 2)),
            bid_vol=np.zeros(4),
            ask_vol=np.zeros(4),
        )


def test_empty_input_returns_empty() -> None:
    result = compute_fmn(
        delta_vol=np.array([]),
        bid_vol=np.array([]),
        ask_vol=np.array([]),
    )
    assert result.shape == (0,)
