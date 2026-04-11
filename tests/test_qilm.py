"""Tests for neurophase.indicators.qilm."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.indicators.qilm import compute_qilm


def test_shape_matches_input() -> None:
    n = 20
    result = compute_qilm(
        open_interest=np.linspace(100.0, 120.0, n),
        volume=np.full(n, 500.0),
        delta_vol=np.full(n, 50.0),
        hidden_vol=np.full(n, 10.0),
        atr=np.full(n, 5.0),
    )
    assert result.shape == (n,)


def test_aligned_bullish_flow_is_positive() -> None:
    """Rising OI + positive delta-volume → positive QILM."""
    oi = np.array([100.0, 110.0, 125.0, 140.0])
    vol = np.array([200.0, 300.0, 400.0, 500.0])
    dv = np.array([20.0, 40.0, 60.0, 80.0])
    hv = np.array([5.0, 5.0, 5.0, 5.0])
    atr = np.array([10.0, 10.0, 10.0, 10.0])
    q = compute_qilm(oi, vol, dv, hv, atr)
    assert np.all(q[1:] > 0)


def test_mismatched_directions_give_negative() -> None:
    """Rising OI but negative delta-vol → weakening move → QILM < 0."""
    oi = np.array([100.0, 110.0, 120.0])
    vol = np.array([200.0, 200.0, 200.0])
    dv = np.array([-20.0, -50.0, -30.0])
    hv = np.array([5.0, 5.0, 5.0])
    atr = np.array([10.0, 10.0, 10.0])
    q = compute_qilm(oi, vol, dv, hv, atr)
    assert np.all(q[1:] < 0)


def test_zero_atr_does_not_crash() -> None:
    """ATR = 0 is regularised with ε and the result remains finite."""
    q = compute_qilm(
        open_interest=np.array([100.0, 101.0]),
        volume=np.array([100.0, 100.0]),
        delta_vol=np.array([1.0, 1.0]),
        hidden_vol=np.array([0.0, 0.0]),
        atr=np.array([0.0, 0.0]),
    )
    assert np.all(np.isfinite(q))


def test_mismatched_length_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_qilm(
            open_interest=np.zeros(5),
            volume=np.zeros(5),
            delta_vol=np.zeros(4),
            hidden_vol=np.zeros(5),
            atr=np.zeros(5),
        )


def test_non_1d_raises() -> None:
    with pytest.raises(ValueError, match="1-D"):
        compute_qilm(
            open_interest=np.zeros((2, 2)),
            volume=np.zeros(4),
            delta_vol=np.zeros(4),
            hidden_vol=np.zeros(4),
            atr=np.zeros(4),
        )


def test_empty_input_returns_empty() -> None:
    q = compute_qilm(
        open_interest=np.array([]),
        volume=np.array([]),
        delta_vol=np.array([]),
        hidden_vol=np.array([]),
        atr=np.array([]),
    )
    assert q.shape == (0,)
