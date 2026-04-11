"""Tests for neurophase.gate.direction_index."""

from __future__ import annotations

import pytest

from neurophase.gate.direction_index import (
    DEFAULT_WEIGHTS,
    Direction,
    DirectionIndexWeights,
    direction_index,
)


def test_positive_signals_give_long() -> None:
    result = direction_index(skew=0.5, curv=0.1, bias=0.2)
    assert result.direction is Direction.LONG
    assert result.value > 0


def test_negative_signals_give_short() -> None:
    result = direction_index(skew=-0.5, curv=-0.1, bias=-0.2)
    assert result.direction is Direction.SHORT
    assert result.value < 0


def test_zero_signals_give_flat() -> None:
    result = direction_index(skew=0.0, curv=0.0, bias=0.0)
    assert result.direction is Direction.FLAT


def test_near_zero_within_tolerance_is_flat() -> None:
    result = direction_index(skew=1e-12, curv=1e-12, bias=0.0, flat_tolerance=1e-6)
    assert result.direction is Direction.FLAT


def test_weights_influence_outcome() -> None:
    """A large skew alone under default weights is enough to flip direction."""
    high_skew_weights = DirectionIndexWeights(w_skew=1.0, w_curv=0.0, w_bias=0.0)
    result = direction_index(skew=0.5, curv=-1.0, bias=-1.0, weights=high_skew_weights)
    assert result.direction is Direction.LONG


def test_rejects_all_zero_weights() -> None:
    with pytest.raises(ValueError, match="At least one weight"):
        DirectionIndexWeights(w_skew=0.0, w_curv=0.0, w_bias=0.0)


def test_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DirectionIndexWeights(w_skew=-0.1, w_curv=0.3, w_bias=0.3)


def test_default_weights_are_simplex_like() -> None:
    # Documented π-system defaults.
    assert DEFAULT_WEIGHTS.w_skew == 0.4
    assert DEFAULT_WEIGHTS.w_curv == 0.3
    assert DEFAULT_WEIGHTS.w_bias == 0.3
