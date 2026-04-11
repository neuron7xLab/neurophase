"""Tests for neurophase.metrics.entropy."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.metrics.entropy import (
    delta_entropy,
    freedman_diaconis_bins,
    renyi_entropy,
    shannon_entropy,
    tsallis_entropy,
)


def test_freedman_diaconis_constant_falls_back() -> None:
    assert freedman_diaconis_bins(np.zeros(64)) == 2


def test_freedman_diaconis_scales_with_sample_size() -> None:
    """Bin count should grow with sample size (∝ n^(1/3) for fixed distribution)."""
    rng = np.random.default_rng(0)
    small = rng.normal(0, 1, 200)
    large = rng.normal(0, 1, 5000)
    assert freedman_diaconis_bins(large) > freedman_diaconis_bins(small)


def test_shannon_uniform_is_maximal() -> None:
    """Uniform sample has higher Shannon entropy than a peaked one when both
    are binned on the SAME support with the SAME bin count."""
    rng = np.random.default_rng(0)
    # Both mapped to [-1, 1] and binned identically.
    uniform = rng.uniform(-1, 1, 5000)
    peaked = np.clip(rng.normal(0, 0.1, 5000), -1.0, 1.0)
    assert shannon_entropy(uniform, bins=64) > shannon_entropy(peaked, bins=64)


def test_shannon_constant_series_is_low() -> None:
    h = shannon_entropy(np.zeros(256))
    assert h >= 0.0
    assert h < 1.0


def test_tsallis_q_must_be_not_one() -> None:
    with pytest.raises(ValueError, match="q must be != 1"):
        tsallis_entropy(np.arange(10, dtype=np.float64), q=1.0)


def test_tsallis_orders_match_shannon_direction() -> None:
    rng = np.random.default_rng(1)
    uniform = rng.uniform(-1, 1, 5000)
    peaked = np.clip(rng.normal(0, 0.1, 5000), -1.0, 1.0)
    assert tsallis_entropy(uniform, q=1.5, bins=64) > tsallis_entropy(peaked, q=1.5, bins=64)


def test_renyi_requires_positive_alpha() -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        renyi_entropy(np.arange(10, dtype=np.float64), alpha=-1.0)


def test_renyi_rejects_alpha_one() -> None:
    with pytest.raises(ValueError, match="alpha must be != 1"):
        renyi_entropy(np.arange(10, dtype=np.float64), alpha=1.0)


def test_renyi_orders_match_shannon_direction() -> None:
    rng = np.random.default_rng(2)
    uniform = rng.uniform(-1, 1, 5000)
    peaked = np.clip(rng.normal(0, 0.1, 5000), -1.0, 1.0)
    assert renyi_entropy(uniform, alpha=2.0, bins=64) > renyi_entropy(peaked, alpha=2.0, bins=64)


def test_delta_entropy_detects_drop() -> None:
    """Trailing window concentrated near zero → ΔH < 0 with fixed binning."""
    rng = np.random.default_rng(3)
    first = rng.uniform(-1, 1, 400)
    second = np.clip(rng.normal(0, 0.05, 400), -1.0, 1.0)
    series = np.concatenate([first, second])
    dh = delta_entropy(series, window=400, bins=64)
    assert dh < 0.0


def test_delta_entropy_requires_enough_samples() -> None:
    with pytest.raises(ValueError, match=r"2.window"):
        delta_entropy(np.zeros(10), window=100)


def test_delta_entropy_rejects_non_positive_window() -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        delta_entropy(np.zeros(100), window=0)
