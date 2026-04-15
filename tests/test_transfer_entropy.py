"""Tests for neurophase.metrics.transfer_entropy."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.metrics.transfer_entropy import (
    TEResult,
    transfer_entropy,
    transfer_entropy_with_significance,
)

# ---------------------------------------------------------------------------
# Fixtures: generative couplings
# ---------------------------------------------------------------------------


def _independent_pair(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n), rng.standard_normal(n)


def _x_drives_y(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """y_t = 0.9 · x_{t-1} + 0.3 · ε — x carries genuine predictive power."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    y = np.empty(n)
    y[0] = noise[0]
    y[1:] = 0.9 * x[:-1] + 0.3 * noise[1:]
    return x, y


# ---------------------------------------------------------------------------
# transfer_entropy — core plug-in estimator
# ---------------------------------------------------------------------------


def test_transfer_entropy_is_non_negative() -> None:
    x, y = _independent_pair()
    assert transfer_entropy(x, y) >= 0.0


def test_transfer_entropy_detects_directed_coupling() -> None:
    """When y_t depends on x_{t-1}, TE(x → y) dominates TE(y → x)."""
    x, y = _x_drives_y()
    te_forward = transfer_entropy(x, y)
    te_backward = transfer_entropy(y, x)
    assert te_forward > te_backward
    assert te_forward - te_backward > 0.01  # clearly separated in nats


def test_transfer_entropy_independent_is_small() -> None:
    """Plug-in TE on independent streams is finite-sample-biased but small."""
    x, y = _independent_pair(n=6000)
    te = transfer_entropy(x, y)
    # Binary quantisation with k=1 on 6000 samples yields bias well under 0.01.
    assert te < 0.01


def test_transfer_entropy_constant_input_is_zero() -> None:
    x = np.zeros(500)
    y = np.arange(500, dtype=np.float64)
    assert transfer_entropy(x, y) == 0.0
    assert transfer_entropy(y, x) == 0.0


def test_transfer_entropy_short_input_is_zero() -> None:
    assert transfer_entropy([1.0, 2.0], [3.0, 4.0], k=1) == 0.0


def test_transfer_entropy_honours_k_and_n_levels() -> None:
    x, y = _x_drives_y(n=3000, seed=7)
    te_default = transfer_entropy(x, y)
    te_ternary = transfer_entropy(x, y, n_levels=3, k=1)
    # Both must be finite, non-negative, and comparable in magnitude.
    assert te_default >= 0.0
    assert te_ternary >= 0.0


def test_transfer_entropy_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="k must be ≥ 1"):
        transfer_entropy([1.0] * 10, [1.0] * 10, k=0)
    with pytest.raises(ValueError, match="n_levels must be ≥ 2"):
        transfer_entropy([1.0] * 10, [1.0] * 10, n_levels=1)
    with pytest.raises(ValueError, match="exceeds safety cap"):
        transfer_entropy([1.0] * 10, [1.0] * 10, k=20, n_levels=8)


def test_transfer_entropy_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share shape"):
        transfer_entropy(np.zeros(10), np.zeros(12))


def test_transfer_entropy_rejects_non_finite() -> None:
    x = np.array([1.0, np.nan, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="finite"):
        transfer_entropy(x, y)


# ---------------------------------------------------------------------------
# transfer_entropy_with_significance — bias correction + p-values
# ---------------------------------------------------------------------------


def test_significance_returns_valid_result() -> None:
    x, y = _x_drives_y(n=3000)
    result = transfer_entropy_with_significance(x, y, n_surrogates=50, seed=0)
    assert isinstance(result, TEResult)
    assert result.te_xy >= 0.0
    assert result.te_yx >= 0.0
    assert 0.0 <= result.p_xy <= 1.0
    assert 0.0 <= result.p_yx <= 1.0
    assert result.te_net == pytest.approx(result.te_xy - result.te_yx)


def test_significance_detects_direction() -> None:
    """On x → y coupling, the forward TE is significant and backward is not."""
    x, y = _x_drives_y(n=5000, seed=11)
    result = transfer_entropy_with_significance(x, y, n_surrogates=200, seed=13)
    assert result.te_xy > 0.0
    assert result.te_net > 0.0
    assert result.p_xy < 0.05
    # Backward direction carries no real coupling → p-value should not be
    # small. A loose bound tolerates Monte-Carlo variation.
    assert result.p_yx > 0.05


def test_significance_bias_correction_shrinks_independent_te() -> None:
    x, y = _independent_pair(n=4000, seed=3)
    uncorrected = transfer_entropy_with_significance(
        x, y, n_surrogates=100, bias_correct=False, seed=5
    )
    corrected = transfer_entropy_with_significance(
        x, y, n_surrogates=100, bias_correct=True, seed=5
    )
    # Bias correction must never inflate the estimate.
    assert corrected.te_xy <= uncorrected.te_xy + 1e-12
    assert corrected.te_yx <= uncorrected.te_yx + 1e-12


def test_significance_is_deterministic_under_seed() -> None:
    x, y = _x_drives_y(n=2000, seed=2)
    a = transfer_entropy_with_significance(x, y, n_surrogates=50, seed=42)
    b = transfer_entropy_with_significance(x, y, n_surrogates=50, seed=42)
    assert a == b


def test_significance_handles_degenerate_input() -> None:
    result = transfer_entropy_with_significance([1.0, 2.0], [3.0, 4.0])
    assert result.te_xy == 0.0
    assert result.te_yx == 0.0
    assert result.p_xy == 1.0
    assert result.p_yx == 1.0


def test_significance_net_sign_flips_on_swap() -> None:
    x, y = _x_drives_y(n=3000, seed=17)
    forward = transfer_entropy_with_significance(x, y, n_surrogates=80, seed=1)
    backward = transfer_entropy_with_significance(y, x, n_surrogates=80, seed=1)
    # Net flow inverts exactly when you swap the interpretation of source/target.
    assert forward.te_net == pytest.approx(-backward.te_net)


def test_significance_rejects_zero_surrogates() -> None:
    with pytest.raises(ValueError, match="n_surrogates must be ≥ 1"):
        transfer_entropy_with_significance(np.zeros(20), np.zeros(20), n_surrogates=0)
