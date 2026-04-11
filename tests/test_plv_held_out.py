"""Tests for the C3 PLV significance retrofit onto ``NullModelHarness``.

Covers:

* Phipson–Smyth smoothing is applied through the shared harness
  (p-value is strictly positive for any finite n_surrogates).
* Determinism: same seed → bit-identical p-values.
* ``HeldOutSplit`` construction + overlap rejection (``HeldOutViolation``).
* ``plv_on_held_out`` enforcement: in-sample evaluation is
  architecturally impossible.
* End-to-end: train+test partition of a locked signal → significant;
  partition of noise → not significant.
* Backward compatibility: legacy ``plv_significance`` signature still
  works and delegates to the harness.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from neurophase.metrics.plv import (
    DEFAULT_PLV_N_SURROGATES,
    HeldOutSplit,
    HeldOutViolation,
    PLVResult,
    plv,
    plv_on_held_out,
    plv_significance,
)

# ---------------------------------------------------------------------------
# Phipson–Smyth smoothing (harness delegation)
# ---------------------------------------------------------------------------


class TestHarnessDelegation:
    def test_p_value_is_strictly_positive(self) -> None:
        """Phipson–Smyth guarantees ``p ≥ 1 / (1 + n)`` — never zero."""
        n = 256
        t = np.linspace(0, 4 * np.pi, n)
        phi_x = t.copy()
        phi_y = t + 0.05  # tight lock
        result = plv_significance(phi_x, phi_y, n_surrogates=200, seed=7)
        assert result.p_value > 0.0
        assert result.p_value >= 1.0 / 201.0

    def test_result_carries_seed(self) -> None:
        result = plv_significance(np.zeros(32), np.zeros(32), n_surrogates=50, seed=123)
        assert result.seed == 123

    def test_rejects_tiny_n_surrogates(self) -> None:
        """The harness floor is 10 — PLV refuses anything smaller for
        the same reason (a p-value from < 10 surrogates is meaningless)."""
        with pytest.raises(ValueError, match="n_surrogates"):
            plv_significance(np.zeros(8), np.zeros(8), n_surrogates=5)

    def test_rejects_bad_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            plv_significance(np.zeros(8), np.zeros(8), alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            plv_significance(np.zeros(8), np.zeros(8), alpha=1.5)

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            plv_significance(np.zeros(8), np.zeros(10))

    def test_rejects_singleton_series(self) -> None:
        with pytest.raises(ValueError, match="length ≥ 2"):
            plv_significance(np.zeros(1), np.zeros(1))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_p_value(self) -> None:
        n = 200
        rng = np.random.default_rng(11)
        phi_x = rng.uniform(-np.pi, np.pi, size=n)
        phi_y = rng.uniform(-np.pi, np.pi, size=n)
        r1 = plv_significance(phi_x, phi_y, n_surrogates=100, seed=777)
        r2 = plv_significance(phi_x, phi_y, n_surrogates=100, seed=777)
        assert r1.p_value == r2.p_value
        assert r1.plv == r2.plv

    def test_different_seeds_give_different_distributions(self) -> None:
        n = 200
        rng = np.random.default_rng(13)
        phi_x = rng.uniform(-np.pi, np.pi, size=n)
        phi_y = rng.uniform(-np.pi, np.pi, size=n)
        r1 = plv_significance(phi_x, phi_y, n_surrogates=100, seed=1)
        r2 = plv_significance(phi_x, phi_y, n_surrogates=100, seed=2)
        assert r1.plv == r2.plv  # observed statistic is seed-independent
        # p-values may agree by accident but are typically different.
        # We just confirm neither crashed.
        assert 0.0 < r1.p_value <= 1.0
        assert 0.0 < r2.p_value <= 1.0


# ---------------------------------------------------------------------------
# HeldOutSplit construction + enforcement
# ---------------------------------------------------------------------------


class TestHeldOutSplitConstruction:
    def test_valid_split(self) -> None:
        s = HeldOutSplit(
            train_indices=np.arange(50, dtype=np.int64),
            test_indices=np.arange(50, 100, dtype=np.int64),
            total_length=100,
        )
        assert s.total_length == 100

    def test_overlap_rejected(self) -> None:
        with pytest.raises(HeldOutViolation, match="overlap"):
            HeldOutSplit(
                train_indices=np.array([0, 1, 2, 3, 4], dtype=np.int64),
                test_indices=np.array([3, 4, 5, 6], dtype=np.int64),
                total_length=10,
            )

    def test_empty_train_rejected(self) -> None:
        with pytest.raises(ValueError, match="≥ 1"):
            HeldOutSplit(
                train_indices=np.array([], dtype=np.int64),
                test_indices=np.array([0, 1], dtype=np.int64),
                total_length=10,
            )

    def test_empty_test_rejected(self) -> None:
        with pytest.raises(ValueError, match="≥ 1"):
            HeldOutSplit(
                train_indices=np.array([0, 1], dtype=np.int64),
                test_indices=np.array([], dtype=np.int64),
                total_length=10,
            )

    def test_negative_index_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            HeldOutSplit(
                train_indices=np.array([-1, 0], dtype=np.int64),
                test_indices=np.array([5, 6], dtype=np.int64),
                total_length=10,
            )

    def test_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            HeldOutSplit(
                train_indices=np.array([0, 1], dtype=np.int64),
                test_indices=np.array([9, 15], dtype=np.int64),
                total_length=10,
            )

    def test_non_1d_rejected(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            HeldOutSplit(
                train_indices=np.zeros((2, 2), dtype=np.int64),
                test_indices=np.array([0], dtype=np.int64),
                total_length=10,
            )

    def test_test_slice(self) -> None:
        s = HeldOutSplit(
            train_indices=np.array([0, 1, 2], dtype=np.int64),
            test_indices=np.array([3, 4, 5], dtype=np.int64),
            total_length=6,
        )
        series = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        np.testing.assert_array_equal(s.test_slice(series), [40.0, 50.0, 60.0])

    def test_test_slice_rejects_wrong_length(self) -> None:
        s = HeldOutSplit(
            train_indices=np.array([0], dtype=np.int64),
            test_indices=np.array([1], dtype=np.int64),
            total_length=2,
        )
        with pytest.raises(ValueError, match="total_length"):
            s.test_slice(np.zeros(5))


# ---------------------------------------------------------------------------
# plv_on_held_out
# ---------------------------------------------------------------------------


class TestPLVOnHeldOut:
    def test_locked_signal_is_significant(self) -> None:
        n = 256
        t = np.linspace(0, 4 * np.pi, n)
        phi_x = t.copy()
        phi_y = t + 0.1  # tight phase lock
        split = HeldOutSplit(
            train_indices=np.arange(n // 2, dtype=np.int64),
            test_indices=np.arange(n // 2, n, dtype=np.int64),
            total_length=n,
        )
        result = plv_on_held_out(phi_x, phi_y, split, n_surrogates=200, seed=31)
        assert isinstance(result, PLVResult)
        assert result.plv > 0.99
        assert result.significant

    def test_noise_is_not_significant(self) -> None:
        rng = np.random.default_rng(41)
        n = 512
        phi_x = rng.uniform(-np.pi, np.pi, size=n)
        phi_y = rng.uniform(-np.pi, np.pi, size=n)
        split = HeldOutSplit(
            train_indices=np.arange(n // 2, dtype=np.int64),
            test_indices=np.arange(n // 2, n, dtype=np.int64),
            total_length=n,
        )
        result = plv_on_held_out(phi_x, phi_y, split, n_surrogates=200, seed=43)
        assert not result.significant

    def test_rejects_mismatched_lengths(self) -> None:
        split = HeldOutSplit(
            train_indices=np.array([0, 1], dtype=np.int64),
            test_indices=np.array([2, 3], dtype=np.int64),
            total_length=4,
        )
        with pytest.raises(ValueError, match="same length"):
            plv_on_held_out(np.zeros(4), np.zeros(6), split)

    def test_rejects_wrong_total_length(self) -> None:
        split = HeldOutSplit(
            train_indices=np.array([0, 1], dtype=np.int64),
            test_indices=np.array([2, 3], dtype=np.int64),
            total_length=4,
        )
        with pytest.raises(ValueError, match="total_length"):
            plv_on_held_out(np.zeros(8), np.zeros(8), split)

    def test_held_out_partition_is_disjoint(self) -> None:
        """The architectural guarantee: after construction the train and
        test indices are disjoint sets. An attempt to construct an
        overlapping split fails **before** any statistic is computed."""
        with pytest.raises(HeldOutViolation):
            HeldOutSplit(
                train_indices=np.array([0, 1, 2], dtype=np.int64),
                test_indices=np.array([2, 3, 4], dtype=np.int64),
                total_length=5,
            )


# ---------------------------------------------------------------------------
# Statistical sanity: locked vs anti-phase
# ---------------------------------------------------------------------------


class TestStatisticalSanity:
    def test_identical_phases_give_plv_one(self) -> None:
        phi = np.linspace(0, 4 * np.pi, 100)
        assert math.isclose(plv(phi, phi), 1.0, abs_tol=1e-12)

    def test_anti_phase_still_has_plv_one(self) -> None:
        """A constant phase offset of π gives PLV=1 — the statistic
        measures *locking*, not direction."""
        phi = np.linspace(0, 4 * np.pi, 100)
        assert math.isclose(plv(phi, phi + math.pi), 1.0, abs_tol=1e-12)

    def test_default_n_surrogates_matches_rd_convention(self) -> None:
        assert DEFAULT_PLV_N_SURROGATES == 1000
