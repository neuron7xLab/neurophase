"""Tests for ``neurophase.calibration.threshold`` (D1)."""

from __future__ import annotations

import dataclasses
import json

import pytest

from neurophase.calibration.threshold import (
    DEFAULT_THRESHOLD_GRID,
    ThresholdEvaluation,
    ThresholdGrid,
    calibrate_gate_threshold,
    youden_j_objective,
)

# ---------------------------------------------------------------------------
# Grid validation
# ---------------------------------------------------------------------------


class TestThresholdGrid:
    def test_default_grid_is_increasing_and_in_range(self) -> None:
        assert len(DEFAULT_THRESHOLD_GRID) >= 10
        assert all(0.0 < t < 1.0 for t in DEFAULT_THRESHOLD_GRID)
        assert list(DEFAULT_THRESHOLD_GRID) == sorted(DEFAULT_THRESHOLD_GRID)

    def test_valid_grid_constructs(self) -> None:
        g = ThresholdGrid(
            thresholds=(0.3, 0.5, 0.7),
            null_seeds=(1, 2, 3, 4),
            locked_seeds=(10, 11, 12, 13),
            n_samples=128,
        )
        assert g.thresholds == (0.3, 0.5, 0.7)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"thresholds": ()},
            {"thresholds": (0.5, 0.3)},  # not increasing
            {"thresholds": (0.0, 0.5)},  # out of range
            {"thresholds": (0.5, 1.0)},  # out of range
            {"null_seeds": ()},
            {"locked_seeds": ()},
            {"n_samples": 16},
        ],
    )
    def test_invalid_grid_rejected(self, kwargs: dict[str, object]) -> None:
        base = {
            "thresholds": (0.3, 0.5, 0.7),
            "null_seeds": (1, 2, 3, 4),
            "locked_seeds": (10, 11, 12, 13),
            "n_samples": 128,
        }
        base.update(kwargs)
        with pytest.raises(ValueError):
            ThresholdGrid(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End-to-end calibration on ground-truth traces
# ---------------------------------------------------------------------------


def _full_grid() -> ThresholdGrid:
    return ThresholdGrid(
        thresholds=DEFAULT_THRESHOLD_GRID,
        null_seeds=tuple(range(100, 120)),
        locked_seeds=tuple(range(200, 220)),
        n_samples=256,
    )


class TestCalibrationEndToEnd:
    def test_best_threshold_is_near_one_for_clean_separation(self) -> None:
        """Locked traces give R_proxy ≡ 1 and null traces give uniform
        R_proxy in [0, 1]. Youden's J is maximized by any threshold
        that admits all locked samples and at most a tiny fraction of
        null samples — the argmax under the 0.05-grid resolution is
        near the top of the grid. We accept anything ≥ 0.6."""
        report = calibrate_gate_threshold(_full_grid(), train_fraction=0.5)
        assert report.best_threshold >= 0.6
        assert report.train_score_at_best > 0.5
        assert report.test_score_at_best > 0.5

    def test_report_has_per_threshold_scores_on_both_splits(self) -> None:
        grid = _full_grid()
        report = calibrate_gate_threshold(grid, train_fraction=0.5)
        assert len(report.train_evaluations) == len(grid.thresholds)
        assert len(report.test_evaluations) == len(grid.thresholds)
        # Score ordering on train: Youden's J must be monotone in how
        # well the threshold separates the two populations. For this
        # clean setup, the sequence should have a clear maximum.
        js = [e.youden_j for e in report.train_evaluations]
        assert max(js) == js[list(grid.thresholds).index(report.best_threshold)]

    def test_generalization_gap_is_reported(self) -> None:
        report = calibrate_gate_threshold(_full_grid(), train_fraction=0.5)
        expected_gap = report.train_score_at_best - report.test_score_at_best
        assert report.generalization_gap == pytest.approx(expected_gap)

    def test_parameter_fingerprint_is_stable(self) -> None:
        report_a = calibrate_gate_threshold(_full_grid(), train_fraction=0.5)
        report_b = calibrate_gate_threshold(_full_grid(), train_fraction=0.5)
        assert report_a.parameter_fingerprint == report_b.parameter_fingerprint

    def test_parameter_fingerprint_distinguishes_grids(self) -> None:
        g1 = _full_grid()
        g2 = ThresholdGrid(
            thresholds=g1.thresholds,
            null_seeds=g1.null_seeds,
            locked_seeds=g1.locked_seeds,
            n_samples=g1.n_samples * 2,  # different
        )
        r1 = calibrate_gate_threshold(g1, train_fraction=0.5)
        r2 = calibrate_gate_threshold(g2, train_fraction=0.5)
        assert r1.parameter_fingerprint != r2.parameter_fingerprint

    def test_report_is_json_serializable(self) -> None:
        report = calibrate_gate_threshold(_full_grid(), train_fraction=0.5)
        payload = report.to_json_dict()
        # json.dumps should succeed without any default= fallback.
        s = json.dumps(payload, sort_keys=True)
        assert "best_threshold" in s
        assert "parameter_fingerprint" in s

    def test_report_is_frozen(self) -> None:
        report = calibrate_gate_threshold(_full_grid(), train_fraction=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.best_threshold = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestCalibrationDeterminism:
    def test_same_grid_same_report(self) -> None:
        r1 = calibrate_gate_threshold(_full_grid(), train_fraction=0.5)
        r2 = calibrate_gate_threshold(_full_grid(), train_fraction=0.5)
        assert r1.best_threshold == r2.best_threshold
        assert r1.train_evaluations == r2.train_evaluations
        assert r1.test_evaluations == r2.test_evaluations


# ---------------------------------------------------------------------------
# Objective swap
# ---------------------------------------------------------------------------


def _tpr_only_objective(e: ThresholdEvaluation) -> float:
    """Maximize locked-TPR regardless of null-FPR (a deliberately lousy objective)."""
    return e.tpr_locked


class TestObjectiveSwap:
    def test_custom_objective_changes_best_threshold(self) -> None:
        """The default objective (Youden's J) balances TPR and FPR.
        A pure-TPR objective always picks the lowest threshold
        because every threshold admits all locked-c=1 samples (they
        all have R_proxy = 1)."""
        youden_report = calibrate_gate_threshold(
            _full_grid(),
            train_fraction=0.5,
            objective=youden_j_objective,
            objective_name="youden_j",
        )
        tpr_report = calibrate_gate_threshold(
            _full_grid(),
            train_fraction=0.5,
            objective=_tpr_only_objective,
            objective_name="tpr_only",
        )
        # For locked=c=1, TPR is ≡ 1 across the whole grid. argmax over
        # a constant vector returns the first index (lowest threshold).
        assert tpr_report.best_threshold == _full_grid().thresholds[0]
        # Youden's J picks a higher threshold because it penalizes
        # null-FPR, which grows as the threshold shrinks.
        assert youden_report.best_threshold > tpr_report.best_threshold
        # The fingerprints differ because objective_name differs.
        assert youden_report.parameter_fingerprint != tpr_report.parameter_fingerprint


# ---------------------------------------------------------------------------
# Split edge cases
# ---------------------------------------------------------------------------


class TestSplitEdgeCases:
    def test_rejects_bad_train_fraction(self) -> None:
        with pytest.raises(ValueError):
            calibrate_gate_threshold(_full_grid(), train_fraction=0.0)
        with pytest.raises(ValueError):
            calibrate_gate_threshold(_full_grid(), train_fraction=1.0)

    def test_rejects_empty_split(self) -> None:
        """A 1-seed grid cannot be split — the helper always keeps ≥ 1
        seed on the train side, leaving the test side empty."""
        grid = ThresholdGrid(
            thresholds=(0.3, 0.5, 0.7),
            null_seeds=(1,),
            locked_seeds=(10,),
            n_samples=64,
        )
        with pytest.raises(ValueError, match="empty"):
            calibrate_gate_threshold(grid, train_fraction=0.5)
