"""Tests for D2 stillness parameter calibration."""

from __future__ import annotations

import dataclasses
import json

import pytest

from neurophase.calibration.stillness import (
    DEFAULT_DELTA_MIN_GRID,
    DEFAULT_EPS_F_GRID,
    DEFAULT_EPS_R_GRID,
    DEFAULT_WINDOW_GRID,
    StillnessCalibrationReport,
    StillnessGrid,
    calibrate_stillness_parameters,
)

# ---------------------------------------------------------------------------
# Grid validation
# ---------------------------------------------------------------------------


class TestStillnessGrid:
    def test_defaults(self) -> None:
        g = StillnessGrid()
        assert g.eps_R_values == DEFAULT_EPS_R_GRID
        assert g.eps_F_values == DEFAULT_EPS_F_GRID
        assert g.delta_min_values == DEFAULT_DELTA_MIN_GRID
        assert g.window_values == DEFAULT_WINDOW_GRID
        assert g.n_samples >= 32
        assert g.dt > 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"eps_R_values": ()},
            {"eps_R_values": (0.0, 1e-3)},
            {"eps_F_values": (-1e-3,)},
            {"delta_min_values": ()},
            {"window_values": ()},
            {"window_values": (1, 4)},
            {"quiet_seeds": ()},
            {"active_seeds": ()},
            {"n_samples": 16},
            {"dt": 0.0},
            {"dt": -0.01},
        ],
    )
    def test_invalid_grid_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            StillnessGrid(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# End-to-end calibration
# ---------------------------------------------------------------------------


def _compact_grid() -> StillnessGrid:
    """Smaller grid for tests — keeps the suite fast."""
    return StillnessGrid(
        eps_R_values=(1e-4, 1e-3, 1e-2),
        eps_F_values=(1e-4, 1e-3, 1e-2),
        delta_min_values=(0.05, 0.10),
        window_values=(4, 8),
        quiet_seeds=tuple(range(200, 208)),
        active_seeds=tuple(range(300, 308)),
        n_samples=128,
        dt=0.01,
    )


class TestCalibrationEndToEnd:
    def test_report_has_per_cell_scores_on_both_splits(self) -> None:
        grid = _compact_grid()
        expected_cells = (
            len(grid.eps_R_values)
            * len(grid.eps_F_values)
            * len(grid.delta_min_values)
            * len(grid.window_values)
        )
        report = calibrate_stillness_parameters(grid, train_fraction=0.5)
        assert len(report.train_evaluations) == expected_cells
        assert len(report.test_evaluations) == expected_cells

    def test_best_cell_has_positive_youden_j(self) -> None:
        """A correctly-working detector must separate quiet from active
        with at least some margin."""
        report = calibrate_stillness_parameters(_compact_grid(), train_fraction=0.5)
        assert report.train_score_at_best > 0.0
        assert report.test_score_at_best >= 0.0

    def test_quiet_traces_are_classified_as_still_most_of_the_time(self) -> None:
        """The best cell must achieve ≥ 80% TPR on the quiet regime."""
        report = calibrate_stillness_parameters(_compact_grid(), train_fraction=0.5)
        assert report.best_cell.tpr_quiet >= 0.8

    def test_generalization_gap_is_reported(self) -> None:
        report = calibrate_stillness_parameters(_compact_grid(), train_fraction=0.5)
        expected = report.train_score_at_best - report.test_score_at_best
        assert report.generalization_gap == pytest.approx(expected)

    def test_report_is_frozen(self) -> None:
        report = calibrate_stillness_parameters(_compact_grid(), train_fraction=0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.train_score_at_best = -1.0  # type: ignore[misc]

    def test_report_is_json_serializable(self) -> None:
        report = calibrate_stillness_parameters(_compact_grid(), train_fraction=0.5)
        payload = report.to_json_dict()
        s = json.dumps(payload, sort_keys=True)
        assert "best_cell" in s
        assert "parameter_fingerprint" in s

    def test_best_cell_is_in_train_evaluations(self) -> None:
        """Sanity: the selected cell must be one of the train grid cells,
        not a synthetic construct."""
        report = calibrate_stillness_parameters(_compact_grid(), train_fraction=0.5)
        assert any(
            e.eps_R == report.best_cell.eps_R
            and e.eps_F == report.best_cell.eps_F
            and e.delta_min == report.best_cell.delta_min
            and e.window == report.best_cell.window
            for e in report.train_evaluations
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestStillnessDeterminism:
    def test_same_grid_same_report(self) -> None:
        r1 = calibrate_stillness_parameters(_compact_grid(), train_fraction=0.5)
        r2 = calibrate_stillness_parameters(_compact_grid(), train_fraction=0.5)
        assert r1.parameter_fingerprint == r2.parameter_fingerprint
        assert r1.best_cell == r2.best_cell
        assert r1.train_evaluations == r2.train_evaluations
        assert r1.test_evaluations == r2.test_evaluations

    def test_fingerprint_distinguishes_grids(self) -> None:
        g1 = _compact_grid()
        # Same everything except larger window grid.
        g2 = StillnessGrid(
            eps_R_values=g1.eps_R_values,
            eps_F_values=g1.eps_F_values,
            delta_min_values=g1.delta_min_values,
            window_values=(4, 8, 16),
            quiet_seeds=g1.quiet_seeds,
            active_seeds=g1.active_seeds,
            n_samples=g1.n_samples,
            dt=g1.dt,
        )
        r1 = calibrate_stillness_parameters(g1, train_fraction=0.5)
        r2 = calibrate_stillness_parameters(g2, train_fraction=0.5)
        assert r1.parameter_fingerprint != r2.parameter_fingerprint


# ---------------------------------------------------------------------------
# Split edge cases
# ---------------------------------------------------------------------------


class TestSplitEdgeCases:
    def test_rejects_bad_train_fraction(self) -> None:
        with pytest.raises(ValueError):
            calibrate_stillness_parameters(_compact_grid(), train_fraction=0.0)
        with pytest.raises(ValueError):
            calibrate_stillness_parameters(_compact_grid(), train_fraction=1.0)

    def test_rejects_single_seed_grid(self) -> None:
        grid = StillnessGrid(
            eps_R_values=(1e-3,),
            eps_F_values=(1e-3,),
            delta_min_values=(0.10,),
            window_values=(4,),
            quiet_seeds=(200,),
            active_seeds=(300,),
            n_samples=64,
            dt=0.01,
        )
        with pytest.raises(ValueError, match="empty"):
            calibrate_stillness_parameters(grid, train_fraction=0.5)


# ---------------------------------------------------------------------------
# Physical reality check — tight params really do reject active traces
# ---------------------------------------------------------------------------


class TestPhysicalReality:
    def test_tight_eps_R_rejects_drifting_R(self) -> None:
        """With eps_R clearly below the drift rate of an active trace, the
        detector correctly never emits STILL on it."""
        report: StillnessCalibrationReport = calibrate_stillness_parameters(
            StillnessGrid(
                eps_R_values=(1e-5,),  # tiny
                eps_F_values=(1e-3,),
                delta_min_values=(0.10,),
                window_values=(4,),
                quiet_seeds=tuple(range(200, 204)),
                active_seeds=tuple(range(300, 304)),
                n_samples=128,
                dt=0.01,
            ),
            train_fraction=0.5,
        )
        # Active trace has dR ≈ 0.10 / 128 ≈ 8e-4 per tick; with eps_R=1e-5
        # the detector fires ACTIVE on every post-warmup sample.
        cell = report.best_cell
        assert cell.fpr_active <= 0.01, (
            f"tight eps_R should give near-zero FPR on active traces, got {cell.fpr_active}"
        )

    def test_loose_eps_R_admits_quiet_traces_as_still(self) -> None:
        report = calibrate_stillness_parameters(
            StillnessGrid(
                eps_R_values=(1e-2,),  # loose enough for noise
                eps_F_values=(1e-2,),
                delta_min_values=(0.10,),
                window_values=(8,),
                quiet_seeds=tuple(range(200, 206)),
                active_seeds=tuple(range(300, 306)),
                n_samples=128,
                dt=0.01,
            ),
            train_fraction=0.5,
        )
        cell = report.best_cell
        assert cell.tpr_quiet >= 0.9, (
            f"loose parameters should give high TPR on quiet traces, got {cell.tpr_quiet}"
        )
