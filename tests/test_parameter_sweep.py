"""H7 — contract tests for the parameter sweep simulation lab.

This test file is the HN31 binding. It locks in:

1. **Determinism.** Two calls to :func:`sweep_parameters` with
   the same grid + seed produce byte-identical reports.
2. **Total enumeration.** Every cell in the Cartesian product
   is exercised exactly once.
3. **Canonical ordering.** Cells appear in
   threshold-major / coupling / trace_seed order.
4. **Monotone admission rate.** For a fixed threshold, higher
   coupling strength yields higher (or equal) mean admission
   rate. This is the load-bearing physical sanity check —
   stronger coupling cannot reduce R_proxy on average.
5. **Threshold monotonicity.** For a fixed coupling strength
   and trace seed, raising the threshold can only decrease
   (or keep equal) the admission rate.
6. **Boundary correctness.** At ``coupling_strength = 1`` the
   mean R_proxy is exactly 1.0 (closed-form ground truth),
   and admission is exactly 100% for any threshold below 1.
7. **Schema validation.** Malformed grids are rejected at
   construction time.
8. **JSON-safe round-trip.** ``to_json_dict`` is a flat
   projection — every value is a primitive or a list of
   primitive dicts.
9. **Frozen dataclasses.** :class:`SweepGrid`,
   :class:`SweepCellResult`, :class:`SweepReport` reject
   attribute reassignment.
"""

from __future__ import annotations

import json

import pytest

from neurophase.benchmarks.parameter_sweep import (
    SweepCellResult,
    SweepError,
    SweepGrid,
    SweepReport,
    sweep_parameters,
)

# ---------------------------------------------------------------------------
# Fixture builder.
# ---------------------------------------------------------------------------


def _grid(
    *,
    thresholds: tuple[float, ...] = (0.30, 0.50, 0.70),
    couplings: tuple[float, ...] = (0.0, 0.5, 1.0),
    seeds: tuple[int, ...] = (1, 2, 3),
    n_samples: int = 64,
) -> SweepGrid:
    return SweepGrid(
        threshold_values=thresholds,
        coupling_strengths=couplings,
        trace_seeds=seeds,
        n_samples=n_samples,
    )


# ---------------------------------------------------------------------------
# 1. Determinism.
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_runs_byte_identical(self) -> None:
        grid = _grid()
        a = sweep_parameters(grid, seed=42)
        b = sweep_parameters(grid, seed=42)
        assert a.to_json_dict() == b.to_json_dict()
        assert a == b

    def test_results_are_in_canonical_order(self) -> None:
        grid = _grid()
        report = sweep_parameters(grid, seed=0)
        # The canonical order is threshold-major, coupling-second,
        # seed-third. Build the expected (threshold, coupling, seed)
        # sequence and check.
        expected = [
            (t, c, s)
            for t in grid.threshold_values
            for c in grid.coupling_strengths
            for s in grid.trace_seeds
        ]
        actual = [(c.threshold, c.coupling_strength, c.seed) for c in report.results]
        assert actual == expected


# ---------------------------------------------------------------------------
# 2. Total enumeration.
# ---------------------------------------------------------------------------


class TestEnumeration:
    def test_total_cells_matches_product(self) -> None:
        grid = _grid(
            thresholds=(0.20, 0.40, 0.60, 0.80),
            couplings=(0.0, 0.25, 0.5, 0.75, 1.0),
            seeds=(10, 20),
        )
        report = sweep_parameters(grid)
        assert grid.total_cells == 4 * 5 * 2
        assert len(report.results) == grid.total_cells

    def test_every_cell_unique(self) -> None:
        grid = _grid()
        report = sweep_parameters(grid)
        keys = [(c.threshold, c.coupling_strength, c.seed) for c in report.results]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# 3. Physical sanity — monotone in coupling strength.
# ---------------------------------------------------------------------------


class TestMonotoneInCoupling:
    def test_higher_coupling_admits_more_on_average(self) -> None:
        grid = _grid(
            thresholds=(0.50,),
            couplings=(0.0, 0.5, 1.0),
            seeds=(1, 2, 3, 4, 5),
            n_samples=128,
        )
        report = sweep_parameters(grid)
        # Average admission rate per coupling strength.
        avgs: dict[float, float] = {}
        for c in grid.coupling_strengths:
            cells = report.by_coupling(c)
            avgs[c] = sum(cell.proportion_above for cell in cells) / len(cells)
        # 0 < 0.5 < 1.0
        assert avgs[0.0] < avgs[0.5] < avgs[1.0]

    def test_mean_R_proxy_monotone_in_coupling(self) -> None:
        grid = _grid(
            thresholds=(0.50,),
            couplings=(0.0, 0.5, 1.0),
            seeds=(1, 2, 3, 4, 5),
            n_samples=128,
        )
        report = sweep_parameters(grid)
        avgs: dict[float, float] = {}
        for c in grid.coupling_strengths:
            cells = report.by_coupling(c)
            avgs[c] = sum(cell.mean_R_proxy for cell in cells) / len(cells)
        assert avgs[0.0] < avgs[0.5] < avgs[1.0]


# ---------------------------------------------------------------------------
# 4. Threshold monotonicity (per cell).
# ---------------------------------------------------------------------------


class TestMonotoneInThreshold:
    def test_admission_decreases_with_threshold(self) -> None:
        grid = _grid(
            thresholds=(0.20, 0.50, 0.80),
            couplings=(0.5,),
            seeds=(7,),
            n_samples=128,
        )
        report = sweep_parameters(grid)
        # All cells share the same (coupling, seed) so the only
        # difference is the threshold. Admission must monotonically
        # decrease as threshold increases.
        admissions = [c.proportion_above for c in report.results]
        for i in range(len(admissions) - 1):
            assert admissions[i] >= admissions[i + 1], f"admission not monotone: {admissions}"


# ---------------------------------------------------------------------------
# 5. Boundary correctness — c=1 is closed-form perfect.
# ---------------------------------------------------------------------------


class TestBoundaryCorrectness:
    def test_full_coupling_yields_unit_R_proxy(self) -> None:
        grid = _grid(
            thresholds=(0.50,),
            couplings=(1.0,),
            seeds=(1, 2, 3),
            n_samples=128,
        )
        report = sweep_parameters(grid)
        for cell in report.results:
            assert cell.mean_R_proxy == pytest.approx(1.0, abs=1e-9)
            assert cell.proportion_above == pytest.approx(1.0, abs=1e-9)

    def test_full_coupling_admits_everything_below_unit_threshold(self) -> None:
        grid = _grid(
            thresholds=(0.10, 0.50, 0.99),
            couplings=(1.0,),
            seeds=(1,),
            n_samples=64,
        )
        report = sweep_parameters(grid)
        for cell in report.results:
            assert cell.n_above_threshold == 64


# ---------------------------------------------------------------------------
# 6. Schema validation.
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_empty_thresholds_rejected(self) -> None:
        with pytest.raises(SweepError, match="threshold_values"):
            SweepGrid(
                threshold_values=(),
                coupling_strengths=(0.5,),
                trace_seeds=(1,),
                n_samples=64,
            )

    def test_threshold_out_of_range_rejected(self) -> None:
        with pytest.raises(SweepError, match=r"\(0, 1\)"):
            SweepGrid(
                threshold_values=(1.2,),
                coupling_strengths=(0.5,),
                trace_seeds=(1,),
                n_samples=64,
            )

    def test_non_increasing_thresholds_rejected(self) -> None:
        with pytest.raises(SweepError, match="strictly increasing"):
            SweepGrid(
                threshold_values=(0.5, 0.4),
                coupling_strengths=(0.5,),
                trace_seeds=(1,),
                n_samples=64,
            )

    def test_coupling_out_of_range_rejected(self) -> None:
        with pytest.raises(SweepError, match=r"coupling_strength"):
            SweepGrid(
                threshold_values=(0.5,),
                coupling_strengths=(1.5,),
                trace_seeds=(1,),
                n_samples=64,
            )

    def test_too_few_samples_rejected(self) -> None:
        with pytest.raises(SweepError, match="n_samples"):
            SweepGrid(
                threshold_values=(0.5,),
                coupling_strengths=(0.5,),
                trace_seeds=(1,),
                n_samples=16,
            )

    def test_empty_seeds_rejected(self) -> None:
        with pytest.raises(SweepError, match="trace_seeds"):
            SweepGrid(
                threshold_values=(0.5,),
                coupling_strengths=(0.5,),
                trace_seeds=(),
                n_samples=64,
            )


# ---------------------------------------------------------------------------
# 7. JSON-safe round-trip.
# ---------------------------------------------------------------------------


class TestJsonProjection:
    def test_to_json_dict_round_trip(self) -> None:
        grid = _grid()
        report = sweep_parameters(grid, seed=42)
        d = report.to_json_dict()
        text = json.dumps(d)
        loaded = json.loads(text)
        assert loaded["seed"] == 42
        assert loaded["grid"]["n_samples"] == 64
        assert len(loaded["results"]) == grid.total_cells

    def test_results_are_flat_dicts(self) -> None:
        grid = _grid()
        report = sweep_parameters(grid)
        d = report.to_json_dict()
        for cell in d["results"]:
            for k, v in cell.items():
                assert isinstance(v, (str, int, float, bool)), (
                    f"cell field {k!r} is not primitive: {v!r}"
                )


# ---------------------------------------------------------------------------
# 8. Frozen dataclasses + per-cell construction validation.
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_grid_is_frozen(self) -> None:
        grid = _grid()
        with pytest.raises((AttributeError, TypeError)):
            grid.n_samples = 999  # type: ignore[misc]

    def test_cell_result_is_frozen(self) -> None:
        report = sweep_parameters(_grid())
        cell = report.results[0]
        with pytest.raises((AttributeError, TypeError)):
            cell.proportion_above = 0.0  # type: ignore[misc]

    def test_report_is_frozen(self) -> None:
        report = sweep_parameters(_grid())
        with pytest.raises((AttributeError, TypeError)):
            report.seed = 999  # type: ignore[misc]

    def test_cell_result_validates_proportion_in_unit_interval(self) -> None:
        with pytest.raises(SweepError, match="proportion_above"):
            SweepCellResult(
                threshold=0.5,
                coupling_strength=0.5,
                seed=1,
                n_samples=10,
                n_above_threshold=5,
                proportion_above=1.5,  # out of range
                mean_R_proxy=0.5,
            )

    def test_report_validates_results_length(self) -> None:
        grid = _grid()
        with pytest.raises(SweepError, match="results length"):
            SweepReport(grid=grid, seed=0, results=())  # zero results


# ---------------------------------------------------------------------------
# 9. Helper API + aesthetic rich __repr__.
# ---------------------------------------------------------------------------


class TestHelperAPI:
    def test_by_threshold_filters(self) -> None:
        grid = _grid()
        report = sweep_parameters(grid)
        for t in grid.threshold_values:
            cells = report.by_threshold(t)
            assert len(cells) == len(grid.coupling_strengths) * len(grid.trace_seeds)
            for cell in cells:
                assert cell.threshold == t

    def test_by_coupling_filters(self) -> None:
        grid = _grid()
        report = sweep_parameters(grid)
        for c in grid.coupling_strengths:
            cells = report.by_coupling(c)
            assert len(cells) == len(grid.threshold_values) * len(grid.trace_seeds)
            for cell in cells:
                assert cell.coupling_strength == c

    def test_repr_surfaces_summary(self) -> None:
        grid = _grid()
        report = sweep_parameters(grid)
        r = repr(report)
        assert r.startswith("SweepReport[")
        assert "cells=27" in r

    def test_cell_repr_format(self) -> None:
        report = sweep_parameters(_grid())
        cell = report.results[0]
        r = repr(cell)
        assert r.startswith("SweepCellResult[")
        assert "θ=" in r
        assert "c=" in r
        assert "admit=" in r
        assert "R̄=" in r
