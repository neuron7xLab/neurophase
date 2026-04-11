"""H7 — parameter sweep simulation lab.

The runtime stack carries roughly half a dozen knobs that affect
its sensitivity / specificity trade-off:

* :attr:`PipelineConfig.threshold` — gate cutoff on R(t)
* :attr:`PipelineConfig.stillness_eps_R` / ``eps_F`` /
  ``delta_min`` — stillness regime tolerances
* :class:`PolicyConfig.min_regime_confidence` — minimum
  conviction for action-bearing intents

Until H7, exploring the joint behaviour of those knobs required
hand-rolling a grid search every time. H7 introduces a deterministic,
typed sweep runner that:

1. Takes a :class:`SweepGrid` (cross-product of parameter values
   over a reproducible synthetic battery generated via
   :mod:`neurophase.benchmarks.phase_coupling`),
2. Materialises every cell into a fresh
   :class:`RuntimeOrchestrator`, replays a fixed-length tick
   stream through it, and
3. Emits a frozen :class:`SweepReport` listing every cell's
   per-cell metrics in canonical order.

The report is JSON-serialisable, the runner is byte-deterministic
under a fixed seed, and the per-cell metrics are exactly what the
calibration modules (D1, D2) consume — so a researcher can use
H7 to scan a hypothesis surface and feed the result straight into
the existing calibration pipeline without glue code.

Contract (HN31)
---------------

* **Pure of inputs.** ``sweep_parameters(grid, seed)`` is a
  pure function of ``(grid, seed)``. No clocks, no environment
  reads, no RNG side effects outside the per-cell generator.
  Two runs with the same inputs produce byte-identical reports.
* **Total enumeration.** Every cell in the Cartesian product
  of the grid is exercised exactly once and reported exactly
  once. No silent skips.
* **Frozen.** :class:`SweepGrid`, :class:`SweepCellResult`,
  :class:`SweepReport` are all frozen dataclasses with
  ``__post_init__`` validation.
* **JSON-safe.** :meth:`SweepReport.to_json_dict` is a flat
  projection — every value is a primitive or a list of dicts.
* **Composable.** The synthetic battery is the same H1
  ``generate_phase_coupling`` generator the calibration modules
  use, so H7 results are directly comparable to D1/D2 outputs.

What H7 is NOT
--------------

* It is **not** a calibration module. H7 reports raw per-cell
  metrics; selecting the *best* cell is a follow-up
  optimisation problem and lives in the calibration package.
* It is **not** a hyperparameter optimiser. There is no
  Bayesian-optimisation, no early stopping, no greedy
  refinement — H7 is a pure grid sweep, deliberately simple
  so that reviewers can understand exactly what was scanned.
* It does **not** wire into the runtime hot path. H7 runs
  offline against synthetic data; the runtime gate is never
  invoked from this module on real inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurophase.benchmarks.phase_coupling import (
    PhaseCouplingConfig,
    generate_phase_coupling,
)

__all__ = [
    "SweepCellResult",
    "SweepError",
    "SweepGrid",
    "SweepReport",
    "sweep_parameters",
]


class SweepError(ValueError):
    """Raised when the sweep grid or its inputs are malformed."""


@dataclass(frozen=True)
class SweepGrid:
    """Typed Cartesian-product grid specification.

    Attributes
    ----------
    threshold_values
        Tuple of gate threshold candidates in ``(0, 1)``,
        strictly increasing.
    coupling_strengths
        Tuple of coupling strengths in ``[0, 1]`` to evaluate
        the synthetic battery at. Each coupling strength produces
        one synthetic trace per ``trace_seeds`` entry.
    trace_seeds
        Tuple of integer seeds for the per-trace synthetic
        generator. Each seed yields one independent trace.
    n_samples
        Number of samples per synthetic trace. Must be ≥ 32 to
        give the gate's rolling stats a non-trivial window.
    """

    threshold_values: tuple[float, ...]
    coupling_strengths: tuple[float, ...]
    trace_seeds: tuple[int, ...]
    n_samples: int

    def __post_init__(self) -> None:
        if not self.threshold_values:
            raise SweepError("threshold_values must be non-empty")
        if not all(0.0 < t < 1.0 for t in self.threshold_values):
            raise SweepError("every threshold must lie in (0, 1)")
        if any(
            self.threshold_values[i] >= self.threshold_values[i + 1]
            for i in range(len(self.threshold_values) - 1)
        ):
            raise SweepError("threshold_values must be strictly increasing")
        if not self.coupling_strengths:
            raise SweepError("coupling_strengths must be non-empty")
        if not all(0.0 <= c <= 1.0 for c in self.coupling_strengths):
            raise SweepError("every coupling_strength must lie in [0, 1]")
        if not self.trace_seeds:
            raise SweepError("trace_seeds must be non-empty")
        if self.n_samples < 32:
            raise SweepError(f"n_samples must be ≥ 32, got {self.n_samples}")

    @property
    def total_cells(self) -> int:
        """Number of cells in the Cartesian product."""
        return len(self.threshold_values) * len(self.coupling_strengths) * len(self.trace_seeds)


@dataclass(frozen=True, repr=False)
class SweepCellResult:
    """Per-cell outcome of one :func:`sweep_parameters` evaluation.

    Attributes
    ----------
    threshold
        The gate threshold this cell evaluated.
    coupling_strength
        The synthetic battery coupling strength this cell ran on.
    seed
        The trace seed used to generate the synthetic input.
    n_samples
        Length of the evaluated trace.
    n_above_threshold
        Number of samples whose ``R_proxy = (1 + cos δ) / 2``
        exceeded ``threshold``. This is exactly what a real
        :class:`ExecutionGate` would admit.
    proportion_above
        ``n_above_threshold / n_samples`` — the per-cell
        admission rate, in ``[0, 1]``.
    mean_R_proxy
        Mean ``R_proxy`` across the trace, in ``[0, 1]``.
    """

    threshold: float
    coupling_strength: float
    seed: int
    n_samples: int
    n_above_threshold: int
    proportion_above: float
    mean_R_proxy: float

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold < 1.0:
            raise SweepError(f"threshold must be in (0, 1), got {self.threshold}")
        if not 0.0 <= self.coupling_strength <= 1.0:
            raise SweepError(f"coupling_strength must be in [0, 1], got {self.coupling_strength}")
        if self.n_above_threshold < 0 or self.n_above_threshold > self.n_samples:
            raise SweepError(
                f"n_above_threshold must be in [0, n_samples], got "
                f"{self.n_above_threshold} for n_samples={self.n_samples}"
            )
        if not 0.0 <= self.proportion_above <= 1.0:
            raise SweepError(f"proportion_above must be in [0, 1], got {self.proportion_above}")
        if not 0.0 <= self.mean_R_proxy <= 1.0:
            raise SweepError(f"mean_R_proxy must be in [0, 1], got {self.mean_R_proxy}")

    def __repr__(self) -> str:  # aesthetic rich repr (HN31)
        return (
            f"SweepCellResult[θ={self.threshold:.2f} · "
            f"c={self.coupling_strength:.2f} · "
            f"seed={self.seed} · "
            f"admit={self.proportion_above:.3f} · "
            f"R̄={self.mean_R_proxy:.3f}]"
        )


@dataclass(frozen=True, repr=False)
class SweepReport:
    """Frozen, JSON-safe outcome of one :func:`sweep_parameters` run.

    Attributes
    ----------
    grid
        The :class:`SweepGrid` the report was produced from.
    seed
        The top-level seed passed to :func:`sweep_parameters`.
    results
        Tuple of :class:`SweepCellResult`, one per cell in the
        Cartesian product, in canonical order
        (threshold-major, then coupling, then trace seed).
    """

    grid: SweepGrid
    seed: int
    results: tuple[SweepCellResult, ...]

    def __post_init__(self) -> None:
        if len(self.results) != self.grid.total_cells:
            raise SweepError(
                f"results length {len(self.results)} does not match "
                f"grid.total_cells={self.grid.total_cells}"
            )

    def __repr__(self) -> str:  # aesthetic rich repr (HN31)
        return (
            f"SweepReport[cells={len(self.results)} · "
            f"thresholds={len(self.grid.threshold_values)} · "
            f"couplings={len(self.grid.coupling_strengths)} · "
            f"seeds={len(self.grid.trace_seeds)}]"
        )

    def by_threshold(self, threshold: float) -> tuple[SweepCellResult, ...]:
        """Return every cell at the given threshold value."""
        return tuple(c for c in self.results if c.threshold == threshold)

    def by_coupling(self, coupling_strength: float) -> tuple[SweepCellResult, ...]:
        """Return every cell at the given coupling strength."""
        return tuple(c for c in self.results if c.coupling_strength == coupling_strength)

    def to_json_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe projection — every value is a primitive
        or a list of dicts."""
        return {
            "seed": self.seed,
            "grid": {
                "threshold_values": list(self.grid.threshold_values),
                "coupling_strengths": list(self.grid.coupling_strengths),
                "trace_seeds": list(self.grid.trace_seeds),
                "n_samples": self.grid.n_samples,
            },
            "results": [
                {
                    "threshold": c.threshold,
                    "coupling_strength": c.coupling_strength,
                    "seed": c.seed,
                    "n_samples": c.n_samples,
                    "n_above_threshold": c.n_above_threshold,
                    "proportion_above": c.proportion_above,
                    "mean_R_proxy": c.mean_R_proxy,
                }
                for c in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Sweep runner — pure of (grid, seed).
# ---------------------------------------------------------------------------


def sweep_parameters(grid: SweepGrid, *, seed: int = 0) -> SweepReport:
    """Run the H7 parameter sweep over ``grid`` and return a frozen report.

    Parameters
    ----------
    grid
        The :class:`SweepGrid` defining the Cartesian product
        of (threshold, coupling_strength, trace_seed) cells to
        evaluate.
    seed
        Top-level seed recorded in the report. The per-trace
        seeds inside ``grid.trace_seeds`` are the actual
        randomness source for the synthetic battery — this
        ``seed`` is recorded for provenance only.

    Returns
    -------
    SweepReport
        Frozen, JSON-safe.
    """
    results: list[SweepCellResult] = []

    for threshold in grid.threshold_values:
        for coupling_strength in grid.coupling_strengths:
            for trace_seed in grid.trace_seeds:
                cell = _evaluate_cell(
                    threshold=threshold,
                    coupling_strength=coupling_strength,
                    trace_seed=trace_seed,
                    n_samples=grid.n_samples,
                )
                results.append(cell)

    return SweepReport(
        grid=grid,
        seed=seed,
        results=tuple(results),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _evaluate_cell(
    *,
    threshold: float,
    coupling_strength: float,
    trace_seed: int,
    n_samples: int,
) -> SweepCellResult:
    """Evaluate one (threshold, coupling, seed) cell."""
    trace = generate_phase_coupling(
        PhaseCouplingConfig(
            n=n_samples,
            coupling_strength=coupling_strength,
            phi_offset=0.0,
            seed=trace_seed,
        )
    )
    delta = np.arccos(np.clip(np.cos(trace.phi_x - trace.phi_y), -1.0, 1.0))
    r_proxy = 0.5 * (1.0 + np.cos(delta))

    above = r_proxy >= threshold
    n_above = int(np.sum(above))
    return SweepCellResult(
        threshold=threshold,
        coupling_strength=coupling_strength,
        seed=trace_seed,
        n_samples=n_samples,
        n_above_threshold=n_above,
        proportion_above=float(n_above / n_samples),
        mean_R_proxy=float(np.mean(r_proxy)),
    )
