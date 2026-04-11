"""D1 — gate threshold calibration over H1 synthetic ground truth.

Problem statement
-----------------

``ExecutionGate.threshold`` controls the ``R(t) < θ ⇒ BLOCKED``
invariant (``I₁``). Before this task the threshold was a hand-picked
default of ``0.65`` — defensible as a starting point, but not a
calibrated value. D1 turns it into a defensible, **out-of-sample
validated** choice.

The calibration pipeline is:

1. Generate a synthetic trace battery via
   :mod:`neurophase.benchmarks.phase_coupling` at a grid of coupling
   strengths ``c ∈ {c_null, c_mid, c_lock, …}``.
2. For each candidate threshold ``θ ∈ grid`` and each trace, compute
   a scalar gate statistic — the proportion of samples whose
   ``R_proxy = (1 + cos δ)/2`` clears ``θ``. This is exactly what a
   real gate would do on that trace.
3. Score each threshold by a caller-specified objective. The default
   objective is **Youden's J statistic** on the {null, locked}
   contrast:

   .. math::

      J(\\theta)
      \\;=\\;
      \\text{TPR}_{\\text{lock}}(\\theta) - \\text{FPR}_{\\text{null}}(\\theta)

   where TPR is the proportion of locked-trace samples admitted
   and FPR is the proportion of null-trace samples incorrectly
   admitted. ``J ∈ [-1, 1]`` and is maximized by the threshold
   that most cleanly separates the two distributions.

4. The caller supplies a **train split** and a **test split** of
   seeds. The grid search runs on train; the best threshold is
   evaluated on test. Both scores are reported. A gap between train
   and test scores is a direct diagnostic for overfitting.

5. The full report is a frozen, JSON-serializable
   :class:`ThresholdCalibrationReport` carrying a parameter
   fingerprint (via :func:`~neurophase.audit.decision_ledger.fingerprint_parameters`)
   for audit-log reproducibility.

What calibration does NOT do
----------------------------

* It does **not** change the default threshold in the library.
  Changing ``DEFAULT_THRESHOLD`` is a separate, deliberate decision
  that requires cross-session replication (T2.6 → T3.3). D1 gives
  the caller *evidence*; the caller picks the policy.
* It does **not** calibrate stillness parameters (those are D2).
* It does **not** learn a parametric model. D1 is a grid search
  with transparent scoring — deliberately simple so reviewers can
  understand exactly what was optimized.

Why a grid instead of a continuous optimizer
--------------------------------------------

For a monotone score a coarse grid + a narrow refined grid covers
the optimum with no risk of local minima. A continuous optimizer
would introduce numerical noise and hide the score surface from the
reviewer. The grid is explicit, auditable, and reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from neurophase.audit.decision_ledger import fingerprint_parameters
from neurophase.benchmarks.phase_coupling import (
    PhaseCouplingConfig,
    generate_phase_coupling,
)

#: Default calibration grid — 21 thresholds from 0.05 to 0.95 in 0.05 steps.
#: Explicit and dense enough for Youden's J to have a clear maximum.
DEFAULT_THRESHOLD_GRID: Final[tuple[float, ...]] = tuple(
    round(0.05 + 0.05 * i, 2)
    for i in range(19)  # 0.05, 0.10, …, 0.95
)


@dataclass(frozen=True)
class ThresholdGrid:
    """Explicit calibration grid.

    Attributes
    ----------
    thresholds
        Candidate thresholds in ``(0, 1)``, strictly increasing.
    null_seeds
        Integer seeds used to generate uncoupled (c=0) traces.
    locked_seeds
        Integer seeds used to generate fully-locked (c=1) traces.
    n_samples
        Length of each synthetic trace.
    """

    thresholds: tuple[float, ...]
    null_seeds: tuple[int, ...]
    locked_seeds: tuple[int, ...]
    n_samples: int

    def __post_init__(self) -> None:
        if not self.thresholds:
            raise ValueError("thresholds must be non-empty")
        if not all(0.0 < t < 1.0 for t in self.thresholds):
            raise ValueError("every threshold must be in (0, 1)")
        if not all(
            self.thresholds[i] < self.thresholds[i + 1] for i in range(len(self.thresholds) - 1)
        ):
            raise ValueError("thresholds must be strictly increasing")
        if not self.null_seeds or not self.locked_seeds:
            raise ValueError("null_seeds and locked_seeds must be non-empty")
        if self.n_samples < 32:
            raise ValueError(f"n_samples must be ≥ 32 for a stable estimate, got {self.n_samples}")


@dataclass(frozen=True)
class ThresholdEvaluation:
    """Per-threshold score on one split (train or test)."""

    threshold: float
    tpr_locked: float
    fpr_null: float
    youden_j: float


@dataclass(frozen=True, repr=False)
class ThresholdCalibrationReport:
    """Frozen, JSON-serializable output of :func:`calibrate_gate_threshold`.

    Attributes
    ----------
    best_threshold
        The threshold that maximized the objective on the **train**
        split. The ``test_score_at_best`` value reports that same
        threshold's performance on the **test** split — the honest
        out-of-sample number.
    train_evaluations
        Per-threshold score on the train split, in grid order.
    test_evaluations
        Per-threshold score on the test split, in grid order. This
        exists so the caller can inspect the full OOS curve, not
        just the best-threshold value.
    train_score_at_best
        Train-split objective value at ``best_threshold``.
    test_score_at_best
        Test-split objective value at ``best_threshold``.
    generalization_gap
        ``train_score_at_best - test_score_at_best``. A gap > 0 is
        the classic overfitting signature — the grid search
        over-selected on train. A gap ≤ 0 is a clean calibration.
    parameter_fingerprint
        SHA256 of the calibration configuration (grid, seeds,
        sample count, objective name) for audit-log reproducibility.
    objective_name
        String identifier of the objective function used.
    """

    best_threshold: float
    train_evaluations: tuple[ThresholdEvaluation, ...]
    test_evaluations: tuple[ThresholdEvaluation, ...]
    train_score_at_best: float
    test_score_at_best: float
    generalization_gap: float
    parameter_fingerprint: str
    objective_name: str

    def __repr__(self) -> str:  # aesthetic rich repr (HN22)
        return (
            f"ThresholdCalibrationReport[best={self.best_threshold:.2f} · "
            f"train={self.train_score_at_best:.3f} · "
            f"test={self.test_score_at_best:.3f} · "
            f"gap={self.generalization_gap:+.3f} · "
            f"{self.objective_name}]"
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return a plain dict suitable for ``json.dumps``."""
        return {
            "best_threshold": self.best_threshold,
            "train_evaluations": [
                {
                    "threshold": e.threshold,
                    "tpr_locked": e.tpr_locked,
                    "fpr_null": e.fpr_null,
                    "youden_j": e.youden_j,
                }
                for e in self.train_evaluations
            ],
            "test_evaluations": [
                {
                    "threshold": e.threshold,
                    "tpr_locked": e.tpr_locked,
                    "fpr_null": e.fpr_null,
                    "youden_j": e.youden_j,
                }
                for e in self.test_evaluations
            ],
            "train_score_at_best": self.train_score_at_best,
            "test_score_at_best": self.test_score_at_best,
            "generalization_gap": self.generalization_gap,
            "parameter_fingerprint": self.parameter_fingerprint,
            "objective_name": self.objective_name,
        }


ObjectiveFn = Callable[[ThresholdEvaluation], float]


def youden_j_objective(evaluation: ThresholdEvaluation) -> float:
    """Youden's J statistic — the default calibration objective."""
    return evaluation.youden_j


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def calibrate_gate_threshold(
    grid: ThresholdGrid,
    *,
    train_fraction: float = 0.5,
    objective: ObjectiveFn | None = None,
    objective_name: str = "youden_j",
) -> ThresholdCalibrationReport:
    """Calibrate a gate threshold over an H1 synthetic battery.

    Parameters
    ----------
    grid
        Explicit grid of thresholds and seeds to sweep.
    train_fraction
        Fraction of each seed list used as the train split. Must be
        in ``(0, 1)``. The remainder is the test split.
    objective
        Scalar objective on a :class:`ThresholdEvaluation`. Defaults
        to Youden's J.
    objective_name
        Identifier of the objective (used only in the fingerprint
        and in the report).

    Returns
    -------
    ThresholdCalibrationReport
        Full provenance of the calibration run.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    fn: ObjectiveFn = objective if objective is not None else youden_j_objective

    null_train, null_test = _split(grid.null_seeds, train_fraction)
    locked_train, locked_test = _split(grid.locked_seeds, train_fraction)
    if not null_train or not null_test or not locked_train or not locked_test:
        raise ValueError(
            "split resulted in an empty train or test partition; "
            "supply more seeds or adjust train_fraction"
        )

    train_scores = _evaluate_grid(
        grid=grid,
        null_seeds=null_train,
        locked_seeds=locked_train,
    )
    test_scores = _evaluate_grid(
        grid=grid,
        null_seeds=null_test,
        locked_seeds=locked_test,
    )

    # Select on train only — the test split is held out from selection
    # and only used for the reported OOS number.
    train_best_idx = int(np.argmax([fn(e) for e in train_scores]))
    best_threshold = train_scores[train_best_idx].threshold
    train_score_at_best = fn(train_scores[train_best_idx])
    test_score_at_best = fn(test_scores[train_best_idx])

    fingerprint = fingerprint_parameters(
        {
            "thresholds": list(grid.thresholds),
            "null_seeds": list(grid.null_seeds),
            "locked_seeds": list(grid.locked_seeds),
            "n_samples": grid.n_samples,
            "train_fraction": train_fraction,
            "objective_name": objective_name,
        }
    )

    return ThresholdCalibrationReport(
        best_threshold=best_threshold,
        train_evaluations=tuple(train_scores),
        test_evaluations=tuple(test_scores),
        train_score_at_best=train_score_at_best,
        test_score_at_best=test_score_at_best,
        generalization_gap=train_score_at_best - test_score_at_best,
        parameter_fingerprint=fingerprint,
        objective_name=objective_name,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _split(seeds: Sequence[int], train_fraction: float) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Deterministic split — first ``⌈train_fraction · n⌉`` seeds → train."""
    n = len(seeds)
    n_train = max(1, min(n - 1, math.ceil(train_fraction * n)))
    return tuple(seeds[:n_train]), tuple(seeds[n_train:])


def _evaluate_grid(
    *,
    grid: ThresholdGrid,
    null_seeds: tuple[int, ...],
    locked_seeds: tuple[int, ...],
) -> list[ThresholdEvaluation]:
    """Compute per-threshold (TPR, FPR, Youden's J) on one split."""
    null_scores = _r_proxy_scores(seeds=null_seeds, coupling_strength=0.0, n=grid.n_samples)
    locked_scores = _r_proxy_scores(seeds=locked_seeds, coupling_strength=1.0, n=grid.n_samples)

    out: list[ThresholdEvaluation] = []
    for theta in grid.thresholds:
        tpr = float((locked_scores >= theta).mean())
        fpr = float((null_scores >= theta).mean())
        out.append(
            ThresholdEvaluation(
                threshold=theta,
                tpr_locked=tpr,
                fpr_null=fpr,
                youden_j=tpr - fpr,
            )
        )
    return out


def _r_proxy_scores(
    *, seeds: tuple[int, ...], coupling_strength: float, n: int
) -> NDArray[np.float64]:
    """Concatenate ``R_proxy = (1 + cos δ)/2`` across all supplied seeds.

    ``δ = |arccos(cos(φ_x - φ_y))|`` is the circular distance used by
    the rest of the pipeline (see
    :mod:`neurophase.analysis.prediction_error`). The per-sample
    ``R_proxy`` is exactly what the stillness detector's ``F_proxy``
    contract operates on, and what the gate's threshold test compares
    against in the two-oscillator limit.
    """
    chunks: list[NDArray[np.float64]] = []
    for seed in seeds:
        trace = generate_phase_coupling(
            PhaseCouplingConfig(
                n=n,
                coupling_strength=coupling_strength,
                phi_offset=0.0,
                seed=seed,
            )
        )
        delta = np.arccos(np.clip(np.cos(trace.phi_x - trace.phi_y), -1.0, 1.0))
        r_proxy = 0.5 * (1.0 + np.cos(delta))
        chunks.append(r_proxy)
    return np.concatenate(chunks).astype(np.float64)
