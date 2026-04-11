"""D2 — stillness parameter calibration.

Addresses the lingering gap from D1: ``StillnessDetector`` has four
hyperparameters (``eps_R``, ``eps_F``, ``delta_min``, ``window``)
that are currently set by eyeballing. D2 turns them into a
defensible, **out-of-sample-validated** choice using the same
methodology as D1 — a transparent grid search over H1 synthetic
traces with an explicit train/test split and a reported
generalization gap.

Objective
---------

Unlike D1, which was a binary classifier problem (locked vs.
uncoupled), D2 is a **regime-discrimination** problem: given a
trace, should the stillness detector emit ``STILL`` or ``ACTIVE``?
Two regimes are constructed synthetically:

* **quiet** — near-constant `R(t) ≈ 0.95`, near-zero `δ(t) ≈ 0.01`.
  The detector **should** emit ``STILL``.
* **active** — ``R(t)`` drifts linearly from `0.85 → 0.95`, `δ(t)`
  stays around `0.01`. The detector **should** emit ``ACTIVE``
  because `|dR/dt|` clears ``eps_R``.

For each candidate parameter 4-tuple, we count:

* ``TPR_quiet`` — fraction of quiet samples (post-warmup) classified
  as ``STILL``.
* ``FPR_active`` — fraction of active samples (post-warmup)
  incorrectly classified as ``STILL``.
* **Youden's J** = ``TPR_quiet − FPR_active``, matching D1's
  calibration metric.

The best parameter set on the train split is selected, and its
performance on the test split is reported as the honest OOS number
— exactly like D1.

What D2 does NOT do
-------------------

* It does **not** change the library defaults in
  ``neurophase.gate.stillness_detector``. Updating the defaults is
  a deliberate policy decision that requires cross-session
  replication; D2 only gives the caller *evidence*.
* It does **not** calibrate ``delta_min`` across the full `[0, π]`
  range — that would be meaningless since `δ` in H1 traces stays
  well below `0.1`. The grid focuses on the physically relevant
  neighbourhood of the current default.
* It does **not** optimize for run-time latency. The cost function
  is purely classification accuracy.

Determinism
-----------

Every calibration run is deterministic: same grid + same seeds →
bit-identical report, including the parameter fingerprint (HN11
re-applied to stillness parameters). Covered by
``tests/test_calibration_stillness.py::TestStillnessDeterminism``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product as cartesian_product
from typing import Final

import numpy as np
from numpy.typing import NDArray

from neurophase.audit.decision_ledger import fingerprint_parameters
from neurophase.gate.stillness_detector import StillnessDetector, StillnessState

#: Default sweep for ``eps_R``.
DEFAULT_EPS_R_GRID: Final[tuple[float, ...]] = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2)
#: Default sweep for ``eps_F`` (same scale — chain-rule product).
DEFAULT_EPS_F_GRID: Final[tuple[float, ...]] = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2)
#: Default sweep for ``delta_min``. Below ``0.01`` the detector would
#: require phase-lock precision below the noise floor of any real
#: bio-sensor; above ``0.15`` the criterion accepts biased locks.
DEFAULT_DELTA_MIN_GRID: Final[tuple[float, ...]] = (0.01, 0.05, 0.10, 0.15)
#: Default sweep for rolling window length. All ≥ 2 as required by
#: the detector.
DEFAULT_WINDOW_GRID: Final[tuple[int, ...]] = (4, 8, 16, 32)


@dataclass(frozen=True)
class StillnessGrid:
    """Explicit D2 calibration grid.

    Attributes
    ----------
    eps_R_values
        Candidate values for ``eps_R`` (tolerance on ``max |dR/dt|``).
    eps_F_values
        Candidate values for ``eps_F`` (tolerance on ``max |dF_proxy/dt|``).
    delta_min_values
        Candidate values for ``delta_min`` (tolerance on ``max δ``).
    window_values
        Candidate rolling-window lengths. Every value must be ≥ 2.
    quiet_seeds
        RNG seeds used to generate quiet traces (one trace per seed).
    active_seeds
        RNG seeds used to generate active traces.
    n_samples
        Length of each synthetic trace. Must be ≥ 32 to give the
        detector room to warm up and emit at least a handful of
        classified samples.
    dt
        Sample step used when constructing the trace. Must be > 0.
    """

    eps_R_values: tuple[float, ...] = DEFAULT_EPS_R_GRID
    eps_F_values: tuple[float, ...] = DEFAULT_EPS_F_GRID
    delta_min_values: tuple[float, ...] = DEFAULT_DELTA_MIN_GRID
    window_values: tuple[int, ...] = DEFAULT_WINDOW_GRID
    quiet_seeds: tuple[int, ...] = tuple(range(200, 208))
    active_seeds: tuple[int, ...] = tuple(range(300, 308))
    n_samples: int = 128
    dt: float = 0.01

    def __post_init__(self) -> None:
        for name, seq in (
            ("eps_R_values", self.eps_R_values),
            ("eps_F_values", self.eps_F_values),
            ("delta_min_values", self.delta_min_values),
        ):
            if not seq:
                raise ValueError(f"{name} must be non-empty")
            if not all(v > 0 for v in seq):
                raise ValueError(f"every {name} entry must be > 0")
        if not self.window_values:
            raise ValueError("window_values must be non-empty")
        if not all(w >= 2 for w in self.window_values):
            raise ValueError("every window_values entry must be ≥ 2")
        if not self.quiet_seeds or not self.active_seeds:
            raise ValueError("quiet_seeds and active_seeds must be non-empty")
        if self.n_samples < 32:
            raise ValueError(f"n_samples must be ≥ 32, got {self.n_samples}")
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")


@dataclass(frozen=True)
class StillnessCellEvaluation:
    """Per-parameter-tuple score on one split."""

    eps_R: float
    eps_F: float
    delta_min: float
    window: int
    tpr_quiet: float
    fpr_active: float
    youden_j: float


@dataclass(frozen=True, repr=False)
class StillnessCalibrationReport:
    """Frozen D2 calibration report.

    Attributes
    ----------
    best_cell
        The parameter 4-tuple (cell) that maximised Youden's J on
        the **train** split.
    train_evaluations
        Per-cell score on the train split, in grid order.
    test_evaluations
        Per-cell score on the test split, in grid order.
    train_score_at_best
        Train J at ``best_cell``.
    test_score_at_best
        Test J at ``best_cell``.
    generalization_gap
        ``train − test``. Positive = possible overfitting.
    parameter_fingerprint
        Deterministic SHA256 of the grid configuration.
    """

    best_cell: StillnessCellEvaluation
    train_evaluations: tuple[StillnessCellEvaluation, ...]
    test_evaluations: tuple[StillnessCellEvaluation, ...]
    train_score_at_best: float
    test_score_at_best: float
    generalization_gap: float
    parameter_fingerprint: str

    def __repr__(self) -> str:  # aesthetic rich repr (HN22)
        cell = self.best_cell
        return (
            f"StillnessCalibrationReport[eps_R={cell.eps_R:g} · "
            f"eps_F={cell.eps_F:g} · "
            f"δ_min={cell.delta_min:.3f} · "
            f"window={cell.window} · "
            f"train_J={self.train_score_at_best:.3f} · "
            f"test_J={self.test_score_at_best:.3f}]"
        )

    def to_json_dict(self) -> dict[str, object]:
        def _row(e: StillnessCellEvaluation) -> dict[str, object]:
            return {
                "eps_R": e.eps_R,
                "eps_F": e.eps_F,
                "delta_min": e.delta_min,
                "window": e.window,
                "tpr_quiet": e.tpr_quiet,
                "fpr_active": e.fpr_active,
                "youden_j": e.youden_j,
            }

        return {
            "best_cell": _row(self.best_cell),
            "train_evaluations": [_row(e) for e in self.train_evaluations],
            "test_evaluations": [_row(e) for e in self.test_evaluations],
            "train_score_at_best": self.train_score_at_best,
            "test_score_at_best": self.test_score_at_best,
            "generalization_gap": self.generalization_gap,
            "parameter_fingerprint": self.parameter_fingerprint,
        }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def calibrate_stillness_parameters(
    grid: StillnessGrid,
    *,
    train_fraction: float = 0.5,
) -> StillnessCalibrationReport:
    """Run the D2 calibration sweep and return a frozen report.

    Parameters
    ----------
    grid
        Explicit :class:`StillnessGrid`.
    train_fraction
        Fraction of each seed list used as the train split. Must be
        in ``(0, 1)``.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    quiet_train, quiet_test = _split(grid.quiet_seeds, train_fraction)
    active_train, active_test = _split(grid.active_seeds, train_fraction)
    if not quiet_train or not quiet_test or not active_train or not active_test:
        raise ValueError(
            "split resulted in an empty train or test partition; "
            "supply more seeds or adjust train_fraction"
        )

    train_scores = _evaluate_grid(
        grid=grid,
        quiet_seeds=quiet_train,
        active_seeds=active_train,
    )
    test_scores = _evaluate_grid(
        grid=grid,
        quiet_seeds=quiet_test,
        active_seeds=active_test,
    )

    js = [e.youden_j for e in train_scores]
    best_idx = int(np.argmax(js))
    best_cell = train_scores[best_idx]
    train_score_at_best = js[best_idx]
    test_score_at_best = test_scores[best_idx].youden_j

    fingerprint = fingerprint_parameters(
        {
            "eps_R_values": list(grid.eps_R_values),
            "eps_F_values": list(grid.eps_F_values),
            "delta_min_values": list(grid.delta_min_values),
            "window_values": list(grid.window_values),
            "quiet_seeds": list(grid.quiet_seeds),
            "active_seeds": list(grid.active_seeds),
            "n_samples": grid.n_samples,
            "dt": grid.dt,
            "train_fraction": train_fraction,
            "objective": "youden_j_still_vs_active",
        }
    )

    return StillnessCalibrationReport(
        best_cell=best_cell,
        train_evaluations=tuple(train_scores),
        test_evaluations=tuple(test_scores),
        train_score_at_best=train_score_at_best,
        test_score_at_best=test_score_at_best,
        generalization_gap=train_score_at_best - test_score_at_best,
        parameter_fingerprint=fingerprint,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _split(seeds: Sequence[int], train_fraction: float) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Deterministic chronological split."""
    n = len(seeds)
    # Use floor so a 2-seed grid with train_fraction=0.5 yields
    # train=1, test=1 (both non-empty) and a 1-seed grid yields
    # an empty test split, which the caller catches.
    n_train = max(1, min(n - 1, int(train_fraction * n)))
    return tuple(seeds[:n_train]), tuple(seeds[n_train:])


def _evaluate_grid(
    *,
    grid: StillnessGrid,
    quiet_seeds: tuple[int, ...],
    active_seeds: tuple[int, ...],
) -> list[StillnessCellEvaluation]:
    """Sweep every parameter tuple and compute (TPR, FPR, J) on one split."""
    quiet_traces = [_generate_quiet_trace(seed, n=grid.n_samples) for seed in quiet_seeds]
    active_traces = [_generate_active_trace(seed, n=grid.n_samples) for seed in active_seeds]

    out: list[StillnessCellEvaluation] = []
    for eps_R, eps_F, delta_min, window in cartesian_product(
        grid.eps_R_values,
        grid.eps_F_values,
        grid.delta_min_values,
        grid.window_values,
    ):
        tpr = _classify_rate(
            traces=quiet_traces,
            eps_R=eps_R,
            eps_F=eps_F,
            delta_min=delta_min,
            window=window,
            dt=grid.dt,
            target=StillnessState.STILL,
        )
        fpr = _classify_rate(
            traces=active_traces,
            eps_R=eps_R,
            eps_F=eps_F,
            delta_min=delta_min,
            window=window,
            dt=grid.dt,
            target=StillnessState.STILL,
        )
        out.append(
            StillnessCellEvaluation(
                eps_R=eps_R,
                eps_F=eps_F,
                delta_min=delta_min,
                window=window,
                tpr_quiet=tpr,
                fpr_active=fpr,
                youden_j=tpr - fpr,
            )
        )
    return out


def _classify_rate(
    *,
    traces: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    eps_R: float,
    eps_F: float,
    delta_min: float,
    window: int,
    dt: float,
    target: StillnessState,
) -> float:
    """Fraction of post-warmup samples where detector emitted ``target``."""
    hits = 0
    total = 0
    for R_series, delta_series in traces:
        detector = StillnessDetector(
            window=window,
            eps_R=eps_R,
            eps_F=eps_F,
            delta_min=delta_min,
            dt=dt,
        )
        for R, delta in zip(R_series, delta_series, strict=True):
            decision = detector.update(R=float(R), delta=float(delta))
            if not decision.window_filled:
                continue  # skip warmup
            total += 1
            if decision.state is target:
                hits += 1
    if total == 0:
        return 0.0
    return hits / total


#: Noise amplitude on the synthetic traces. Chosen so that at
#: ``dt = 0.01`` the finite-difference derivative stays well below
#: the tightest ``eps_R`` / ``eps_F`` in the default grid (1e-4).
#: With ``σ = 1e-7`` the one-step derivative has max
#: ``~sqrt(2) · σ / dt ≈ 1.4e-5`` — two orders of magnitude below
#: the tightest tolerance, so noise alone never crosses a threshold.
_SYNTH_NOISE_SIGMA: Final[float] = 1e-7


def _generate_quiet_trace(seed: int, *, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Near-constant R ≈ 0.95, near-zero δ ≈ 0.01. Should be STILL."""
    rng = np.random.default_rng(seed)
    R = np.full(n, 0.95, dtype=np.float64) + rng.normal(0, _SYNTH_NOISE_SIGMA, size=n)
    delta = np.full(n, 0.01, dtype=np.float64) + np.abs(rng.normal(0, _SYNTH_NOISE_SIGMA, size=n))
    R = np.clip(R, 0.0, 1.0)
    delta = np.clip(delta, 0.0, float(np.pi))
    return R, delta


def _generate_active_trace(seed: int, *, n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """R drifting 0.85 → 0.95 linearly; δ stays small. Should be ACTIVE.

    The drift over ``n`` samples at ``dt = 0.01`` gives a per-tick
    derivative of ``(0.95 − 0.85) / (n · dt) · dt ≈ 0.1 / n``. For
    ``n = 128`` this is ``≈ 7.8e-4``, which cleanly clears the
    default ``eps_R = 1e-3`` → ``ACTIVE`` on all cells that use a
    tighter or equal tolerance.
    """
    rng = np.random.default_rng(seed)
    R = np.linspace(0.85, 0.95, n, dtype=np.float64)
    R += rng.normal(0, _SYNTH_NOISE_SIGMA, size=n)
    delta = np.full(n, 0.01, dtype=np.float64) + np.abs(rng.normal(0, _SYNTH_NOISE_SIGMA, size=n))
    R = np.clip(R, 0.0, 1.0)
    delta = np.clip(delta, 0.0, float(np.pi))
    return R, delta
