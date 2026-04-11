"""Phase Locking Value — the falsification metric.

    PLV = |mean( exp(i · (φ_x(t) − φ_y(t))) )|,   PLV ∈ [0, 1]

    0 → no phase relationship (random phase difference)
    1 → perfect phase locking

The falsifiable predicate of the neurophase hypothesis:

    PLV(φ_neural, φ_market) > 0   on held-out intraday data.

Significance is assessed via a surrogate test. **As of PR #15 (task C3)
the surrogate test is delegated to :class:`~neurophase.validation.null_model.NullModelHarness`**,
which supplies Phipson–Smyth ``(1 + k) / (1 + n)`` p-value smoothing,
seeded determinism, and the shared harness contract.

Held-out discipline
-------------------

Invariant **I₂** of ``docs/theory/scientific_basis.md`` requires PLV to
be computed on held-out data only; in-sample PLV is not reported. This
module provides two enforcement layers:

* :func:`plv` — unconditioned: computes the statistic on whatever the
  caller passes. Callable from any context.
* :func:`plv_significance` — backwards-compatible surrogate test that
  now routes through ``NullModelHarness``. Returns the Phipson–Smyth
  smoothed p-value; the old naive estimator is gone.
* :class:`HeldOutSplit` + :func:`plv_on_held_out` — the new
  enforcement wrapper. ``HeldOutSplit`` represents an explicit
  train/test partition with non-overlapping, index-addressed slices.
  ``plv_on_held_out`` refuses to compute PLV on indices that overlap
  the training partition — an attempt to do so raises
  :class:`HeldOutViolation` before any statistic is computed.

The three layers compose: high-level consumers should prefer
``plv_on_held_out`` for anything that will be reported as evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from neurophase.validation.null_model import NullModelHarness
from neurophase.validation.surrogates import cyclic_shift

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Default number of surrogate samples for :func:`plv_significance`.
DEFAULT_PLV_N_SURROGATES: Final[int] = 1000


class HeldOutViolation(ValueError):  # noqa: N818 — domain term, not a generic error suffix
    """Raised when a PLV call is issued on indices inside the training set.

    This is a hard error — it fires **before** the statistic is
    computed, so no in-sample PLV ever escapes to a downstream
    consumer. The name is deliberately *HeldOutViolation* (not
    *HeldOutError*) because it encodes a load-bearing scientific
    contract violation, not a generic failure mode.
    """


@dataclass(frozen=True)
class PLVResult:
    """Significance-tested PLV result.

    Attributes
    ----------
    plv : float
        Observed Phase Locking Value in ``[0, 1]``.
    p_value : float
        Phipson–Smyth smoothed right-tailed p-value from the surrogate
        harness: ``(1 + #{s ≥ observed}) / (1 + n)``. Never exactly zero.
    n_surrogates : int
        Number of surrogate samples used.
    significant : bool
        ``True`` when ``p_value < alpha``.
    alpha : float
        Significance level used.
    seed : int
        Seed used to drive the cyclic-shift surrogate generator.
    """

    plv: float
    p_value: float
    n_surrogates: int
    significant: bool
    alpha: float
    seed: int


@dataclass(frozen=True)
class HeldOutSplit:
    """Explicit train / test partition of a time series.

    Attributes
    ----------
    train_indices
        Indices used for model calibration, parameter selection, or
        threshold tuning. **Forbidden** for evidence reporting.
    test_indices
        Indices used for evidence reporting. Must not overlap with
        ``train_indices``.
    total_length
        Total length of the underlying series; both index arrays must
        be subsets of ``range(total_length)``.
    """

    train_indices: IntArray
    test_indices: IntArray
    total_length: int

    def __post_init__(self) -> None:
        train = np.asarray(self.train_indices, dtype=np.int64)
        test = np.asarray(self.test_indices, dtype=np.int64)
        if train.ndim != 1 or test.ndim != 1:
            raise ValueError("train_indices and test_indices must be 1-D")
        if train.size == 0 or test.size == 0:
            raise ValueError("train_indices and test_indices must each contain ≥ 1 index")
        if train.min() < 0 or test.min() < 0:
            raise ValueError("indices must be non-negative")
        if train.max() >= self.total_length or test.max() >= self.total_length:
            raise ValueError(f"index out of range for total_length={self.total_length}")
        # The load-bearing invariant: disjoint train / test.
        overlap = np.intersect1d(train, test, assume_unique=False)
        if overlap.size > 0:
            raise HeldOutViolation(
                f"train and test indices overlap at {overlap.size} positions: "
                f"first overlap at index {int(overlap[0])}"
            )

    def test_slice(self, series: FloatArray) -> FloatArray:
        """Return the test partition of ``series``."""
        arr = np.asarray(series, dtype=np.float64)
        if arr.size != self.total_length:
            raise ValueError(
                f"series length {arr.size} does not match split total_length {self.total_length}"
            )
        test = np.asarray(self.test_indices, dtype=np.int64)
        return arr[test]


# ---------------------------------------------------------------------------
# PLV statistic
# ---------------------------------------------------------------------------


def plv(phi_x: npt.ArrayLike, phi_y: npt.ArrayLike) -> float:
    """Compute the Phase Locking Value between two phase series.

    Parameters
    ----------
    phi_x, phi_y : array_like, shape (T,)
        Instantaneous phase series in radians.

    Returns
    -------
    float
        PLV in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the two series have different lengths or are empty.
    """
    x = np.asarray(phi_x, dtype=np.float64)
    y = np.asarray(phi_y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"phi_x and phi_y must have the same shape, got {x.shape} vs {y.shape}")
    if x.size == 0:
        raise ValueError("phase series must be non-empty")
    return float(np.abs(np.mean(np.exp(1j * (x - y)))))


def _plv_statistic(x: FloatArray, y: FloatArray) -> float:
    """PLV evaluated as a harness statistic on two pre-validated 1-D arrays."""
    return float(np.abs(np.mean(np.exp(1j * (x - y)))))


def plv_significance(
    phi_x: npt.ArrayLike,
    phi_y: npt.ArrayLike,
    n_surrogates: int = DEFAULT_PLV_N_SURROGATES,
    alpha: float = 0.05,
    seed: int = 42,
) -> PLVResult:
    """Significance-test the PLV of ``(phi_x, phi_y)`` via the shared harness.

    As of PR #15 this function delegates to
    :class:`~neurophase.validation.null_model.NullModelHarness` with a
    ``cyclic_shift`` surrogate generator. The returned ``p_value`` is
    Phipson–Smyth smoothed and therefore strictly positive for any
    finite ``n_surrogates``.

    Parameters
    ----------
    phi_x, phi_y : array_like, shape (T,)
        Instantaneous phase series (caller's responsibility to pass
        held-out data — see :func:`plv_on_held_out` for enforcement).
    n_surrogates : int
        Number of surrogate samples. Must be ≥ 10 (harness floor).
    alpha : float
        Significance level in ``(0, 1)``.
    seed : int
        Deterministic seed for the cyclic-shift surrogate RNG.

    Returns
    -------
    PLVResult
    """
    if n_surrogates < 10:
        raise ValueError(f"n_surrogates must be ≥ 10 for a meaningful p-value, got {n_surrogates}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    x = np.asarray(phi_x, dtype=np.float64)
    y = np.asarray(phi_y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"phi_x and phi_y must have the same shape, got {x.shape} vs {y.shape}")
    if x.size < 2:
        raise ValueError(f"phase series must have length ≥ 2, got {x.size}")

    harness = NullModelHarness(n_surrogates=n_surrogates, alpha=alpha)
    rng = np.random.default_rng(seed)
    result = harness.test(
        x,
        y,
        statistic=_plv_statistic,
        surrogate_fn=lambda arr: cyclic_shift(arr, rng=rng),
        seed=seed,
    )
    return PLVResult(
        plv=result.observed,
        p_value=result.p_value,
        n_surrogates=n_surrogates,
        significant=result.rejected,
        alpha=alpha,
        seed=seed,
    )


def plv_on_held_out(
    phi_x: npt.ArrayLike,
    phi_y: npt.ArrayLike,
    split: HeldOutSplit,
    *,
    n_surrogates: int = DEFAULT_PLV_N_SURROGATES,
    alpha: float = 0.05,
    seed: int = 42,
) -> PLVResult:
    """Compute PLV significance strictly on the held-out partition.

    This is the **enforcement wrapper** that makes invariant I₂ a
    pre-condition rather than a post-hoc discipline. The function:

    1. Validates the split (``HeldOutSplit.__post_init__`` raises
       ``HeldOutViolation`` if train / test overlap).
    2. Slices ``phi_x`` and ``phi_y`` down to the test indices.
    3. Delegates to :func:`plv_significance`.

    At no point is the training partition visible to the statistic —
    in-sample PLV is architecturally impossible through this entry
    point.

    Parameters
    ----------
    phi_x, phi_y : array_like, shape (total_length,)
        Full phase series; the function extracts ``split.test_indices``.
    split : HeldOutSplit
        The train / test partition. Must have been constructed with
        disjoint index arrays (enforced at construction time).
    """
    x = np.asarray(phi_x, dtype=np.float64)
    y = np.asarray(phi_y, dtype=np.float64)
    if x.size != y.size:
        raise ValueError(f"phi_x and phi_y must have the same length, got {x.size} vs {y.size}")
    test_x = split.test_slice(x)
    test_y = split.test_slice(y)
    return plv_significance(
        test_x,
        test_y,
        n_surrogates=n_surrogates,
        alpha=alpha,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Rolling PLV (unchanged — retained for existing consumers)
# ---------------------------------------------------------------------------


def rolling_plv(
    phi_x: npt.ArrayLike,
    phi_y: npt.ArrayLike,
    window: int,
) -> FloatArray:
    """Compute PLV in a rolling window.

    Parameters
    ----------
    phi_x, phi_y : array_like, shape (T,)
    window : int
        Window length in samples. Must be positive and ≤ T.

    Returns
    -------
    FloatArray, shape (T − window + 1,)
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    x = np.asarray(phi_x, dtype=np.float64)
    y = np.asarray(phi_y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"phi_x and phi_y must have the same shape, got {x.shape} vs {y.shape}")
    T = x.size
    if window > T:
        raise ValueError(f"window={window} larger than series length={T}")
    n_out = T - window + 1
    result = np.empty(n_out, dtype=np.float64)
    for i in range(n_out):
        result[i] = float(np.abs(np.mean(np.exp(1j * (x[i : i + window] - y[i : i + window])))))
    return result
