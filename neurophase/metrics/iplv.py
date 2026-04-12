"""iPLV — imaginary part of Phase Locking Value.

    iPLV = |(1/T) · Σ Im(exp(i·(φ_x(t) − φ_y(t))))|

iPLV is insensitive to zero-lag (volume conduction) artifacts because
the imaginary part of a zero-lag coherency is exactly zero. Any non-zero
iPLV therefore reflects genuine phase coupling with a non-zero time lag.

Mathematical property: iPLV ≤ PLV for all inputs (triangle inequality
on the imaginary projection).

For any evidence claim above "Tentative", iPLV must be used alongside
PLV to guard against volume-conduction false positives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from neurophase.metrics.plv import HeldOutSplit
from neurophase.validation.null_model import NullModelHarness
from neurophase.validation.surrogates import cyclic_shift

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class iPLVResult:  # noqa: N801 — standard neuroscience abbreviation
    """Significance-tested iPLV result.

    Attributes
    ----------
    iplv : float
        Imaginary PLV ∈ [0, 1].
    plv : float
        Standard PLV for comparison.
    n_samples : int
        Number of samples used.
    p_value : float
        Phipson–Smyth smoothed p-value from surrogate test.
    significant : bool
        True when p_value < alpha.
    alpha : float
        Significance level used.
    seed : int
        Seed for the surrogate RNG.
    """

    iplv: float
    plv: float
    n_samples: int
    p_value: float
    significant: bool
    alpha: float
    seed: int


def iplv(phi_x: npt.ArrayLike, phi_y: npt.ArrayLike) -> float:
    """Compute the imaginary PLV between two phase series.

    Parameters
    ----------
    phi_x, phi_y : array_like, shape (T,)
        Instantaneous phase series in radians.

    Returns
    -------
    float
        iPLV ∈ [0, 1].
    """
    x = np.asarray(phi_x, dtype=np.float64)
    y = np.asarray(phi_y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"phi_x and phi_y must have the same shape, got {x.shape} vs {y.shape}")
    if x.size == 0:
        raise ValueError("phase series must be non-empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("phi_x must contain only finite values")
    if not np.all(np.isfinite(y)):
        raise ValueError("phi_y must contain only finite values")
    coherency = np.exp(1j * (x - y))
    return float(np.abs(np.mean(np.imag(coherency))))


def _iplv_statistic(x: FloatArray, y: FloatArray) -> float:
    """iPLV evaluated as a harness statistic."""
    coherency = np.exp(1j * (x - y))
    return float(np.abs(np.mean(np.imag(coherency))))


def _plv_statistic(x: FloatArray, y: FloatArray) -> float:
    """PLV evaluated as a harness statistic."""
    return float(np.abs(np.mean(np.exp(1j * (x - y)))))


def iplv_significance(
    phi_x: npt.ArrayLike,
    phi_y: npt.ArrayLike,
    *,
    n_surrogates: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> iPLVResult:
    """Significance-test iPLV via surrogate null model.

    Uses the same NullModelHarness + cyclic_shift approach as
    :func:`~neurophase.metrics.plv.plv_significance`, but the
    statistic is iPLV instead of PLV.

    Parameters
    ----------
    phi_x, phi_y : array_like, shape (T,)
        Instantaneous phase series.
    n_surrogates : int
        Number of surrogate resamples.
    alpha : float
        Significance level.
    seed : int
        Deterministic seed.

    Returns
    -------
    iPLVResult
    """
    x = np.asarray(phi_x, dtype=np.float64)
    y = np.asarray(phi_y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"phi_x and phi_y must have the same shape, got {x.shape} vs {y.shape}")
    if x.size < 2:
        raise ValueError(f"phase series must have length ≥ 2, got {x.size}")
    if not np.all(np.isfinite(x)):
        raise ValueError("phi_x must contain only finite values")
    if not np.all(np.isfinite(y)):
        raise ValueError("phi_y must contain only finite values")

    # Compute standard PLV for comparison
    plv_val = _plv_statistic(x, y)

    # Run iPLV through the surrogate harness
    harness = NullModelHarness(n_surrogates=n_surrogates, alpha=alpha)
    rng = np.random.default_rng(seed)
    result = harness.test(
        x,
        y,
        statistic=_iplv_statistic,
        surrogate_fn=lambda arr: cyclic_shift(arr, rng=rng),
        seed=seed,
    )

    return iPLVResult(
        iplv=result.observed,
        plv=plv_val,
        n_samples=x.size,
        p_value=result.p_value,
        significant=result.rejected,
        alpha=alpha,
        seed=seed,
    )


def iplv_on_held_out(
    phi_x: npt.ArrayLike,
    phi_y: npt.ArrayLike,
    split: HeldOutSplit,
    *,
    n_surrogates: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> iPLVResult:
    """Compute iPLV significance strictly on the held-out partition.

    Mirrors :func:`~neurophase.metrics.plv.plv_on_held_out` — enforces
    HeldOutSplit at construction. In-sample iPLV is architecturally
    impossible through this entry point.

    Raises
    ------
    HeldOutViolation
        If train and test indices overlap.
    """
    x = np.asarray(phi_x, dtype=np.float64)
    y = np.asarray(phi_y, dtype=np.float64)
    if x.size != y.size:
        raise ValueError(f"phi_x and phi_y must have the same length, got {x.size} vs {y.size}")
    test_x = split.test_slice(x)
    test_y = split.test_slice(y)
    return iplv_significance(
        test_x,
        test_y,
        n_surrogates=n_surrogates,
        alpha=alpha,
        seed=seed,
    )
