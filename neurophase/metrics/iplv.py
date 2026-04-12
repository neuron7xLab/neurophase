"""iPLV + PPC — bias-free phase coupling metrics.

iPLV (imaginary PLV):
    iPLV = |(1/T) · Σ Im(exp(i·(φ_x(t) − φ_y(t))))|

    Insensitive to zero-lag (volume conduction) artifacts because
    Im(coherency) = 0 at zero lag. Any non-zero iPLV reflects genuine
    phase coupling with a non-zero time delay.

PPC (Pairwise Phase Consistency, Vinck et al. 2010):
    PPC = (N · PLV² − 1) / (N − 1)

    Unbiased estimator of PLV² — E[PPC] = true_PLV² for all N.
    Standard PLV is positively biased: E[PLV] > 0 even when true
    coupling = 0, with bias O(1/√N). PPC removes this bias and is
    the primary inference metric. PLV is retained for reference only.

    Reference: Vinck et al. (2010) doi:10.1016/j.neuroimage.2010.01.073

Mathematical properties:
    iPLV ≤ PLV  for all inputs (imaginary projection)
    PPC ∈ [0, 1]  after clamping (can be slightly negative at null)
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
    """Significance-tested coupling result with bias-free PPC.

    Attributes
    ----------
    iplv : float
        Imaginary PLV ∈ [0, 1] — volume conduction guard.
    plv : float
        Standard PLV — biased at finite N. Retained for reference;
        use ``ppc`` for inference (Vinck et al. 2010).
    ppc : float
        Pairwise Phase Consistency ∈ [0, 1] — unbiased estimator
        of PLV². Primary inference metric. ``(N·PLV² − 1) / (N − 1)``,
        clamped to [0, 1].
    n_samples : int
        Number of samples used.
    p_value : float
        Phipson–Smyth smoothed p-value from surrogate test on PPC.
    significant : bool
        True when p_value < alpha.
    alpha : float
        Significance level used.
    seed : int
        Seed for the surrogate RNG.
    """

    iplv: float
    plv: float
    ppc: float
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


def compute_ppc(phi_x: npt.ArrayLike, phi_y: npt.ArrayLike) -> float:
    """Compute PPC — unbiased estimator of PLV² (Vinck et al. 2010).

    PPC = (N · PLV² − 1) / (N − 1)

    Clamped to [0, 1]. At null coupling (true PLV = 0), E[PPC] = 0
    regardless of sample size, unlike PLV which has positive bias.

    Parameters
    ----------
    phi_x, phi_y : array_like, shape (T,)
        Instantaneous phase series in radians.

    Returns
    -------
    float
        PPC ∈ [0, 1].
    """
    x = np.asarray(phi_x, dtype=np.float64)
    y = np.asarray(phi_y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"phi_x and phi_y must have the same shape, got {x.shape} vs {y.shape}")
    n = x.size
    if n < 2:
        raise ValueError(f"need at least 2 samples for PPC, got {n}")
    if not np.all(np.isfinite(x)):
        raise ValueError("phi_x must contain only finite values")
    if not np.all(np.isfinite(y)):
        raise ValueError("phi_y must contain only finite values")
    plv_val = float(np.abs(np.mean(np.exp(1j * (x - y)))))
    ppc_raw = (n * plv_val**2 - 1) / (n - 1)
    return float(np.clip(ppc_raw, 0.0, 1.0))


def _ppc_statistic(x: FloatArray, y: FloatArray) -> float:
    """PPC evaluated as a harness statistic."""
    n = x.size
    plv_val = float(np.abs(np.mean(np.exp(1j * (x - y)))))
    ppc_raw = (n * plv_val**2 - 1) / (n - 1)
    return float(max(0.0, ppc_raw))


def _iplv_statistic(x: FloatArray, y: FloatArray) -> float:
    """iPLV evaluated as a harness statistic."""
    coherency = np.exp(1j * (x - y))
    return float(np.abs(np.mean(np.imag(coherency))))


def _plv_statistic(x: FloatArray, y: FloatArray) -> float:
    """PLV evaluated as a harness statistic (biased — for reference only)."""
    return float(np.abs(np.mean(np.exp(1j * (x - y)))))


def iplv_significance(
    phi_x: npt.ArrayLike,
    phi_y: npt.ArrayLike,
    *,
    n_surrogates: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> iPLVResult:
    """Significance-test coupling via PPC surrogate null model.

    The surrogate test runs on PPC (bias-free), not PLV.
    iPLV and PLV are computed alongside for diagnostics.

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

    # Compute all three metrics
    plv_val = _plv_statistic(x, y)
    iplv_val = _iplv_statistic(x, y)
    ppc_val = _ppc_statistic(x, y)

    # Surrogate test on PPC (bias-free primary metric)
    harness = NullModelHarness(n_surrogates=n_surrogates, alpha=alpha)
    rng = np.random.default_rng(seed)
    result = harness.test(
        x,
        y,
        statistic=_ppc_statistic,
        surrogate_fn=lambda arr: cyclic_shift(arr, rng=rng),
        seed=seed,
    )

    return iPLVResult(
        iplv=iplv_val,
        plv=plv_val,
        ppc=ppc_val,
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
    """Compute coupling significance strictly on the held-out partition.

    Mirrors :func:`~neurophase.metrics.plv.plv_on_held_out` — enforces
    HeldOutSplit at construction. In-sample computation is architecturally
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
