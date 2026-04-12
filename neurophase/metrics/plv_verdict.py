"""Three-gate PLV/PPC verdict — independent verification of phase coupling.

Three independent verification paths, each answering a different question:

1. **Rayleigh test** (PATH 1): Is the phase-difference distribution
   non-uniform? Purely analytical, no surrogates.

2. **Dual surrogate test** (PATH 2): Does PPC survive *both* cyclic-shift
   and time-reversal null models? If only cyclic-shift rejects but
   time-reversal does not, coupling exists but is not directional.

3. **Analytical ground truth** (PATH 3): Does measured PPC match the
   Bessel-function prediction? Only applicable in synthetic mode where
   coupling_k is known.

Verdict rules:
    CONFIRMED  — all applicable gates pass → publish-grade claim
    MARGINAL   — 1–2 gates pass           → Strongly Plausible
    REJECTED   — 0 gates pass             → null not rejected
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from neurophase.benchmarks.ppc_analytical import theoretical_ppc
from neurophase.metrics.iplv import _ppc_statistic, compute_ppc
from neurophase.metrics.rayleigh import rayleigh_test
from neurophase.validation.null_model import NullModelHarness
from neurophase.validation.surrogates import cyclic_shift, time_reversal

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DualSurrogateResult:
    """PPC significance tested with two independent surrogate strategies.

    Attributes
    ----------
    ppc : float
        Observed PPC on the data.
    p_cyclic_shift : float
        p-value from cyclic-shift surrogate null model.
    p_time_reversal : float
        p-value from time-reversal surrogate null model.
    both_significant : bool
        True when both p-values < alpha.
    directional : bool
        True when time-reversal p < alpha (coupling is causal/directed).
    alpha : float
        Significance level used.
    """

    ppc: float
    p_cyclic_shift: float
    p_time_reversal: float
    both_significant: bool
    directional: bool
    alpha: float


def dual_surrogate_test(
    phi_x: npt.ArrayLike,
    phi_y: npt.ArrayLike,
    *,
    n_surrogates: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> DualSurrogateResult:
    """Test PPC with both cyclic-shift and time-reversal surrogates.

    Parameters
    ----------
    phi_x, phi_y : array_like, shape (T,)
        Phase series.
    n_surrogates : int
        Surrogates per strategy.
    alpha : float
        Significance level.
    seed : int
        Base seed (time-reversal uses seed+1 for independence).

    Returns
    -------
    DualSurrogateResult
    """
    x = np.asarray(phi_x, dtype=np.float64)
    y = np.asarray(phi_y, dtype=np.float64)

    ppc_val = compute_ppc(x, y)
    harness = NullModelHarness(n_surrogates=n_surrogates, alpha=alpha)

    # Cyclic-shift surrogate
    rng_cs = np.random.default_rng(seed)
    result_cs = harness.test(
        x, y,
        statistic=_ppc_statistic,
        surrogate_fn=lambda arr: cyclic_shift(arr, rng=rng_cs),
        seed=seed,
    )

    # Time-reversal surrogate (different seed for independence)
    rng_tr = np.random.default_rng(seed + 1)
    result_tr = harness.test(
        x, y,
        statistic=_ppc_statistic,
        surrogate_fn=lambda arr: time_reversal(arr, rng=rng_tr),
        seed=seed + 1,
    )

    return DualSurrogateResult(
        ppc=ppc_val,
        p_cyclic_shift=result_cs.p_value,
        p_time_reversal=result_tr.p_value,
        both_significant=result_cs.rejected and result_tr.rejected,
        directional=result_tr.rejected,
        alpha=alpha,
    )


@dataclass(frozen=True)
class PLVVerdict:
    """Three-gate verdict on phase coupling evidence.

    Attributes
    ----------
    ppc : float
        Observed PPC.
    rayleigh_p : float
        p-value from Rayleigh test (PATH 1).
    theory_delta : float | None
        |measured − theoretical| PPC. None when coupling_k unknown.
    dual_surrogate : DualSurrogateResult
        Dual surrogate test result (PATH 2).
    verdict : str
        One of CONFIRMED, MARGINAL, REJECTED.
    gates_passed : int
        Number of gates passed (0–3).
    """

    ppc: float
    rayleigh_p: float
    theory_delta: float | None
    dual_surrogate: DualSurrogateResult
    verdict: str
    gates_passed: int


def compute_verdict(
    phi_neural: npt.ArrayLike,
    phi_market: npt.ArrayLike,
    *,
    coupling_k: float | None = None,
    noise_sigma: float = 1.0,
    n_surrogates: int = 1000,
    alpha: float = 0.05,
    theory_tolerance: float = 0.10,
    seed: int = 42,
) -> PLVVerdict:
    """Compute the three-gate PLV/PPC verdict.

    Parameters
    ----------
    phi_neural, phi_market : array_like, shape (T,)
        Phase series.
    coupling_k : float | None
        Known coupling strength (synthetic mode). None for real data
        (gate 3 auto-passes).
    noise_sigma : float
        Noise σ for theoretical PPC prediction.
    n_surrogates : int
        Surrogates per strategy in the dual test.
    alpha : float
        Significance level.
    theory_tolerance : float
        Max |measured − theoretical| PPC for gate 3 to pass.
    seed : int
        Base seed for surrogate RNGs.

    Returns
    -------
    PLVVerdict
    """
    x = np.asarray(phi_neural, dtype=np.float64)
    y = np.asarray(phi_market, dtype=np.float64)

    ppc_val = compute_ppc(x, y)

    # Gate 1: Rayleigh test
    delta_phi = x - y
    rayleigh = rayleigh_test(delta_phi, alpha=alpha)

    # Gate 2: Dual surrogate test
    dual = dual_surrogate_test(
        x, y,
        n_surrogates=n_surrogates,
        alpha=alpha,
        seed=seed,
    )

    # Gate 3: Analytical ground truth (only if k is known)
    theory_delta: float | None = None
    if coupling_k is not None:
        theory = theoretical_ppc(coupling_k, noise_sigma)
        theory_delta = abs(ppc_val - theory)

    # Count gates
    gate_1 = rayleigh.significant
    gate_2 = dual.both_significant
    gate_3 = True  # default pass for real data
    if coupling_k is not None and theory_delta is not None:
        gate_3 = theory_delta < theory_tolerance

    gates = sum([gate_1, gate_2, gate_3])

    if gate_1 and gate_2 and gate_3:
        verdict = "CONFIRMED"
    elif gate_1 or gate_2:
        verdict = "MARGINAL"
    else:
        verdict = "REJECTED"

    return PLVVerdict(
        ppc=ppc_val,
        rayleigh_p=rayleigh.p_value,
        theory_delta=theory_delta,
        dual_surrogate=dual,
        verdict=verdict,
        gates_passed=gates,
    )
