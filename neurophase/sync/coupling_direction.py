"""Directional coupling diagnostics on brain × market phase pairs.

Consumes the scalar series produced by
:class:`~neurophase.sync.coupled_brain_market.CoupledBrainMarketSystem`
— either the sub-population mean phases ``(ψ_brain, ψ_market)`` or any
other paired scalar observable — and returns a single structured
verdict that closes the epistemic loop the kernel's symmetric metrics
(PLV, iPLV, ISM) leave open:

    *They are synchronised. Who leads?*

The verdict layers two orthogonal coordinates on top of ``R(t)``:

* **Directionality** via transfer entropy
  (:mod:`neurophase.metrics.transfer_entropy`) — signed net flow with
  two one-sided p-values against circular-shift surrogates.
* **Regime** via branching ratio
  (:mod:`neurophase.metrics.branching_ratio`) — σ of ``|ΔR|`` classified
  against the (0.95, 1.05) critical band.

Sign convention
---------------
``net_flow_brain_to_market = TE(brain → market) − TE(market → brain)``.
Positive ⇒ the brain side carries information about the market's near
future beyond what the market's own past predicts; negative ⇒ the
market leads the brain. Zero ⇒ symmetric or undetectable coupling.

Boundary contract
-----------------
This module is a **pure read-only diagnostic** layered on top of
already-computed phase series. It does **not** drive the gate, mutate
state, or perform RNG draws outside the surrogate-significance call.
The kernel's five-state FSM remains the sole execution authority;
``CouplingDirection`` exists so downstream policy (Program G / I-layer)
can consume a typed, p-valued coupling snapshot without re-deriving it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from neurophase.metrics.branching_ratio import (
    BranchingRatioEMA,
    CriticalPhase,
    branching_ratio,
    critical_phase,
)
from neurophase.metrics.transfer_entropy import (
    TEResult,
    transfer_entropy_with_significance,
)

FloatArray = npt.NDArray[np.float64]

DEFAULT_K: Final[int] = 1
DEFAULT_N_LEVELS: Final[int] = 2
DEFAULT_N_SURROGATES: Final[int] = 200
DEFAULT_SEED: Final[int] = 42


@dataclass(frozen=True)
class CouplingDirection:
    """Layered directional + regime verdict on a brain × market segment.

    Attributes
    ----------
    te_brain_to_market, te_market_to_brain : float
        Bias-corrected transfer entropies in nats. Non-negative.
    net_flow_brain_to_market : float
        ``te_brain_to_market − te_market_to_brain``. Positive ⇒ brain
        leads, negative ⇒ market leads, zero ⇒ undetectable asymmetry.
    p_brain_to_market, p_market_to_brain : float
        One-sided empirical p-values against the circular-shift null.
    sigma_R : float
        Branching ratio σ of ``|ΔR|`` over the segment, or ``1.0`` when
        no ``R(t)`` series is supplied (honest-null).
    phase_R : CriticalPhase
        Critical-band classification of ``sigma_R``.
    n_samples : int
        Length of the input phase series (after pairing & validation).
    n_surrogates : int
        Number of circular-shift surrogates per direction used during TE.
    k : int
        TE history depth.
    n_levels : int
        TE quantile-binning alphabet size.
    """

    te_brain_to_market: float
    te_market_to_brain: float
    net_flow_brain_to_market: float
    p_brain_to_market: float
    p_market_to_brain: float
    sigma_R: float
    phase_R: CriticalPhase
    n_samples: int
    n_surrogates: int
    k: int
    n_levels: int

    @property
    def brain_leads(self) -> bool:
        """``True`` when net flow is significantly positive at α = 0.05."""
        return self.net_flow_brain_to_market > 0.0 and self.p_brain_to_market < 0.05

    @property
    def market_leads(self) -> bool:
        """``True`` when net flow is significantly negative at α = 0.05."""
        return self.net_flow_brain_to_market < 0.0 and self.p_market_to_brain < 0.05

    def summary(self) -> str:
        """One-line human-readable summary suitable for ledger / logs."""
        if self.brain_leads:
            arrow = "brain → market"
            p = self.p_brain_to_market
        elif self.market_leads:
            arrow = "market → brain"
            p = self.p_market_to_brain
        else:
            arrow = "symmetric / undetectable"
            p = max(self.p_brain_to_market, self.p_market_to_brain)
        sigma_part = f"σ_R={self.sigma_R:.3f} [{self.phase_R}]"  # noqa: RUF001
        return f"{arrow} (net={self.net_flow_brain_to_market:+.4f} nats, p={p:.3g}); {sigma_part}"


def analyse_coupling(
    phase_brain: npt.ArrayLike,
    phase_market: npt.ArrayLike,
    *,
    order_parameter: npt.ArrayLike | None = None,
    k: int = DEFAULT_K,
    n_levels: int = DEFAULT_N_LEVELS,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    seed: int | None = DEFAULT_SEED,
) -> CouplingDirection:
    """Compute the layered directional + regime verdict on a phase pair.

    Parameters
    ----------
    phase_brain, phase_market : array_like
        Scalar phase series of identical length. Typically the sub-
        population mean phases produced by
        :meth:`CoupledBrainMarketSystem.run`. Quantile binning makes the
        TE estimator rank-invariant, so phase wrap-around does not bias
        the result as long as the wrap is consistent across both inputs.
    order_parameter : array_like | None
        Joint order parameter ``R(t)`` over the same window. When
        supplied, ``σ`` is computed on ``|ΔR|`` (the canonical
        activity-like signal of an order parameter excursion). When
        ``None``, ``sigma_R = 1.0`` and ``phase_R = CRITICAL`` —
        the honest-null for "no excursion data was supplied".
    k : int
        TE history depth per variable.
    n_levels : int
        TE quantile-binning alphabet size.
    n_surrogates : int
        Number of circular-shift surrogates per direction.
    seed : int | None
        RNG seed for the surrogate sweep. ``None`` ⇒ fresh
        non-determinism. Default :data:`DEFAULT_SEED` (42) keeps repeat
        analyses on the same window byte-identical.

    Returns
    -------
    CouplingDirection
        Frozen verdict carrying both directional TE and regime σ.

    Raises
    ------
    ValueError
        When the brain / market series do not share length, or when
        ``order_parameter`` is supplied with a different length, or
        when any input contains non-finite values.
    """
    brain_arr = np.asarray(phase_brain, dtype=np.float64).ravel()
    market_arr = np.asarray(phase_market, dtype=np.float64).ravel()

    if brain_arr.shape != market_arr.shape:
        raise ValueError(
            f"phase_brain and phase_market must share shape; "
            f"got {brain_arr.shape} vs {market_arr.shape}"
        )
    if not (np.all(np.isfinite(brain_arr)) and np.all(np.isfinite(market_arr))):
        raise ValueError("phase series must be finite")

    te = transfer_entropy_with_significance(
        brain_arr,
        market_arr,
        k=k,
        n_levels=n_levels,
        n_surrogates=n_surrogates,
        seed=seed,
    )
    sigma_R, phase_R = _branching_of_R(order_parameter, expected_size=brain_arr.size)

    return CouplingDirection(
        te_brain_to_market=te.te_xy,
        te_market_to_brain=te.te_yx,
        net_flow_brain_to_market=te.te_net,
        p_brain_to_market=te.p_xy,
        p_market_to_brain=te.p_yx,
        sigma_R=sigma_R,
        phase_R=phase_R,
        n_samples=int(brain_arr.size),
        n_surrogates=te.n_surrogates,
        k=te.k,
        n_levels=te.n_levels,
    )


# ---------------------------------------------------------------------------
# Internal: regime coordinate from R(t)
# ---------------------------------------------------------------------------


def _branching_of_R(
    order_parameter: npt.ArrayLike | None, *, expected_size: int
) -> tuple[float, CriticalPhase]:
    """Branching ratio of ``|ΔR|`` with explicit honest-null."""
    if order_parameter is None:
        return 1.0, CriticalPhase.CRITICAL
    R = np.asarray(order_parameter, dtype=np.float64).ravel()
    if R.size != expected_size:
        raise ValueError(
            f"order_parameter must match phase length; got {R.size}, expected {expected_size}"
        )
    if not np.all(np.isfinite(R)):
        raise ValueError("order_parameter must be finite")
    if R.size < 3:
        return 1.0, CriticalPhase.CRITICAL
    activity = np.abs(np.diff(R))
    sigma = branching_ratio(activity)
    return sigma, critical_phase(sigma)


# Re-export the streaming estimator at this layer so consumers building
# online pipelines get a single import surface.
__all__ = [
    "DEFAULT_K",
    "DEFAULT_N_LEVELS",
    "DEFAULT_N_SURROGATES",
    "DEFAULT_SEED",
    "BranchingRatioEMA",
    "CouplingDirection",
    "CriticalPhase",
    "TEResult",
    "analyse_coupling",
]
