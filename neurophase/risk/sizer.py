"""Composite position sizer.

Combines three signals into a single capital fraction:

    1. CVaR-budget constraint:
           max_capital ≤ risk_per_trade / CVaR_p
    2. Synchronization scaling:
           scale_R = clamp((R - θ) / (1 - θ), 0, 1)
       Size grows linearly from zero at the gate boundary to full
       allocation when R = 1.
    3. Multifractal deflation:
           scale_m = max(1 - γ · m_instability, 0)
       High h(q) spread ⇒ wider regime heterogeneity ⇒ shrink size.

Final size is the product, clamped to [0, max_leverage · equity_fraction].

This module is deliberately *stateless* — it neither knows nor cares
about order books or execution paths. It converts a risk budget and
physics readings into a capital fraction. Integration with an actual
OMS is downstream of neurophase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class RiskProfile:
    """Risk budget for a single trade.

    Attributes
    ----------
    risk_per_trade : float
        Fraction of equity the trader is willing to lose in the tail
        scenario (e.g. 0.01 → 1 %). Must be in (0, 1).
    confidence : float
        Tail confidence level for CVaR (e.g. 0.99). Must be in (0, 1).
    max_leverage : float
        Hard cap on the capital fraction. Must be > 0.
    multifractal_penalty : float
        γ in ``scale_m = max(1 - γ · m_instability, 0)``. Zero disables.
    """

    risk_per_trade: float = 0.01
    confidence: float = 0.99
    max_leverage: float = 3.0
    multifractal_penalty: float = 1.5

    def __post_init__(self) -> None:
        if not 0.0 < self.risk_per_trade < 1.0:
            raise ValueError(f"risk_per_trade must be in (0, 1), got {self.risk_per_trade}")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {self.confidence}")
        if self.max_leverage <= 0:
            raise ValueError(f"max_leverage must be > 0, got {self.max_leverage}")
        if self.multifractal_penalty < 0:
            raise ValueError(f"multifractal_penalty must be >= 0, got {self.multifractal_penalty}")


@dataclass(frozen=True)
class PositionSize:
    """Per-trade sizing breakdown.

    Attributes
    ----------
    fraction : float
        Final capital fraction in [0, max_leverage]. Zero means do not trade.
    cvar_cap : float
        Capital cap implied by CVaR budget alone.
    scale_R : float
        Synchronization multiplier ∈ [0, 1].
    scale_m : float
        Multifractal multiplier ∈ [0, 1].
    reason : str
        Short explanation of the dominant constraint.
    """

    fraction: float
    cvar_cap: float
    scale_R: float
    scale_m: float
    reason: str


DEFAULT_PROFILE: Final[RiskProfile] = RiskProfile()


def size_position(
    R: float,
    threshold: float,
    cvar: float,
    multifractal_instability_value: float = 0.0,
    profile: RiskProfile = DEFAULT_PROFILE,
) -> PositionSize:
    """Turn a physics state + CVaR into a position size.

    The caller is expected to have already passed the execution gate
    (R ≥ θ). If it has not, this function returns a zero size with a
    ``"gate blocked"`` reason — a defensive null.

    Parameters
    ----------
    R : float
        Current Kuramoto order parameter.
    threshold : float
        Gate threshold θ.
    cvar : float
        Expected Shortfall at ``profile.confidence``. Must be > 0.
    multifractal_instability_value : float
        Output of ``risk.mfdfa.multifractal_instability``. Zero disables
        the multifractal deflator.
    profile : RiskProfile

    Returns
    -------
    PositionSize
    """
    if threshold >= 1.0 or threshold < 0.0:
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")
    if cvar <= 0:
        return PositionSize(
            fraction=0.0,
            cvar_cap=0.0,
            scale_R=0.0,
            scale_m=0.0,
            reason="CVaR must be positive; refusing to size on degenerate tail",
        )
    if R < threshold:  # noqa: SIM300 — physical semantics: R vs gate
        return PositionSize(
            fraction=0.0,
            cvar_cap=0.0,
            scale_R=0.0,
            scale_m=0.0,
            reason="gate blocked: R < θ",
        )

    cvar_cap = profile.risk_per_trade / cvar
    span = 1.0 - threshold
    scale_R = (R - threshold) / span if span > 0 else 1.0
    scale_R = max(0.0, min(scale_R, 1.0))
    scale_m = max(0.0, 1.0 - profile.multifractal_penalty * multifractal_instability_value)
    scale_m = min(scale_m, 1.0)

    fraction = cvar_cap * scale_R * scale_m
    capped = min(fraction, profile.max_leverage)

    reason: str
    if capped == 0.0:
        reason = "all-zero multiplier — no exposure"
    elif capped < fraction:
        reason = f"capped at max_leverage={profile.max_leverage}"
    elif scale_m < 1e-9:
        reason = "multifractal deflator collapsed size to zero"
    elif scale_R < 1.0:
        reason = f"scaled by synchronisation headroom {scale_R:.3f}"
    else:
        reason = "cvar-budget constrained"

    return PositionSize(
        fraction=float(capped),
        cvar_cap=float(cvar_cap),
        scale_R=float(scale_R),
        scale_m=float(scale_m),
        reason=reason,
    )
