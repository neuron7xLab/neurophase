"""Direction Index — combines statistical and topological asymmetries.

Once the emergent-phase criterion has fired, we still need to choose a
direction. The Direction Index combines three orthogonal signals:

    DI(t) = w_s · Skew(X_t) + w_c · Δ_curv(G_t) + w_b · Bias(agent)

where:
    Skew(X_t)    — sample skewness of recent returns
    Δ_curv(G_t)  — Ollivier-Ricci curvature asymmetry across subgraphs
    Bias(agent)  — running expected-return bias accumulated by the agent

Default weights from the π-system reference:
    w_s = 0.4, w_c = 0.3, w_b = 0.3

The downstream rule is binary:

    DI(t) > 0  →  LONG
    DI(t) < 0  →  SHORT
    DI(t) = 0  →  FLAT (no direction picked)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Final


class Direction(Enum):
    """Trade direction after emergent phase + DI evaluation."""

    LONG = auto()
    SHORT = auto()
    FLAT = auto()


@dataclass(frozen=True)
class DirectionIndexWeights:
    """Weights for the Direction Index linear combination.

    Weights must be finite but not constrained to a simplex — the sign of
    DI is what matters. The defaults follow the π-system reference.
    """

    w_skew: float = 0.4
    w_curv: float = 0.3
    w_bias: float = 0.3

    def __post_init__(self) -> None:
        if any(w < 0 for w in (self.w_skew, self.w_curv, self.w_bias)):
            raise ValueError("Direction weights must be non-negative")
        if self.w_skew == 0 and self.w_curv == 0 and self.w_bias == 0:
            raise ValueError("At least one weight must be positive")


@dataclass(frozen=True)
class DirectionDecision:
    """Per-component breakdown of the Direction Index evaluation."""

    direction: Direction
    value: float
    skew: float
    curv: float
    bias: float


DEFAULT_WEIGHTS: Final[DirectionIndexWeights] = DirectionIndexWeights()


def direction_index(
    skew: float,
    curv: float,
    bias: float,
    weights: DirectionIndexWeights = DEFAULT_WEIGHTS,
    flat_tolerance: float = 1e-9,
) -> DirectionDecision:
    """Compute the Direction Index and resolve it to a direction label.

    Parameters
    ----------
    skew : float
        Sample skewness of recent returns.
    curv : float
        Topological curvature asymmetry Δ_curv.
    bias : float
        Agent-maintained expected-return bias.
    weights : DirectionIndexWeights
        Linear combination weights.
    flat_tolerance : float
        Magnitudes below this threshold resolve to ``FLAT`` — an honest
        null rather than arbitrary sign assignment.

    Returns
    -------
    DirectionDecision
    """
    value = weights.w_skew * skew + weights.w_curv * curv + weights.w_bias * bias
    if abs(value) <= flat_tolerance:
        direction = Direction.FLAT
    elif value > 0:
        direction = Direction.LONG
    else:
        direction = Direction.SHORT
    return DirectionDecision(
        direction=direction,
        value=float(value),
        skew=float(skew),
        curv=float(curv),
        bias=float(bias),
    )
