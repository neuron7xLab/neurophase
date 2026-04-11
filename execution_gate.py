"""Execution gate: block trading when trader and market are desynchronized.

This is the operational core of the system. It is not a risk rule.
It is a physical measurement translated into a binary permission.

INVARIANT (cannot be overridden):
    R(t) < threshold → execution_allowed = False

The gate operates on the composite order parameter R(t) computed from
the joint Kuramoto network of market + neural oscillators.

States:
    READY         R(t) ≥ threshold. Execution permitted.
    BLOCKED       R(t) < threshold. Execution blocked.
    SENSOR_ABSENT Bio-sensor unavailable. Execution blocked.
    DEGRADED      R(t) is NaN or out of range. Execution blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

import numpy as np

DEFAULT_THRESHOLD: Final[float] = 0.65


class GateState(Enum):
    READY = auto()
    BLOCKED = auto()
    SENSOR_ABSENT = auto()
    DEGRADED = auto()


@dataclass(frozen=True)
class GateDecision:
    state: GateState
    execution_allowed: bool
    R: float | None
    threshold: float
    reason: str

    def __post_init__(self) -> None:
        # Enforce the invariant at construction time.
        if self.execution_allowed and self.state != GateState.READY:
            raise ValueError(
                "Invariant violated: execution_allowed=True requires state=READY"
            )


class ExecutionGate:
    """Hard execution gate based on the Kuramoto order parameter R(t).

    Parameters
    ----------
    threshold : float
        Minimum R(t) for execution. Default 0.65.
        Lower values mean the system tolerates more desynchronization.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.threshold = threshold

    def evaluate(self, R: float | None, sensor_present: bool = True) -> GateDecision:
        """Evaluate current gate state.

        Parameters
        ----------
        R : float | None
            Current order parameter value. None if computation failed.
        sensor_present : bool
            Whether bio-sensor data is available.

        Returns
        -------
        GateDecision
            Immutable decision with execution_allowed flag.
        """
        if not sensor_present:
            return GateDecision(
                state=GateState.SENSOR_ABSENT,
                execution_allowed=False,
                R=None,
                threshold=self.threshold,
                reason="Bio-sensor absent. Connect hardware to enable trading.",
            )

        if R is None or not np.isfinite(R) or not 0.0 <= R <= 1.0:
            return GateDecision(
                state=GateState.DEGRADED,
                execution_allowed=False,
                R=R,
                threshold=self.threshold,
                reason=f"R(t) = {R!r} is invalid. Cannot assess synchronization.",
            )

        if R < self.threshold:
            return GateDecision(
                state=GateState.BLOCKED,
                execution_allowed=False,
                R=R,
                threshold=self.threshold,
                reason=(
                    f"R(t) = {R:.4f} < threshold = {self.threshold:.4f}. "
                    "Trader and market are desynchronized."
                ),
            )

        return GateDecision(
            state=GateState.READY,
            execution_allowed=True,
            R=R,
            threshold=self.threshold,
            reason=f"R(t) = {R:.4f} ≥ threshold = {self.threshold:.4f}. Synchronized.",
        )
