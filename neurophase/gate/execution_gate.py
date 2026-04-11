"""Execution gate: block trading when trader and market are desynchronized.

This is the operational core of the system. It is not a risk rule.
It is a physical measurement translated into a binary permission.

Invariants (cannot be overridden — enforced at ``GateDecision.__post_init__``):

* ``I₁``: ``R(t) < threshold``          ⇒ ``execution_allowed = False``
* ``I₂``: bio-sensor absent             ⇒ ``execution_allowed = False``
* ``I₃``: ``R(t)`` invalid / NaN / OOR  ⇒ ``execution_allowed = False``
* ``I₄``: stillness (no new information) ⇒ ``execution_allowed = False``

The gate operates on the composite order parameter ``R(t)`` computed from
the joint Kuramoto network of market + neural oscillators. When a
``StillnessDetector`` (``I₄``) is attached the gate additionally
classifies ``READY`` into ``READY`` (active, execute) or ``UNNECESSARY``
(still, no new information justifies action).

States
------

* ``SENSOR_ABSENT``  — bio-sensor unavailable, ``I₂``.
* ``DEGRADED``       — ``R(t)`` is NaN, None, or out of range, ``I₃``.
* ``BLOCKED``        — ``R(t) < threshold``, ``I₁``.
* ``READY``          — ``R(t) ≥ threshold`` and system is active.
* ``UNNECESSARY``    — ``R(t) ≥ threshold`` but dynamics are still, ``I₄``.

Only ``READY`` produces ``execution_allowed = True``. Every other state,
including ``UNNECESSARY``, is non-permissive. The difference between
``BLOCKED`` and ``UNNECESSARY`` is purely informational: ``BLOCKED``
means "acting now would be lossy"; ``UNNECESSARY`` means "acting now
would add no information". Both are forbidden by the gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

import numpy as np

from neurophase.gate.stillness_detector import (
    StillnessDecision,
    StillnessDetector,
    StillnessState,
)

DEFAULT_THRESHOLD: Final[float] = 0.65


class GateState(Enum):
    """Gate state as a closed enumeration (5-state after ``I₄``)."""

    READY = auto()
    BLOCKED = auto()
    SENSOR_ABSENT = auto()
    DEGRADED = auto()
    UNNECESSARY = auto()


@dataclass(frozen=True)
class GateDecision:
    """Immutable gate decision enforcing ``execution_allowed`` invariants.

    Constructing a decision with ``execution_allowed=True`` while the
    state is **not** ``READY`` raises ``ValueError`` — the invariant
    holds at the type boundary, not only at runtime. This covers
    ``I₁``–``I₄`` uniformly: ``BLOCKED``, ``SENSOR_ABSENT``, ``DEGRADED``
    and ``UNNECESSARY`` can never be accidentally marked permissive.
    """

    state: GateState
    execution_allowed: bool
    R: float | None
    threshold: float
    reason: str
    #: Optional stillness-layer provenance (only populated when a
    #: ``StillnessDetector`` was attached to the gate).
    stillness_state: StillnessState | None = None

    def __post_init__(self) -> None:
        if self.execution_allowed and self.state is not GateState.READY:
            raise ValueError("Invariant violated: execution_allowed=True requires state=READY")


class ExecutionGate:
    """Hard execution gate based on the Kuramoto order parameter ``R(t)``.

    Parameters
    ----------
    threshold
        Minimum ``R(t)`` for execution. Must be in ``(0, 1)``. Lower
        values mean the system tolerates more desynchronization.
    stillness_detector
        Optional ``StillnessDetector`` that classifies the ``READY``
        regime into ``READY`` (active) or ``UNNECESSARY`` (still).
        When omitted, the gate behaves exactly as the pre-``I₄`` 4-state
        gate — no behavioral change for existing callers.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        stillness_detector: StillnessDetector | None = None,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")
        self.threshold: float = threshold
        self.stillness_detector: StillnessDetector | None = stillness_detector

    def evaluate(
        self,
        R: float | None,
        sensor_present: bool = True,
        delta: float | None = None,
    ) -> GateDecision:
        """Evaluate the gate state from ``R(t)``, sensor presence, and optional ``δ(t)``.

        Evaluation order (strict — each check short-circuits):

        1. ``sensor_present=False`` → ``SENSOR_ABSENT`` (``I₂``).
        2. ``R`` invalid / None / out-of-range → ``DEGRADED`` (``I₃``).
        3. ``R < threshold`` → ``BLOCKED`` (``I₁``).
        4. ``R ≥ threshold``:

           4.1 no stillness detector attached → ``READY``.
           4.2 stillness detector attached but ``δ`` missing or invalid →
               ``READY`` with reason ``"stillness evaluation skipped"``.
               **Critical**: missing ``δ`` never degrades to ``DEGRADED``
               or to ``BLOCKED`` — the stillness layer is optional and
               its absence must never be conflated with a hardware
               fault.
           4.3 ``stillness_detector.update(R, δ)`` → ``STILL`` →
               ``UNNECESSARY``; ``ACTIVE`` → ``READY``.

        Parameters
        ----------
        R
            Current order parameter. ``None`` signals a failed computation.
        sensor_present
            Whether bio-sensor data is available.
        delta
            Optional circular distance between brain and market mean
            phases. Required only when a ``StillnessDetector`` is
            attached and the caller wants the ``I₄`` layer to run.

        Returns
        -------
        GateDecision
            Immutable decision carrying the ``execution_allowed`` flag
            and (optionally) the stillness provenance.
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

        if R < self.threshold:  # noqa: SIM300 — physical semantics: R compared to threshold
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

        # R ≥ threshold. Defer to the stillness layer if one is attached.
        return self._classify_ready(R=R, delta=delta)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _classify_ready(self, *, R: float, delta: float | None) -> GateDecision:
        """Split the ``READY`` regime via the optional ``I₄`` layer."""
        if self.stillness_detector is None:
            return GateDecision(
                state=GateState.READY,
                execution_allowed=True,
                R=R,
                threshold=self.threshold,
                reason=f"R(t) = {R:.4f} ≥ threshold = {self.threshold:.4f}. Synchronized.",
            )

        if delta is None or not np.isfinite(delta) or not 0.0 <= delta <= np.pi + 1e-12:
            # Optional layer cannot run. Fall back to 4-state READY
            # without marking the gate DEGRADED — the hardware is fine.
            return GateDecision(
                state=GateState.READY,
                execution_allowed=True,
                R=R,
                threshold=self.threshold,
                reason=(
                    f"R(t) = {R:.4f} ≥ threshold = {self.threshold:.4f}. "
                    "Synchronized; stillness evaluation skipped (delta missing/invalid)."
                ),
            )

        decision: StillnessDecision = self.stillness_detector.update(R=R, delta=delta)
        if decision.state is StillnessState.STILL:
            return GateDecision(
                state=GateState.UNNECESSARY,
                execution_allowed=False,
                R=R,
                threshold=self.threshold,
                reason=(
                    f"R(t) = {R:.4f} ≥ threshold = {self.threshold:.4f}, "
                    f"but stillness layer rejects execution: {decision.reason}"
                ),
                stillness_state=decision.state,
            )
        return GateDecision(
            state=GateState.READY,
            execution_allowed=True,
            R=R,
            threshold=self.threshold,
            reason=(
                f"R(t) = {R:.4f} ≥ threshold = {self.threshold:.4f}. "
                f"Synchronized and active: {decision.reason}"
            ),
            stillness_state=decision.state,
        )
