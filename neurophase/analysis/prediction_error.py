"""Prediction-error monitor — Friston/Clark predictive processing.

Maps the divergence between the brain's internal market model and the
actual market phase onto a scalar **prediction error** ``δ(t)`` and a
three-state cognitive classification (``SYNCHRONIZED`` /
``DIVERGING`` / ``SURRENDERED``).

Circular distance
-----------------
The prediction error is the circular distance between two phases::

    δ(t) = arccos( cos(ψ_brain(t) − ψ_market(t)) )

This maps any phase difference into ``[0, π]``:

* ``δ = 0``   — phases coincide (perfect prediction).
* ``δ = π/2`` — half-cycle disagreement.
* ``δ = π``   — anti-phase (maximum disagreement).

An R(t) proxy is derived directly from the circular distance:

    R_proxy(t) = (1 + cos δ(t)) / 2 ∈ [0, 1]

which matches the two-oscillator limit of the Kuramoto order parameter
and lets this monitor be used stand-alone without running the full
coupled Kuramoto network.

Cognitive state classification
------------------------------
* ``SYNCHRONIZED`` — ``R_proxy ≥ sync_threshold`` (≈ 0.65 by default).
* ``DIVERGING``    — ``surrender_threshold ≤ R_proxy < sync_threshold``.
* ``SURRENDERED``  — ``R_proxy < surrender_threshold`` (≈ 0.35 by default).

``SURRENDERED`` explicitly names the regime in which ``ExecutionGate``
must hard-block any action: the trader's internal model and the market
have structurally diverged and acting on the internal model is
statistically harmful.

Sources
-------
Friston (2010); Clark (2013, 2026); Petalas et al., *Psychon. Bull. Rev.*
(2020); R&D report sections 4 and 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Final, TypedDict

import numpy as np
import pandas as pd


class PredictionErrorResult(TypedDict):
    """Structured result of a single :meth:`PredictionErrorMonitor.update`."""

    t: float
    psi_brain: float
    psi_market: float
    delta: float
    R_proxy: float
    cognitive_state: CognitiveState


#: Default boundaries on the ``R_proxy`` classification.
DEFAULT_SYNC_THRESHOLD: Final[float] = 0.65
DEFAULT_SURRENDER_THRESHOLD: Final[float] = 0.35


class CognitiveState(Enum):
    """Three-band cognitive state derived from ``R_proxy``."""

    SYNCHRONIZED = auto()
    DIVERGING = auto()
    SURRENDERED = auto()


@dataclass(frozen=True)
class PredictionErrorSample:
    """One monitor update output, stored as a row in the session archive."""

    t: float
    psi_brain: float
    psi_market: float
    delta: float
    R_proxy: float
    cognitive_state: CognitiveState


class PredictionErrorMonitor:
    """Online prediction-error monitor.

    Parameters
    ----------
    sync_threshold
        ``R_proxy`` at or above this value → ``SYNCHRONIZED``.
    surrender_threshold
        ``R_proxy`` strictly below this value → ``SURRENDERED``.
    dt
        Assumed step between consecutive updates, used for the ``t``
        column in ``history()``. Set to ``None`` to require callers to
        pass an explicit timestamp.
    """

    def __init__(
        self,
        sync_threshold: float = DEFAULT_SYNC_THRESHOLD,
        surrender_threshold: float = DEFAULT_SURRENDER_THRESHOLD,
        dt: float | None = 1.0,
    ) -> None:
        if not 0.0 < surrender_threshold < sync_threshold < 1.0:
            raise ValueError(
                "require 0 < surrender_threshold < sync_threshold < 1, "
                f"got surrender={surrender_threshold}, sync={sync_threshold}"
            )
        if dt is not None and dt <= 0:
            raise ValueError(f"dt must be > 0 if provided, got {dt}")

        self.sync_threshold: float = sync_threshold
        self.surrender_threshold: float = surrender_threshold
        self.dt: float | None = dt

        self._history: list[PredictionErrorSample] = []
        self._n_updates: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        psi_brain: float,
        psi_market: float,
        t: float | None = None,
    ) -> PredictionErrorResult:
        """Ingest one ``(ψ_brain, ψ_market)`` pair and return the analysis.

        Parameters
        ----------
        psi_brain
            Instantaneous brain sub-population mean phase (radians).
        psi_market
            Instantaneous market sub-population mean phase (radians).
        t
            Optional explicit timestamp. If ``None`` the monitor uses
            ``n_updates * dt`` (requires ``dt`` to have been set at
            construction time).

        Returns
        -------
        dict
            Keys: ``delta``, ``R_proxy``, ``cognitive_state``,
            ``psi_brain``, ``psi_market``, ``t``.
        """
        if not np.isfinite(psi_brain) or not np.isfinite(psi_market):
            raise ValueError(
                f"phases must be finite (got ψ_brain={psi_brain}, ψ_market={psi_market})"
            )

        if t is None:
            if self.dt is None:
                raise ValueError("explicit timestamp required when dt was not provided")
            t = float(self._n_updates * self.dt)

        delta = _circular_distance(psi_brain, psi_market)
        R_proxy = float(0.5 * (1.0 + np.cos(delta)))
        state = self._classify(R_proxy)

        sample = PredictionErrorSample(
            t=float(t),
            psi_brain=float(psi_brain),
            psi_market=float(psi_market),
            delta=delta,
            R_proxy=R_proxy,
            cognitive_state=state,
        )
        self._history.append(sample)
        self._n_updates += 1

        return {
            "t": float(t),
            "psi_brain": float(psi_brain),
            "psi_market": float(psi_market),
            "delta": delta,
            "R_proxy": R_proxy,
            "cognitive_state": state,
        }

    def history(self) -> pd.DataFrame:
        """Return the full session history as a pandas DataFrame.

        The schema is stable and suitable for the session archive:

        ``t``, ``psi_brain``, ``psi_market``, ``delta``, ``R_proxy``,
        ``cognitive_state``.
        """
        if not self._history:
            return pd.DataFrame(
                columns=[
                    "t",
                    "psi_brain",
                    "psi_market",
                    "delta",
                    "R_proxy",
                    "cognitive_state",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "t": s.t,
                    "psi_brain": s.psi_brain,
                    "psi_market": s.psi_market,
                    "delta": s.delta,
                    "R_proxy": s.R_proxy,
                    "cognitive_state": s.cognitive_state.name,
                }
                for s in self._history
            ]
        )

    def reset(self) -> None:
        """Clear the session archive (e.g. at a session boundary)."""
        self._history = []
        self._n_updates = 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _classify(self, R_proxy: float) -> CognitiveState:
        if R_proxy >= self.sync_threshold:
            return CognitiveState.SYNCHRONIZED
        if R_proxy < self.surrender_threshold:
            return CognitiveState.SURRENDERED
        return CognitiveState.DIVERGING


# ----------------------------------------------------------------------
# Pure helper
# ----------------------------------------------------------------------


def _circular_distance(psi_a: float, psi_b: float) -> float:
    """Shortest angular distance between two phases, in ``[0, π]``."""
    # arccos(cos Δ) is numerically robust for any signed phase difference.
    return float(np.arccos(np.clip(np.cos(psi_a - psi_b), -1.0, 1.0)))
