"""External γ-verification witness backed by :mod:`neosynaptex`.

``GammaWitness`` wraps a :class:`NeosynaptexResetAdapter` inside a private
``Neosynaptex`` instance and exposes a single observation call that
returns a frozen :class:`GammaWitnessReport`. The witness is strictly
**advisory**: its verdict is attached to every :class:`KLRFrame` but
never blocks or mutates any gate decision.

Design contracts (mirrored in ``INVARIANTS.yaml`` as ``NEO-I1`` /
``NEO-I2``):

* The witness never mutates the :class:`SystemState` it inspects
  (``NEO-I1``).
* Any verdict — including ``INCOHERENT`` and ``INSUFFICIENT_DATA`` — is
  purely advisory and must never alter ``KLRFrame.decision`` or any
  execution gate state (``NEO-I2``).
* During warmup (fewer than ``window`` observations) the witness emits a
  deterministic placeholder report with ``phase = "WARMUP"``,
  ``verdict = "INSUFFICIENT_DATA"`` and ``gamma_external = 0.0``.
* The witness never raises: any failure inside ``neosynaptex`` or a
  missing optional dependency is caught and surfaced as the same
  placeholder report, keeping ``KLRPipeline.tick()`` exception-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from neurophase.reset.neosynaptex_adapter import NeosynaptexResetAdapter
from neurophase.reset.state import SystemState

__all__ = [
    "COHERENCE_THRESHOLD",
    "DEFAULT_WINDOW",
    "GammaWitness",
    "GammaWitnessReport",
]

#: Window size forwarded to ``Neosynaptex``. Matches the NFI default.
DEFAULT_WINDOW: int = 16

#: Cross-coherence threshold that separates ``COHERENT`` from ``INCOHERENT``.
COHERENCE_THRESHOLD: float = 0.80


@dataclass(frozen=True)
class GammaWitnessReport:
    """Frozen result returned by :meth:`GammaWitness.observe`.

    Attributes
    ----------
    gamma_external:
        Mean γ reported by the external ``neosynaptex`` substrate. ``0.0``
        during warmup or when no witness signal is available.
    phase:
        ``WARMUP`` during warmup; otherwise one of ``INITIALIZING``,
        ``METASTABLE``, ``CONVERGING``, ``DIVERGING``, ``COLLAPSING``,
        ``DRIFTING`` or ``DEGENERATE`` as emitted by neosynaptex.
    coherence:
        Cross-domain coherence reported by neosynaptex. ``0.0`` during
        warmup or when the single-domain projection cannot produce a
        coherence signal.
    verdict:
        ``COHERENT`` | ``INCOHERENT`` | ``INSUFFICIENT_DATA``.
    """

    gamma_external: float
    phase: str
    coherence: float
    verdict: str


def _warmup_report() -> GammaWitnessReport:
    return GammaWitnessReport(
        gamma_external=0.0,
        phase="WARMUP",
        coherence=0.0,
        verdict="INSUFFICIENT_DATA",
    )


def _derive_verdict(coherence: float) -> str:
    """Map a neosynaptex ``cross_coherence`` value onto the tri-state verdict.

    * Finite coherence ≥ :data:`COHERENCE_THRESHOLD` → ``COHERENT``.
    * Finite coherence below the threshold                → ``INCOHERENT``.
    * Non-finite coherence (single-domain projection, empty window, …)
      → ``INSUFFICIENT_DATA``.
    """

    if not np.isfinite(coherence):
        return "INSUFFICIENT_DATA"
    if coherence >= COHERENCE_THRESHOLD:
        return "COHERENT"
    return "INCOHERENT"


class GammaWitness:
    """Advisory γ witness that wraps a private ``neosynaptex`` instance.

    The witness is lazily initialised: the first call to :meth:`observe`
    attempts to import ``neosynaptex`` and register the adapter. If the
    optional dependency is missing, the witness transitions to a permanently
    degraded state and every subsequent call returns a placeholder
    ``INSUFFICIENT_DATA`` report — :class:`KLRPipeline` is unaffected and
    every existing invariant continues to hold.
    """

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        if window < 8:
            raise ValueError(f"window must be >= 8, got {window}")
        self._window: int = window
        self._adapter: NeosynaptexResetAdapter = NeosynaptexResetAdapter()
        self._nx: Any | None = None
        self._disabled: bool = False
        self._init_attempted: bool = False
        self._n_obs: int = 0

    # ------------------------------------------------------------------
    # Read-only introspection
    # ------------------------------------------------------------------
    @property
    def window(self) -> int:
        return self._window

    @property
    def is_disabled(self) -> bool:
        return self._disabled

    @property
    def n_obs(self) -> int:
        return self._n_obs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def observe(self, state: SystemState) -> GammaWitnessReport:
        """Feed ``state`` into the witness and return a fresh report.

        The call is strictly read-only with respect to ``state`` (NEO-I1)
        and never raises to the caller (NEO-I2).
        """

        if not self._init_attempted:
            self._init_attempted = True
            self._try_initialize()

        if self._disabled or self._nx is None:
            return _warmup_report()

        try:
            self._adapter.update(state)
        except Exception:
            return _warmup_report()

        self._n_obs += 1
        if self._n_obs < self._window:
            return _warmup_report()

        try:
            nx_state = self._nx.observe()
        except Exception:
            # Permanently degrade the witness on internal failure so that
            # subsequent ticks remain fast and exception-free.
            self._disabled = True
            return _warmup_report()

        gamma_raw = float(getattr(nx_state, "gamma_mean", float("nan")))
        coherence_raw = float(getattr(nx_state, "cross_coherence", float("nan")))
        phase_raw = str(getattr(nx_state, "phase", "INITIALIZING"))

        gamma_external = gamma_raw if np.isfinite(gamma_raw) else 0.0
        coherence = coherence_raw if np.isfinite(coherence_raw) else 0.0
        verdict = _derive_verdict(coherence_raw)

        return GammaWitnessReport(
            gamma_external=gamma_external,
            phase=phase_raw,
            coherence=coherence,
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _try_initialize(self) -> None:
        try:
            import neosynaptex
        except Exception:
            self._disabled = True
            return

        try:
            nx = neosynaptex.Neosynaptex(window=self._window)
            nx.register(self._adapter)
        except Exception:
            self._disabled = True
            self._nx = None
            return

        self._nx = nx
