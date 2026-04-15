"""Streaming consumer for :class:`CouplingDirection` snapshots.

:func:`analyse_coupling` is a one-shot estimator over a finite window.
:class:`CouplingObserver` wraps it as a sliding-window observer that
ingests one :class:`CoupledStep` at a time and emits a fresh verdict
every time the buffer is full and a configurable hop has elapsed.

Lifecycle
---------

::

    obs = CouplingObserver(window=512, hop=128, n_surrogates=200, seed=7)
    for _ in range(N):
        step = system.step_record()       # CoupledStep
        snapshot = obs.observe(step)      # CouplingDirection | None
        if snapshot is not None:
            audit.emit("coupling_snapshot", snapshot.summary())

Boundary contract
-----------------
Like :func:`analyse_coupling`, the observer is **pure read-only**: it
does not drive the gate, does not mutate the input system, and the
only RNG draws are the surrogate sweep inside the wrapped TE
significance call. Failure of the wrapped call propagates as a normal
:class:`ValueError`; the observer holds no buffered state across such
failures (the ring is committed before the call).

Why hop?
--------
Re-running surrogate-corrected TE every single step would dominate
the per-tick CPU budget. ``hop`` decouples emission cadence from
ingest cadence: the buffer always contains the *latest* ``window``
samples, but a snapshot is only computed every ``hop`` steps after
the buffer warms up. ``hop = window`` recovers strictly disjoint
windows; ``hop = 1`` recovers per-step emission.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from neurophase.sync.coupled_brain_market import CoupledStep
from neurophase.sync.coupling_direction import (
    DEFAULT_K,
    DEFAULT_N_LEVELS,
    DEFAULT_N_SURROGATES,
    DEFAULT_SEED,
    CouplingDirection,
    analyse_coupling,
)


@dataclass
class CouplingObserver:
    """Sliding-window observer that emits :class:`CouplingDirection`.

    Parameters
    ----------
    window : int
        Number of consecutive ``CoupledStep`` samples held in the ring
        buffer. Must be ≥ 4 — the smallest length that lets the wrapped
        TE estimator see at least one (k = 1) joint cell after history
        offsetting and the branching ratio see at least one ``ΔR``.
    hop : int
        Steps between consecutive snapshot emissions, counted from the
        moment the buffer first fills. ``hop = 1`` emits every tick;
        ``hop = window`` emits over disjoint windows. Must be ≥ 1.
    k, n_levels, n_surrogates : int
        Forwarded to :func:`analyse_coupling`. Defaults match the
        package-wide :data:`DEFAULT_K`, :data:`DEFAULT_N_LEVELS`,
        :data:`DEFAULT_N_SURROGATES`.
    seed : int | None
        Forwarded to :func:`analyse_coupling`. ``None`` ⇒ fresh
        non-determinism per snapshot. Default :data:`DEFAULT_SEED`
        keeps repeat windows byte-identical.
    """

    window: int
    hop: int = 1
    k: int = DEFAULT_K
    n_levels: int = DEFAULT_N_LEVELS
    n_surrogates: int = DEFAULT_N_SURROGATES
    seed: int | None = DEFAULT_SEED
    _psi_brain: deque[float] = field(init=False)
    _psi_market: deque[float] = field(init=False)
    _R: deque[float] = field(init=False)
    _steps_since_emit: int = field(default=0, init=False)
    _emissions: int = field(default=0, init=False)
    _last_snapshot: CouplingDirection | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.window < 4:
            raise ValueError(f"window must be ≥ 4; got {self.window!r}")
        if self.hop < 1:
            raise ValueError(f"hop must be ≥ 1; got {self.hop!r}")
        self._psi_brain = deque(maxlen=self.window)
        self._psi_market = deque(maxlen=self.window)
        self._R = deque(maxlen=self.window)

    # ------------------------------------------------------------------
    # Read-only inspection
    # ------------------------------------------------------------------

    @property
    def filled(self) -> bool:
        """True once the ring buffer holds ``window`` samples."""
        return len(self._psi_brain) >= self.window

    @property
    def emissions(self) -> int:
        """Number of snapshots emitted since construction or :meth:`reset`."""
        return self._emissions

    @property
    def last_snapshot(self) -> CouplingDirection | None:
        """The most recent emitted snapshot, or ``None`` before warm-up."""
        return self._last_snapshot

    # ------------------------------------------------------------------
    # Streaming entry
    # ------------------------------------------------------------------

    def observe(self, step: CoupledStep) -> CouplingDirection | None:
        """Ingest one ``CoupledStep`` and emit a snapshot when due.

        Returns the freshly computed :class:`CouplingDirection` on the
        ticks where ``filled and steps_since_last_emit ≥ hop``, and
        ``None`` otherwise.
        """
        self._psi_brain.append(float(step.psi_brain))
        self._psi_market.append(float(step.psi_market))
        self._R.append(float(step.R))

        if not self.filled:
            return None

        self._steps_since_emit += 1
        if self._steps_since_emit < self.hop and self._emissions > 0:
            return None
        # Emit either on the first warm-up tick or whenever hop has elapsed.
        snapshot = analyse_coupling(
            list(self._psi_brain),
            list(self._psi_market),
            order_parameter=list(self._R),
            k=self.k,
            n_levels=self.n_levels,
            n_surrogates=self.n_surrogates,
            seed=self.seed,
        )
        self._steps_since_emit = 0
        self._emissions += 1
        self._last_snapshot = snapshot
        return snapshot

    def reset(self) -> None:
        """Clear the buffer, snapshot history, and emission counter."""
        self._psi_brain.clear()
        self._psi_market.clear()
        self._R.clear()
        self._steps_since_emit = 0
        self._emissions = 0
        self._last_snapshot = None


__all__ = ["CouplingObserver"]
