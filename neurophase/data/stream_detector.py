"""B2 + B6 — packet-level temporal fault detection and state machine.

This module sits **between** :class:`TemporalValidator` (B1) and
downstream consumers. It does two things:

1. **Packet-level bookkeeping (B2).** While B1 classifies each
   timestamp into a :class:`TimeQuality`, B2 runs a rolling counter
   over the resulting decision stream and exposes aggregate
   statistics: how many samples were valid, how many gaps the stream
   had, how many duplicates, and so on. This is the observability
   layer that tools and dashboards query — callers do not have to
   maintain their own counters.

2. **Time-quality state machine (B6).** The detector classifies the
   **stream as a whole** into one of four time-quality regimes,
   with explicit transitions and a minimum-residence-time
   hysteresis to prevent chatter on the boundary. The stream-level
   regime is coarser than the per-sample decision — a single
   isolated glitch does not move the regime, but a persistent run
   of faults does.

Stream-level regimes
--------------------

.. code-block:: text

    WARMUP   — not enough samples to assess stream quality
    HEALTHY  — recent window is dominated by VALID samples
    DEGRADED — recent window has enough faults to reject phase
    OFFLINE  — persistent fault regime (every recent sample is non-VALID)

Only ``HEALTHY`` permits the gate to emit a non-DEGRADED state
downstream. ``DEGRADED`` and ``OFFLINE`` short-circuit the gate into
``DEGRADED`` via the existing B1 ``time_quality`` contract. The
distinction between ``DEGRADED`` and ``OFFLINE`` is purely
informational: both block execution, but ``OFFLINE`` means the
upstream stream has stopped producing valid packets entirely
(suggesting a hardware disconnect rather than a transient glitch).

Relation to B1
--------------

B1 (``TemporalValidator``) answers "is *this sample* valid?". B2+B6
answers "is *this stream* healthy?". The two questions are
independent — a stream can be healthy overall while still rejecting
the occasional bad sample, and a stream can be degraded overall
while still containing some valid samples.

The two layers compose cleanly: feed each ``TemporalQualityDecision``
from a :class:`TemporalValidator` into a :class:`TemporalStreamDetector`,
and read the stream regime via :attr:`TemporalStreamDetector.regime`.

No SciPy. No randomness. Deterministic. ``__slots__`` for zero-overhead
construction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

from neurophase.data.temporal_validator import TemporalQualityDecision, TimeQuality

#: Default rolling-window length (samples).
DEFAULT_STREAM_WINDOW: Final[int] = 32

#: Default maximum proportion of non-VALID samples in the window
#: before the regime moves to ``DEGRADED``.
DEFAULT_MAX_FAULT_RATE: Final[float] = 0.25

#: Default minimum residence time (samples) before a regime change
#: commits. Prevents chatter on the boundary.
DEFAULT_HOLD_STEPS: Final[int] = 4


class StreamRegime(Enum):
    """Stream-level time-quality regime."""

    WARMUP = auto()
    HEALTHY = auto()
    DEGRADED = auto()
    OFFLINE = auto()


@dataclass(frozen=True)
class StreamQualityStats:
    """Aggregate packet-level statistics over the rolling window.

    Attributes
    ----------
    total
        Total samples currently in the window.
    valid
        Count of ``VALID`` samples in the window.
    gapped
        Count of ``GAPPED`` samples.
    stale
        Count of ``STALE`` samples.
    reversed
        Count of ``REVERSED`` samples.
    duplicate
        Count of ``DUPLICATE`` samples.
    invalid
        Count of ``INVALID`` samples (non-finite timestamps).
    warmup
        Count of ``WARMUP`` samples.
    fault_rate
        Proportion of non-``VALID`` samples in the window. ``0.0``
        means everything is valid, ``1.0`` means nothing is.
    """

    total: int
    valid: int
    gapped: int
    stale: int
    reversed: int
    duplicate: int
    invalid: int
    warmup: int
    fault_rate: float


@dataclass(frozen=True)
class StreamQualityDecision:
    """One update output of :meth:`TemporalStreamDetector.update`.

    Attributes
    ----------
    regime
        The classified stream-level regime after this update.
    stats
        Aggregate statistics over the current rolling window.
    last_quality
        The per-sample quality of the most recent packet.
    held
        ``True`` iff the raw classification wanted a different regime
        but the hysteresis lock vetoed the change.
    reason
        Human-readable explanation. First token is a stable tag:
        ``warmup:`` / ``healthy:`` / ``degraded:`` / ``offline:``.
    """

    regime: StreamRegime
    stats: StreamQualityStats
    last_quality: TimeQuality
    held: bool
    reason: str


class TemporalStreamDetector:
    """Stream-level temporal-quality state machine (B2 + B6).

    Parameters
    ----------
    window
        Number of recent samples to consider. Must be ``≥ 4``.
    max_fault_rate
        Proportion of non-VALID samples above which the stream
        transitions out of ``HEALTHY``. Must be in ``(0, 1)``.
    hold_steps
        Minimum residence time (samples) in each regime before a
        change commits. ``0`` disables hysteresis. Must be ``≥ 0``.
    """

    __slots__ = (
        "_buffer",
        "_hold_remaining",
        "_last_regime",
        "_n_updates",
        "hold_steps",
        "max_fault_rate",
        "window",
    )

    def __init__(
        self,
        window: int = DEFAULT_STREAM_WINDOW,
        max_fault_rate: float = DEFAULT_MAX_FAULT_RATE,
        hold_steps: int = DEFAULT_HOLD_STEPS,
    ) -> None:
        if window < 4:
            raise ValueError(f"window must be ≥ 4, got {window}")
        if not 0.0 < max_fault_rate < 1.0:
            raise ValueError(f"max_fault_rate must be in (0, 1), got {max_fault_rate}")
        if hold_steps < 0:
            raise ValueError(f"hold_steps must be ≥ 0, got {hold_steps}")

        self.window: int = window
        self.max_fault_rate: float = max_fault_rate
        self.hold_steps: int = hold_steps

        self._buffer: deque[TimeQuality] = deque(maxlen=window)
        self._n_updates: int = 0
        self._last_regime: StreamRegime = StreamRegime.WARMUP
        self._hold_remaining: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, decision: TemporalQualityDecision) -> StreamQualityDecision:
        """Ingest one B1 decision and return the stream-level classification.

        The detector never inspects the *content* of the
        :class:`TemporalQualityDecision` beyond its ``quality`` field,
        so the caller is free to drive the detector from any source
        that produces decisions (a real validator, a unit test, a
        replay ledger).
        """
        quality = decision.quality
        self._buffer.append(quality)
        self._n_updates += 1

        stats = self._compute_stats()
        raw_regime = self._raw_classification(stats)
        final_regime, held = self._apply_hysteresis(raw_regime)

        reason = self._reason(
            regime=final_regime,
            stats=stats,
            raw_regime=raw_regime,
            held=held,
        )
        return StreamQualityDecision(
            regime=final_regime,
            stats=stats,
            last_quality=quality,
            held=held,
            reason=reason,
        )

    def reset(self) -> None:
        """Discard all state. Use at stream boundaries."""
        self._buffer.clear()
        self._n_updates = 0
        self._last_regime = StreamRegime.WARMUP
        self._hold_remaining = 0

    # ------------------------------------------------------------------
    # Read-only diagnostics
    # ------------------------------------------------------------------

    @property
    def regime(self) -> StreamRegime:
        """Current committed regime."""
        return self._last_regime

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def window_filled(self) -> bool:
        return len(self._buffer) >= self.window

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_stats(self) -> StreamQualityStats:
        total = len(self._buffer)
        counts: dict[TimeQuality, int] = dict.fromkeys(TimeQuality, 0)
        for q in self._buffer:
            counts[q] += 1
        fault_count = total - counts[TimeQuality.VALID]
        fault_rate = float(fault_count / total) if total > 0 else 0.0
        return StreamQualityStats(
            total=total,
            valid=counts[TimeQuality.VALID],
            gapped=counts[TimeQuality.GAPPED],
            stale=counts[TimeQuality.STALE],
            reversed=counts[TimeQuality.REVERSED],
            duplicate=counts[TimeQuality.DUPLICATE],
            invalid=counts[TimeQuality.INVALID],
            warmup=counts[TimeQuality.WARMUP],
            fault_rate=fault_rate,
        )

    def _raw_classification(self, stats: StreamQualityStats) -> StreamRegime:
        if not self.window_filled:
            return StreamRegime.WARMUP
        if stats.valid == 0:
            return StreamRegime.OFFLINE
        if stats.fault_rate > self.max_fault_rate:
            return StreamRegime.DEGRADED
        return StreamRegime.HEALTHY

    def _apply_hysteresis(self, raw_regime: StreamRegime) -> tuple[StreamRegime, bool]:
        if self.hold_steps == 0:
            self._last_regime = raw_regime
            return raw_regime, False

        if raw_regime is self._last_regime:
            if self._hold_remaining > 0:
                self._hold_remaining -= 1
            return raw_regime, False

        # The WARMUP → anything transition is never held back. Hysteresis
        # only applies between the three steady-state regimes.
        if self._last_regime is StreamRegime.WARMUP:
            self._last_regime = raw_regime
            self._hold_remaining = self.hold_steps
            return raw_regime, False

        if self._hold_remaining > 0:
            self._hold_remaining -= 1
            return self._last_regime, True

        self._last_regime = raw_regime
        self._hold_remaining = self.hold_steps
        return raw_regime, False

    def _reason(
        self,
        *,
        regime: StreamRegime,
        stats: StreamQualityStats,
        raw_regime: StreamRegime,
        held: bool,
    ) -> str:
        if held:
            return (
                f"held: stream regime {regime.name.lower()} locked "
                f"({self._hold_remaining} steps remaining); raw → {raw_regime.name.lower()}"
            )
        tag = regime.name.lower()
        return f"{tag}: fault_rate={stats.fault_rate:.3f}, valid={stats.valid}/{stats.total}"
