"""Temporal integrity gate (B1) — preconditions for phase computation.

Phase is a time-dependent quantity. A phase computed from a stream
whose timestamps are non-monotonic, gapped, or stale is not a phase
— it is numeric noise with phase-shaped output. This module is the
load-bearing enforcement point that sits **upstream** of the Kuramoto
/ PLV / stillness layers and refuses to emit a validity stamp unless
the incoming stream meets four hard contracts:

1. **Monotonicity** — timestamps must be strictly non-decreasing.
2. **Gap bound** — consecutive samples may not be separated by more
   than ``max_gap_seconds``.
3. **Staleness bound** — the most recent sample may not lag the
   caller-supplied wall clock by more than ``max_staleness_seconds``.
4. **Finiteness** — timestamps must be finite (no NaN, no ±∞).

The validator returns a :class:`TemporalQualityDecision` whose
``quality`` field is one of :class:`TimeQuality` values. Only
``VALID`` decisions permit downstream phase computation. Every other
value is non-permissive and causes :class:`ExecutionGate` to settle
in ``DEGRADED`` with a ``temporal:…`` reason tag.

Relation to the four gate invariants
------------------------------------

B1 does **not** introduce a fifth invariant. It enforces a
**precondition** for `I₁`–`I₄`: "no phase without valid time".
Concretely, an ``ExecutionGate`` configured with a temporal validator
will return ``DEGRADED`` (`I₃`) on any non-VALID quality — so
temporal corruption is reported through the existing invariant chain
rather than a new one. This keeps the public state machine at five
states.

Design notes
------------

* **Stateful rolling buffer.** The validator stores only the last
  timestamp and a rolling deque of the most recent ``warmup_samples``
  timestamps. Memory usage is O(warmup_samples), independent of
  stream length.
* **Duplicates.** Exact duplicates (``ts == last_ts``) are treated as
  ``DUPLICATE``. Strictly backward timestamps are ``REVERSED``. This
  is deliberate: a duplicate is recoverable (skip the sample), a
  reversal is a clock fault and must be surfaced.
* **Warmup.** The first sample cannot be assessed for gap or
  monotonicity and is returned as ``WARMUP``. Subsequent samples
  until ``warmup_samples`` have accumulated also return ``WARMUP``
  so the caller has one well-defined transition point. Warmup is
  **not** ``VALID``; a gate that has not yet warmed up must not
  compute phase.
* **Reference clock.** Staleness is optional. If the caller supplies
  ``reference_now``, staleness is ``now - ts``; otherwise the
  staleness check is skipped and ``STALE`` is unreachable.
* **No SciPy.** Uses only ``collections.deque`` and built-in ``math``.
* **Determinism.** The validator has no randomness; same input
  sequence → same decision sequence.

Sources
-------

No external sources — this is a pure data-hygiene contract. The
doctrine is stated formally in
``docs/theory/time_integrity.md``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

#: Default maximum gap between consecutive samples, in seconds.
DEFAULT_MAX_GAP_SECONDS: Final[float] = 1.0

#: Default maximum staleness of the most recent sample relative to
#: the reference clock, in seconds. Set to ``math.inf`` to disable.
DEFAULT_MAX_STALENESS_SECONDS: Final[float] = math.inf

#: Default number of samples that must accumulate before the validator
#: will emit a non-``WARMUP`` decision. Must be ``≥ 2``.
DEFAULT_WARMUP_SAMPLES: Final[int] = 2


class TimeQuality(Enum):
    """Closed enumeration of temporal quality regimes.

    * ``VALID``     — all four contracts satisfied; downstream phase
                        computation is permitted.
    * ``GAPPED``    — monotonic but consecutive gap exceeds
                        ``max_gap_seconds``.
    * ``STALE``     — the most recent sample lags the reference clock
                        by more than ``max_staleness_seconds``.
    * ``REVERSED``  — strictly backward timestamp (``ts < last_ts``).
    * ``DUPLICATE`` — exact duplicate timestamp (``ts == last_ts``).
    * ``WARMUP``    — warmup window not yet filled; validity
                        undefined, downstream must wait.
    * ``INVALID``   — timestamp is non-finite (NaN, ±∞).
    """

    VALID = auto()
    GAPPED = auto()
    STALE = auto()
    REVERSED = auto()
    DUPLICATE = auto()
    WARMUP = auto()
    INVALID = auto()


class TemporalError(ValueError):
    """Raised when the validator is asked to process an unrecoverable input.

    Currently raised on constructor misuse only — all *sample-level*
    faults are surfaced as a ``TemporalQualityDecision`` with a
    non-VALID quality, never as an exception. This keeps the hot path
    exception-free.
    """


@dataclass(frozen=True)
class TemporalQualityDecision:
    """Immutable outcome of a single :meth:`TemporalValidator.validate` call.

    Attributes
    ----------
    quality
        The classification — one of :class:`TimeQuality`.
    ts
        The timestamp that was validated (in seconds).
    last_ts
        The previous timestamp, or ``None`` on the very first sample.
    gap_seconds
        ``ts - last_ts`` if ``last_ts`` is available, else ``None``.
    staleness_seconds
        ``reference_now - ts`` if a reference clock was supplied, else ``None``.
    warmup_remaining
        Non-negative remainder of the warmup window, or ``0`` once warmed up.
    reason
        Human-readable explanation. First token is a stable parseable tag:
        ``valid:`` / ``warmup:`` / ``gapped:`` / ``stale:`` / ``reversed:`` /
        ``duplicate:`` / ``invalid:``.
    """

    quality: TimeQuality
    ts: float
    last_ts: float | None
    gap_seconds: float | None
    staleness_seconds: float | None
    warmup_remaining: int
    reason: str

    @property
    def is_valid(self) -> bool:
        """``True`` iff downstream phase computation may proceed."""
        return self.quality is TimeQuality.VALID


class TemporalValidator:
    """Rolling temporal-integrity validator for a single stream.

    Parameters
    ----------
    max_gap_seconds
        Maximum allowed gap between consecutive samples, in seconds.
        Must be strictly positive and finite.
    max_staleness_seconds
        Maximum allowed staleness relative to the reference clock.
        ``math.inf`` disables the staleness check.
    warmup_samples
        Number of samples required before the validator emits a
        non-``WARMUP`` decision. Must be ``≥ 2``.
    """

    __slots__ = (
        "_history",
        "_last_ts",
        "_n_seen",
        "max_gap_seconds",
        "max_staleness_seconds",
        "warmup_samples",
    )

    def __init__(
        self,
        max_gap_seconds: float = DEFAULT_MAX_GAP_SECONDS,
        max_staleness_seconds: float = DEFAULT_MAX_STALENESS_SECONDS,
        warmup_samples: int = DEFAULT_WARMUP_SAMPLES,
    ) -> None:
        if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise TemporalError(f"max_gap_seconds must be > 0 and finite, got {max_gap_seconds}")
        if max_staleness_seconds <= 0:
            raise TemporalError(f"max_staleness_seconds must be > 0, got {max_staleness_seconds}")
        if warmup_samples < 2:
            raise TemporalError(f"warmup_samples must be ≥ 2, got {warmup_samples}")

        self.max_gap_seconds: float = max_gap_seconds
        self.max_staleness_seconds: float = max_staleness_seconds
        self.warmup_samples: int = warmup_samples

        self._last_ts: float | None = None
        self._history: deque[float] = deque(maxlen=warmup_samples)
        self._n_seen: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, ts: float, *, reference_now: float | None = None) -> TemporalQualityDecision:
        """Validate one timestamp and return the resulting decision.

        Parameters
        ----------
        ts
            The timestamp to validate (seconds since an arbitrary epoch).
        reference_now
            Optional wall-clock reference, used only for staleness.

        Returns
        -------
        TemporalQualityDecision
            Immutable record of the classification and its provenance.
        """
        # Clause 0: finiteness. Unrecoverable at this level but surfaced
        # as a quality state, not an exception — the hot path stays
        # exception-free and downstream (gate) decides the policy.
        if not math.isfinite(ts):
            return TemporalQualityDecision(
                quality=TimeQuality.INVALID,
                ts=ts,
                last_ts=self._last_ts,
                gap_seconds=None,
                staleness_seconds=None,
                warmup_remaining=max(0, self.warmup_samples - self._n_seen),
                reason=f"invalid: timestamp is non-finite ({ts!r})",
            )

        last_ts = self._last_ts

        # Clause 1: monotonicity — reversed vs duplicate.
        if last_ts is not None:
            if ts < last_ts:
                return TemporalQualityDecision(
                    quality=TimeQuality.REVERSED,
                    ts=ts,
                    last_ts=last_ts,
                    gap_seconds=ts - last_ts,
                    staleness_seconds=self._staleness(ts, reference_now),
                    warmup_remaining=max(0, self.warmup_samples - self._n_seen),
                    reason=(
                        f"reversed: ts={ts:.6f} < last_ts={last_ts:.6f} (Δ={ts - last_ts:+.6f}s)"
                    ),
                )
            if ts == last_ts:
                return TemporalQualityDecision(
                    quality=TimeQuality.DUPLICATE,
                    ts=ts,
                    last_ts=last_ts,
                    gap_seconds=0.0,
                    staleness_seconds=self._staleness(ts, reference_now),
                    warmup_remaining=max(0, self.warmup_samples - self._n_seen),
                    reason=f"duplicate: ts == last_ts == {ts:.6f}",
                )

        # Beyond this point the sample is monotonically ahead.
        gap = None if last_ts is None else ts - last_ts

        # Clause 2: gap bound.
        if gap is not None and gap > self.max_gap_seconds:
            # Commit the sample to history anyway so future monotonicity
            # checks are evaluated against the most recent observation.
            self._commit(ts)
            return TemporalQualityDecision(
                quality=TimeQuality.GAPPED,
                ts=ts,
                last_ts=last_ts,
                gap_seconds=gap,
                staleness_seconds=self._staleness(ts, reference_now),
                warmup_remaining=max(0, self.warmup_samples - self._n_seen),
                reason=(f"gapped: Δ={gap:.6f}s exceeds max_gap_seconds={self.max_gap_seconds:.6f}"),
            )

        # Clause 3: staleness bound.
        staleness = self._staleness(ts, reference_now)
        if staleness is not None and staleness > self.max_staleness_seconds:
            self._commit(ts)
            return TemporalQualityDecision(
                quality=TimeQuality.STALE,
                ts=ts,
                last_ts=last_ts,
                gap_seconds=gap,
                staleness_seconds=staleness,
                warmup_remaining=max(0, self.warmup_samples - self._n_seen),
                reason=(
                    f"stale: staleness={staleness:.6f}s exceeds "
                    f"max_staleness_seconds={self.max_staleness_seconds:.6f}"
                ),
            )

        # Commit the sample into history before deciding warmup vs VALID.
        self._commit(ts)

        # Clause 4: warmup.
        if self._n_seen < self.warmup_samples:
            return TemporalQualityDecision(
                quality=TimeQuality.WARMUP,
                ts=ts,
                last_ts=last_ts,
                gap_seconds=gap,
                staleness_seconds=staleness,
                warmup_remaining=self.warmup_samples - self._n_seen,
                reason=(f"warmup: need {self.warmup_samples} samples, have {self._n_seen}"),
            )

        return TemporalQualityDecision(
            quality=TimeQuality.VALID,
            ts=ts,
            last_ts=last_ts,
            gap_seconds=gap,
            staleness_seconds=staleness,
            warmup_remaining=0,
            reason="valid: all four temporal contracts satisfied",
        )

    def reset(self) -> None:
        """Discard all history (e.g. at a stream boundary).

        After reset, the validator is back in the WARMUP phase and the
        next ``warmup_samples`` calls will emit ``WARMUP`` decisions.
        """
        self._last_ts = None
        self._history.clear()
        self._n_seen = 0

    # ------------------------------------------------------------------
    # Read-only diagnostics
    # ------------------------------------------------------------------

    @property
    def n_seen(self) -> int:
        """Total number of samples ingested since construction / reset."""
        return self._n_seen

    @property
    def last_ts(self) -> float | None:
        """Most recent committed timestamp, or ``None``."""
        return self._last_ts

    @property
    def is_warm(self) -> bool:
        """``True`` once the warmup window has been filled."""
        return self._n_seen >= self.warmup_samples

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _commit(self, ts: float) -> None:
        """Append a committed timestamp to history."""
        self._last_ts = ts
        self._history.append(ts)
        self._n_seen += 1

    @staticmethod
    def _staleness(ts: float, reference_now: float | None) -> float | None:
        """Return ``reference_now - ts`` when possible."""
        if reference_now is None or not math.isfinite(reference_now):
            return None
        return reference_now - ts
