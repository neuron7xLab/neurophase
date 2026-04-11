"""Data-integrity layer — preconditions for any phase computation.

This subpackage hosts the temporal validity infrastructure that must
run **before** any Kuramoto / PLV / stillness computation: phase is a
time-dependent quantity and a phase computed from non-monotonic,
gapped, or stale samples is physically meaningless.

Public API:

* :class:`TemporalValidator` — rolling validator for a single stream.
* :class:`TimeQuality` — five-way enum of the validity regimes.
* :class:`TemporalQualityDecision` — immutable validator output.
* :class:`TemporalError` — raised when an input breaks non-recoverable contracts.
"""

from __future__ import annotations

from neurophase.data.temporal_validator import (
    DEFAULT_MAX_GAP_SECONDS,
    DEFAULT_MAX_STALENESS_SECONDS,
    DEFAULT_WARMUP_SAMPLES,
    TemporalError,
    TemporalQualityDecision,
    TemporalValidator,
    TimeQuality,
)

__all__ = [
    "DEFAULT_MAX_GAP_SECONDS",
    "DEFAULT_MAX_STALENESS_SECONDS",
    "DEFAULT_WARMUP_SAMPLES",
    "TemporalError",
    "TemporalQualityDecision",
    "TemporalValidator",
    "TimeQuality",
]
