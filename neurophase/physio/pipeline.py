"""End-to-end physio replay pipeline.

Wires:

    RRReplayReader  ->  HRVWindow  ->  PhysioGate  ->  PhysioFrame

into one call, :meth:`PhysioReplayPipeline.run`, that produces one
canonical-style :class:`PhysioFrame` per input sample (after the buffer
has warmed up enough to produce features) plus a final summary.

No orchestrator, no session manifest, no ledger are constructed here —
those are heavier kernel components intended for the market-side
Kuramoto path. The physio slice is deliberately minimal; it reuses
only the load-bearing :class:`ExecutionGate` through :class:`PhysioGate`.

Frames are JSON-serialisable via :meth:`PhysioFrame.to_json_dict` so a
caller can append them to any audit sink (file, stdout, or an
existing ledger) without this module taking a dependency on a sink.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neurophase.physio.features import DEFAULT_WINDOW_SIZE, HRVFeatures, HRVWindow
from neurophase.physio.gate import (
    DEFAULT_THRESHOLD_ABSTAIN,
    DEFAULT_THRESHOLD_ALLOW,
    PhysioDecision,
    PhysioGate,
    PhysioGateState,
)
from neurophase.physio.replay import ReplayIngestError, RRReplayReader, RRSample

CANONICAL_FRAME_SCHEMA_VERSION: str = "physio-v1"


@dataclass(frozen=True)
class PhysioFrame:
    """Canonical-style immutable envelope for one physio tick."""

    schema_version: str
    tick_index: int
    timestamp_s: float
    rr_ms: float
    features: HRVFeatures
    decision: PhysioDecision
    labels: tuple[str, ...] = field(default_factory=tuple)

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-safe dict. Enums are serialised by name, tuples by list."""
        return {
            "schema_version": self.schema_version,
            "tick_index": self.tick_index,
            "timestamp_s": self.timestamp_s,
            "rr_ms": self.rr_ms,
            "features": asdict(self.features),
            "decision": {
                "state": self.decision.state.name,
                "execution_allowed": self.decision.execution_allowed,
                "confidence": self.decision.confidence,
                "threshold_allow": self.decision.threshold_allow,
                "threshold_abstain": self.decision.threshold_abstain,
                "reason": self.decision.reason,
                "kernel_state": self.decision.kernel_state.name,
            },
            "labels": list(self.labels),
        }


@dataclass(frozen=True)
class PhysioRunSummary:
    """Aggregate summary of one replay run."""

    n_samples_ingested: int
    n_samples_rejected: int
    n_frames_emitted: int
    state_counts: dict[str, int]
    first_state_tick: dict[str, int]
    n_execution_allowed: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "n_samples_ingested": self.n_samples_ingested,
            "n_samples_rejected": self.n_samples_rejected,
            "n_frames_emitted": self.n_frames_emitted,
            "state_counts": dict(self.state_counts),
            "first_state_tick": dict(self.first_state_tick),
            "n_execution_allowed": self.n_execution_allowed,
        }


class PhysioSession:
    """Shared incremental physio core used by BOTH replay and live paths.

    This class owns the single source of truth for:

    * rolling HRV window (:class:`HRVWindow`)
    * admission gate (:class:`PhysioGate`)
    * canonical frame emission (:class:`PhysioFrame`)

    Both :class:`PhysioReplayPipeline` (file-driven) and the live
    session runner in :mod:`neurophase.physio.live` (LSL-driven)
    consume this class via :meth:`step`. There is intentionally only
    one state vocabulary, one feature computation, and one gate
    policy; there is no "live-specific scoring". Refactoring anything
    that diverges replay vs live from this core is forbidden.
    """

    __slots__ = ("_gate", "_window")

    def __init__(
        self,
        *,
        window_size: int = DEFAULT_WINDOW_SIZE,
        threshold_allow: float = DEFAULT_THRESHOLD_ALLOW,
        threshold_abstain: float = DEFAULT_THRESHOLD_ABSTAIN,
    ) -> None:
        self._window = HRVWindow(window_size=window_size)
        self._gate = PhysioGate(
            threshold_allow=threshold_allow,
            threshold_abstain=threshold_abstain,
        )

    def step(self, sample: RRSample, *, tick_index: int) -> PhysioFrame:
        """Advance by one sample; return the emitted :class:`PhysioFrame`.

        Pure incremental semantics: given the same ``(sample, tick_index)``
        history, two :class:`PhysioSession` instances produce byte-identical
        :class:`PhysioFrame` sequences, regardless of whether they were
        fed from a replay CSV or from a live LSL stream.
        """
        self._window.push(sample)
        features: HRVFeatures = self._window.features()
        decision: PhysioDecision = self._gate.evaluate(features)
        return PhysioFrame(
            schema_version=CANONICAL_FRAME_SCHEMA_VERSION,
            tick_index=tick_index,
            timestamp_s=sample.timestamp_s,
            rr_ms=sample.rr_ms,
            features=features,
            decision=decision,
        )


class PhysioReplayPipeline(PhysioSession):
    """Replay-only driver: reads a CSV into the shared :class:`PhysioSession`."""

    def run_iterable(
        self, samples: Iterable[RRSample]
    ) -> tuple[list[PhysioFrame], PhysioRunSummary]:
        """Consume a sample iterable and return (frames, summary)."""
        frames: list[PhysioFrame] = []
        n_rejected = 0
        state_counts: Counter[str] = Counter()
        first_seen: dict[str, int] = {}

        for tick_index, sample in enumerate(samples):
            frame = self.step(sample, tick_index=tick_index)
            frames.append(frame)
            name = frame.decision.state.name
            state_counts[name] += 1
            if name not in first_seen:
                first_seen[name] = tick_index

        n_allowed = state_counts.get(PhysioGateState.EXECUTE_ALLOWED.name, 0)
        summary = PhysioRunSummary(
            n_samples_ingested=len(frames) + n_rejected,
            n_samples_rejected=n_rejected,
            n_frames_emitted=len(frames),
            state_counts=dict(state_counts),
            first_state_tick=dict(first_seen),
            n_execution_allowed=n_allowed,
        )
        return frames, summary

    def run_csv(self, path: str | Path) -> tuple[list[PhysioFrame], PhysioRunSummary]:
        """Consume a replay CSV file and return (frames, summary).

        Ingest errors (malformed row, impossible RR, non-monotonic
        timestamp) are re-raised as :class:`ReplayIngestError`;
        callers decide whether to abort or skip-and-continue.
        """
        reader = RRReplayReader(path)
        return self.run_iterable(reader)


__all__ = [
    "CANONICAL_FRAME_SCHEMA_VERSION",
    "PhysioFrame",
    "PhysioReplayPipeline",
    "PhysioRunSummary",
    "PhysioSession",
    "ReplayIngestError",
]
