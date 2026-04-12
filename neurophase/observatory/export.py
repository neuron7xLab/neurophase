"""Outbound witness / event export.

Contracts for an external collector to consume kernel observations
without reaching into runtime state.

Design invariants
-----------------

* **Typed.** Every outbound object is a frozen dataclass
  (``ObservatoryEvent``). The wire shape is a versioned dict.
* **Versioned.** ``OBSERVATORY_SCHEMA_VERSION`` pins the shape; any
  change bumps the version. Independent of
  ``CANONICAL_FRAME_SCHEMA_VERSION`` so the two can evolve separately.
* **Replay-safe.** ``export_frame(frame)`` is a pure projection of the
  runtime-canonical dict plus observatory metadata. No clock, no RNG.
* **Audit-safe.** The payload carries the ledger tip and the canonical
  frame schema version, so a collector can cross-reference the kernel's
  append-only ledger.
* **No runtime coupling.** The export boundary has no handle on the
  orchestrator / pipeline / gate internals. It accepts an
  :class:`OrchestratedFrame` and returns a dict. Sinks (network,
  stdout, file, test buffer) plug in via the :class:`ObservatorySink`
  protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

from neurophase.contracts import (
    CANONICAL_FRAME_SCHEMA_VERSION,
    as_canonical_dict,
    validate_canonical_dict,
)
from neurophase.runtime.orchestrator import OrchestratedFrame

#: Observatory schema version. Bumped independently of the canonical
#: frame schema so collectors can upgrade on their own cadence.
OBSERVATORY_SCHEMA_VERSION: Final[str] = "1.0.0"

#: Literal event kind emitted on every runtime tick.
_EVENT_KIND_RUNTIME_TICK: Final[str] = "runtime.tick"


@dataclass(frozen=True)
class ObservatoryEvent:
    """Frozen outbound event.

    Attributes
    ----------
    kind
        Event kind (currently always ``"runtime.tick"``).
    schema_version
        Observatory schema version.
    frame_schema_version
        Canonical frame schema version carried in ``payload``. Present
        as a top-level field so a collector can route without parsing
        the payload.
    source
        Emitter id. Defaults to ``"neurophase"`` but integrators may
        set a more specific id (e.g. ``"neurophase@host-42"``).
    payload
        Canonical frame dict (same shape as
        :func:`~neurophase.contracts.as_canonical_dict` output).
    """

    kind: str
    schema_version: str
    frame_schema_version: str
    source: str
    payload: Mapping[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe projection."""
        return {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "frame_schema_version": self.frame_schema_version,
            "source": self.source,
            "payload": dict(self.payload),
        }


class ObservatorySink(Protocol):
    """Minimal interface for an observatory consumer.

    A sink is any callable-shaped object that accepts an
    :class:`ObservatoryEvent` and returns ``None``. A production
    integration wraps an HTTP client, a kafka producer, or a
    filesystem writer in this shape.
    """

    def send(self, event: ObservatoryEvent) -> None: ...  # pragma: no cover — protocol


def export_frame(frame: OrchestratedFrame, *, source: str = "neurophase") -> ObservatoryEvent:
    """Project an :class:`OrchestratedFrame` to a versioned observatory event.

    Pure function: no side effects, no I/O. Callers that want to emit
    the event pass the result to an :class:`ObservatorySink`.

    Parameters
    ----------
    frame
        A runtime-emitted ``OrchestratedFrame``.
    source
        Emitter identifier recorded on the event. Must be non-empty.

    Raises
    ------
    ValueError
        If ``source`` is empty or the frame fails canonical validation.
    """
    if not source:
        raise ValueError("observatory.export_frame: source must be non-empty")

    payload = as_canonical_dict(frame)
    # Defensive: the export boundary is where we validate one more time.
    # A bug in as_canonical_dict must never reach an external collector.
    validate_canonical_dict(payload)

    return ObservatoryEvent(
        kind=_EVENT_KIND_RUNTIME_TICK,
        schema_version=OBSERVATORY_SCHEMA_VERSION,
        frame_schema_version=CANONICAL_FRAME_SCHEMA_VERSION,
        source=source,
        payload=payload,
    )


class ObservatoryExporter:
    """Stateless convenience wrapper: export a frame and hand it to a sink.

    Purely a coordination object — the sink owns the transport. The
    exporter itself never caches, buffers, or retries. Callers that
    need those behaviours layer them into the sink.
    """

    __slots__ = ("_sink", "source")

    def __init__(self, sink: ObservatorySink, *, source: str = "neurophase") -> None:
        if not source:
            raise ValueError("ObservatoryExporter.source must be non-empty")
        self._sink = sink
        self.source: str = source

    def emit(self, frame: OrchestratedFrame) -> ObservatoryEvent:
        """Export ``frame`` and hand the event to the sink.

        Returns the emitted event so the caller can inspect it.
        """
        event = export_frame(frame, source=self.source)
        self._sink.send(event)
        return event
