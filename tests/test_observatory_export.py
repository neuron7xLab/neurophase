"""Observatory export contract tests."""

from __future__ import annotations

import json

import pytest

from neurophase.api import (
    OrchestratorConfig,
    PipelineConfig,
    PolicyConfig,
    RuntimeOrchestrator,
)
from neurophase.contracts import CANONICAL_FRAME_SCHEMA_VERSION
from neurophase.observatory import (
    OBSERVATORY_SCHEMA_VERSION,
    ObservatoryEvent,
    ObservatoryExporter,
    export_frame,
)


class _BufferSink:
    def __init__(self) -> None:
        self.buffer: list[ObservatoryEvent] = []

    def send(self, event: ObservatoryEvent) -> None:
        self.buffer.append(event)


def _orch() -> RuntimeOrchestrator:
    return RuntimeOrchestrator(
        OrchestratorConfig(
            pipeline=PipelineConfig(warmup_samples=2, stream_window=4, enable_stillness=False),
            policy=PolicyConfig(),
        )
    )


def _ready_frame() -> object:
    orch = _orch()
    frame = None
    for i in range(30):
        frame = orch.tick(timestamp=float(i) * 0.01, R=0.9, delta=0.01)
        if frame.execution_allowed:
            return frame
    raise AssertionError("fixture failed to reach READY")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_version_is_semver() -> None:
    parts = OBSERVATORY_SCHEMA_VERSION.split(".")
    assert len(parts) == 3
    for p in parts:
        assert p.isdigit()


def test_observatory_version_is_independent_of_frame_version() -> None:
    """The two versions may or may not match — but they are declared
    independently so a minor canonical-schema bump does not force a
    collector upgrade."""
    assert OBSERVATORY_SCHEMA_VERSION
    assert CANONICAL_FRAME_SCHEMA_VERSION
    # They're separate names; this test just pins that fact.


# ---------------------------------------------------------------------------
# Pure function: export_frame
# ---------------------------------------------------------------------------


def test_export_frame_produces_versioned_event() -> None:
    event = export_frame(_ready_frame(), source="neurophase@test")  # type: ignore[arg-type]
    assert isinstance(event, ObservatoryEvent)
    assert event.kind == "runtime.tick"
    assert event.schema_version == OBSERVATORY_SCHEMA_VERSION
    assert event.frame_schema_version == CANONICAL_FRAME_SCHEMA_VERSION
    assert event.source == "neurophase@test"


def test_event_to_json_dict_is_json_serialisable() -> None:
    event = export_frame(_ready_frame())  # type: ignore[arg-type]
    d = event.to_json_dict()
    # Must round-trip through json without loss.
    encoded = json.dumps(d, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded == d


def test_event_carries_canonical_payload_with_audit_fields() -> None:
    event = export_frame(_ready_frame())  # type: ignore[arg-type]
    required = {
        "schema_version",
        "tick_index",
        "timestamp",
        "gate_state",
        "execution_allowed",
        "regime_label",
        "action_intent",
        "ledger_record_hash",
    }
    missing = required - set(event.payload.keys())
    assert not missing, f"observatory payload missing required keys: {missing}"


def test_export_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="source"):
        export_frame(_ready_frame(), source="")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Exporter + sink plumbing
# ---------------------------------------------------------------------------


def test_exporter_delivers_event_to_sink() -> None:
    sink = _BufferSink()
    exp = ObservatoryExporter(sink, source="neurophase@host")
    frame = _ready_frame()
    event = exp.emit(frame)  # type: ignore[arg-type]
    assert len(sink.buffer) == 1
    assert sink.buffer[0] is event
    assert event.source == "neurophase@host"


def test_exporter_never_buffers() -> None:
    """The exporter must not hold state between emits."""
    sink = _BufferSink()
    exp = ObservatoryExporter(sink)
    orch = _orch()
    for i in range(40):
        frame = orch.tick(timestamp=float(i) * 0.01, R=0.9, delta=0.01)
        if frame.execution_allowed:
            exp.emit(frame)
    assert len(sink.buffer) > 0
    # Each event is a fresh object (no mutable state leaked)
    ids = {id(e) for e in sink.buffer}
    assert len(ids) == len(sink.buffer)


def test_exporter_rejects_empty_source() -> None:
    sink = _BufferSink()
    with pytest.raises(ValueError, match="source"):
        ObservatoryExporter(sink, source="")


def test_two_exports_are_byte_identical_under_replay() -> None:
    sink_a = _BufferSink()
    sink_b = _BufferSink()

    for sink in (sink_a, sink_b):
        exp = ObservatoryExporter(sink, source="neurophase")
        orch = _orch()
        for i in range(16):
            frame = orch.tick(timestamp=float(i) * 0.01, R=0.9, delta=0.01)
            if frame.execution_allowed:
                exp.emit(frame)

    payload_a = [e.to_json_dict() for e in sink_a.buffer]
    payload_b = [e.to_json_dict() for e in sink_b.buffer]
    assert json.dumps(payload_a, sort_keys=True) == json.dumps(payload_b, sort_keys=True)
