"""Canonical runtime frame — contract tests.

Phase 4 of the kernelization protocol declared
:class:`~neurophase.runtime.orchestrator.OrchestratedFrame` the single
canonical typed runtime envelope, and
:data:`neurophase.contracts.CANONICAL_FRAME_SCHEMA_VERSION` the single
on-wire schema. This suite pins both properties as tests so that:

* any drift in the set of fields must be made explicit (version bump +
  schema + doc update in the same commit);
* serialization / validation round-trip without silent mutation;
* replay safety (two runs → identical canonical dict sequences) holds.
"""

from __future__ import annotations

import json

import pytest

from neurophase.api import (
    OrchestratorConfig,
    PipelineConfig,
    PolicyConfig,
    RuntimeOrchestrator,
)
from neurophase.contracts import (
    CANONICAL_FRAME_SCHEMA_VERSION,
    CanonicalFrameSchema,
    SchemaValidationError,
    as_canonical_dict,
    validate_canonical_dict,
)
from neurophase.contracts.frame import CANONICAL_SCHEMA


def _build_orchestrator() -> RuntimeOrchestrator:
    return RuntimeOrchestrator(
        OrchestratorConfig(
            pipeline=PipelineConfig(warmup_samples=2, enable_stillness=False),
            policy=PolicyConfig(),
        )
    )


def _drive_n(orch: RuntimeOrchestrator, n: int) -> list:  # type: ignore[type-arg]
    frames = []
    for i in range(n):
        frames.append(orch.tick(timestamp=float(i) * 0.01, R=0.9, delta=0.01))
    return frames


# ---------------------------------------------------------------------------
# Schema identity
# ---------------------------------------------------------------------------


def test_schema_version_is_frozen_string() -> None:
    assert isinstance(CANONICAL_FRAME_SCHEMA_VERSION, str)
    assert CANONICAL_FRAME_SCHEMA_VERSION
    parts = CANONICAL_FRAME_SCHEMA_VERSION.split(".")
    assert len(parts) == 3, (
        f"schema version must be semver-style: {CANONICAL_FRAME_SCHEMA_VERSION!r}"
    )
    for part in parts:
        assert part.isdigit(), f"non-numeric segment in version: {part!r}"


def test_canonical_schema_exposes_required_keys() -> None:
    assert isinstance(CANONICAL_SCHEMA, CanonicalFrameSchema)
    assert CANONICAL_SCHEMA.version == CANONICAL_FRAME_SCHEMA_VERSION
    # Fields the v1.1 protocol explicitly demands on the canonical envelope.
    must_have = {
        "schema_version",
        "timestamp",
        "tick_index",
        "time_quality",
        "stream_regime",
        "gate_state",
        "gate_reason",
        "execution_allowed",
        "regime_label",
        "regime_confidence",
        "action_intent",
        "ledger_record_hash",
        # KLR / witness: optional domain extension, but must be named.
        "klr_decision",
        "klr_ntk_rank_delta",
        "klr_warning",
    }
    missing = must_have - CANONICAL_SCHEMA.required_keys
    assert not missing, f"canonical schema is missing fields: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_canonical_dict_has_schema_version_first_and_required_keys() -> None:
    orch = _build_orchestrator()
    [frame] = _drive_n(orch, 1)
    d = as_canonical_dict(frame)
    assert "schema_version" in d
    assert d["schema_version"] == CANONICAL_FRAME_SCHEMA_VERSION
    # Exact key set must match the declared schema.
    assert set(d.keys()) == CANONICAL_SCHEMA.required_keys


def test_canonical_dict_is_json_safe() -> None:
    orch = _build_orchestrator()
    frames = _drive_n(orch, 5)
    for frame in frames:
        d = as_canonical_dict(frame)
        # Must round-trip through json without loss.
        encoded = json.dumps(d, sort_keys=True)
        decoded = json.loads(encoded)
        assert decoded == d


def test_canonical_dict_exposes_declared_fields() -> None:
    orch = _build_orchestrator()
    [frame] = _drive_n(orch, 1)
    d = as_canonical_dict(frame)

    # Every value is either None (for nullable fields) or a primitive.
    primitive = (str, int, float, bool)
    for key, value in d.items():
        if value is None:
            assert key in CANONICAL_SCHEMA.nullable_keys, (
                f"key {key!r} is None but not in the nullable set"
            )
        else:
            assert isinstance(value, primitive), (
                f"key {key!r} has non-primitive value of type {type(value).__name__}"
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_accepts_freshly_serialized_frame() -> None:
    orch = _build_orchestrator()
    frames = _drive_n(orch, 10)
    for frame in frames:
        validate_canonical_dict(as_canonical_dict(frame))  # must not raise


def test_validate_rejects_unexpected_key() -> None:
    orch = _build_orchestrator()
    [frame] = _drive_n(orch, 1)
    d = as_canonical_dict(frame)
    d["hallucinated_field"] = 1
    with pytest.raises(SchemaValidationError, match="unexpected keys"):
        validate_canonical_dict(d)


def test_validate_rejects_missing_key() -> None:
    orch = _build_orchestrator()
    [frame] = _drive_n(orch, 1)
    d = as_canonical_dict(frame)
    del d["gate_state"]
    with pytest.raises(SchemaValidationError, match="missing required keys"):
        validate_canonical_dict(d)


def test_validate_rejects_non_nullable_none() -> None:
    orch = _build_orchestrator()
    [frame] = _drive_n(orch, 1)
    d = as_canonical_dict(frame)
    d["gate_state"] = None  # gate_state is non-nullable
    with pytest.raises(SchemaValidationError, match="non-nullable"):
        validate_canonical_dict(d)


def test_validate_rejects_wrong_type() -> None:
    orch = _build_orchestrator()
    [frame] = _drive_n(orch, 1)
    d = as_canonical_dict(frame)
    d["tick_index"] = "zero"
    with pytest.raises(SchemaValidationError, match="tick_index"):
        validate_canonical_dict(d)


def test_validate_rejects_wrong_schema_version() -> None:
    orch = _build_orchestrator()
    [frame] = _drive_n(orch, 1)
    d = as_canonical_dict(frame)
    d["schema_version"] = "0.9.0"
    with pytest.raises(SchemaValidationError, match="schema_version mismatch"):
        validate_canonical_dict(d)


def test_validate_rejects_non_mapping() -> None:
    with pytest.raises(SchemaValidationError, match="expected a mapping"):
        validate_canonical_dict("not a mapping")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Replay safety
# ---------------------------------------------------------------------------


def test_two_orchestrators_emit_identical_canonical_sequences() -> None:
    """Deterministic contract: same config + same input → byte-identical
    canonical dict sequence."""
    orch_a = _build_orchestrator()
    orch_b = _build_orchestrator()

    seq_a = [as_canonical_dict(f) for f in _drive_n(orch_a, 16)]
    seq_b = [as_canonical_dict(f) for f in _drive_n(orch_b, 16)]

    encoded_a = json.dumps(seq_a, sort_keys=True)
    encoded_b = json.dumps(seq_b, sort_keys=True)
    assert encoded_a == encoded_b, "canonical frame sequence is not deterministic"


# ---------------------------------------------------------------------------
# Absence of alternative frame types on the runtime path
# ---------------------------------------------------------------------------


def test_orchestrated_frame_is_the_only_runtime_envelope() -> None:
    """The canonical typed envelope is ``OrchestratedFrame``.

    ``neurophase.api`` must expose exactly one runtime-envelope class
    (``OrchestratedFrame``), plus the inner ``DecisionFrame`` as the
    pipeline-stage record. ``NeuralFrame`` is ingress data, not a
    runtime frame, so it must **not** appear on the blessed façade.
    """
    import neurophase.api as api

    assert "OrchestratedFrame" in api.__all__
    assert "DecisionFrame" in api.__all__
    assert "NeuralFrame" not in api.__all__
    assert "KLRFrame" not in api.__all__
