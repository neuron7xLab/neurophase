"""Canonical runtime frame — serialization contract.

The runtime canonical frame is an :class:`OrchestratedFrame`. This module
is the **serialization side** of that contract: it defines the exact
JSON shape every external consumer (ledger replay, observatory export,
downstream adapter) can rely on.

One envelope, one schema, one version
-------------------------------------

The PHASE 1 kernelization audit flagged four competing frame types
(``DecisionFrame``, ``OrchestratedFrame``, ``NeuralFrame``, ``KLRFrame``)
as closure blocker **C2**. PHASE 4 resolves C2 by declaring that:

1. ``OrchestratedFrame`` is the single canonical **typed in-memory
   envelope**. No alternative payload may be used as the runtime frame.
2. ``CanonicalFrameSchema`` — defined here — is the single canonical
   **on-wire schema**. Every serialized frame carries the
   ``schema_version`` key so consumers can detect drift and refuse to
   operate on a frame they cannot fully validate.
3. ``NeuralFrame`` is the bio-sensor **ingress** payload (upstream of
   the canonical frame) and lives in ``oscillators/neural_protocol``.
   It is not a runtime frame — it is an input.
4. ``KLRFrame`` is a parallel subsystem payload (``reset/``). It is
   advisory and surfaces into the canonical frame only through the
   three explicit ``klr_*`` fields (``klr_decision``,
   ``klr_ntk_rank_delta``, ``klr_warning``) on ``DecisionFrame``.
5. ``DecisionFrame`` remains the inner pipeline record and is exposed
   through ``OrchestratedFrame.pipeline_frame`` — it is *not* a
   separate top-level frame type.

Versioning
----------

The schema version is a semver-style string. Bumping the **major**
version is a breaking change for consumers. Adding optional fields with
safe defaults may keep the major version intact but should bump the
**minor**. Every bump requires an explicit entry in ``CHANGELOG.md``
and an update to ``docs/RUNTIME_CANONICAL_FRAME.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from neurophase.runtime.orchestrator import OrchestratedFrame


#: Canonical schema version. Bump via ``CHANGELOG.md``. Consumers must
#: check this value before interpreting any other field.
CANONICAL_FRAME_SCHEMA_VERSION: Final[str] = "1.0.0"


class SchemaValidationError(ValueError):
    """Raised when a dict does not conform to the canonical frame schema.

    The error is a ``ValueError`` subclass because schema drift is a
    *data* error, not a bug. Callers should catch it at the ingress
    boundary and reject the frame, not crash.
    """


@dataclass(frozen=True)
class _FieldSpec:
    """One required / optional key in the canonical schema."""

    name: str
    kinds: tuple[type, ...]
    required: bool
    nullable: bool = False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


_FIELDS: tuple[_FieldSpec, ...] = (
    # Meta
    _FieldSpec("schema_version", (str,), required=True),
    # Tick identity
    _FieldSpec("tick_index", (int,), required=True),
    _FieldSpec("timestamp", (int, float), required=True),
    # Raw inputs (nullable — gate still emits a DEGRADED frame on None R)
    _FieldSpec("R", (int, float), required=True, nullable=True),
    _FieldSpec("delta", (int, float), required=True, nullable=True),
    # Temporal layer (B1)
    _FieldSpec("time_quality", (str,), required=True),
    _FieldSpec("temporal_reason", (str,), required=True),
    # Stream layer (B2 + B6)
    _FieldSpec("stream_regime", (str,), required=True),
    _FieldSpec("stream_reason", (str,), required=True),
    _FieldSpec("stream_fault_rate", (int, float), required=True),
    # Gate layer (I₁–I₄)
    _FieldSpec("gate_state", (str,), required=True),
    _FieldSpec("gate_reason", (str,), required=True),
    _FieldSpec("execution_allowed", (bool,), required=True),
    # Ledger tip (F1)
    _FieldSpec("ledger_record_hash", (str,), required=True, nullable=True),
    # Regime layer (G1) — nullable because missing R/delta skips this layer
    _FieldSpec("regime_label", (str,), required=True, nullable=True),
    _FieldSpec("regime_confidence", (int, float), required=True, nullable=True),
    _FieldSpec("regime_warm", (bool,), required=True, nullable=True),
    _FieldSpec("regime_reason", (str,), required=True, nullable=True),
    # Policy layer (I1) — nullable for the same reason as regime
    _FieldSpec("action_intent", (str,), required=True, nullable=True),
    # KLR subsystem — optional domain extension (advisory only)
    _FieldSpec("klr_decision", (str,), required=True, nullable=True),
    _FieldSpec("klr_ntk_rank_delta", (int, float), required=True, nullable=True),
    _FieldSpec("klr_warning", (str,), required=True, nullable=True),
)


@dataclass(frozen=True)
class CanonicalFrameSchema:
    """Frozen declaration of the canonical runtime-frame schema.

    Attributes
    ----------
    version
        Schema version string (e.g. ``"1.0.0"``).
    required_keys
        The exact set of keys a valid serialized frame must contain.
    nullable_keys
        Subset of ``required_keys`` whose values may be ``None``.

    Notes
    -----
    The schema is intentionally *flat*: no nested dicts with their own
    optional-ness rules. Every non-null value is a primitive (``str``,
    ``int``, ``float``, or ``bool``). This keeps the contract small,
    auditable by eye, and trivially portable to any language /
    serialization format.
    """

    version: str
    required_keys: frozenset[str]
    nullable_keys: frozenset[str]


CANONICAL_SCHEMA: Final[CanonicalFrameSchema] = CanonicalFrameSchema(
    version=CANONICAL_FRAME_SCHEMA_VERSION,
    required_keys=frozenset(f.name for f in _FIELDS if f.required),
    nullable_keys=frozenset(f.name for f in _FIELDS if f.nullable),
)


_FIELDS_BY_NAME: Final[Mapping[str, _FieldSpec]] = MappingProxyType({f.name: f for f in _FIELDS})


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def as_canonical_dict(frame: OrchestratedFrame) -> dict[str, Any]:
    """Serialize an ``OrchestratedFrame`` to the canonical flat dict.

    The returned dict always carries ``schema_version`` as its first
    key and follows the shape declared by :data:`CANONICAL_SCHEMA`.

    Parameters
    ----------
    frame
        The :class:`~neurophase.runtime.orchestrator.OrchestratedFrame`
        emitted by :meth:`RuntimeOrchestrator.tick`.

    Returns
    -------
    dict[str, Any]
        Flat, JSON-safe representation conforming to the canonical
        schema.
    """
    pf = frame.pipeline_frame
    regime = frame.regime
    action = frame.action

    out: dict[str, Any] = {
        "schema_version": CANONICAL_FRAME_SCHEMA_VERSION,
        "tick_index": pf.tick_index,
        "timestamp": pf.timestamp,
        "R": pf.R,
        "delta": pf.delta,
        "time_quality": pf.temporal.quality.name,
        "temporal_reason": pf.temporal.reason,
        "stream_regime": pf.stream.regime.name,
        "stream_reason": pf.stream.reason,
        "stream_fault_rate": pf.stream.stats.fault_rate,
        "gate_state": pf.gate.state.name,
        "gate_reason": pf.gate.reason,
        "execution_allowed": pf.gate.execution_allowed,
        "ledger_record_hash": (
            pf.ledger_record.record_hash if pf.ledger_record is not None else None
        ),
        "regime_label": regime.label.name if regime is not None else None,
        "regime_confidence": regime.confidence_score if regime is not None else None,
        "regime_warm": regime.warm if regime is not None else None,
        "regime_reason": regime.reason if regime is not None else None,
        "action_intent": action.intent.name if action is not None else None,
        "klr_decision": pf.klr_decision,
        "klr_ntk_rank_delta": pf.klr_ntk_rank_delta,
        "klr_warning": pf.klr_warning,
    }
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_canonical_dict(d: Mapping[str, Any]) -> None:
    """Assert ``d`` conforms to the canonical schema.

    Checks in order:

    1. ``d`` has no unexpected keys.
    2. Every required key is present.
    3. ``schema_version`` matches the canonical version (strict equality
       — a minor-version bump is still a bump).
    4. Each value is either ``None`` (if the key is nullable) or an
       instance of one of the declared primitive types.

    Raises
    ------
    SchemaValidationError
        On any of the above violations. The message names the offending
        key to make log triage trivial.
    """
    if not isinstance(d, Mapping):
        raise SchemaValidationError(f"expected a mapping, got {type(d).__name__}")

    keys = set(d.keys())
    unexpected = keys - CANONICAL_SCHEMA.required_keys
    if unexpected:
        raise SchemaValidationError(f"unexpected keys in canonical frame: {sorted(unexpected)}")
    missing = CANONICAL_SCHEMA.required_keys - keys
    if missing:
        raise SchemaValidationError(f"missing required keys in canonical frame: {sorted(missing)}")

    version = d["schema_version"]
    if not isinstance(version, str):
        raise SchemaValidationError(
            f"schema_version must be a string, got {type(version).__name__}"
        )
    if version != CANONICAL_FRAME_SCHEMA_VERSION:
        raise SchemaValidationError(
            f"schema_version mismatch: frame={version!r} "
            f"expected={CANONICAL_FRAME_SCHEMA_VERSION!r}. "
            "Rejecting to prevent silent semantic drift."
        )

    for key, value in d.items():
        spec = _FIELDS_BY_NAME[key]
        if value is None:
            if not spec.nullable:
                raise SchemaValidationError(
                    f"canonical frame key {key!r} is non-nullable but got None"
                )
            continue
        if not isinstance(value, spec.kinds):
            kinds_str = ", ".join(k.__name__ for k in spec.kinds)
            raise SchemaValidationError(
                f"canonical frame key {key!r} expected one of ({kinds_str}), "
                f"got {type(value).__name__}"
            )
        # bool is a subclass of int — guard against silent coercion
        # for fields that only accept int/float but not bool.
        if isinstance(value, bool) and bool not in spec.kinds:
            raise SchemaValidationError(f"canonical frame key {key!r} expected numeric, got bool")
