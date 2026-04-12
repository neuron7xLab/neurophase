"""contracts — versioned runtime envelope schema.

The ``contracts`` subpackage holds the **single canonical serialization
schema** for the runtime envelope (see
:class:`~neurophase.runtime.orchestrator.OrchestratedFrame`). It is the
inverse of the typed in-memory object: one version-pinned shape that
every external consumer — ledger, replay, observatory export, downstream
adapter — can rely on.

The package intentionally contains no frame *type*: the canonical typed
envelope is ``OrchestratedFrame``. ``contracts`` only owns the
**schema** (the set of keys, types, required-ness, and version) and the
functions that serialize to / validate against it.

Public surface::

    from neurophase.contracts import (
        CANONICAL_FRAME_SCHEMA_VERSION,
        CanonicalFrameSchema,
        SchemaValidationError,
        as_canonical_dict,
        validate_canonical_dict,
    )
"""

from __future__ import annotations

from neurophase.contracts.frame import (
    CANONICAL_FRAME_SCHEMA_VERSION,
    CanonicalFrameSchema,
    SchemaValidationError,
    as_canonical_dict,
    validate_canonical_dict,
)

__all__ = [
    "CANONICAL_FRAME_SCHEMA_VERSION",
    "CanonicalFrameSchema",
    "SchemaValidationError",
    "as_canonical_dict",
    "validate_canonical_dict",
]
