"""observatory — outbound witness / event export contract.

The kernel can be wired as a node in an external self-observing
system. This subpackage owns the **outbound** boundary: a typed,
versioned, replay-safe export surface that a remote collector can
consume without reaching into the runtime.

Public surface::

    from neurophase.observatory import (
        OBSERVATORY_SCHEMA_VERSION,
        ObservatoryEvent,
        ObservatoryExporter,
        ObservatorySink,
        export_frame,
    )

See ``docs/OBSERVATORY_EXPORT.md``.
"""

from __future__ import annotations

from neurophase.observatory.export import (
    OBSERVATORY_SCHEMA_VERSION,
    ObservatoryEvent,
    ObservatoryExporter,
    ObservatorySink,
    export_frame,
)

__all__ = [
    "OBSERVATORY_SCHEMA_VERSION",
    "ObservatoryEvent",
    "ObservatoryExporter",
    "ObservatorySink",
    "export_frame",
]
