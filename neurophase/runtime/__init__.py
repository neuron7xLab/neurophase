"""Online runtime system — Program E of the Evolution Board.

The runtime layer composes every load-bearing module into a single
stateful streaming pipeline that consumes raw (timestamp, R, delta)
tuples and emits typed :class:`DecisionFrame` objects carrying the
full provenance of the decision.

Public API:

* :class:`DecisionFrame` — typed runtime envelope with full provenance.
* :class:`StreamingPipeline` — stateful composition of
  ``TemporalValidator → TemporalStreamDetector → ExecutionGate``.
* :class:`PipelineConfig` — immutable configuration.
"""

from __future__ import annotations

from neurophase.runtime.memory_audit import (
    ComponentMemoryFootprint,
    MemoryAuditError,
    MemoryAuditReport,
    audit_runtime_memory,
)
from neurophase.runtime.orchestrator import (
    OrchestratedFrame,
    OrchestratorConfig,
    RuntimeOrchestrator,
)
from neurophase.runtime.pipeline import (
    DecisionFrame,
    PipelineConfig,
    StreamingPipeline,
)

__all__ = [
    "ComponentMemoryFootprint",
    "DecisionFrame",
    "MemoryAuditError",
    "MemoryAuditReport",
    "OrchestratedFrame",
    "OrchestratorConfig",
    "PipelineConfig",
    "RuntimeOrchestrator",
    "StreamingPipeline",
    "audit_runtime_memory",
]
