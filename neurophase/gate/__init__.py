"""Execution gate, emergent phase detector, and direction index."""

from __future__ import annotations

from neurophase.gate.direction_index import (
    DEFAULT_WEIGHTS,
    Direction,
    DirectionDecision,
    DirectionIndexWeights,
    direction_index,
)
from neurophase.gate.emergent_phase import (
    DEFAULT_CRITERIA,
    EmergentPhaseCriteria,
    EmergentPhaseDecision,
    detect_emergent_phase,
)
from neurophase.gate.execution_gate import (
    DEFAULT_THRESHOLD,
    ExecutionGate,
    GateDecision,
    GateState,
)

__all__ = [
    "DEFAULT_CRITERIA",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WEIGHTS",
    "Direction",
    "DirectionDecision",
    "DirectionIndexWeights",
    "EmergentPhaseCriteria",
    "EmergentPhaseDecision",
    "ExecutionGate",
    "GateDecision",
    "GateState",
    "detect_emergent_phase",
    "direction_index",
]
