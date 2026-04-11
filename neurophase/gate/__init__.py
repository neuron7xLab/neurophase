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
from neurophase.gate.stillness_detector import (
    DEFAULT_DELTA_MIN,
    DEFAULT_DT,
    DEFAULT_EPS_F,
    DEFAULT_EPS_R,
    DEFAULT_WINDOW,
    StillnessDecision,
    StillnessDetector,
    StillnessState,
    free_energy_proxy,
)

__all__ = [
    "DEFAULT_CRITERIA",
    "DEFAULT_DELTA_MIN",
    "DEFAULT_DT",
    "DEFAULT_EPS_F",
    "DEFAULT_EPS_R",
    "DEFAULT_THRESHOLD",
    "DEFAULT_WEIGHTS",
    "DEFAULT_WINDOW",
    "Direction",
    "DirectionDecision",
    "DirectionIndexWeights",
    "EmergentPhaseCriteria",
    "EmergentPhaseDecision",
    "ExecutionGate",
    "GateDecision",
    "GateState",
    "StillnessDecision",
    "StillnessDetector",
    "StillnessState",
    "detect_emergent_phase",
    "direction_index",
    "free_energy_proxy",
]
