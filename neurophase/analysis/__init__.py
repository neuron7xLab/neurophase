"""Analysis layer — online cognitive / decision diagnostics.

Contains monitors that translate phase-level outputs of the coupled
brain × market system into higher-level cognitive signals (prediction
error, cognitive state, session archive rows).
"""

from __future__ import annotations

from neurophase.analysis.prediction_error import (
    CognitiveState,
    PredictionErrorMonitor,
    PredictionErrorResult,
    PredictionErrorSample,
)

__all__ = [
    "CognitiveState",
    "PredictionErrorMonitor",
    "PredictionErrorResult",
    "PredictionErrorSample",
]
