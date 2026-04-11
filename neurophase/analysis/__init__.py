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
from neurophase.analysis.regime import (
    DEFAULT_REGIME_THRESHOLDS,
    RegimeClassifier,
    RegimeLabel,
    RegimeState,
    RegimeThresholds,
)
from neurophase.analysis.regime_transitions import (
    InsufficientHistoryError,
    RegimeTransitionMatrix,
    RegimeTransitionTracker,
    TransitionEvent,
)

__all__ = [
    "DEFAULT_REGIME_THRESHOLDS",
    "CognitiveState",
    "InsufficientHistoryError",
    "PredictionErrorMonitor",
    "PredictionErrorResult",
    "PredictionErrorSample",
    "RegimeClassifier",
    "RegimeLabel",
    "RegimeState",
    "RegimeThresholds",
    "RegimeTransitionMatrix",
    "RegimeTransitionTracker",
    "TransitionEvent",
]
