"""Parameter calibration — Program D of the Evolution Board.

Turns the arbitrary ``DEFAULT_THRESHOLD = 0.65`` into a defensible,
out-of-sample-validated number by sweeping candidate thresholds
over synthetic traces with known ground truth (H1) and selecting
the one that optimizes a transparent objective function.

Public API:

* :class:`ThresholdGrid` — explicit, caller-controlled grid of
  candidate thresholds.
* :class:`ThresholdCalibrationReport` — frozen report with full
  provenance (parameter fingerprint, per-threshold scores, best
  choice, OOS metric).
* :func:`calibrate_gate_threshold` — the orchestration function.
"""

from __future__ import annotations

from neurophase.calibration.stillness import (
    DEFAULT_DELTA_MIN_GRID,
    DEFAULT_EPS_F_GRID,
    DEFAULT_EPS_R_GRID,
    DEFAULT_WINDOW_GRID,
    StillnessCalibrationReport,
    StillnessCellEvaluation,
    StillnessGrid,
    calibrate_stillness_parameters,
)
from neurophase.calibration.threshold import (
    DEFAULT_THRESHOLD_GRID,
    ThresholdCalibrationReport,
    ThresholdEvaluation,
    ThresholdGrid,
    calibrate_gate_threshold,
)

__all__ = [
    "DEFAULT_DELTA_MIN_GRID",
    "DEFAULT_EPS_F_GRID",
    "DEFAULT_EPS_R_GRID",
    "DEFAULT_THRESHOLD_GRID",
    "DEFAULT_WINDOW_GRID",
    "StillnessCalibrationReport",
    "StillnessCellEvaluation",
    "StillnessGrid",
    "ThresholdCalibrationReport",
    "ThresholdEvaluation",
    "ThresholdGrid",
    "calibrate_gate_threshold",
    "calibrate_stillness_parameters",
]
