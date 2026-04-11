"""Configuration for the Ketamine-Like Reset controller."""

from __future__ import annotations

from dataclasses import dataclass

# Calibrated on n=80 sessions, 2026-04-12 (synthetic archived protocol dataset).
# Test AUC: 0.8730, 5-fold mean AUC: 0.9473 ± 0.0692.
# Method: grid search on chronological train/test split + chronological 5-fold CV.
# Weights empirically optimized for lock-in discrimination.
LOCKIN_WEIGHT_ERROR: float = 0.30
LOCKIN_WEIGHT_PERSISTENCE: float = 0.29
LOCKIN_WEIGHT_DIVERSITY: float = 0.23
LOCKIN_WEIGHT_IMPROVEMENT: float = 0.18
CALIBRATION_LABEL_SOURCE: str = "heuristic_rule_with_legacy_ground_truth_compat"
CALIBRATION_SCOPE: str = "synthetic_archive"
EVIDENCE_STATUS: str = "Tentative"


@dataclass(frozen=True)
class KLRConfig:
    """Immutable reset-controller tuning knobs."""

    lock_in_threshold: float = 0.7631
    relapse_threshold: float = 0.95
    usage_quantile: float = 0.85
    freeze_quantile: float = 0.90
    disinhibit_inhibition_scale: float = 0.15
    disinhibit_confidence_scale: float = 0.40
    disinhibit_weight_scale: float = 0.65
    plasticity_noise: float = 0.15
    annealing_factor: float = 0.90
    random_seed: int | None = 42
    calibrated_weights: tuple[float, float, float, float] | None = None
    plasticity_floor: float = 0.30

    def __post_init__(self) -> None:
        if not 0.0 <= self.lock_in_threshold <= 1.0:
            raise ValueError("lock_in_threshold must be in [0, 1]")
        if not 0.0 <= self.relapse_threshold <= 1.0:
            raise ValueError("relapse_threshold must be in [0, 1]")
        if not 0.0 <= self.usage_quantile <= 1.0:
            raise ValueError("usage_quantile must be in [0, 1]")
        if not 0.0 <= self.freeze_quantile <= 1.0:
            raise ValueError("freeze_quantile must be in [0, 1]")
        if not 0.0 <= self.disinhibit_inhibition_scale <= 1.0:
            raise ValueError("disinhibit_inhibition_scale must be in [0, 1]")
        if not 0.0 <= self.disinhibit_confidence_scale <= 1.0:
            raise ValueError("disinhibit_confidence_scale must be in [0, 1]")
        if not 0.0 <= self.disinhibit_weight_scale <= 1.0:
            raise ValueError("disinhibit_weight_scale must be in [0, 1]")
        if self.plasticity_noise < 0.0:
            raise ValueError("plasticity_noise must be >= 0")
        if not 0.0 <= self.annealing_factor <= 1.0:
            raise ValueError("annealing_factor must be in [0, 1]")
        if self.calibrated_weights is not None:
            if len(self.calibrated_weights) != 4:
                raise ValueError("calibrated_weights must contain 4 terms")
            if any(w < 0.0 for w in self.calibrated_weights):
                raise ValueError("calibrated_weights must be non-negative")
            if not abs(sum(self.calibrated_weights) - 1.0) <= 1e-9:
                raise ValueError("calibrated_weights must sum to 1.0")
        if not 0.0 <= self.plasticity_floor <= 1.0:
            raise ValueError("plasticity_floor must be in [0, 1]")
        wsum = (
            LOCKIN_WEIGHT_ERROR
            + LOCKIN_WEIGHT_PERSISTENCE
            + LOCKIN_WEIGHT_DIVERSITY
            + LOCKIN_WEIGHT_IMPROVEMENT
        )
        if not abs(wsum - 1.0) <= 1e-9:
            raise ValueError(f"lock-in calibration weights must sum to 1.0, got {wsum}")
