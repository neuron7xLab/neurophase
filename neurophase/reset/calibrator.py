"""Ledger-driven calibration of lock-in score weights."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from neurophase.reset.config import (
    LOCKIN_WEIGHT_DIVERSITY,
    LOCKIN_WEIGHT_ERROR,
    LOCKIN_WEIGHT_IMPROVEMENT,
    LOCKIN_WEIGHT_PERSISTENCE,
)
from neurophase.reset.ledger import LedgerEntry


@dataclass(frozen=True)
class CalibrationResult:
    weights: tuple[float, float, float, float]
    confidence_interval: tuple[float, float]
    n_samples: int


class LockinScoreCalibrator:
    def __init__(self, min_samples: int = 50) -> None:
        self.min_samples = min_samples
        self.default_weights = (
            LOCKIN_WEIGHT_ERROR,
            LOCKIN_WEIGHT_PERSISTENCE,
            LOCKIN_WEIGHT_DIVERSITY,
            LOCKIN_WEIGHT_IMPROVEMENT,
        )

    def calibrate(self, ledger: list[LedgerEntry]) -> CalibrationResult:
        labeled = [e for e in ledger if e.decision in {"SUCCESS", "ROLLBACK"}]
        if len(labeled) < self.min_samples:
            return CalibrationResult(self.default_weights, (0.0, 0.0), len(labeled))
        try:
            linear_model_mod: Any = import_module("sklearn.linear_model")
        except ImportError:
            return CalibrationResult(self.default_weights, (0.0, 0.0), len(labeled))

        x = np.array(
            [
                [
                    e.metrics_snapshot.get("error", 0.0),
                    e.metrics_snapshot.get("persistence", 0.0),
                    1.0 - e.metrics_snapshot.get("diversity", 0.0),
                    1.0 - e.metrics_snapshot.get("improvement", 0.0),
                ]
                for e in labeled
            ],
            dtype=np.float64,
        )
        y = np.array([1 if e.decision == "ROLLBACK" else 0 for e in labeled], dtype=np.int64)

        logistic_cls: Any = getattr(linear_model_mod, "LogisticRegression", None)
        if logistic_cls is None:
            return CalibrationResult(self.default_weights, (0.0, 0.0), len(labeled))
        model = logistic_cls(max_iter=500, solver="lbfgs")
        model.fit(x, y)
        coefs = np.abs(model.coef_[0])
        norm: float = float(np.sum(coefs))
        if norm <= 1e-12:
            weights = self.default_weights
        else:
            w = coefs / norm
            weights = (float(w[0]), float(w[1]), float(w[2]), float(w[3]))

        preds = model.predict_proba(x)[:, 1]
        ci = (float(np.percentile(preds, 2.5)), float(np.percentile(preds, 97.5)))
        return CalibrationResult(weights=weights, confidence_interval=ci, n_samples=len(labeled))
