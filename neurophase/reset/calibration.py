"""Calibration utilities for lock-in score weighting.

Implements dataset building, grid search and chronological cross-validation
for KLR lock-in calibration artifacts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from statistics import mean
from typing import cast

from neurophase.reset.config import (
    LOCKIN_WEIGHT_DIVERSITY,
    LOCKIN_WEIGHT_ERROR,
    LOCKIN_WEIGHT_IMPROVEMENT,
    LOCKIN_WEIGHT_PERSISTENCE,
)
from neurophase.reset.metrics import SystemMetrics


@dataclass(frozen=True)
class CalibrationRow:
    session_id: str
    error: float
    persistence: float
    diversity: float
    improvement: float
    ground_truth: bool
    timestamp: str
    heuristic_ground_truth: bool | None = None
    expert_label: bool | None = None


def confusion_matrix(y_true: list[bool], y_pred: list[bool]) -> dict[str, int]:
    tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt and yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if (not yt) and (not yp))
    fp = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if (not yt) and yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt and (not yp))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def validate_calibration_rows(rows: list[CalibrationRow]) -> dict[str, float]:
    if not rows:
        raise ValueError("calibration rows are empty")

    session_ids = [r.session_id for r in rows]
    if len(set(session_ids)) != len(session_ids):
        raise ValueError("duplicate session_id in calibration rows")

    timestamps = [r.timestamp for r in rows]
    if timestamps != sorted(timestamps):
        raise ValueError("rows must be chronological (non-decreasing timestamp)")

    y = [
        bool(r.heuristic_ground_truth if r.heuristic_ground_truth is not None else r.ground_truth)
        for r in rows
    ]
    prevalence = float(mean(1.0 if v else 0.0 for v in y))
    if prevalence <= 0.0 or prevalence >= 1.0:
        raise ValueError("calibration rows must include both positive and negative classes")

    return {
        "n_rows": float(len(rows)),
        "prevalence": prevalence,
    }


def normalize_metric(value: float, *, lo: float = 0.0, hi: float = 1.0) -> float:
    if hi <= lo:
        raise ValueError("invalid normalization bounds")
    clamped = min(max(value, lo), hi)
    return (clamped - lo) / (hi - lo)


def lockin_score(
    *,
    error: float,
    persistence: float,
    diversity: float,
    improvement: float,
    w_error: float,
    w_persistence: float,
    w_diversity: float,
    w_improvement: float,
) -> float:
    return (
        w_error * error
        + w_persistence * persistence
        + w_diversity * (1.0 - diversity)
        + w_improvement * (1.0 - improvement)
    )


def roc_auc(y_true: list[bool], y_score: list[float]) -> float:
    pos = [s for y, s in zip(y_true, y_score, strict=False) if y]
    neg = [s for y, s in zip(y_true, y_score, strict=False) if not y]
    if not pos or not neg:
        raise ValueError("indeterminate fold: AUC requires both positive and negative classes")

    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def detect_lockin(metrics: SystemMetrics, threshold: float) -> bool:
    return (
        lockin_score(
            error=metrics.error,
            persistence=metrics.persistence,
            diversity=metrics.diversity,
            improvement=metrics.improvement,
            w_error=LOCKIN_WEIGHT_ERROR,
            w_persistence=LOCKIN_WEIGHT_PERSISTENCE,
            w_diversity=LOCKIN_WEIGHT_DIVERSITY,
            w_improvement=LOCKIN_WEIGHT_IMPROVEMENT,
        )
        >= threshold
    )


def explain_lockin_score(metrics: SystemMetrics, threshold: float) -> dict[str, float]:
    error_term = LOCKIN_WEIGHT_ERROR * metrics.error
    persistence_term = LOCKIN_WEIGHT_PERSISTENCE * metrics.persistence
    inverse_diversity_term = LOCKIN_WEIGHT_DIVERSITY * (1.0 - metrics.diversity)
    inverse_improvement_term = LOCKIN_WEIGHT_IMPROVEMENT * (1.0 - metrics.improvement)
    total = error_term + persistence_term + inverse_diversity_term + inverse_improvement_term
    return {
        "error_term": float(error_term),
        "persistence_term": float(persistence_term),
        "inverse_diversity_term": float(inverse_diversity_term),
        "inverse_improvement_term": float(inverse_improvement_term),
        "total_score": float(total),
        "threshold": float(threshold),
        "triggered": float(total >= threshold),
    }


def confusion_stats(y_true: list[bool], y_score: list[float], threshold: float) -> dict[str, float]:
    y_pred = [s >= threshold for s in y_score]
    tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt and yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if (not yt) and (not yp))
    fp = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if (not yt) and yp)
    fn = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt and (not yp))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    youden = tpr + tnr - 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "youden": youden,
    }


def optimize_weights(
    train: list[CalibrationRow],
    test: list[CalibrationRow],
    *,
    w_error_grid: Iterable[float],
    w_persistence_grid: Iterable[float],
    w_diversity_grid: Iterable[float],
) -> dict[str, float | list[float] | int]:
    best: dict[str, float | list[float] | int] | None = None
    iterations = 0

    for w_error, w_persistence, w_diversity in product(
        w_error_grid, w_persistence_grid, w_diversity_grid
    ):
        w_improvement = 1.0 - (w_error + w_persistence + w_diversity)
        if not 0.0 <= w_improvement <= 1.0:
            continue

        weights = [w_error, w_persistence, w_diversity, w_improvement]
        train_scores = [
            lockin_score(
                error=r.error,
                persistence=r.persistence,
                diversity=r.diversity,
                improvement=r.improvement,
                w_error=w_error,
                w_persistence=w_persistence,
                w_diversity=w_diversity,
                w_improvement=w_improvement,
            )
            for r in train
        ]
        train_truth = [r.ground_truth for r in train]
        unique_thresholds = sorted(set(train_scores))

        best_threshold = unique_thresholds[0]
        best_youden = -1.0
        for threshold in unique_thresholds:
            y = confusion_stats(train_truth, train_scores, threshold)["youden"]
            if y > best_youden:
                best_youden = y
                best_threshold = threshold

        test_scores = [
            lockin_score(
                error=r.error,
                persistence=r.persistence,
                diversity=r.diversity,
                improvement=r.improvement,
                w_error=w_error,
                w_persistence=w_persistence,
                w_diversity=w_diversity,
                w_improvement=w_improvement,
            )
            for r in test
        ]
        test_truth = [r.ground_truth for r in test]

        auc = roc_auc(test_truth, test_scores)
        stats = confusion_stats(test_truth, test_scores, best_threshold)
        candidate: dict[str, float | list[float] | int] = {
            "best_weights": weights,
            "best_auc": auc,
            "best_threshold": best_threshold,
            "precision": stats["precision"],
            "recall": stats["recall"],
            "f1": stats["f1"],
        }

        iterations += 1
        if best is None or float(cast(float, candidate["best_auc"])) > float(
            cast(float, best["best_auc"])
        ):
            best = candidate

    if best is None:
        raise ValueError("no valid weight combinations found")
    best["grid_iterations"] = iterations
    return best
