from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurophase.reset import (
    Curriculum,
    KetamineLikeResetController,
    KLRConfig,
    SystemMetrics,
    SystemState,
)
from neurophase.reset.calibration import (
    CalibrationRow,
    confusion_matrix,
    normalize_metric,
    roc_auc,
    validate_calibration_rows,
)


def _dummy_state(n: int = 4) -> SystemState:
    return SystemState(
        weights=np.eye(n, dtype=float),
        confidence=np.full(n, 0.5, dtype=float),
        usage=np.full(n, 0.5, dtype=float),
        utility=np.full(n, 0.5, dtype=float),
        inhibition=np.full(n, 0.5, dtype=float),
        topology=np.ones((n, n), dtype=float),
    )


def _dummy_curriculum(n: int = 4) -> Curriculum:
    return Curriculum(
        target_bias=np.zeros(n, dtype=float),
        corrective_signal=np.zeros(n, dtype=float),
        stress_pattern=np.zeros(n, dtype=float),
    )


def test_calibrated_lockin_detection_on_historical_data() -> None:
    dataset_path = Path("data/calibration_dataset.json")
    assert dataset_path.is_file(), "calibration dataset is missing"

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert data["metadata"]["label_source"] == "heuristic_rule_with_legacy_ground_truth_compat"
    assert data["metadata"]["calibration_scope"] == "synthetic_archive"
    assert data["metadata"]["evidence_status"] == "Tentative"
    test_sessions = data["test"]
    train_sessions = data["train"]

    rows = [
        CalibrationRow(
            session_id=r["session_id"],
            error=r["metrics"]["error"],
            persistence=r["metrics"]["persistence"],
            diversity=r["metrics"]["diversity"],
            improvement=r["metrics"]["improvement"],
            ground_truth=bool(r["ground_truth"]),
            heuristic_ground_truth=bool(r.get("heuristic_ground_truth", r["ground_truth"])),
            expert_label=r.get("expert_label", None),
            timestamp=r["timestamp"],
        )
        for r in (train_sessions + test_sessions)
    ]
    summary = validate_calibration_rows(rows)
    assert int(summary["n_rows"]) == 80
    assert 0.0 < summary["prevalence"] < 1.0

    config = KLRConfig()
    controller = KetamineLikeResetController(config)

    y_true: list[bool] = []
    y_pred: list[bool] = []

    for session in test_sessions:
        metrics = SystemMetrics(**session["metrics"], noise=0.0, reward=0.0)
        should_trigger = bool(session.get("heuristic_ground_truth", session["ground_truth"]))
        assert session.get("expert_label", None) is None

        did_trigger = controller.detect_lockin(metrics)
        y_true.append(should_trigger)
        y_pred.append(did_trigger)

    tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if yt and yp)
    tn = sum(1 for yt, yp in zip(y_true, y_pred, strict=False) if (not yt) and (not yp))

    acc = (tp + tn) / max(1, len(y_true))
    assert acc >= 0.75, f"expected holdout accuracy >= 0.75, got {acc:.3f}"
    assert np.isclose(config.lock_in_threshold, 0.7631)
    cm = confusion_matrix(y_true, y_pred)
    assert cm["tp"] >= 1 and cm["tn"] >= 1

    # Smoke-check full intervention still runs with calibrated defaults.
    state, report = controller.run(
        _dummy_state(), SystemMetrics(0.9, 0.9, 0.2, 0.1, 0.0, 0.0), _dummy_curriculum()
    )
    assert report.status in {"SUCCESS", "ROLLBACK", "SKIPPED"}
    assert state.weights.shape == (4, 4)


def test_fold_indeterminate_guardrail() -> None:
    try:
        roc_auc([True, True, True], [0.1, 0.2, 0.3])
    except ValueError as exc:
        assert "indeterminate fold" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for single-class fold")


def test_normalize_metric_bounds() -> None:
    assert np.isclose(normalize_metric(0.5), 0.5)
    assert np.isclose(normalize_metric(-1.0), 0.0)
    assert np.isclose(normalize_metric(2.0), 1.0)
