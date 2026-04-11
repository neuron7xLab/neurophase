from __future__ import annotations

from neurophase.reset.calibrator import LockinScoreCalibrator
from neurophase.reset.ledger import LedgerEntry


def _entry(i: int, decision: str) -> LedgerEntry:
    return LedgerEntry(
        timestamp=float(i),
        state_hash=str(i),
        metrics_snapshot={
            "error": 0.9 if decision == "ROLLBACK" else 0.2,
            "persistence": 0.8 if decision == "ROLLBACK" else 0.2,
            "diversity": 0.2 if decision == "ROLLBACK" else 0.9,
            "improvement": 0.1 if decision == "ROLLBACK" else 0.9,
        },
        decision=decision,
        relapse_ratio=1.0,
        improvement_ratio=0.0,
        reason="x",
        new_frozen_nodes=[],
    )


def test_calibrator_fallback_and_normalized_weights() -> None:
    c = LockinScoreCalibrator(min_samples=50)
    small = [_entry(i, "SUCCESS") for i in range(10)]
    out = c.calibrate(small)
    assert out.n_samples == 10
    assert abs(sum(out.weights) - 1.0) < 1e-9

    data = [_entry(i, "ROLLBACK" if i % 2 else "SUCCESS") for i in range(120)]
    out2 = c.calibrate(data)
    assert out2.n_samples == 120
    assert abs(sum(out2.weights) - 1.0) < 1e-9
