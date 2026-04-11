from __future__ import annotations

import numpy as np

from neurophase.reset import (
    Curriculum,
    KetamineLikeResetController,
    KLRConfig,
    SystemMetrics,
    SystemState,
)


def _state() -> SystemState:
    return SystemState(
        weights=np.full((4, 4), 0.25),
        confidence=np.array([0.9, 0.6, 0.6, 0.6]),
        usage=np.array([0.9, 0.8, 0.2, 0.3]),
        utility=np.array([0.9, 0.6, 0.6, 0.6]),
        inhibition=np.full(4, 0.6),
        topology=np.ones((4, 4)),
        gamma=0.4,
    )


def test_commit_and_rollback_by_relapse() -> None:
    c = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.1, relapse_threshold=0.95))
    m = SystemMetrics(0.9, 0.9, 0.1, 0.1, 0.0, 0.0)

    success_curr = Curriculum(np.zeros(4), np.array([0.2, 0.1, -0.1, 0.0]), np.zeros(4))
    _, ok = c.run(_state(), m, success_curr)
    assert ok.status in {"SUCCESS", "SKIPPED"}

    fail_curr = Curriculum(np.ones(4) * 10, np.zeros(4), np.ones(4))
    _, bad = c.run(_state(), m, fail_curr)
    assert bad.status in {"ROLLBACK", "SKIPPED"}


def test_shape_corruption_non_finite_collapse_to_rollback() -> None:
    c = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.1))
    m = SystemMetrics(0.9, 0.9, 0.1, 0.1, 0.0, 0.0)
    bad_curr = Curriculum(np.zeros(3), np.zeros(3), np.zeros(3))
    _, r = c.run(_state(), m, bad_curr)
    assert r.status == "ROLLBACK"
    assert "Execution fault" in r.reason or "shape" in r.reason
