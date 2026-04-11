from __future__ import annotations

import numpy as np

from neurophase.reset import (
    Curriculum,
    KetamineLikeResetController,
    KLRPipeline,
    SystemMetrics,
    SystemState,
)


def _state() -> SystemState:
    n = 6
    return SystemState(
        weights=np.full((n, n), 1.0 / n),
        confidence=np.full(n, 0.5),
        usage=np.array([0.99, 0.98, 0.95, 0.1, 0.1, 0.1]),
        utility=np.array([0.9, 0.85, 0.8, 0.2, 0.2, 0.2]),
        inhibition=np.full(n, 0.8),
        topology=np.ones((n, n)),
    )


def test_pipeline_beats_patch_lockin_score() -> None:
    metrics = SystemMetrics(0.9, 0.9, 0.2, 0.1, 0.1, 0.1)
    old = KetamineLikeResetController()
    s_old = _state()
    cur = Curriculum(
        target_bias=s_old.utility / np.sum(s_old.utility),
        corrective_signal=s_old.utility - np.mean(s_old.utility),
        stress_pattern=s_old.usage / np.sum(s_old.usage),
    )
    _, old_report = old.run(s_old, metrics, cur)

    new = KLRPipeline(_state())
    frame = new.tick(metrics)
    assert frame.report.lockin_score <= old_report.lockin_score
