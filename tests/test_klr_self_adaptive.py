from __future__ import annotations

import numpy as np

from neurophase.reset import KLRPipeline, SystemMetrics, SystemState


def _state() -> SystemState:
    n = 8
    return SystemState(
        weights=np.full((n, n), 1.0 / n),
        confidence=np.full(n, 0.5),
        usage=np.array([0.99, 0.98, 0.9, 0.1, 0.1, 0.1, 0.2, 0.2]),
        utility=np.linspace(0.9, 0.1, n),
        inhibition=np.full(n, 0.8),
        topology=np.ones((n, n)),
    )


def test_self_adaptive_behavior() -> None:
    p = KLRPipeline(_state())
    m = SystemMetrics(0.95, 0.9, 0.1, 0.1, 0.1, 0.1)
    deltas = []
    rollbacks_after_500 = 0
    for i in range(700):
        frame = p.tick(m)
        deltas.append(frame.ntk_rank_delta)
        if i > 500 and frame.report.status == "ROLLBACK":
            rollbacks_after_500 += 1
    assert p.adaptive_threshold.current() != 0.72
    assert float(np.mean(deltas)) >= 0.0
    assert rollbacks_after_500 == 0
