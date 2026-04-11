from __future__ import annotations

import numpy as np

from neurophase.reset import KLRPipeline, SystemMetrics, SystemState


def _state() -> SystemState:
    n = 5
    return SystemState(
        weights=np.full((n, n), 1.0 / n),
        confidence=np.full(n, 0.5),
        usage=np.linspace(0.1, 0.9, n),
        utility=np.linspace(0.2, 0.8, n),
        inhibition=np.full(n, 0.7),
        topology=np.ones((n, n)),
    )


def test_pipeline_is_deterministic() -> None:
    m = SystemMetrics(0.9, 0.9, 0.2, 0.2, 0.1, 0.1)
    p1 = KLRPipeline(_state())
    p2 = KLRPipeline(_state())

    seq1 = [p1.tick(m).report.status for _ in range(40)]
    seq2 = [p2.tick(m).report.status for _ in range(40)]
    assert seq1 == seq2
    assert np.allclose(p1.twin_state.active.weights, p2.twin_state.active.weights)
