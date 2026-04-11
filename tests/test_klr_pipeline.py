from __future__ import annotations

import numpy as np

from neurophase.reset import KLRPipeline, SystemMetrics, SystemState


def _state() -> SystemState:
    n = 6
    return SystemState(
        weights=np.full((n, n), 1.0 / n),
        confidence=np.full(n, 0.6),
        usage=np.linspace(0.1, 0.9, n),
        utility=np.linspace(0.2, 0.8, n),
        inhibition=np.full(n, 0.5),
        topology=np.ones((n, n)),
    )


def _metrics() -> SystemMetrics:
    return SystemMetrics(0.9, 0.9, 0.2, 0.2, 0.1, 0.1)


def test_pipeline_runs_and_rows_invariant() -> None:
    p = KLRPipeline(_state())
    for _ in range(1000):
        frame = p.tick(_metrics())
        assert frame.report.status in {"SUCCESS", "SKIPPED", "ROLLBACK"}
    assert np.allclose(p.twin_state.active.weights.sum(axis=1), 1.0)
    ntk_rank = p.ntk_monitor.rank_proxy(p.twin_state.active.weights)
    assert ntk_rank >= p.config.plasticity_floor


def test_no_mutation_during_refractory() -> None:
    p = KLRPipeline(_state())
    p.refractory.register_outcome("SUCCESS", 1.0)
    before = p.twin_state.active.weights.copy()
    _ = p.tick(_metrics())
    assert np.allclose(before, p.twin_state.active.weights)
