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
    n = 6
    return SystemState(
        weights=np.full((n, n), 1.0 / n, dtype=float),
        confidence=np.linspace(0.4, 0.9, n),
        usage=np.linspace(0.2, 0.9, n),
        utility=np.linspace(0.5, 0.8, n),
        inhibition=np.full(n, 0.7),
        topology=np.ones((n, n), dtype=float),
    )


def test_bit_identical_for_identical_inputs_three_runs() -> None:
    metrics = SystemMetrics(0.92, 0.90, 0.15, 0.10, 0.0, 0.0)
    curriculum = Curriculum(
        target_bias=np.zeros(6),
        corrective_signal=np.array([0.2, -0.1, 0.1, 0.0, 0.05, -0.02]),
        stress_pattern=np.zeros(6),
    )

    outs = []
    for _ in range(3):
        c = KetamineLikeResetController(KLRConfig(random_seed=999))
        s, r = c.run(_state(), metrics, curriculum)
        outs.append((s.weights.copy(), r.seed_trace, r.status, r.improvement_ratio))

    first = outs[0]
    for item in outs[1:]:
        assert np.allclose(first[0], item[0])
        assert first[1:] == item[1:]
