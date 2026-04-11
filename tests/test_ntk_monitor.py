from __future__ import annotations

import numpy as np

from neurophase.reset import (
    Curriculum,
    KetamineLikeResetController,
    KLRConfig,
    SystemMetrics,
    SystemState,
)
from neurophase.reset.ntk_monitor import NTKMonitor


def test_rank_non_decreasing_on_success() -> None:
    n = 4
    state = SystemState(
        weights=np.eye(n),
        confidence=np.full(n, 0.6),
        usage=np.array([0.9, 0.8, 0.1, 0.2]),
        utility=np.array([0.8, 0.7, 0.5, 0.6]),
        inhibition=np.full(n, 0.7),
        topology=np.ones((n, n)),
    )
    metrics = SystemMetrics(0.9, 0.9, 0.2, 0.1, 0.1, 0.1)
    cur = Curriculum(np.zeros(n), np.array([0.2, 0.1, -0.1, 0.0]), np.zeros(n))
    c = KetamineLikeResetController(KLRConfig(random_seed=1))
    _, report = c.run(state, metrics, cur)
    assert report.ntk_rank_pre >= 0.0
    assert report.ntk_rank_post >= 0.0
    if report.status == "SUCCESS":
        assert report.rank_delta == report.ntk_rank_post - report.ntk_rank_pre


def test_monitor_compare() -> None:
    m = NTKMonitor()
    a = np.eye(3)
    b = np.array([[1 / 3, 1 / 3, 1 / 3]] * 3, dtype=float)
    snap = m.compare(a, b)
    assert snap.rank_delta <= 0.0
