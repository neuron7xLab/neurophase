from __future__ import annotations

import numpy as np

from neurophase.reset import Curriculum, KLREnsemble, SystemMetrics, SystemState


def _state() -> SystemState:
    n = 4
    return SystemState(
        weights=np.eye(n),
        confidence=np.array([0.9, 0.7, 0.5, 0.6]),
        usage=np.array([0.9, 0.8, 0.2, 0.3]),
        utility=np.array([0.8, 0.75, 0.4, 0.5]),
        inhibition=np.ones(n),
        topology=np.ones((n, n)),
    )


def test_skip_vote_outcome() -> None:
    e = KLREnsemble()
    curriculum = Curriculum(np.zeros(4), np.zeros(4), np.ones(4))
    low = SystemMetrics(0.1, 0.2, 0.8, 0.9, 0.0, 0.0)
    _, decision = e.run(_state(), low, curriculum)
    assert decision.decision == "SKIP"


def test_rollback_vote_outcome() -> None:
    e = KLREnsemble()
    curriculum = Curriculum(np.ones(4) * 10, np.zeros(4), np.ones(4))
    high = SystemMetrics(0.95, 0.95, 0.1, 0.1, 0.0, 0.0)
    _, decision = e.run(_state(), high, curriculum)
    assert decision.decision in {"ROLLBACK", "CONSERVATIVE"}


def test_success_or_approve_path() -> None:
    e = KLREnsemble()
    curriculum = Curriculum(np.zeros(4), np.array([0.2, 0.1, -0.1, 0.0]), np.zeros(4))
    high = SystemMetrics(0.95, 0.95, 0.1, 0.1, 0.0, 0.0)
    _, decision = e.run(_state(), high, curriculum)
    assert decision.decision in {"APPROVE", "CONSERVATIVE"}


def test_split_conservative_path_possible() -> None:
    e = KLREnsemble()
    curriculum = Curriculum(
        np.array([1.0, 0.0, 0.0, 1.0]),
        np.array([0.1, -0.3, 0.05, -0.1]),
        np.array([1.0, 0.0, 1.0, 0.0]),
    )
    high = SystemMetrics(0.85, 0.82, 0.2, 0.2, 0.0, 0.0)
    _, decision = e.run(_state(), high, curriculum)
    assert 0.0 <= decision.consensus_score <= 1.0
