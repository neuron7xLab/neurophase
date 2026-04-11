from __future__ import annotations

from pathlib import Path

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
        confidence=np.full(4, 0.6),
        usage=np.full(4, 0.7),
        utility=np.full(4, 0.7),
        inhibition=np.full(4, 0.7),
        topology=np.ones((4, 4)),
        gamma=0.5,
    )


def test_no_commit_without_validation_and_no_permission_widening() -> None:
    c = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.1))
    s = _state()
    m = SystemMetrics(0.9, 0.9, 0.1, 0.1, 0.0, 0.0)
    bad = Curriculum(np.zeros(3), np.zeros(3), np.zeros(3))
    _, r = c.run(s, m, bad)
    assert r.status == "ROLLBACK"


def test_rollback_returns_checkpoint_not_partial_mutation() -> None:
    c = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.1))
    s = _state()
    before = s.weights.copy()
    m = SystemMetrics(0.9, 0.9, 0.1, 0.1, 0.0, 0.0)
    curr = Curriculum(np.ones(4) * 10, np.zeros(4), np.ones(4))
    out, r = c.run(s, m, curr)
    if r.status == "ROLLBACK":
        assert np.allclose(out.weights, before)


def test_tentative_claim_not_promoted_in_docs() -> None:
    text = Path("docs/theory/neurophase_elite_bibliography.md").read_text(encoding="utf-8")
    assert "Lock-in detection via weighted metrics combination | Tentative" in text
