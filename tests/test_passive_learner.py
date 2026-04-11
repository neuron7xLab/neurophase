from __future__ import annotations

import numpy as np

from neurophase.reset.ledger import LedgerEntry
from neurophase.reset.passive_learner import PassiveLearner
from neurophase.reset.state import SystemState


def _state() -> SystemState:
    n = 4
    return SystemState(
        weights=np.full((n, n), 0.25),
        confidence=np.full(n, 0.5),
        usage=np.array([0.9, 0.1, 0.2, 0.3]),
        utility=np.array([0.8, 0.5, 0.6, 0.4]),
        inhibition=np.full(n, 0.5),
        topology=np.ones((n, n)),
    )


def test_replay_mutates_passive_only_and_preserves_row_sums() -> None:
    passive = _state()
    before = passive.weights.copy()
    learner = PassiveLearner()
    ledger = [
        LedgerEntry(0.0, "h", {}, "SUCCESS", 0.0, 1.0, "ok", []),
        LedgerEntry(1.0, "h", {}, "ROLLBACK", 1.0, 0.0, "no", []),
    ]
    learner.replay(passive, ledger)
    assert not np.allclose(before, passive.weights)
    assert np.allclose(passive.weights.sum(axis=1), 1.0)


def test_empty_ledger_noop() -> None:
    passive = _state()
    before = passive.weights.copy()
    PassiveLearner().replay(passive, [])
    assert np.allclose(before, passive.weights)
