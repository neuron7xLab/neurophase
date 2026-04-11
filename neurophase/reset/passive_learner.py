"""Off-policy learner for the passive twin state."""

from __future__ import annotations

import numpy as np

from neurophase.reset.ledger import LedgerEntry
from neurophase.reset.state import SystemState


class PassiveLearner:
    def __init__(self, lr: float = 0.01, replay_batch_size: int = 10) -> None:
        self.lr = lr
        self.replay_batch_size = replay_batch_size

    def replay(self, passive: SystemState, ledger: list[LedgerEntry]) -> None:
        success = [e for e in ledger if e.decision == "SUCCESS"]
        if not success:
            return
        batch = success[-self.replay_batch_size :]

        novelty = 1.0 / (passive.usage + 1e-8)
        novelty = novelty / (float(np.max(novelty)) + 1e-8)

        for _ in batch:
            update = self.lr * novelty[:, None] * np.outer(passive.utility, passive.utility)
            if passive.frozen is not None:
                mask = (~passive.frozen)[:, None] & (~passive.frozen)[None, :]
                update = update * mask
            passive.weights += update
            row_sums = passive.weights.sum(axis=1, keepdims=True)
            passive.weights = np.divide(passive.weights, np.where(row_sums == 0.0, 1.0, row_sums))
