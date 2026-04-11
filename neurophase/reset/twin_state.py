"""Twin-state orchestration for active/passive plasticity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurophase.reset.ledger import LedgerEntry
from neurophase.reset.ntk_monitor import NTKMonitor
from neurophase.reset.passive_learner import PassiveLearner
from neurophase.reset.state import SystemState


def _clone_state(state: SystemState) -> SystemState:
    return SystemState(
        weights=np.copy(state.weights),
        confidence=np.copy(state.confidence),
        usage=np.copy(state.usage),
        utility=np.copy(state.utility),
        inhibition=np.copy(state.inhibition),
        topology=np.copy(state.topology),
        frozen=None if state.frozen is None else np.copy(state.frozen),
        gamma=state.gamma,
        metadata=state.metadata.copy(),
    )


@dataclass
class TwinStateManager:
    active: SystemState
    passive: SystemState
    step_counter: int = 0
    swap_interval: int = 200

    def __post_init__(self) -> None:
        self._ntk = NTKMonitor()
        self._learner = PassiveLearner()

    def tick(self, ledger_entries: list[LedgerEntry], refractory_active: bool) -> bool:
        self.step_counter += 1
        self._learner.replay(self.passive, ledger_entries)

        if refractory_active or self.step_counter % self.swap_interval != 0:
            return False

        active_rank = self._ntk.rank_proxy(self.active.weights)
        passive_rank = self._ntk.rank_proxy(self.passive.weights)
        if passive_rank > active_rank:
            prev_active = _clone_state(self.active)
            self.active = _clone_state(self.passive)
            self.passive = prev_active
            return True
        return False
