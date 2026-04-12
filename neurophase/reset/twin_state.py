"""Twin-state orchestration for active/passive plasticity."""

from __future__ import annotations

from dataclasses import dataclass

from neurophase.reset.ledger import LedgerEntry
from neurophase.reset.ntk_monitor import NTKMonitor
from neurophase.reset.passive_learner import PassiveLearner
from neurophase.reset.state import SystemState, clone_state


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
            prev_active = clone_state(self.active)
            self.active = clone_state(self.passive)
            self.passive = prev_active
            return True
        return False
