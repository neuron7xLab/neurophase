"""Ensemble consensus wrapper for KLR."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from neurophase.reset.config import KLRConfig
from neurophase.reset.controller import KetamineLikeResetController, ResetReport
from neurophase.reset.curriculum import Curriculum
from neurophase.reset.metrics import SystemMetrics
from neurophase.reset.state import SystemState


@dataclass(frozen=True)
class EnsembleDecision:
    decision: str
    consensus_score: float
    report: ResetReport


class KLREnsemble:
    def __init__(self, n_members: int = 3) -> None:
        if n_members != 3:
            raise ValueError("current strict voting protocol requires n_members=3")
        thresholds = [0.70, 0.72, 0.74]
        self.members = [
            KetamineLikeResetController(KLRConfig(lock_in_threshold=t)) for t in thresholds
        ]

    def run(
        self, state: SystemState, metrics: SystemMetrics, curriculum: Curriculum
    ) -> tuple[SystemState, EnsembleDecision]:
        outputs = [m.run(deepcopy(state), metrics, curriculum) for m in self.members]
        reports = [r for _, r in outputs]
        states = [s for s, _ in outputs]

        counts: dict[str, int] = {}
        for r in reports:
            counts[r.status] = counts.get(r.status, 0) + 1

        if counts.get("SUCCESS", 0) >= 2:
            success = sorted(
                [(s, r) for s, r in outputs if r.status == "SUCCESS"],
                key=lambda sr: sr[1].improvement_ratio,
            )
            chosen_state, chosen_report = success[len(success) // 2]
            return chosen_state, EnsembleDecision("APPROVE", counts["SUCCESS"] / 3.0, chosen_report)

        if counts.get("ROLLBACK", 0) == 3:
            return states[0], EnsembleDecision("ROLLBACK", 1.0, reports[0])

        if counts.get("SKIPPED", 0) >= 2:
            skipped = next(r for r in reports if r.status == "SKIPPED")
            return state, EnsembleDecision("SKIP", counts["SKIPPED"] / 3.0, skipped)

        conservative = max(reports, key=lambda r: r.relapse_ratio)
        idx = reports.index(conservative)
        agreement = max(counts.values())
        return states[idx], EnsembleDecision("CONSERVATIVE", agreement / 3.0, conservative)
