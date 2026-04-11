"""Rollback ledger for KLR attempts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time

from neurophase.reset.controller import ResetReport
from neurophase.reset.metrics import SystemMetrics
from neurophase.reset.state import SystemState


@dataclass(frozen=True)
class LedgerEntry:
    timestamp: float
    state_hash: str
    metrics_snapshot: dict[str, float]
    decision: str
    relapse_ratio: float
    improvement_ratio: float
    reason: str
    new_frozen_nodes: list[int]


class RollbackLedger:
    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._entries: list[LedgerEntry] = []

    def record_attempt(
        self, state: SystemState, metrics: SystemMetrics, result: ResetReport
    ) -> None:
        frozen = [] if state.frozen is None else [int(i) for i, v in enumerate(state.frozen) if v]
        entry = LedgerEntry(
            timestamp=time(),
            state_hash=hashlib.sha256(state.weights.tobytes()).hexdigest(),
            metrics_snapshot={
                "error": metrics.error,
                "persistence": metrics.persistence,
                "diversity": metrics.diversity,
                "improvement": metrics.improvement,
                "noise": metrics.noise,
                "reward": metrics.reward,
                "lockin_score": result.lockin_score,
            },
            decision=result.status,
            relapse_ratio=result.relapse_ratio,
            improvement_ratio=result.improvement_ratio,
            reason=result.reason,
            new_frozen_nodes=frozen,
        )
        self._entries.append(entry)
        if len(self._entries) > self.capacity:
            self._entries = self._entries[-self.capacity :]

    def query_by_reason(self, reason: str) -> list[LedgerEntry]:
        return [e for e in self._entries if e.reason == reason]

    def export_json(self, path: Path | str) -> None:
        payload = [asdict(e) for e in self._entries]
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def statistics(self) -> dict[str, float | int | dict[str, int]]:
        total = len(self._entries)
        rollback = sum(1 for e in self._entries if e.decision == "ROLLBACK")
        reasons: dict[str, int] = {}
        for e in self._entries:
            reasons[e.reason] = reasons.get(e.reason, 0) + 1
        top = dict(sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:3])
        return {
            "total": total,
            "rollback_rate": (rollback / total) if total else 0.0,
            "top_reasons": top,
        }
