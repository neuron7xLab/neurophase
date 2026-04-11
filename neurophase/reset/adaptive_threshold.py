"""Self-adaptive lock-in threshold derived from ledger outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurophase.reset.ledger import LedgerEntry


@dataclass
class AdaptiveThreshold:
    fallback_threshold: float = 0.72
    min_threshold: float = 0.55
    max_threshold: float = 0.90
    min_history: int = 20
    update_interval: int = 50
    window_size: int = 100
    _current: float = 0.72
    _frozen: bool = False
    _last_update_count: int = 0

    def current(self) -> float:
        return float(self._current)

    def freeze(self) -> None:
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False

    def update(self, history: list[LedgerEntry]) -> float:
        if self._frozen:
            return self.current()
        if len(history) < self.min_history:
            self._current = self.fallback_threshold
            return self.current()
        if len(history) - self._last_update_count < self.update_interval:
            return self.current()

        window = history[-self.window_size :]
        rollback_scores = [
            e.metrics_snapshot.get("lockin_score", 0.0) for e in window if e.decision == "ROLLBACK"
        ]
        success_scores = [
            e.metrics_snapshot.get("lockin_score", 0.0) for e in window if e.decision == "SUCCESS"
        ]

        candidate = self._current
        if rollback_scores:
            candidate = float(np.percentile(rollback_scores, 25))
        elif success_scores:
            candidate = float(np.percentile(success_scores, 75))

        self._current = float(np.clip(candidate, self.min_threshold, self.max_threshold))
        self._last_update_count = len(history)
        return self.current()

    def serialize(self) -> dict[str, float | int | bool]:
        return {
            "fallback_threshold": self.fallback_threshold,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "min_history": self.min_history,
            "update_interval": self.update_interval,
            "window_size": self.window_size,
            "current": self._current,
            "frozen": self._frozen,
            "last_update_count": self._last_update_count,
        }

    @classmethod
    def deserialize(cls, payload: dict[str, float | int | bool]) -> AdaptiveThreshold:
        obj = cls(
            fallback_threshold=float(payload["fallback_threshold"]),
            min_threshold=float(payload["min_threshold"]),
            max_threshold=float(payload["max_threshold"]),
            min_history=int(payload["min_history"]),
            update_interval=int(payload["update_interval"]),
            window_size=int(payload["window_size"]),
            _current=float(payload["current"]),
            _frozen=bool(payload["frozen"]),
            _last_update_count=int(payload["last_update_count"]),
        )
        return obj
