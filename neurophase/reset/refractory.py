"""Refractory gate to avoid over-frequent interventions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RefractoryGate:
    success_lock_seconds: float = 3600.0
    rollback_lock_seconds: float = 7200.0
    override_seconds: float = 21600.0
    _locked_at: float | None = None
    _unlock_at: float | None = None

    def can_intervene(self, now: float) -> bool:
        self.force_unlock_after_threshold(now)
        return self._unlock_at is None or now >= self._unlock_at

    def seconds_until_ready(self, now: float) -> float:
        self.force_unlock_after_threshold(now)
        if self._unlock_at is None:
            return 0.0
        return max(0.0, self._unlock_at - now)

    def register_outcome(self, decision: str, now: float) -> None:
        self._locked_at = now
        if decision == "SUCCESS":
            self._unlock_at = now + self.success_lock_seconds
        elif decision == "ROLLBACK":
            self._unlock_at = now + self.rollback_lock_seconds
        else:
            self._unlock_at = now

    def force_unlock_after_threshold(self, now: float) -> None:
        if self._locked_at is None:
            return
        if now - self._locked_at >= self.override_seconds:
            self._unlock_at = now
