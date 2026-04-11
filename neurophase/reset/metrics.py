"""Diagnostic metrics for lock-in detection."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SystemMetrics:
    """Metrics used by the lock-in classifier."""

    error: float
    persistence: float
    diversity: float
    improvement: float
    noise: float
    reward: float

    def __post_init__(self) -> None:
        finite = (
            self.error,
            self.persistence,
            self.diversity,
            self.improvement,
            self.noise,
            self.reward,
        )
        if not all(math.isfinite(v) for v in finite):
            raise ValueError("all metrics must be finite")
        for name in ("error", "persistence", "diversity", "improvement"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.noise < 0.0:
            raise ValueError("noise must be >= 0")
