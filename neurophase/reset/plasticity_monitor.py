"""Monitor of diversity/rank trends during continual adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PlasticityReport:
    diversity_index: float
    rank_trend: float
    freeze_ratio: float
    injection_triggered: bool
    ntk_rank_current: float
    ntk_rank_vs_floor: float
    warning: str | None


class PlasticityMonitor:
    def __init__(self, monitor_interval: int = 25) -> None:
        self.monitor_interval = monitor_interval
        self._rank_history: list[float] = []

    def _gini(self, x: NDArray[np.float64]) -> float:
        arr = np.sort(np.asarray(x, dtype=np.float64))
        if arr.size == 0:
            return 0.0
        if np.all(arr == 0):
            return 0.0
        n = arr.size
        idx = np.arange(1, n + 1)
        return float((2 * np.sum(idx * arr) / (n * np.sum(arr))) - (n + 1) / n)

    def compute(
        self,
        usage: NDArray[np.float64],
        frozen: NDArray[np.bool_],
        ntk_rank: float,
        *,
        plasticity_floor: float,
        injection_triggered: bool,
    ) -> PlasticityReport:
        self._rank_history.append(float(ntk_rank))
        if len(self._rank_history) > 128:
            self._rank_history = self._rank_history[-128:]

        diversity_index = float(1.0 - self._gini(usage))
        recent = self._rank_history[-10:]
        if len(recent) >= 2:
            x: NDArray[np.float64] = np.arange(len(recent), dtype=np.float64)
            slope = float(np.polyfit(x, np.array(recent, dtype=np.float64), 1)[0])
        else:
            slope = 0.0
        freeze_ratio = float(np.sum(frozen) / frozen.size) if frozen.size else 0.0
        warning = "negative_rank_trend" if slope < -0.01 else None
        return PlasticityReport(
            diversity_index=diversity_index,
            rank_trend=slope,
            freeze_ratio=freeze_ratio,
            injection_triggered=injection_triggered,
            ntk_rank_current=float(ntk_rank),
            ntk_rank_vs_floor=float(ntk_rank - plasticity_floor),
            warning=warning,
        )
