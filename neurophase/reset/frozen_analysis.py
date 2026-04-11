"""Frozen-node diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurophase.reset.curriculum import Curriculum
from neurophase.reset.state import SystemState


@dataclass(frozen=True)
class FrozenWarning:
    message: str
    nodes: list[int]


class FrozenNodeAnalyzer:
    def verify_consolidation(
        self, pre: SystemState, post: SystemState, curriculum: Curriculum
    ) -> tuple[bool, list[FrozenWarning]]:
        if post.frozen is None:
            return True, []
        warnings: list[FrozenWarning] = []
        frozen_idx = [int(i) for i, f in enumerate(post.frozen) if f]
        if not frozen_idx:
            return True, warnings

        if np.any(post.utility[post.frozen] < 0.75):
            bad = [int(i) for i in np.where(post.frozen & (post.utility < 0.75))[0]]
            warnings.append(FrozenWarning("utility_below_threshold", bad))

        drift = np.abs(post.weights - pre.weights).mean(axis=1)
        if np.any(drift[post.frozen] > 0.05):
            bad = [int(i) for i in np.where(post.frozen & (drift > 0.05))[0]]
            warnings.append(FrozenWarning("drift_rate_too_high", bad))

        ratio = float(np.mean(post.frozen))
        if ratio > 0.5:
            warnings.append(FrozenWarning("frozen_ratio_exceeds_half", frozen_idx))

        _ = curriculum  # explicit in contract, currently not required by checks
        return len(warnings) == 0, warnings

    def detect_premature_freeze(self, history: list[SystemState]) -> list[int]:
        if len(history) < 2:
            return []
        n = history[-1].weights.shape[0]
        if history[-1].frozen is None:
            return []
        flagged: list[int] = []
        for i in range(n):
            frozen_steps = [bool(s.frozen[i]) for s in history if s.frozen is not None]
            if not frozen_steps[-1]:
                continue
            util = [float(s.utility[i]) for s in history]
            if max(util) < 0.75:
                flagged.append(i)
                continue
            drift = [
                abs(float(history[j].weights[i].mean() - history[j - 1].weights[i].mean()))
                for j in range(1, len(history))
            ]
            if drift and max(drift) > 0.05:
                flagged.append(i)
        return sorted(set(flagged))
