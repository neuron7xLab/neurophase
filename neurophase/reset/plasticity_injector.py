"""Continual plasticity injection for low-usage nodes."""

from __future__ import annotations

import hashlib

import numpy as np

from neurophase.reset.config import KLRConfig
from neurophase.reset.state import SystemState


class PlasticityInjector:
    def maybe_inject(
        self,
        state: SystemState,
        ntk_rank_normalized: float,
        config: KLRConfig,
    ) -> bool:
        if ntk_rank_normalized >= config.plasticity_floor:
            return False

        deficit = config.plasticity_floor - ntk_rank_normalized
        n = state.weights.shape[0]
        n_reinit = max(1, int(deficit * n))
        order = np.argsort(state.usage)
        selected = [
            int(i) for i in order[:n_reinit] if state.frozen is None or not bool(state.frozen[i])
        ]
        if not selected:
            return False

        digest = hashlib.sha256(
            state.weights.tobytes() + str(ntk_rank_normalized).encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)

        for idx in selected:
            state.weights[idx, :] = rng.dirichlet(np.ones(n, dtype=np.float64))
            state.usage[idx] = 0.0
            state.confidence[idx] = 0.5

        row_sums = state.weights.sum(axis=1, keepdims=True)
        state.weights = np.divide(state.weights, np.where(row_sums == 0.0, 1.0, row_sums))
        return True
