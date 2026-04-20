"""Market coupling diagnostics for KLR orchestration."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from neurophase.reset.config import KLRConfig
from neurophase.reset.curriculum import Curriculum


class MarketPhase(Enum):
    COHERENT = auto()
    FRAGMENTED = auto()
    BREAKDOWN = auto()


class MarketCouplingValidator:
    def verify_regime_alignment(
        self, brain_state: NDArray[np.float64], market_phase: NDArray[np.float64]
    ) -> tuple[bool, float]:
        c = np.corrcoef(brain_state, market_phase)
        eig = np.linalg.eigvals(c)
        gamma = float(np.max(np.real(eig)) / max(1, c.shape[0]))
        return gamma <= 1.15, gamma

    def should_reset_on_regime_break(self, coherence: float) -> bool:
        return coherence < 0.35

    def adapt_config_for_market_phase(
        self, config: KLRConfig, market_phase: MarketPhase
    ) -> KLRConfig:
        if market_phase is MarketPhase.BREAKDOWN:
            return KLRConfig(lock_in_threshold=min(1.0, config.lock_in_threshold + 0.05))
        return config

    def curriculum_from_market_regime(self, market_phase: MarketPhase, n: int) -> Curriculum:
        if market_phase is MarketPhase.COHERENT:
            return Curriculum(
                target_bias=np.full(n, 0.8, dtype=np.float64),
                corrective_signal=np.full(n, 0.2, dtype=np.float64),
                stress_pattern=np.full(n, 0.1, dtype=np.float64),
            )
        if market_phase is MarketPhase.FRAGMENTED:
            return Curriculum(
                target_bias=np.full(n, 0.5, dtype=np.float64),
                corrective_signal=np.linspace(-0.2, 0.2, n, dtype=np.float64),
                stress_pattern=np.full(n, 0.2, dtype=np.float64),
            )
        return Curriculum(
            target_bias=np.zeros(n, dtype=np.float64),
            corrective_signal=np.zeros(n, dtype=np.float64),
            stress_pattern=np.full(n, 0.4, dtype=np.float64),
        )
