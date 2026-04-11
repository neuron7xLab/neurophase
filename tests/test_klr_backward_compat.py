from __future__ import annotations

import numpy as np

from neurophase import KLRConfig as RootConfig
from neurophase.reset import (
    Curriculum,
    KetamineLikeResetController,
    KLRConfig,
    SystemMetrics,
    SystemState,
)
from neurophase.state.klr_reset import KLRConfig as ShimConfig


def test_import_surfaces_and_run_intervention_dict() -> None:
    assert RootConfig is KLRConfig
    assert ShimConfig is KLRConfig

    s = SystemState(
        weights=np.full((4, 4), 0.25),
        confidence=np.full(4, 0.6),
        usage=np.full(4, 0.7),
        utility=np.full(4, 0.7),
        inhibition=np.full(4, 0.7),
        topology=np.ones((4, 4)),
        gamma=0.4,
    )
    m = SystemMetrics(0.1, 0.2, 0.9, 0.9, 0.0, 0.0)
    c = Curriculum(np.zeros(4), np.zeros(4), np.zeros(4))

    _, report = KetamineLikeResetController().run_intervention(s, m, c)
    assert "status" in report and "threshold_used" in report
