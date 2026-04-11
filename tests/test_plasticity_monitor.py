from __future__ import annotations

import numpy as np

from neurophase.reset.plasticity_monitor import PlasticityMonitor


def test_plasticity_report_fields() -> None:
    m = PlasticityMonitor()
    rep = m.compute(
        np.array([1.0, 1.0, 1.0]),
        np.array([False, False, False]),
        0.5,
        plasticity_floor=0.3,
        injection_triggered=False,
    )
    assert rep.freeze_ratio == 0.0
    assert rep.diversity_index <= 1.0
    assert not rep.injection_triggered
    assert rep.ntk_rank_vs_floor > 0.0


def test_warning_on_negative_trend() -> None:
    m = PlasticityMonitor()
    for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        rep = m.compute(
            np.array([1.0, 2.0, 3.0]),
            np.array([False, False, False]),
            r,
            plasticity_floor=0.3,
            injection_triggered=r < 0.3,
        )
    assert rep.warning is not None
