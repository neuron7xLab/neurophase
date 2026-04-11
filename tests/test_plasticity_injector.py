from __future__ import annotations

import numpy as np

from neurophase.reset.config import KLRConfig
from neurophase.reset.plasticity_injector import PlasticityInjector
from neurophase.reset.state import SystemState


def _state() -> SystemState:
    n = 10
    return SystemState(
        weights=np.full((n, n), 0.1),
        confidence=np.full(n, 0.6),
        usage=np.linspace(0.0, 0.9, n),
        utility=np.full(n, 0.5),
        inhibition=np.full(n, 0.5),
        topology=np.ones((n, n)),
        frozen=np.array([True] + [False] * 9),
    )


def test_no_inject_above_floor() -> None:
    state = _state()
    cfg = KLRConfig(plasticity_floor=0.30)
    changed = PlasticityInjector().maybe_inject(state, ntk_rank_normalized=0.35, config=cfg)
    assert not changed


def test_inject_below_floor() -> None:
    state = _state()
    cfg = KLRConfig(plasticity_floor=0.30)
    changed = PlasticityInjector().maybe_inject(state, ntk_rank_normalized=0.0, config=cfg)
    assert changed
    assert np.count_nonzero(state.usage == 0.0) >= 1
    assert np.allclose(state.weights.sum(axis=1), 1.0)


def test_n_reinit_proportional_to_deficit() -> None:
    cfg = KLRConfig(plasticity_floor=0.30)
    inj = PlasticityInjector()
    s_low_deficit = _state()
    s_high_deficit = _state()
    _ = inj.maybe_inject(s_low_deficit, ntk_rank_normalized=0.25, config=cfg)
    _ = inj.maybe_inject(s_high_deficit, ntk_rank_normalized=0.05, config=cfg)
    low_count = int(np.count_nonzero(s_low_deficit.usage == 0.0))
    high_count = int(np.count_nonzero(s_high_deficit.usage == 0.0))
    assert high_count >= low_count


def test_frozen_untouched() -> None:
    s1 = _state()
    frozen_row_before = s1.weights[0, :].copy()
    cfg = KLRConfig(plasticity_floor=0.30)
    _ = PlasticityInjector().maybe_inject(s1, ntk_rank_normalized=0.0, config=cfg)
    assert np.allclose(frozen_row_before, s1.weights[0, :])
