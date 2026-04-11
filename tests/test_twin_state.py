from __future__ import annotations

import numpy as np

from neurophase.reset.state import SystemState
from neurophase.reset.twin_state import TwinStateManager


def _state(scale: float) -> SystemState:
    n = 4
    w = np.full((n, n), 1.0 / n) if scale < 0.5 else np.eye(n)
    return SystemState(
        weights=w,
        confidence=np.full(n, 0.5),
        usage=np.full(n, 0.4),
        utility=np.full(n, 0.5),
        inhibition=np.full(n, 0.5),
        topology=np.ones((n, n)),
    )


def test_swap_occurs_on_higher_passive_rank() -> None:
    mgr = TwinStateManager(active=_state(0.3), passive=_state(0.9), swap_interval=1)
    swapped = mgr.tick([], refractory_active=False)
    assert swapped


def test_no_swap_in_refractory() -> None:
    mgr = TwinStateManager(active=_state(0.3), passive=_state(0.9), swap_interval=1)
    swapped = mgr.tick([], refractory_active=True)
    assert not swapped
