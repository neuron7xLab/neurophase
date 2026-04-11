from __future__ import annotations

import numpy as np
import pytest

from neurophase.reset import SystemState


def _valid(n: int = 4) -> SystemState:
    return SystemState(
        weights=np.full((n, n), 1.0 / n),
        confidence=np.full(n, 0.5),
        usage=np.full(n, 0.5),
        utility=np.full(n, 0.5),
        inhibition=np.full(n, 0.5),
        topology=np.ones((n, n)),
        frozen=None,
        gamma=0.4,
    )


def test_valid_minimal_state_and_frozen_init() -> None:
    s = _valid()
    assert s.frozen is not None
    assert s.frozen.dtype == bool


def test_invalid_shapes_and_ranges_rejected() -> None:
    with pytest.raises(ValueError):
        SystemState(
            weights=np.ones((4, 3)),
            confidence=np.ones(4),
            usage=np.ones(4),
            utility=np.ones(4),
            inhibition=np.ones(4),
            topology=np.ones((4, 4)),
        )
    with pytest.raises(ValueError):
        s = _valid()
        s.confidence[0] = 2.0
        SystemState(**s.__dict__)


def test_finite_enforcement_and_gamma_bounds() -> None:
    with pytest.raises(ValueError):
        s = _valid()
        s.weights[0, 0] = np.nan
        SystemState(**s.__dict__)
    with pytest.raises(ValueError):
        s = _valid()
        s.gamma = 1.2
        SystemState(**s.__dict__)


def test_row_stochastic_semantics_positive_row_sum() -> None:
    with pytest.raises(ValueError):
        s = _valid()
        s.weights[0, :] = 0.0
        SystemState(**s.__dict__)
