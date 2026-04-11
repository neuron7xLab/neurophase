from __future__ import annotations

import numpy as np
import pytest

from neurophase.reset import (
    Curriculum,
    KetamineLikeResetController,
    KLRConfig,
    SystemMetrics,
    SystemState,
)


def _base_state(n: int = 4) -> SystemState:
    return SystemState(
        weights=np.full((n, n), 1.0 / n),
        confidence=np.full(n, 0.6),
        usage=np.full(n, 0.7),
        utility=np.full(n, 0.7),
        inhibition=np.full(n, 0.7),
        topology=np.ones((n, n)),
    )


def _metrics_high() -> SystemMetrics:
    return SystemMetrics(0.9, 0.9, 0.1, 0.1, 0.0, 0.0)


def _curr(n: int = 4) -> Curriculum:
    return Curriculum(np.zeros(n), np.zeros(n), np.zeros(n))


def test_nan_in_weights_rejected() -> None:
    with pytest.raises(ValueError):
        s = _base_state()
        s.weights[0, 0] = np.nan
        SystemState(**s.__dict__)


def test_inf_in_metrics_safe_terminal_state() -> None:
    # SystemMetrics contract rejects non-finite inputs at construction
    # time — failing fast at the type boundary is the safe behavior.
    with pytest.raises(ValueError, match="finite"):
        SystemMetrics(np.inf, 0.9, 0.1, 0.1, 0.0, 0.0)


def test_all_nodes_frozen_safe() -> None:
    s = _base_state()
    s.frozen[:] = True
    c = KetamineLikeResetController()
    _, report = c.run(s, _metrics_high(), _curr())
    assert report.status in {"ROLLBACK", "SKIPPED", "SUCCESS"}


def test_zero_diversity_metrics() -> None:
    c = KetamineLikeResetController()
    _, report = c.run(_base_state(), SystemMetrics(0.8, 0.8, 0.0, 0.1, 0.0, 0.0), _curr())
    assert report.status in {"ROLLBACK", "SKIPPED", "SUCCESS"}


def test_negative_weights_input_safe() -> None:
    s = _base_state()
    s.weights[0, 1] = -0.5
    c = KetamineLikeResetController()
    _, report = c.run(s, _metrics_high(), _curr())
    assert report.status in {"ROLLBACK", "SUCCESS", "SKIPPED"}


def test_refractory_path_double_run() -> None:
    c = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.1))
    s = _base_state()
    c.run(s, _metrics_high(), _curr())
    _, report2 = c.run(s, _metrics_high(), _curr())
    assert report2.status == "SKIPPED"


def test_curriculum_shape_mismatch_value_error_path() -> None:
    c = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.1))
    bad = Curriculum(np.zeros(3), np.zeros(3), np.zeros(3))
    _, report = c.run(_base_state(), _metrics_high(), bad)
    assert report.status == "ROLLBACK"


def test_curriculum_all_zeros() -> None:
    c = KetamineLikeResetController()
    _, report = c.run(_base_state(), _metrics_high(), _curr())
    assert report.status in {"ROLLBACK", "SUCCESS", "SKIPPED"}
