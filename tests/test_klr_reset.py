from __future__ import annotations

import numpy as np

from neurophase.reset import (
    Curriculum,
    KetamineLikeResetController,
    KLRConfig,
    ResetReport,
    ResetState,
    SystemMetrics,
    SystemState,
)


def _state() -> SystemState:
    n = 4
    return SystemState(
        weights=np.array(
            [
                [0.40, 0.10, 0.20, 0.30],
                [0.20, 0.40, 0.20, 0.20],
                [0.25, 0.25, 0.25, 0.25],
                [0.35, 0.15, 0.25, 0.25],
            ],
            dtype=float,
        ),
        confidence=np.array([0.90, 0.60, 0.55, 0.70], dtype=float),
        usage=np.array([0.95, 0.40, 0.30, 0.85], dtype=float),
        utility=np.array([0.80, 0.62, 0.58, 0.75], dtype=float),
        inhibition=np.array([0.90, 0.60, 0.55, 0.80], dtype=float),
        topology=np.ones((n, n), dtype=float),
    )


def _metrics_lockin() -> SystemMetrics:
    return SystemMetrics(
        error=0.90,
        persistence=0.90,
        diversity=0.20,
        improvement=0.10,
        noise=0.10,
        reward=0.20,
    )


def test_skip_when_stable_metrics() -> None:
    state = _state()
    metrics = SystemMetrics(0.1, 0.2, 0.9, 0.9, 0.1, 0.8)
    curriculum = Curriculum(
        target_bias=np.ones(4),
        corrective_signal=np.zeros(4),
        stress_pattern=np.ones(4),
    )
    controller = KetamineLikeResetController(KLRConfig(random_seed=7))

    out_state, report = controller.run(state, metrics, curriculum)

    assert out_state is state
    assert isinstance(report, ResetReport)
    assert report.status == "SKIPPED"
    assert 0.0 <= report.lockin_score < controller.config.lock_in_threshold
    assert report.threshold_used == controller.config.lock_in_threshold
    assert controller.state_machine is ResetState.IDLE


def test_success_path_reports_structured_result() -> None:
    state = _state()
    curriculum = Curriculum(
        target_bias=np.zeros(4, dtype=float),
        corrective_signal=np.array([0.10, 0.20, -0.10, 0.10], dtype=float),
        stress_pattern=np.zeros(4, dtype=float),
    )
    controller = KetamineLikeResetController(KLRConfig(random_seed=13))

    out_state, report = controller.run(state, _metrics_lockin(), curriculum)

    assert out_state is state
    assert report.status == "SUCCESS"
    assert 0.0 <= report.improvement_ratio <= 1.0
    assert 0.0 <= report.gamma_after <= 1.0
    assert report.lockin_score >= 0.72
    assert controller.state_machine is ResetState.STABLE
    assert np.allclose(out_state.weights.sum(axis=1), 1.0)


def test_rollback_path_and_backward_compat_api() -> None:
    state = _state()
    base = np.copy(state.weights)
    curriculum = Curriculum(
        target_bias=np.array([10.0, 10.0, 10.0, 10.0], dtype=float),
        corrective_signal=np.zeros(4, dtype=float),
        stress_pattern=np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
    )
    controller = KetamineLikeResetController(KLRConfig(random_seed=19))

    out_state, report = controller.run_intervention(state, _metrics_lockin(), curriculum)

    assert report["status"] == "ROLLBACK"
    assert "lockin_score" in report
    assert controller.state_machine is ResetState.ROLLBACK
    assert np.allclose(out_state.weights, base)


def test_run_is_deterministic_for_same_seed() -> None:
    curriculum = Curriculum(
        target_bias=np.zeros(4, dtype=float),
        corrective_signal=np.array([0.10, 0.20, -0.10, 0.10], dtype=float),
        stress_pattern=np.zeros(4, dtype=float),
    )
    c1 = KetamineLikeResetController(KLRConfig(random_seed=23))
    c2 = KetamineLikeResetController(KLRConfig(random_seed=23))

    s1, r1 = c1.run(_state(), _metrics_lockin(), curriculum)
    s2, r2 = c2.run(_state(), _metrics_lockin(), curriculum)

    assert np.allclose(s1.weights, s2.weights)
    assert r1.status == r2.status
    assert np.isclose(r1.gamma_after, r2.gamma_after)


def test_explain_lockin_terms_stable() -> None:
    c = KetamineLikeResetController()
    exp = c.explain_lockin(_metrics_lockin())
    assert exp["total_score"] >= exp["threshold"]
    assert exp["triggered"] == 1.0
