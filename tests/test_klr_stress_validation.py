from __future__ import annotations

import numpy as np

from neurophase.reset import Curriculum, KetamineLikeResetController, SystemMetrics, SystemState


def _random_state(rng: np.random.Generator, n: int = 8) -> SystemState:
    w = rng.random((n, n))
    w = w / w.sum(axis=1, keepdims=True)
    return SystemState(
        weights=w,
        confidence=rng.uniform(0.0, 1.0, n),
        usage=rng.uniform(0.0, 1.0, n),
        utility=rng.uniform(0.0, 1.0, n),
        inhibition=rng.uniform(0.0, 1.0, n),
        topology=np.ones((n, n), dtype=float),
        gamma=float(rng.uniform(0.0, 1.0)),
    )


def test_stress_validation_fail_closed_and_finite() -> None:
    rng = np.random.default_rng(20260411)

    counts = {"SUCCESS": 0, "ROLLBACK": 0, "SKIPPED": 0}
    for _ in range(200):
        controller = KetamineLikeResetController()
        state = _random_state(rng)
        metrics = SystemMetrics(
            error=float(rng.uniform(0.0, 1.0)),
            persistence=float(rng.uniform(0.0, 1.0)),
            diversity=float(rng.uniform(0.0, 1.0)),
            improvement=float(rng.uniform(0.0, 1.0)),
            noise=float(rng.uniform(0.0, 1.0)),
            reward=float(rng.uniform(-1.0, 1.0)),
        )
        curriculum = Curriculum(
            target_bias=rng.uniform(0.0, 1.0, 8),
            corrective_signal=rng.uniform(-0.2, 0.2, 8),
            stress_pattern=rng.uniform(0.0, 1.0, 8),
        )

        out, report = controller.run(state, metrics, curriculum)
        assert report.status in counts
        counts[report.status] += 1
        assert np.isfinite(report.relapse_ratio)
        assert np.isfinite(report.improvement_ratio)
        assert np.isfinite(report.gamma_after)
        assert out.weights.shape == (8, 8)
        assert np.isfinite(out.weights).all()

    non_zero = sum(1 for v in counts.values() if v > 0)
    assert non_zero >= 2
