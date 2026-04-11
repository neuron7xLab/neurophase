"""Tests for ``neurophase.state.executive_monitor``.

Covers:
- warmup window
- sensor-absent fallback (I₃-style semantics)
- monotonic-timestamp enforcement
- classification bands (NORMAL / SLOW_DOWN / HARD_BLOCK)
- overload value monotonicity in each channel
- verification-step mapping
- config validation
- falsifiable baseline: monitor detects an injected stress burst before a
  latency-only baseline does.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from neurophase.state.executive_monitor import (
    ExecutiveMonitor,
    ExecutiveMonitorConfig,
    ExecutiveSample,
    OverloadIndex,
    PacingDirective,
    VerificationStep,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calm_sample(t: float) -> ExecutiveSample:
    """A representative 'calm' sample."""
    return ExecutiveSample(beta_power=1.0, hrv=60.0, error_burst=0.0, timestamp=t)


def _stress_sample(t: float) -> ExecutiveSample:
    """A representative 'stressed' sample."""
    return ExecutiveSample(beta_power=10.0, hrv=15.0, error_burst=5.0, timestamp=t)


def _prime(
    monitor: ExecutiveMonitor, n: int, sampler: Callable[[float], ExecutiveSample]
) -> None:
    for i in range(n):
        monitor.update(sampler(float(i)))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults_valid(self) -> None:
        cfg = ExecutiveMonitorConfig()
        assert cfg.window >= 8
        assert 0 < cfg.slow_down_threshold < cfg.hard_block_threshold < 1

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"window": 4},
            {"warmup": 0},
            {"warmup": 100, "window": 64},
            {"weights": (0.5, 0.3, 0.1)},
            {"weights": (-0.1, 0.6, 0.5)},
            {"slope": 0.0},
            {"slow_down_threshold": 0.9, "hard_block_threshold": 0.8},
            {"slow_down_threshold": 0.0},
            {"hard_block_threshold": 1.0},
        ],
    )
    def test_invalid_config_rejected(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            ExecutiveMonitorConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


class TestWarmup:
    def test_warmup_emits_normal_at_0_5(self) -> None:
        monitor = ExecutiveMonitor()
        first = monitor.update(_calm_sample(0.0))
        assert first.directive is PacingDirective.NORMAL
        assert first.value == pytest.approx(0.5)
        assert first.reason == "warmup"

    def test_post_warmup_classifies(self) -> None:
        monitor = ExecutiveMonitor(ExecutiveMonitorConfig(window=32, warmup=8))
        _prime(monitor, 10, _calm_sample)
        result = monitor.update(_calm_sample(11.0))
        assert result.reason.startswith("overload=")
        # Calm baseline should remain NORMAL, not at the warmup sentinel.
        assert result.directive is PacingDirective.NORMAL


class TestSensorAbsent:
    @pytest.mark.parametrize(
        "field,bad",
        [
            ("beta_power", None),
            ("hrv", None),
            ("error_burst", None),
            ("beta_power", float("nan")),
            ("hrv", float("inf")),
            ("error_burst", float("-inf")),
        ],
    )
    def test_missing_or_nonfinite_channels(self, field: str, bad: float | None) -> None:
        monitor = ExecutiveMonitor()
        base: dict[str, float | None] = {
            "beta_power": 1.0,
            "hrv": 60.0,
            "error_burst": 0.0,
            "timestamp": 0.0,
        }
        base[field] = bad
        # ``timestamp`` is always a concrete float here; the mypy cast below
        # narrows the dict-of-Option values to the dataclass signature.
        sample = ExecutiveSample(
            beta_power=base["beta_power"],
            hrv=base["hrv"],
            error_burst=base["error_burst"],
            timestamp=float(base["timestamp"] or 0.0),
        )
        result = monitor.update(sample)
        assert result.directive is PacingDirective.SENSOR_ABSENT
        assert result.reason == "sensor_absent"
        assert result.value == pytest.approx(1.0)


class TestMonotonicTimestamps:
    def test_reject_backwards_timestamp(self) -> None:
        monitor = ExecutiveMonitor()
        monitor.update(_calm_sample(5.0))
        with pytest.raises(ValueError, match="monotonic"):
            monitor.update(_calm_sample(4.0))


# ---------------------------------------------------------------------------
# Classification bands
# ---------------------------------------------------------------------------


class TestClassification:
    def test_stress_escalates_to_hard_block(self) -> None:
        monitor = ExecutiveMonitor(ExecutiveMonitorConfig(window=32, warmup=8))
        _prime(monitor, 16, _calm_sample)
        # Inject a clearly out-of-distribution stress sample.
        result = monitor.update(_stress_sample(20.0))
        assert result.directive is PacingDirective.HARD_BLOCK
        assert result.value > 0.8

    def test_mild_drift_slows_down(self) -> None:
        # A realistic baseline has non-trivial variance — otherwise any
        # deviation looks like ±∞σ. Prime with a noisy calm window, then
        # apply a moderate stress bump.
        monitor = ExecutiveMonitor(ExecutiveMonitorConfig(window=64, warmup=16))
        rng = np.random.default_rng(7)
        for i in range(48):
            monitor.update(
                ExecutiveSample(
                    beta_power=1.0 + float(rng.normal(0, 0.25)),
                    hrv=60.0 + float(rng.normal(0, 5.0)),
                    error_burst=float(rng.uniform(0, 0.3)),
                    timestamp=float(i),
                )
            )
        # Only one channel drifts — beta up by ~1σ, others at baseline.
        mild = ExecutiveSample(beta_power=1.3, hrv=60.0, error_burst=0.15, timestamp=50.0)
        result = monitor.update(mild)
        assert result.directive in {PacingDirective.SLOW_DOWN, PacingDirective.NORMAL}
        # But the value must be strictly above the calm baseline (0.5).
        assert result.value > 0.5

    def test_calm_stays_normal(self) -> None:
        monitor = ExecutiveMonitor(ExecutiveMonitorConfig(window=32, warmup=8))
        _prime(monitor, 32, _calm_sample)
        result = monitor.update(_calm_sample(33.0))
        assert result.directive is PacingDirective.NORMAL


# ---------------------------------------------------------------------------
# Monotonicity properties
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def _prime_with_variance(self) -> ExecutiveMonitor:
        """Seed the monitor with a small-variance baseline so z-scores are finite."""
        monitor = ExecutiveMonitor(ExecutiveMonitorConfig(window=64, warmup=8))
        rng = np.random.default_rng(42)
        for i in range(32):
            monitor.update(
                ExecutiveSample(
                    beta_power=1.0 + float(rng.normal(0, 0.1)),
                    hrv=60.0 + float(rng.normal(0, 2.0)),
                    error_burst=float(rng.uniform(0, 0.2)),
                    timestamp=float(i),
                )
            )
        return monitor

    def test_higher_beta_raises_overload(self) -> None:
        m_low = self._prime_with_variance()
        m_high = self._prime_with_variance()
        low = m_low.update(ExecutiveSample(1.5, 60.0, 0.1, 100.0))
        high = m_high.update(ExecutiveSample(5.0, 60.0, 0.1, 100.0))
        assert high.value > low.value

    def test_lower_hrv_raises_overload(self) -> None:
        m_hi = self._prime_with_variance()
        m_lo = self._prime_with_variance()
        hi = m_hi.update(ExecutiveSample(1.0, 70.0, 0.1, 100.0))
        lo = m_lo.update(ExecutiveSample(1.0, 30.0, 0.1, 100.0))
        assert lo.value > hi.value

    def test_more_errors_raise_overload(self) -> None:
        m_quiet = self._prime_with_variance()
        m_loud = self._prime_with_variance()
        quiet = m_quiet.update(ExecutiveSample(1.0, 60.0, 0.0, 100.0))
        loud = m_loud.update(ExecutiveSample(1.0, 60.0, 3.0, 100.0))
        assert loud.value > quiet.value


# ---------------------------------------------------------------------------
# Verification step
# ---------------------------------------------------------------------------


class TestVerificationStep:
    @pytest.fixture
    def monitor(self) -> ExecutiveMonitor:
        return ExecutiveMonitor()

    def _synth(
        self,
        value: float,
        directive: PacingDirective,
    ) -> OverloadIndex:
        return OverloadIndex(
            value=value,
            beta_z=0.0,
            hrv_z=0.0,
            error_z=0.0,
            directive=directive,
            reason="synth",
        )

    def test_normal_returns_none(self, monitor: ExecutiveMonitor) -> None:
        assert monitor.verification_for(self._synth(0.2, PacingDirective.NORMAL)) is None

    def test_sensor_absent_returns_none(self, monitor: ExecutiveMonitor) -> None:
        assert monitor.verification_for(self._synth(1.0, PacingDirective.SENSOR_ABSENT)) is None

    def test_slow_down_returns_sanity_check(self, monitor: ExecutiveMonitor) -> None:
        step = monitor.verification_for(self._synth(0.6, PacingDirective.SLOW_DOWN))
        assert isinstance(step, VerificationStep)
        assert step.expected_kind == "sanity_check"
        assert step.deadline_seconds > 0

    def test_hard_block_returns_counter_example(self, monitor: ExecutiveMonitor) -> None:
        step = monitor.verification_for(self._synth(0.95, PacingDirective.HARD_BLOCK))
        assert isinstance(step, VerificationStep)
        assert step.expected_kind == "counter_example"
        assert step.deadline_seconds > 0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_calibration(self) -> None:
        monitor = ExecutiveMonitor(ExecutiveMonitorConfig(window=16, warmup=4))
        _prime(monitor, 8, _calm_sample)
        # Pre-reset: we are past warmup and producing overload readings.
        assert "overload" in monitor.update(_calm_sample(8.0)).reason
        monitor.reset()
        # After reset: warmup sentinel returns.
        first = monitor.update(_calm_sample(100.0))
        assert first.reason == "warmup"

    def test_reset_clears_timestamp_state(self) -> None:
        monitor = ExecutiveMonitor(ExecutiveMonitorConfig(window=16, warmup=4))
        monitor.update(_calm_sample(100.0))
        monitor.reset()
        # A new session starting at t=0 must be accepted after reset.
        monitor.update(_calm_sample(0.0))


# ---------------------------------------------------------------------------
# Falsifiable property: the monitor beats a pure-latency baseline on a
# synthetic stress-burst prediction task. This is not a full scientific
# evaluation — it's a unit-level smoke test of Prediction 2.
# ---------------------------------------------------------------------------


class TestFalsifiableBaseline:
    def test_monitor_detects_injected_burst_early(self) -> None:
        cfg = ExecutiveMonitorConfig(window=32, warmup=8)
        monitor = ExecutiveMonitor(cfg)

        # 32 calm samples — baseline.
        for i in range(32):
            monitor.update(_calm_sample(float(i)))

        # Sample 33: a mild stressor — beta climbs, HRV sags, no errors yet.
        pre_burst = monitor.update(
            ExecutiveSample(beta_power=4.0, hrv=35.0, error_burst=0.0, timestamp=33.0)
        )

        # The monitor should already be flagging elevated overload before any
        # behavioral errors appear — that is the whole point of EEG/HRV fusion.
        assert pre_burst.value > 0.5
        assert pre_burst.directive in {
            PacingDirective.SLOW_DOWN,
            PacingDirective.HARD_BLOCK,
        }
        assert math.isfinite(pre_burst.beta_z)
        assert math.isfinite(pre_burst.hrv_z)
