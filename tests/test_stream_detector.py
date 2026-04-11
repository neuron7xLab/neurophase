"""Tests for ``neurophase.data.stream_detector`` (B2 + B6)."""

from __future__ import annotations

import dataclasses

import pytest

from neurophase.data.stream_detector import (
    DEFAULT_HOLD_STEPS,
    DEFAULT_MAX_FAULT_RATE,
    DEFAULT_STREAM_WINDOW,
    StreamQualityDecision,
    StreamRegime,
    TemporalStreamDetector,
)
from neurophase.data.temporal_validator import (
    TemporalQualityDecision,
    TemporalValidator,
    TimeQuality,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decision(quality: TimeQuality) -> TemporalQualityDecision:
    return TemporalQualityDecision(
        quality=quality,
        ts=0.0,
        last_ts=None,
        gap_seconds=None,
        staleness_seconds=None,
        warmup_remaining=0,
        reason=f"{quality.name.lower()}: fixture",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        d = TemporalStreamDetector()
        assert d.window == DEFAULT_STREAM_WINDOW
        assert d.max_fault_rate == DEFAULT_MAX_FAULT_RATE
        assert d.hold_steps == DEFAULT_HOLD_STEPS
        assert d.regime is StreamRegime.WARMUP
        assert d.n_updates == 0
        assert d.window_filled is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"window": 3},
            {"window": 0},
            {"max_fault_rate": 0.0},
            {"max_fault_rate": 1.0},
            {"max_fault_rate": -0.1},
            {"hold_steps": -1},
        ],
    )
    def test_invalid_config_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            TemporalStreamDetector(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    def test_warmup_before_window_full(self) -> None:
        d = TemporalStreamDetector(window=8, hold_steps=0)
        for _ in range(7):
            out = d.update(_decision(TimeQuality.VALID))
            assert out.regime is StreamRegime.WARMUP
            assert out.reason.startswith("warmup:")

    def test_warmup_transitions_to_healthy(self) -> None:
        d = TemporalStreamDetector(window=4, hold_steps=0)
        for _ in range(4):
            out = d.update(_decision(TimeQuality.VALID))
        assert out.regime is StreamRegime.HEALTHY
        assert d.window_filled is True


# ---------------------------------------------------------------------------
# HEALTHY / DEGRADED / OFFLINE classification
# ---------------------------------------------------------------------------


class TestClassification:
    def test_all_valid_is_healthy(self) -> None:
        d = TemporalStreamDetector(window=4, max_fault_rate=0.25, hold_steps=0)
        for _ in range(4):
            out = d.update(_decision(TimeQuality.VALID))
        assert out.regime is StreamRegime.HEALTHY
        assert out.stats.valid == 4
        assert out.stats.fault_rate == 0.0

    def test_above_fault_rate_is_degraded(self) -> None:
        d = TemporalStreamDetector(window=4, max_fault_rate=0.25, hold_steps=0)
        # 3 VALID + 1 GAPPED = 25% fault rate — exactly at the
        # threshold (strict > is used, so this is HEALTHY).
        for q in (
            TimeQuality.VALID,
            TimeQuality.VALID,
            TimeQuality.VALID,
            TimeQuality.GAPPED,
        ):
            out = d.update(_decision(q))
        assert out.regime is StreamRegime.HEALTHY
        # Replacing one more VALID with GAPPED → 50% fault rate.
        out = d.update(_decision(TimeQuality.GAPPED))
        assert out.regime is StreamRegime.DEGRADED
        assert out.reason.startswith("degraded:")

    def test_all_non_valid_is_offline(self) -> None:
        d = TemporalStreamDetector(window=4, max_fault_rate=0.25, hold_steps=0)
        for _ in range(4):
            out = d.update(_decision(TimeQuality.STALE))
        assert out.regime is StreamRegime.OFFLINE
        assert out.stats.valid == 0
        assert out.reason.startswith("offline:")

    def test_stats_track_every_category(self) -> None:
        d = TemporalStreamDetector(window=8, max_fault_rate=0.9, hold_steps=0)
        sequence = [
            TimeQuality.VALID,
            TimeQuality.GAPPED,
            TimeQuality.STALE,
            TimeQuality.REVERSED,
            TimeQuality.DUPLICATE,
            TimeQuality.INVALID,
            TimeQuality.WARMUP,
            TimeQuality.VALID,
        ]
        for q in sequence:
            out = d.update(_decision(q))
        assert out.stats.total == 8
        assert out.stats.valid == 2
        assert out.stats.gapped == 1
        assert out.stats.stale == 1
        assert out.stats.reversed == 1
        assert out.stats.duplicate == 1
        assert out.stats.invalid == 1
        assert out.stats.warmup == 1
        assert out.stats.fault_rate == pytest.approx(6.0 / 8.0)


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_hold_prevents_immediate_flip(self) -> None:
        """Once HEALTHY commits, an immediate degraded-worthy sample is held back."""
        d = TemporalStreamDetector(window=4, max_fault_rate=0.10, hold_steps=3)
        # Warmup and commit HEALTHY with 4 VALID samples.
        for _ in range(4):
            d.update(_decision(TimeQuality.VALID))
        assert d.regime is StreamRegime.HEALTHY

        # Feed a GAPPED sample. Buffer becomes [V,V,V,G]:
        # fault_rate = 0.25 > max_fault_rate = 0.10 → raw=DEGRADED.
        # But hysteresis should hold HEALTHY for the next few updates.
        out_a = d.update(_decision(TimeQuality.GAPPED))
        assert out_a.regime is StreamRegime.HEALTHY  # held
        assert out_a.held is True
        assert "held" in out_a.reason

    def test_warmup_to_steady_is_never_held(self) -> None:
        d = TemporalStreamDetector(window=4, hold_steps=3)
        for _ in range(4):
            out = d.update(_decision(TimeQuality.VALID))
        # The WARMUP → HEALTHY transition must not be held.
        assert out.held is False
        assert out.regime is StreamRegime.HEALTHY

    def test_hold_zero_is_no_op(self) -> None:
        d = TemporalStreamDetector(window=4, hold_steps=0)
        for _ in range(4):
            d.update(_decision(TimeQuality.VALID))
        out = d.update(_decision(TimeQuality.GAPPED))
        assert out.held is False


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_state(self) -> None:
        d = TemporalStreamDetector(window=4, hold_steps=0)
        for _ in range(4):
            d.update(_decision(TimeQuality.VALID))
        pre_reset_regime: StreamRegime = d.regime
        assert pre_reset_regime is StreamRegime.HEALTHY
        d.reset()
        post_reset_regime: StreamRegime = d.regime
        assert post_reset_regime is StreamRegime.WARMUP
        assert d.n_updates == 0
        assert d.window_filled is False


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_stream_decision_frozen(self) -> None:
        d = TemporalStreamDetector(window=4, hold_steps=0)
        out = d.update(_decision(TimeQuality.VALID))
        assert isinstance(out, StreamQualityDecision)
        with pytest.raises(dataclasses.FrozenInstanceError):
            out.regime = StreamRegime.OFFLINE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# End-to-end: B1 TemporalValidator → B2/B6 TemporalStreamDetector
# ---------------------------------------------------------------------------


class TestB1B2Composition:
    def test_healthy_monotonic_stream_stays_healthy(self) -> None:
        validator = TemporalValidator(max_gap_seconds=1.0, warmup_samples=2)
        detector = TemporalStreamDetector(window=8, max_fault_rate=0.25, hold_steps=0)
        for ts in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            tq = validator.validate(ts)
            out = detector.update(tq)
        assert out.regime is StreamRegime.HEALTHY
        assert out.stats.valid >= 4

    def test_burst_of_gaps_moves_regime_to_degraded(self) -> None:
        """A mix of valid + gapped samples with enough gaps to exceed
        the fault_rate threshold — but still enough valids to avoid
        the all-fault OFFLINE branch."""
        validator = TemporalValidator(max_gap_seconds=0.2, warmup_samples=2)
        detector = TemporalStreamDetector(window=6, max_fault_rate=0.25, hold_steps=0)
        # Alternating valid / gap sequence: every other packet is a
        # big jump. After warmup the window is ~50% gapped, ~50% valid.
        timestamps = [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 2.0, 2.1, 3.0]
        out = None
        for ts in timestamps:
            tq = validator.validate(ts)
            out = detector.update(tq)
        assert out is not None
        assert out.regime is StreamRegime.DEGRADED
        # But NOT offline — there are still some valid packets.
        assert out.stats.valid >= 1

    def test_reversed_stream_is_offline_eventually(self) -> None:
        """A stream where every packet is REVERSED eventually becomes OFFLINE."""
        validator = TemporalValidator(warmup_samples=2)
        detector = TemporalStreamDetector(window=4, hold_steps=0)
        # Bootstrap with two good samples so last_ts is set.
        for ts in [10.0, 11.0]:
            detector.update(validator.validate(ts))
        # Now feed 4 reversed samples.
        for ts in [5.0, 4.0, 3.0, 2.0]:
            out = detector.update(validator.validate(ts))
        assert out.regime is StreamRegime.OFFLINE
