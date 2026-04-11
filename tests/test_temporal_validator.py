"""Tests for ``neurophase.data.temporal_validator`` (B1).

Covers every clause of the temporal integrity contract:

* monotonicity (forward / reversed / duplicate)
* gap bound
* staleness bound
* finiteness
* warmup
* reset
* frozen-dataclass invariant
* reason-string first-token contract
* defensive construction
* integration-level contract: same input sequence → same decision sequence

Plus the integration test: ``ExecutionGate`` returns ``DEGRADED`` with a
``temporal:…`` reason tag whenever the caller supplies a non-VALID
``TemporalQualityDecision``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from neurophase.data.temporal_validator import (
    DEFAULT_MAX_GAP_SECONDS,
    DEFAULT_MAX_STALENESS_SECONDS,
    DEFAULT_WARMUP_SAMPLES,
    TemporalError,
    TemporalQualityDecision,
    TemporalValidator,
    TimeQuality,
)
from neurophase.gate.execution_gate import ExecutionGate, GateState
from neurophase.gate.stillness_detector import StillnessDetector

# ---------------------------------------------------------------------------
# Construction and defaults
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_defaults(self) -> None:
        v = TemporalValidator()
        assert v.max_gap_seconds == DEFAULT_MAX_GAP_SECONDS
        assert v.max_staleness_seconds == DEFAULT_MAX_STALENESS_SECONDS
        assert v.warmup_samples == DEFAULT_WARMUP_SAMPLES
        assert v.n_seen == 0
        assert v.last_ts is None
        assert v.is_warm is False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_gap_seconds": 0.0},
            {"max_gap_seconds": -1.0},
            {"max_gap_seconds": math.inf},
            {"max_gap_seconds": math.nan},
            {"max_staleness_seconds": 0.0},
            {"max_staleness_seconds": -1.0},
            {"warmup_samples": 1},
            {"warmup_samples": 0},
            {"warmup_samples": -5},
        ],
    )
    def test_rejects_bad_config(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(TemporalError):
            TemporalValidator(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Finiteness — INVALID
# ---------------------------------------------------------------------------


class TestFiniteness:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_first_sample_is_invalid(self, bad: float) -> None:
        v = TemporalValidator()
        d = v.validate(bad)
        assert d.quality is TimeQuality.INVALID
        assert d.reason.startswith("invalid:")
        # Non-finite never enters history.
        assert v.n_seen == 0
        assert v.last_ts is None

    def test_nonfinite_mid_stream_does_not_corrupt_last_ts(self) -> None:
        v = TemporalValidator(warmup_samples=2)
        v.validate(0.0)
        v.validate(1.0)  # now VALID
        bad = v.validate(float("nan"))
        assert bad.quality is TimeQuality.INVALID
        # last_ts must still be the last legal sample.
        assert v.last_ts == 1.0
        assert v.n_seen == 2


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    def test_first_sample_is_warmup(self) -> None:
        v = TemporalValidator(warmup_samples=3)
        d = v.validate(0.0)
        assert d.quality is TimeQuality.WARMUP
        assert d.reason.startswith("warmup:")
        assert d.warmup_remaining == 2
        assert d.last_ts is None
        assert d.gap_seconds is None

    def test_warmup_completes_exactly_at_warmup_samples(self) -> None:
        v = TemporalValidator(warmup_samples=3)
        v.validate(0.0)
        v.validate(0.5)
        third = v.validate(1.0)
        assert third.quality is TimeQuality.VALID
        assert v.is_warm is True
        assert third.reason.startswith("valid:")


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


class TestMonotonicity:
    def test_reversed_timestamp(self) -> None:
        v = TemporalValidator(warmup_samples=2)
        v.validate(1.0)
        v.validate(2.0)  # VALID
        d = v.validate(1.5)
        assert d.quality is TimeQuality.REVERSED
        assert d.reason.startswith("reversed:")
        assert d.gap_seconds is not None and d.gap_seconds < 0
        # History must NOT advance on a reversed sample.
        assert v.last_ts == 2.0

    def test_duplicate_timestamp(self) -> None:
        v = TemporalValidator(warmup_samples=2)
        v.validate(1.0)
        v.validate(2.0)
        d = v.validate(2.0)
        assert d.quality is TimeQuality.DUPLICATE
        assert d.reason.startswith("duplicate:")
        assert d.gap_seconds == 0.0
        assert v.last_ts == 2.0  # unchanged

    def test_duplicate_is_distinct_from_reversed(self) -> None:
        v = TemporalValidator(warmup_samples=2)
        v.validate(1.0)
        v.validate(2.0)
        dup = v.validate(2.0)
        rev = v.validate(1.9)
        assert dup.quality is TimeQuality.DUPLICATE
        assert rev.quality is TimeQuality.REVERSED


# ---------------------------------------------------------------------------
# Gap bound
# ---------------------------------------------------------------------------


class TestGapBound:
    def test_gap_within_tolerance_is_valid(self) -> None:
        v = TemporalValidator(max_gap_seconds=1.0, warmup_samples=2)
        v.validate(0.0)
        d = v.validate(0.9)
        assert d.quality is TimeQuality.VALID

    def test_gap_at_tolerance_is_valid(self) -> None:
        """The spec uses ``>``, not ``≥``, so ``Δ == max_gap`` is allowed."""
        v = TemporalValidator(max_gap_seconds=1.0, warmup_samples=2)
        v.validate(0.0)
        d = v.validate(1.0)
        assert d.quality is TimeQuality.VALID

    def test_gap_above_tolerance_is_gapped(self) -> None:
        v = TemporalValidator(max_gap_seconds=1.0, warmup_samples=2)
        v.validate(0.0)
        d = v.validate(1.5)
        assert d.quality is TimeQuality.GAPPED
        assert d.reason.startswith("gapped:")
        assert d.gap_seconds == pytest.approx(1.5)
        # Gapped samples still commit so the validator can recover.
        assert v.last_ts == 1.5

    def test_recovery_from_gap(self) -> None:
        v = TemporalValidator(max_gap_seconds=1.0, warmup_samples=2)
        v.validate(0.0)
        v.validate(5.0)  # GAPPED but committed
        d = v.validate(5.5)
        assert d.quality is TimeQuality.VALID


# ---------------------------------------------------------------------------
# Staleness bound
# ---------------------------------------------------------------------------


class TestStalenessBound:
    def test_staleness_skipped_when_no_reference(self) -> None:
        v = TemporalValidator(max_staleness_seconds=0.5, warmup_samples=2)
        v.validate(0.0)
        d = v.validate(1.0)
        assert d.quality is TimeQuality.VALID
        assert d.staleness_seconds is None

    def test_fresh_sample_is_valid(self) -> None:
        v = TemporalValidator(max_staleness_seconds=1.0, warmup_samples=2)
        v.validate(0.0, reference_now=0.1)
        d = v.validate(1.0, reference_now=1.1)
        assert d.quality is TimeQuality.VALID
        assert d.staleness_seconds == pytest.approx(0.1)

    def test_stale_sample(self) -> None:
        v = TemporalValidator(max_staleness_seconds=0.2, warmup_samples=2)
        v.validate(0.0, reference_now=0.1)
        d = v.validate(1.0, reference_now=2.0)
        assert d.quality is TimeQuality.STALE
        assert d.reason.startswith("stale:")
        assert d.staleness_seconds == pytest.approx(1.0)
        # Stale samples still commit.
        assert v.last_ts == 1.0

    def test_infinite_staleness_bound_disables_check(self) -> None:
        v = TemporalValidator(max_staleness_seconds=math.inf, warmup_samples=2)
        v.validate(0.0, reference_now=0.0)
        d = v.validate(1.0, reference_now=1e9)
        assert d.quality is TimeQuality.VALID


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_history(self) -> None:
        v = TemporalValidator(warmup_samples=2)
        v.validate(0.0)
        v.validate(1.0)
        assert v.is_warm is True
        v.reset()
        assert v.is_warm is False
        assert v.n_seen == 0
        assert v.last_ts is None
        # After reset, a sample that would have been REVERSED against the
        # old history is now perfectly fine as a first sample.
        d = v.validate(0.5)
        assert d.quality is TimeQuality.WARMUP


# ---------------------------------------------------------------------------
# Determinism / replay contract
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_decision_sequence(self) -> None:
        sequence = [0.0, 0.5, 1.0, 1.2, 1.25, 1.0, 5.0, 5.3]

        def run() -> list[TimeQuality]:
            v = TemporalValidator(max_gap_seconds=1.0, warmup_samples=2)
            return [v.validate(t).quality for t in sequence]

        a = run()
        b = run()
        assert a == b


# ---------------------------------------------------------------------------
# Reason-string contract
# ---------------------------------------------------------------------------


class TestReasonContract:
    @pytest.mark.parametrize(
        "scenario,expected_prefix",
        [
            ("warmup", "warmup:"),
            ("valid", "valid:"),
            ("gapped", "gapped:"),
            ("reversed", "reversed:"),
            ("duplicate", "duplicate:"),
            ("invalid", "invalid:"),
        ],
    )
    def test_first_token_tag(self, scenario: str, expected_prefix: str) -> None:
        v = TemporalValidator(max_gap_seconds=1.0, warmup_samples=2)
        match scenario:
            case "warmup":
                d = v.validate(0.0)
            case "valid":
                v.validate(0.0)
                d = v.validate(0.5)
            case "gapped":
                v.validate(0.0)
                d = v.validate(5.0)
            case "reversed":
                v.validate(0.0)
                v.validate(1.0)
                d = v.validate(0.5)
            case "duplicate":
                v.validate(0.0)
                v.validate(1.0)
                d = v.validate(1.0)
            case "invalid":
                d = v.validate(float("nan"))
        assert d.reason.startswith(expected_prefix)


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------


class TestFrozenDecision:
    def test_decision_is_frozen(self) -> None:
        d = TemporalQualityDecision(
            quality=TimeQuality.VALID,
            ts=0.0,
            last_ts=None,
            gap_seconds=None,
            staleness_seconds=None,
            warmup_remaining=0,
            reason="valid: test",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.quality = TimeQuality.GAPPED  # type: ignore[misc]

    def test_is_valid_property(self) -> None:
        d_valid = TemporalQualityDecision(
            quality=TimeQuality.VALID,
            ts=0.0,
            last_ts=None,
            gap_seconds=None,
            staleness_seconds=None,
            warmup_remaining=0,
            reason="valid: test",
        )
        assert d_valid.is_valid is True
        d_gapped = TemporalQualityDecision(
            quality=TimeQuality.GAPPED,
            ts=1.0,
            last_ts=0.0,
            gap_seconds=1.0,
            staleness_seconds=None,
            warmup_remaining=0,
            reason="gapped: test",
        )
        assert d_gapped.is_valid is False


# ---------------------------------------------------------------------------
# ExecutionGate integration — the core contract of B1
# ---------------------------------------------------------------------------


def _valid_decision() -> TemporalQualityDecision:
    return TemporalQualityDecision(
        quality=TimeQuality.VALID,
        ts=1.0,
        last_ts=0.0,
        gap_seconds=1.0,
        staleness_seconds=0.0,
        warmup_remaining=0,
        reason="valid: test",
    )


def _non_valid_decision(q: TimeQuality) -> TemporalQualityDecision:
    return TemporalQualityDecision(
        quality=q,
        ts=1.0,
        last_ts=0.0,
        gap_seconds=1.0,
        staleness_seconds=0.0,
        warmup_remaining=1 if q is TimeQuality.WARMUP else 0,
        reason=f"{q.name.lower()}: test",
    )


class TestGateIntegration:
    def test_gate_unchanged_when_time_quality_not_supplied(self) -> None:
        gate = ExecutionGate(threshold=0.65)
        d = gate.evaluate(R=0.80)
        assert d.state is GateState.READY
        assert d.execution_allowed is True

    def test_gate_ready_with_valid_time_quality(self) -> None:
        gate = ExecutionGate(threshold=0.65)
        d = gate.evaluate(R=0.80, time_quality=_valid_decision())
        assert d.state is GateState.READY
        assert d.execution_allowed is True

    @pytest.mark.parametrize(
        "quality",
        [
            TimeQuality.GAPPED,
            TimeQuality.STALE,
            TimeQuality.REVERSED,
            TimeQuality.DUPLICATE,
            TimeQuality.WARMUP,
            TimeQuality.INVALID,
        ],
    )
    def test_gate_degraded_on_non_valid_time_quality(self, quality: TimeQuality) -> None:
        gate = ExecutionGate(threshold=0.65)
        d = gate.evaluate(R=0.99, time_quality=_non_valid_decision(quality))
        assert d.state is GateState.DEGRADED
        assert d.execution_allowed is False
        assert d.reason.startswith("temporal: ")

    def test_temporal_check_overrides_ready(self) -> None:
        """Temporal precondition takes priority over every downstream invariant."""
        gate = ExecutionGate(
            threshold=0.65,
            stillness_detector=StillnessDetector(window=3),
        )
        d = gate.evaluate(R=0.99, delta=0.01, time_quality=_non_valid_decision(TimeQuality.GAPPED))
        assert d.state is GateState.DEGRADED

    def test_temporal_check_overrides_sensor_absent(self) -> None:
        """Even sensor_absent loses to temporal — both are non-permissive,
        but temporal fires first because a non-VALID stream means we
        cannot even trust that sensor_present makes sense."""
        gate = ExecutionGate(threshold=0.65)
        d = gate.evaluate(
            R=0.99,
            sensor_present=False,
            time_quality=_non_valid_decision(TimeQuality.REVERSED),
        )
        assert d.state is GateState.DEGRADED

    def test_end_to_end_pipeline_with_validator(self) -> None:
        """Full pipeline: TemporalValidator → ExecutionGate.

        Drive the validator with a synthetic stream and confirm that
        valid ticks produce READY decisions while bad ticks produce
        DEGRADED with temporal reason tags.
        """
        gate = ExecutionGate(threshold=0.65)
        validator = TemporalValidator(max_gap_seconds=1.0, warmup_samples=2)

        sequence: list[tuple[float, str]] = [
            (0.0, "warmup"),  # first sample
            (0.5, "valid"),  # warmup complete, VALID
            (1.0, "valid"),
            (5.0, "gapped"),  # big jump
            (5.2, "valid"),
            (4.0, "reversed"),
            (5.2, "duplicate"),
        ]

        seen: list[GateState] = []
        for ts, _label in sequence:
            tq = validator.validate(ts)
            decision = gate.evaluate(R=0.90, time_quality=tq)
            seen.append(decision.state)

        # Warmup + 3 bad → 4 DEGRADED; 3 VALID → 3 READY. But warmup
        # counts as DEGRADED at the gate level.
        assert seen.count(GateState.READY) == 3
        assert seen.count(GateState.DEGRADED) == 4
