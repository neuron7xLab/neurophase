"""Tests for neurophase.physio.gate — fail-closed signal-quality gate."""

from __future__ import annotations

import pytest

from neurophase.gate.execution_gate import GateState
from neurophase.physio.features import (
    DEFAULT_WINDOW_SIZE,
    MIN_WINDOW_SIZE,
    HRVFeatures,
    HRVWindow,
)
from neurophase.physio.gate import PhysioDecision, PhysioGate, PhysioGateState
from neurophase.physio.replay import RRSample


def _features(
    *,
    confidence: float,
    rmssd_ms: float = 35.0,
    window_size: int = DEFAULT_WINDOW_SIZE,
    continuity_fraction: float = 1.0,
    stability: float = 0.95,
    mean_rr_ms: float = 820.0,
    std_rr_ms: float = 40.0,
    rmssd_plausible: bool = True,
) -> HRVFeatures:
    return HRVFeatures(
        rmssd_ms=rmssd_ms,
        mean_rr_ms=mean_rr_ms,
        std_rr_ms=std_rr_ms,
        stability=stability,
        continuity_fraction=continuity_fraction,
        confidence=confidence,
        window_size=window_size,
        rmssd_plausible=rmssd_plausible,
    )


class TestConstruction:
    def test_invalid_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError, match="threshold_abstain"):
            PhysioGate(threshold_allow=0.5, threshold_abstain=0.5)
        with pytest.raises(ValueError, match="threshold_abstain"):
            PhysioGate(threshold_allow=0.3, threshold_abstain=0.8)
        with pytest.raises(ValueError, match="threshold_abstain"):
            PhysioGate(threshold_allow=1.0, threshold_abstain=0.5)


class TestDecisionInvariant:
    def test_execution_allowed_requires_execute_allowed_state(self) -> None:
        with pytest.raises(ValueError, match="Invariant"):
            PhysioDecision(
                state=PhysioGateState.EXECUTE_REDUCED,
                execution_allowed=True,
                confidence=0.6,
                threshold_allow=0.8,
                threshold_abstain=0.5,
                reason="forced",
                kernel_state=GateState.READY,
            )

    def test_execute_allowed_state_requires_execution_allowed(self) -> None:
        with pytest.raises(ValueError, match="Invariant"):
            PhysioDecision(
                state=PhysioGateState.EXECUTE_ALLOWED,
                execution_allowed=False,
                confidence=0.9,
                threshold_allow=0.8,
                threshold_abstain=0.5,
                reason="forced",
                kernel_state=GateState.READY,
            )


class TestGateMapping:
    def test_insufficient_buffer_degrades(self) -> None:
        gate = PhysioGate()
        feats = _features(
            confidence=0.0,
            window_size=MIN_WINDOW_SIZE - 1,
            continuity_fraction=(MIN_WINDOW_SIZE - 1) / DEFAULT_WINDOW_SIZE,
            rmssd_ms=0.0,
            rmssd_plausible=False,
        )
        d = gate.evaluate(feats)
        assert d.state is PhysioGateState.SENSOR_DEGRADED
        assert d.execution_allowed is False

    def test_rmssd_outside_envelope_degrades(self) -> None:
        gate = PhysioGate()
        feats = _features(confidence=0.6, rmssd_ms=400.0, rmssd_plausible=False)
        d = gate.evaluate(feats)
        assert d.state is PhysioGateState.SENSOR_DEGRADED
        assert d.execution_allowed is False

    def test_low_confidence_abstains(self) -> None:
        gate = PhysioGate()
        feats = _features(confidence=0.30)
        d = gate.evaluate(feats)
        assert d.state is PhysioGateState.ABSTAIN
        assert d.execution_allowed is False

    def test_middle_confidence_reduced(self) -> None:
        gate = PhysioGate(threshold_allow=0.8, threshold_abstain=0.5)
        feats = _features(confidence=0.65)
        d = gate.evaluate(feats)
        assert d.state is PhysioGateState.EXECUTE_REDUCED
        assert d.execution_allowed is False

    def test_high_confidence_allows(self) -> None:
        gate = PhysioGate(threshold_allow=0.8, threshold_abstain=0.5)
        feats = _features(confidence=0.90)
        d = gate.evaluate(feats)
        assert d.state is PhysioGateState.EXECUTE_ALLOWED
        assert d.execution_allowed is True

    def test_exact_allow_boundary_is_inclusive(self) -> None:
        gate = PhysioGate(threshold_allow=0.8, threshold_abstain=0.5)
        d = gate.evaluate(_features(confidence=0.80))
        assert d.state is PhysioGateState.EXECUTE_ALLOWED

    def test_exact_abstain_boundary_is_inclusive_for_reduced(self) -> None:
        # ExecutionGate admits R >= threshold. Physio kernel threshold
        # equals threshold_abstain, so confidence == threshold_abstain
        # yields kernel READY + physio EXECUTE_REDUCED.
        gate = PhysioGate(threshold_allow=0.8, threshold_abstain=0.5)
        d = gate.evaluate(_features(confidence=0.50))
        assert d.state is PhysioGateState.EXECUTE_REDUCED

    def test_just_below_abstain_boundary_abstains(self) -> None:
        gate = PhysioGate(threshold_allow=0.8, threshold_abstain=0.5)
        d = gate.evaluate(_features(confidence=0.49999))
        assert d.state is PhysioGateState.ABSTAIN


class TestDeterminism:
    def test_same_features_same_decision(self) -> None:
        gate = PhysioGate()
        feats = _features(confidence=0.7)
        a = gate.evaluate(feats)
        b = gate.evaluate(feats)
        assert a == b

    def test_replay_stream_deterministic(self) -> None:
        """Two independent pipelines fed the same RR stream must agree state-by-state."""
        win1, win2 = HRVWindow(), HRVWindow()
        gate1, gate2 = PhysioGate(), PhysioGate()
        base = 820.0
        t = 0.0
        states1: list[PhysioGateState] = []
        states2: list[PhysioGateState] = []
        for i in range(60):
            rr = base + (10.0 if i % 2 == 0 else -10.0)
            t += rr / 1000.0
            s = RRSample(timestamp_s=t, rr_ms=rr, row_index=i)
            win1.push(s)
            win2.push(s)
            states1.append(gate1.evaluate(win1.features()).state)
            states2.append(gate2.evaluate(win2.features()).state)
        assert states1 == states2


class TestNoSignalNeverExecutes:
    """Critical fail-closed invariant: an empty or near-empty buffer must never
    produce EXECUTE_ALLOWED, and the physio gate must never emit
    execution_allowed=True outside the EXECUTE_ALLOWED state."""

    def test_empty_buffer_no_execute(self) -> None:
        gate = PhysioGate()
        win = HRVWindow()
        d = gate.evaluate(win.features())
        assert d.state is not PhysioGateState.EXECUTE_ALLOWED
        assert d.execution_allowed is False

    def test_warmup_buffer_never_executes(self) -> None:
        gate = PhysioGate()
        win = HRVWindow()
        t = 0.0
        for i in range(MIN_WINDOW_SIZE - 1):
            t += 0.820
            win.push(RRSample(timestamp_s=t, rr_ms=820.0, row_index=i))
            d = gate.evaluate(win.features())
            assert d.state is not PhysioGateState.EXECUTE_ALLOWED
            assert d.execution_allowed is False
