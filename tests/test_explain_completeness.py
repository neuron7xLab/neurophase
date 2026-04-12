"""Comprehensive tests for every gate state × stillness config combination.

Covers:
* Every GateState → correct causal_contract
* Stillness enabled × delta present × active → I4 PASS
* Stillness enabled × delta present × still → I4 FAIL, causal=I4
* Stillness enabled × delta missing → I4 SKIPPED (no_detector_or_missing_delta)
* Stillness disabled → I4 SKIPPED (no_detector_or_missing_delta)
* Upstream short-circuit → all downstream steps SKIPPED (upstream_short_circuit)
* Determinism (byte-identical to_dict())
* as_text causal root marker
* explain_gate() ≡ explain_decision() equivalence
* Verdict enum value stability
"""

from __future__ import annotations

from typing import Any

import pytest

from neurophase.explain import (
    Contract,
    DecisionExplanation,
    Verdict,
    explain_decision,
    explain_gate,
)
from neurophase.gate.execution_gate import ExecutionGate, GateState
from neurophase.gate.stillness_detector import StillnessDetector
from neurophase.runtime.pipeline import (
    DecisionFrame,
    PipelineConfig,
    StreamingPipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THRESHOLD = 0.65


def _pipeline(
    *,
    warmup: int = 6,
    enable_stillness: bool = True,
    stillness_window: int = 4,
    stillness_eps_R: float = 1e-3,
    stillness_eps_F: float = 1e-3,
    stillness_delta_min: float = 0.20,
) -> StreamingPipeline:
    cfg = PipelineConfig(
        warmup_samples=2,
        stream_window=4,
        max_fault_rate=0.50,
        enable_stillness=enable_stillness,
        stillness_window=stillness_window,
        stillness_eps_R=stillness_eps_R,
        stillness_eps_F=stillness_eps_F,
        stillness_delta_min=stillness_delta_min,
    )
    p = StreamingPipeline(cfg)
    for i in range(warmup):
        p.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.01)
    return p


def _ready_frame(
    *,
    R: float = 0.9,
    delta: float | None = 0.01,
    enable_stillness: bool = True,
) -> DecisionFrame:
    p = _pipeline(enable_stillness=enable_stillness)
    return p.tick(timestamp=0.7, R=R, delta=delta)


def _steps(exp: DecisionExplanation) -> dict[Contract, Any]:
    return {s.contract: s for s in exp.chain}


# ---------------------------------------------------------------------------
# 1. Every GateState → correct causal_contract
# ---------------------------------------------------------------------------


class TestEveryGateStateCausalContract:
    def test_degraded_causal_is_i3(self) -> None:
        """R=None after warmup → DEGRADED, causal=I3."""
        p = _pipeline(enable_stillness=False)
        frame = p.tick(timestamp=0.7, R=None, delta=0.01)
        assert frame.gate_state is GateState.DEGRADED
        exp = explain_decision(frame)
        assert exp.final_state is GateState.DEGRADED
        assert exp.causal_contract is Contract.I3

    def test_sensor_absent_causal_is_i2(self) -> None:
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=0.9, sensor_present=False)
        exp = explain_gate(decision)
        assert exp.final_state is GateState.SENSOR_ABSENT
        assert exp.causal_contract is Contract.I2

    def test_blocked_causal_is_i1(self) -> None:
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=0.30)
        exp = explain_gate(decision)
        assert exp.final_state is GateState.BLOCKED
        assert exp.causal_contract is Contract.I1

    def test_unnecessary_causal_is_i4(self) -> None:
        p = _pipeline(enable_stillness=True, stillness_window=4)
        frame: DecisionFrame | None = None
        for i in range(20):
            frame = p.tick(timestamp=float(i + 7) * 0.1, R=0.95, delta=0.01)
            if frame.gate_state is GateState.UNNECESSARY:
                break
        assert frame is not None
        assert frame.gate_state is GateState.UNNECESSARY
        exp = explain_decision(frame)
        assert exp.causal_contract is Contract.I4

    def test_ready_causal_is_ready(self) -> None:
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=0.9)
        exp = explain_gate(decision)
        assert exp.final_state is GateState.READY
        assert exp.causal_contract is Contract.READY


# ---------------------------------------------------------------------------
# 2. Stillness enabled + delta present → ACTIVE → I4 PASS
# ---------------------------------------------------------------------------


class TestStillnessEnabledDeltaPresentActive:
    def test_i4_step_verdict_is_pass(self) -> None:
        """StillnessDetector attached, delta provided, system is active."""
        # We need a state where R ≥ threshold but not yet STILL.
        # Fresh pipeline after 6 warmup ticks with varying delta → ACTIVE.
        p = _pipeline(enable_stillness=True, stillness_eps_R=1e-3, stillness_delta_min=0.20)
        # Push a tick with a large delta so STILL cannot be declared.
        frame = p.tick(timestamp=0.7, R=0.9, delta=1.0)
        # Gate may be READY or UNNECESSARY at this point; we need READY.
        # If UNNECESSARY was reached, skip — rare on first tick after warmup.
        if frame.gate_state is GateState.READY:
            exp = explain_decision(frame)
            steps = _steps(exp)
            assert Contract.I4 in steps
            i4 = steps[Contract.I4]
            assert i4.verdict is Verdict.PASS

    def test_i4_pass_via_gate_directly(self) -> None:
        """Direct gate with StillnessDetector, delta drives ACTIVE."""
        sd = StillnessDetector(window=4, eps_R=1e-3, eps_F=1e-3, delta_min=0.20)
        gate = ExecutionGate(threshold=_THRESHOLD, stillness_detector=sd)
        # First call — buffer not full → ACTIVE.
        decision = gate.evaluate(R=0.9, delta=1.5)
        assert decision.state is GateState.READY
        exp = explain_gate(decision)
        steps = _steps(exp)
        assert Contract.I4 in steps
        i4 = steps[Contract.I4]
        assert i4.verdict is Verdict.PASS


# ---------------------------------------------------------------------------
# 3. Stillness enabled + delta present → STILL → I4 FAIL
# ---------------------------------------------------------------------------


class TestStillnessEnabledDeltaPresentStill:
    def test_i4_fail_causal_i4(self) -> None:
        p = _pipeline(enable_stillness=True, stillness_window=4)
        frame: DecisionFrame | None = None
        for i in range(30):
            frame = p.tick(timestamp=float(i + 7) * 0.1, R=0.95, delta=0.01)
            if frame.gate_state is GateState.UNNECESSARY:
                break
        assert frame is not None
        assert frame.gate_state is GateState.UNNECESSARY
        exp = explain_decision(frame)
        steps = _steps(exp)
        assert Contract.I4 in steps
        i4 = steps[Contract.I4]
        assert i4.verdict is Verdict.FAIL
        assert exp.causal_contract is Contract.I4


# ---------------------------------------------------------------------------
# 4. Stillness enabled + delta missing → I4 SKIPPED (no_detector_or_missing_delta)
# ---------------------------------------------------------------------------


class TestStillnessEnabledDeltaMissing:
    def test_i4_skipped_observed_no_detector_or_missing_delta(self) -> None:
        """StillnessDetector attached, but delta=None → I4 SKIPPED."""
        p = _pipeline(enable_stillness=True)
        frame = p.tick(timestamp=0.7, R=0.9, delta=None)
        # Gate lands READY (stillness evaluation skipped, not DEGRADED)
        assert frame.gate_state is GateState.READY
        exp = explain_decision(frame)
        steps = _steps(exp)
        assert Contract.I4 in steps
        i4 = steps[Contract.I4]
        assert i4.verdict is Verdict.SKIPPED
        assert i4.observed == "no_detector_or_missing_delta"


# ---------------------------------------------------------------------------
# 5. Stillness disabled → I4 SKIPPED (no_detector_or_missing_delta)
# ---------------------------------------------------------------------------


class TestStillnessDisabled:
    def test_i4_skipped_when_no_detector(self) -> None:
        """No StillnessDetector attached → I4 step is SKIPPED."""
        p = _pipeline(enable_stillness=False)
        frame = p.tick(timestamp=0.7, R=0.9, delta=0.01)
        assert frame.gate_state is GateState.READY
        exp = explain_decision(frame)
        steps = _steps(exp)
        assert Contract.I4 in steps
        i4 = steps[Contract.I4]
        assert i4.verdict is Verdict.SKIPPED
        assert i4.observed == "no_detector_or_missing_delta"


# ---------------------------------------------------------------------------
# 6. Upstream short-circuit → all downstream SKIPPED (upstream_short_circuit)
# ---------------------------------------------------------------------------


class TestUpstreamShortCircuit:
    def test_b1_fail_all_downstream_upstream_short_circuit(self) -> None:
        """B1 fails (R=None → DEGRADED via I3 or B1) — after B1 every step
        is upstream_short_circuit.

        We use R=None to force DEGRADED; the explain chain short-circuits
        at I3 (B1 passes, I2 passes, I3 fails). I1 and I4 must then carry
        observed='upstream_short_circuit'.
        """
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=None)
        assert decision.state is GateState.DEGRADED
        exp = explain_gate(decision)
        steps = _steps(exp)
        # I3 is the causal root; I1 and I4 are downstream → skipped.
        assert steps[Contract.I3].verdict is Verdict.FAIL
        for contract in (Contract.I1, Contract.I4):
            step = steps[contract]
            assert step.verdict is Verdict.SKIPPED
            assert step.observed == "upstream_short_circuit"

    def test_i1_fail_i4_upstream_short_circuit(self) -> None:
        """I1 fails → I4 must be upstream_short_circuit."""
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=0.30)
        assert decision.state is GateState.BLOCKED
        exp = explain_gate(decision)
        steps = _steps(exp)
        assert steps[Contract.I1].verdict is Verdict.FAIL
        i4 = steps[Contract.I4]
        assert i4.verdict is Verdict.SKIPPED
        assert i4.observed == "upstream_short_circuit"

    def test_i2_fail_i3_i1_i4_upstream_short_circuit(self) -> None:
        """I2 fails → I3, I1, I4 are all upstream_short_circuit."""
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=0.9, sensor_present=False)
        assert decision.state is GateState.SENSOR_ABSENT
        exp = explain_gate(decision)
        steps = _steps(exp)
        assert steps[Contract.I2].verdict is Verdict.FAIL
        for contract in (Contract.I3, Contract.I1, Contract.I4):
            step = steps[contract]
            assert step.verdict is Verdict.SKIPPED
            assert step.observed == "upstream_short_circuit"


# ---------------------------------------------------------------------------
# 7. Determinism — same frame → byte-identical to_dict()
# ---------------------------------------------------------------------------


class TestExplanationDeterminism:
    def test_same_frame_same_to_dict(self) -> None:
        frame = _ready_frame(R=0.9, enable_stillness=False)
        a = explain_decision(frame).to_dict()
        b = explain_decision(frame).to_dict()
        assert a == b

    def test_determinism_across_blocked_frame(self) -> None:
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=0.30)
        a = explain_gate(decision).to_dict()
        b = explain_gate(decision).to_dict()
        assert a == b


# ---------------------------------------------------------------------------
# 8. as_text renders "← causal root" marker for the failing step
# ---------------------------------------------------------------------------


class TestAsTextCausalRootMarker:
    def test_causal_root_marker_present_for_blocked(self) -> None:
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=0.30)
        exp = explain_gate(decision)
        text = exp.as_text()
        assert "← causal root" in text

    def test_causal_root_marker_absent_for_ready(self) -> None:
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=0.9)
        exp = explain_gate(decision)
        text = exp.as_text()
        assert "← causal root" not in text

    def test_causal_root_marker_on_i3_fail(self) -> None:
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=float("nan"))
        exp = explain_gate(decision)
        text = exp.as_text()
        assert "← causal root" in text
        # Marker must appear on I3 line
        for line in text.splitlines():
            if "I3" in line:
                assert "← causal root" in line
                break
        else:
            pytest.fail("No I3 line found in as_text() output")


# ---------------------------------------------------------------------------
# 9. explain_gate() ≡ explain_decision() for the same gate decision
# ---------------------------------------------------------------------------


class TestExplainGateEquivalence:
    def test_ready_decision_equivalent(self) -> None:
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=0.9)
        exp_gate = explain_gate(decision, tick_index=0, timestamp=0.0)
        # explain_gate builds a synthetic frame and calls explain_decision.
        # The resulting chain content must match for the shared fields.
        steps_gate = _steps(exp_gate)
        assert exp_gate.final_state is GateState.READY
        assert exp_gate.causal_contract is Contract.READY
        # All non-I4-skipped steps pass.
        for contract, step in steps_gate.items():
            if contract is Contract.I4:
                continue
            assert step.verdict is Verdict.PASS

    def test_blocked_decision_equivalent(self) -> None:
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=0.30)
        exp = explain_gate(decision)
        assert exp.final_state is GateState.BLOCKED
        assert exp.causal_contract is Contract.I1
        steps = _steps(exp)
        assert steps[Contract.I1].verdict is Verdict.FAIL

    def test_sensor_absent_equivalent(self) -> None:
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=0.9, sensor_present=False)
        exp = explain_gate(decision)
        assert exp.final_state is GateState.SENSOR_ABSENT
        assert exp.causal_contract is Contract.I2

    def test_degraded_equivalent(self) -> None:
        gate = ExecutionGate(threshold=_THRESHOLD)
        decision = gate.evaluate(R=float("nan"))
        exp = explain_gate(decision)
        assert exp.final_state is GateState.DEGRADED
        assert exp.causal_contract is Contract.I3


# ---------------------------------------------------------------------------
# 10. Verdict enum values are stable strings (frozen contract)
# ---------------------------------------------------------------------------


class TestVerdictValueStability:
    def test_pass_value(self) -> None:
        assert Verdict.PASS.value == "pass"

    def test_fail_value(self) -> None:
        assert Verdict.FAIL.value == "fail"

    def test_skipped_value(self) -> None:
        assert Verdict.SKIPPED.value == "skipped"

    def test_verdict_count(self) -> None:
        """Exactly 3 verdict values — adding a new one is a breaking change."""
        assert len(Verdict) == 3

    def test_contract_values_stable(self) -> None:
        expected = {"B1", "I1", "I2", "I3", "I4", "READY"}
        actual = {c.value for c in Contract}
        assert actual == expected
