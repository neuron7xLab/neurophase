"""Tests for the interpretability layer ``neurophase.explain``."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from neurophase.explain import (
    Contract,
    ExplanationStep,
    Verdict,
    explain_decision,
    explain_gate,
)
from neurophase.gate.execution_gate import ExecutionGate, GateState
from neurophase.runtime.pipeline import (
    DecisionFrame,
    PipelineConfig,
    StreamingPipeline,
)

# ---------------------------------------------------------------------------
# Helper — drive the real pipeline to a specific final state.
# ---------------------------------------------------------------------------


def _drive_pipeline(
    *,
    R: float | None,
    delta: float | None = 0.01,
    enable_stillness: bool = True,
) -> DecisionFrame:
    """Prime the pipeline with 6 healthy ticks then emit one probe frame."""
    cfg = PipelineConfig(
        warmup_samples=2,
        stream_window=4,
        max_fault_rate=0.50,
        enable_stillness=enable_stillness,
        stillness_window=4,
        stillness_eps_R=1e-3,
        stillness_eps_F=1e-3,
        stillness_delta_min=0.20,
    )
    p = StreamingPipeline(cfg)
    for i in range(6):
        p.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.01)
    return p.tick(timestamp=0.7, R=R, delta=delta)


# ---------------------------------------------------------------------------
# Base contract: every frame produces a valid explanation.
# ---------------------------------------------------------------------------


class TestBasicContract:
    def test_ready_frame_has_all_pass_chain(self) -> None:
        frame = _drive_pipeline(R=0.99, enable_stillness=False)
        assert frame.gate_state is GateState.READY
        exp = explain_decision(frame)
        assert exp.final_state is GateState.READY
        assert exp.execution_allowed is True
        assert exp.causal_contract is Contract.READY
        # Every non-I4-skipped step must be pass.
        for step in exp.chain:
            if step.contract is Contract.I4:
                continue
            assert step.verdict is Verdict.PASS, step

    def test_blocked_frame_has_i1_as_causal_root(self) -> None:
        frame = _drive_pipeline(R=0.30)
        assert frame.gate_state is GateState.BLOCKED
        exp = explain_decision(frame)
        assert exp.final_state is GateState.BLOCKED
        assert exp.causal_contract is Contract.I1
        # The I1 step must be FAIL; later steps must be SKIPPED.
        steps_by_contract = {s.contract: s for s in exp.chain}
        assert steps_by_contract[Contract.I1].verdict is Verdict.FAIL
        assert steps_by_contract[Contract.I4].verdict is Verdict.SKIPPED

    def test_degraded_frame_has_b1_as_causal_root(self) -> None:
        """An R=None probe after warmup lands in DEGRADED via I3
        (R invalid), not B1 — B1 is satisfied because the stream is
        healthy. Verify the chain attributes the failure to I3."""
        frame = _drive_pipeline(R=None)
        assert frame.gate_state is GateState.DEGRADED
        exp = explain_decision(frame)
        assert exp.final_state is GateState.DEGRADED
        assert exp.causal_contract is Contract.I3
        # Earlier steps pass.
        steps_by_contract = {s.contract: s for s in exp.chain}
        assert steps_by_contract[Contract.B1].verdict is Verdict.PASS
        assert steps_by_contract[Contract.I2].verdict is Verdict.PASS
        assert steps_by_contract[Contract.I3].verdict is Verdict.FAIL

    def test_unnecessary_frame_has_i4_as_causal_root(self) -> None:
        # Drive to UNNECESSARY by supplying a perfectly still signal.
        cfg = PipelineConfig(
            warmup_samples=2,
            stream_window=4,
            max_fault_rate=0.50,
            enable_stillness=True,
            stillness_window=4,
            stillness_eps_R=1e-3,
            stillness_eps_F=1e-3,
            stillness_delta_min=0.20,
        )
        p = StreamingPipeline(cfg)
        frame: DecisionFrame | None = None
        for i in range(20):
            frame = p.tick(timestamp=float(i) * 0.1, R=0.95, delta=0.01)
            if frame.gate_state is GateState.UNNECESSARY:
                break
        assert frame is not None
        assert frame.gate_state is GateState.UNNECESSARY
        exp = explain_decision(frame)
        assert exp.causal_contract is Contract.I4
        steps_by_contract = {s.contract: s for s in exp.chain}
        assert steps_by_contract[Contract.I4].verdict is Verdict.FAIL


# ---------------------------------------------------------------------------
# Determinism — same frame → byte-identical explanation.
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_frame_same_explanation(self) -> None:
        frame = _drive_pipeline(R=0.99, enable_stillness=False)
        exp_a = explain_decision(frame)
        exp_b = explain_decision(frame)
        assert exp_a == exp_b
        assert exp_a.to_dict() == exp_b.to_dict()
        assert exp_a.as_text() == exp_b.as_text()

    def test_as_dict_is_json_serializable(self) -> None:
        frame = _drive_pipeline(R=0.30)
        exp = explain_decision(frame)
        payload = exp.to_dict()
        s = json.dumps(payload, sort_keys=True)
        assert "causal_contract" in s
        assert "chain" in s
        reloaded = json.loads(s)
        assert reloaded["final_state"] == "BLOCKED"
        assert reloaded["causal_contract"] == "I1"

    def test_as_dict_is_flat(self) -> None:
        """No nested dataclass instances — every value is JSON-safe."""
        frame = _drive_pipeline(R=0.30)
        exp = explain_decision(frame)
        payload = exp.to_dict()
        for key, value in payload.items():
            assert not dataclasses.is_dataclass(value), (
                f"field {key!r} is a dataclass — flat-contract violated"
            )
        # chain is a list of dicts.
        for step_dict in payload["chain"]:
            assert isinstance(step_dict, dict)
            assert not dataclasses.is_dataclass(step_dict)


# ---------------------------------------------------------------------------
# Text rendering.
# ---------------------------------------------------------------------------


class TestTextRendering:
    def test_as_text_contains_tick_and_final(self) -> None:
        frame = _drive_pipeline(R=0.30)
        exp = explain_decision(frame)
        text = exp.as_text()
        assert f"tick {frame.tick_index}" in text
        assert "final: BLOCKED" in text
        assert "execution_allowed=False" in text

    def test_causal_root_is_marked(self) -> None:
        frame = _drive_pipeline(R=0.30)
        exp = explain_decision(frame)
        text = exp.as_text()
        assert "causal root" in text

    def test_ready_frame_text_has_no_causal_marker(self) -> None:
        frame = _drive_pipeline(R=0.99, enable_stillness=False)
        exp = explain_decision(frame)
        text = exp.as_text()
        assert "causal root" not in text
        assert "final: READY" in text


# ---------------------------------------------------------------------------
# Convenience wrapper — explain_gate on a bare GateDecision.
# ---------------------------------------------------------------------------


class TestExplainGate:
    def test_ready_decision(self) -> None:
        gate = ExecutionGate(threshold=0.65)
        decision = gate.evaluate(R=0.90)
        exp = explain_gate(decision, tick_index=7, timestamp=1.5)
        assert exp.final_state is GateState.READY
        assert exp.tick_index == 7
        assert exp.timestamp == 1.5
        assert exp.execution_allowed is True

    def test_blocked_decision(self) -> None:
        gate = ExecutionGate(threshold=0.65)
        decision = gate.evaluate(R=0.30)
        exp = explain_gate(decision)
        assert exp.final_state is GateState.BLOCKED
        assert exp.causal_contract is Contract.I1

    def test_sensor_absent_decision(self) -> None:
        gate = ExecutionGate(threshold=0.65)
        decision = gate.evaluate(R=0.99, sensor_present=False)
        exp = explain_gate(decision)
        assert exp.final_state is GateState.SENSOR_ABSENT
        assert exp.causal_contract is Contract.I2

    def test_degraded_decision(self) -> None:
        gate = ExecutionGate(threshold=0.65)
        decision = gate.evaluate(R=float("nan"))
        exp = explain_gate(decision)
        assert exp.final_state is GateState.DEGRADED
        assert exp.causal_contract is Contract.I3


# ---------------------------------------------------------------------------
# Frozen invariants.
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_explanation_is_frozen(self) -> None:
        frame = _drive_pipeline(R=0.99, enable_stillness=False)
        exp = explain_decision(frame)
        with pytest.raises(dataclasses.FrozenInstanceError):
            exp.tick_index = -1  # type: ignore[misc]

    def test_step_is_frozen(self) -> None:
        step = ExplanationStep(
            contract=Contract.I1,
            verdict=Verdict.PASS,
            observed="0.9",
            detail="test",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            step.verdict = Verdict.FAIL  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration with the F2 replay engine: explanations must replay byte-
# identically across two independent runs of the same pipeline.
# ---------------------------------------------------------------------------


class TestExplainReplay:
    def test_explanations_are_bit_identical_on_replay(self, tmp_path: Path) -> None:
        """Two pipelines fed the same inputs must produce the same
        sequence of explanations. This is the interpretability
        complement to F3 determinism."""

        def run() -> list[dict[str, object]]:
            cfg = PipelineConfig(warmup_samples=2, stream_window=4, enable_stillness=False)
            p = StreamingPipeline(cfg)
            out: list[dict[str, object]] = []
            for i, R in enumerate([0.9, 0.9, 0.9, 0.9, 0.9, 0.3, 0.9, 0.9]):
                frame = p.tick(timestamp=float(i) * 0.1, R=R, delta=0.01)
                out.append(explain_decision(frame).to_dict())
            return out

        a = run()
        b = run()
        assert a == b
