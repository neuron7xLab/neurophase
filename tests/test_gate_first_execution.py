"""Gate-first execution law.

Pins the contract: **no downstream action may be derived from anything
other than a gate-approved ``OrchestratedFrame``**. Violations are
either architectural (a helper that bypasses the gate) or behavioural
(the runtime allows a ``DEGRADED`` / ``BLOCKED`` / ``SENSOR_ABSENT`` /
``UNNECESSARY`` frame to produce execution). This suite forbids both.
"""

from __future__ import annotations

import math

import pytest

from neurophase.api import (
    GateState,
    OrchestratorConfig,
    PipelineConfig,
    PolicyConfig,
    RuntimeOrchestrator,
)
from neurophase.bridges import (
    ClockSync,
    DownstreamAdapter,
    DownstreamAdapterError,
    EegIngress,
    FrameMux,
    MarketIngress,
    MarketTick,
    NeuralSample,
)
from neurophase.policy.action import ActionIntent

# ---------------------------------------------------------------------------
# Behavioural: non-READY ticks never carry execution_allowed=True
# ---------------------------------------------------------------------------


def _orch() -> RuntimeOrchestrator:
    return RuntimeOrchestrator(
        OrchestratorConfig(
            pipeline=PipelineConfig(warmup_samples=2, stream_window=4, enable_stillness=False),
            policy=PolicyConfig(),
        )
    )


def test_none_R_routes_to_degraded_not_ready() -> None:
    orch = _orch()
    frame = orch.tick(timestamp=0.0, R=None, delta=0.01)
    assert frame.pipeline_frame.gate.state != GateState.READY
    assert frame.execution_allowed is False


def test_low_R_never_allows_execution() -> None:
    orch = _orch()
    frame = None
    for i in range(20):
        frame = orch.tick(timestamp=float(i) * 0.01, R=0.1, delta=0.01)
    assert frame is not None
    assert frame.execution_allowed is False
    assert frame.pipeline_frame.gate.state in {GateState.DEGRADED, GateState.BLOCKED}


def test_invalid_R_fails_closed() -> None:
    """NaN R must fail-closed: either the orchestrator rejects the tick
    (raises), or the gate routes it through a non-READY state. Silent
    promotion to ``execution_allowed=True`` is forbidden.

    Note: the current stack raises at the regime layer — that is still
    fail-closed (the exception is never caught and ``execution_allowed``
    is never set to True). A future hardening pass may move rejection
    earlier (temporal / gate) but must never make this test pass by
    returning an approved frame.
    """
    orch = _orch()
    try:
        frame = orch.tick(timestamp=0.0, R=math.nan, delta=0.01)
    except (ValueError, TypeError):
        return  # orchestrator refused — acceptable fail-closed outcome
    assert frame.execution_allowed is False, (
        "NaN R produced an execution-allowed frame; silent promotion is forbidden"
    )


def test_warmup_never_emits_execution_allowed() -> None:
    orch = _orch()
    frame = orch.tick(timestamp=0.0, R=0.99, delta=0.001)
    assert frame.execution_allowed is False
    # Second tick still inside stream-warmup window
    frame = orch.tick(timestamp=0.01, R=0.99, delta=0.001)
    assert frame.execution_allowed is False


# ---------------------------------------------------------------------------
# Behavioural: ActionPolicy never proposes anything other than HOLD on a
# refused frame.
# ---------------------------------------------------------------------------


def test_policy_emits_hold_when_gate_refuses() -> None:
    orch = _orch()
    for i in range(3):
        frame = orch.tick(timestamp=float(i) * 0.01, R=0.1, delta=0.01)
        if frame.action is not None:
            assert frame.action.intent == ActionIntent.HOLD, (
                f"policy emitted {frame.action.intent.name} while gate refused"
            )


# ---------------------------------------------------------------------------
# Architectural: the downstream adapter is the only egress and it is
# gate-first by construction.
# ---------------------------------------------------------------------------


def _ready_frame() -> object:
    orch = _orch()
    frame = None
    for i in range(30):
        frame = orch.tick(timestamp=float(i) * 0.01, R=0.9, delta=0.01)
        if frame.execution_allowed:
            return frame
    raise AssertionError("fixture failed to reach READY")


def test_downstream_dispatches_only_ready_frames() -> None:
    sent: list[object] = []

    def transport(payload: dict) -> str:  # type: ignore[type-arg]
        sent.append(payload)
        return "ack"

    adapter = DownstreamAdapter(transport)
    frame = _ready_frame()
    adapter.dispatch(frame)  # type: ignore[arg-type]
    assert len(sent) == 1


def test_downstream_refuses_every_non_ready_state() -> None:
    sent: list[object] = []

    def transport(payload: dict) -> str:  # type: ignore[type-arg]
        sent.append(payload)
        return "ack"

    orch = _orch()
    adapter = DownstreamAdapter(transport)
    for i in range(5):
        frame = orch.tick(timestamp=float(i) * 0.01, R=0.5, delta=0.01)
        if not frame.execution_allowed:
            with pytest.raises(DownstreamAdapterError, match="gate refused"):
                adapter.dispatch(frame)
    assert sent == [], "non-READY frames must not be dispatched"


# ---------------------------------------------------------------------------
# Architectural: ingress failures never bypass the gate.
# ---------------------------------------------------------------------------


def test_frame_mux_failure_never_produces_canonical_tick() -> None:
    """A bridge failure must abort the cycle — the kernel must never be
    fed a half-validated input that might later produce an allowed
    frame."""

    def bad_market() -> MarketTick:
        raise RuntimeError("feed crashed")

    def good_neural() -> NeuralSample:
        return NeuralSample(timestamp=1.0, phase=0.0, source_id="Fz")

    mux = FrameMux(
        eeg=EegIngress(good_neural, source_id="Fz"),
        market=MarketIngress(bad_market, source_id="feed"),
        clock=ClockSync(max_drift_seconds=0.05),
    )
    with pytest.raises(Exception, match="feed crashed"):
        mux.poll()
