"""Closure-ready causal path — end-to-end integration test.

The single full-stack scenario that exercises every load-bearing layer
in the canonical order::

    ingress contract (FrameMux)
      → canonical frame (OrchestratedFrame)
      → gate (I₁..I₄)
      → regime (G1)
      → action intent (I1)
      → ledger / witness (F1, as_canonical_dict)
      → downstream adapter (bridges.DownstreamAdapter)

Two runs with identical driver state produce byte-identical canonical
dict sequences — the replay contract.

**Important.** This is a *closure-ready causal path*, not a full
closed-loop cognition machine. There is no two-way live bus between the
downstream and the ingress. The chain is a one-way causal sequence that
can be replayed, audited, and verified against the ledger.
"""

from __future__ import annotations

import json

from neurophase.api import (
    OrchestratorConfig,
    PipelineConfig,
    PolicyConfig,
    RuntimeOrchestrator,
)
from neurophase.bridges import (
    ClockSync,
    DownstreamAdapter,
    EegIngress,
    FrameMux,
    MarketIngress,
    MarketTick,
    NeuralSample,
)
from neurophase.contracts import as_canonical_dict, validate_canonical_dict


def _build_mux(start_ts: float = 0.0) -> tuple[FrameMux, list[float]]:
    """Driver fixture: paired deterministic ingresses + shared clock.

    Returns the mux and the shared tick-counter list so a second
    identical run can be reconstructed.
    """
    counter = [0]

    def neural() -> NeuralSample:
        i = counter[0]
        return NeuralSample(
            timestamp=start_ts + i * 0.01,
            phase=0.0,
            source_id="Fz",
        )

    def market() -> MarketTick:
        i = counter[0]
        counter[0] = i + 1
        return MarketTick(
            timestamp=start_ts + i * 0.01,
            R=0.9,
            delta=0.01,
            source_id="feed",
        )

    mux = FrameMux(
        eeg=EegIngress(neural, source_id="Fz"),
        market=MarketIngress(market, source_id="feed"),
        clock=ClockSync(max_drift_seconds=0.05),
    )
    return mux, counter


def _build_orchestrator() -> RuntimeOrchestrator:
    return RuntimeOrchestrator(
        OrchestratorConfig(
            pipeline=PipelineConfig(warmup_samples=2, stream_window=4, enable_stillness=False),
            policy=PolicyConfig(),
        )
    )


def _drive_session(n: int = 16) -> list[dict]:  # type: ignore[type-arg]
    """Full ingress → gate → canonical → downstream session.

    Returns the sequence of canonical dicts dispatched to the
    transport. A non-READY frame is still canonically serialised and
    validated, but skipped by the downstream adapter (tracked via the
    ``dispatched`` flag on the transport log).
    """
    mux, _counter = _build_mux()
    orch = _build_orchestrator()

    sent: list[dict] = []  # type: ignore[type-arg]

    def transport(payload: dict) -> str:  # type: ignore[type-arg]
        sent.append(dict(payload))
        return f"ack-{len(sent)}"

    adapter = DownstreamAdapter(transport)

    for _ in range(n):
        muxed = mux.poll()
        frame = orch.tick(
            timestamp=muxed.timestamp,
            R=muxed.R,
            delta=muxed.delta,
        )
        # Canonical serialisation + validation run on every frame
        # regardless of gate verdict — that is the audit surface.
        payload = as_canonical_dict(frame)
        validate_canonical_dict(payload)
        # Downstream dispatch only runs on READY frames (gate-first).
        if frame.execution_allowed:
            adapter.dispatch(frame)

    return sent


def test_closure_path_runs_end_to_end() -> None:
    sent = _drive_session(n=16)
    assert sent, "no frames reached the downstream adapter — gate never opened"
    # Every dispatched payload must carry the canonical schema version
    # and be self-validating.
    for payload in sent:
        validate_canonical_dict(payload)


def test_closure_path_is_deterministic_under_replay() -> None:
    """Two runs with identical state emit byte-identical dispatches."""
    a = _drive_session(n=24)
    b = _drive_session(n=24)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True), (
        "closure-ready path is not deterministic under replay"
    )


def test_closure_path_never_dispatches_non_ready_frames() -> None:
    """Every dispatched payload must have ``execution_allowed=True``.

    Asserted directly on the payloads since the downstream adapter
    is the only egress and it enforces the gate-first law.
    """
    sent = _drive_session(n=32)
    for payload in sent:
        assert payload["execution_allowed"] is True
        assert payload["gate_state"] == "READY"


def test_closure_path_payloads_carry_audit_fields() -> None:
    sent = _drive_session(n=16)
    assert sent, "fixture did not dispatch any frames"
    required = {
        "schema_version",
        "tick_index",
        "timestamp",
        "gate_state",
        "execution_allowed",
        "regime_label",
        "action_intent",
    }
    for payload in sent:
        missing = required - set(payload.keys())
        assert not missing, f"canonical payload missing audit fields: {missing}"
