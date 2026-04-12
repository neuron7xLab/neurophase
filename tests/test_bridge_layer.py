"""Bridge-layer contract tests.

Pins the fail-closed behaviour of every bridge:

* ``NeuralSample`` / ``MarketTick`` reject non-finite, out-of-range,
  and empty source_id values at construction.
* ``EegIngress`` / ``MarketIngress`` raise on driver exception,
  ``None`` return, wrong type, and source_id mismatch.
* ``ClockSync`` refuses a pair drifting beyond tolerance.
* ``FrameMux`` composes them into one ``poll()`` and surfaces every
  failure as a :class:`BridgeError`.
* ``DownstreamAdapter`` never dispatches a frame whose gate refused,
  validates the payload, and surfaces transport exceptions.
"""

from __future__ import annotations

import math

import pytest

from neurophase.api import (
    OrchestratorConfig,
    PipelineConfig,
    PolicyConfig,
    RuntimeOrchestrator,
)
from neurophase.bridges import (
    BridgeError,
    ClockDesyncError,
    ClockSync,
    DownstreamAdapter,
    DownstreamAdapterError,
    EegIngress,
    FrameMux,
    MarketIngress,
    MarketTick,
    NeuralSample,
)

# ---------------------------------------------------------------------------
# NeuralSample / MarketTick
# ---------------------------------------------------------------------------


def test_neural_sample_accepts_valid_inputs() -> None:
    ns = NeuralSample(timestamp=1.0, phase=1.2, source_id="Fz")
    assert ns.source_id == "Fz"


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"timestamp": math.inf, "phase": 0.0, "source_id": "a"}, "timestamp"),
        ({"timestamp": math.nan, "phase": 0.0, "source_id": "a"}, "timestamp"),
        ({"timestamp": 0.0, "phase": math.nan, "source_id": "a"}, "phase"),
        ({"timestamp": 0.0, "phase": 10.0, "source_id": "a"}, "outside"),
        ({"timestamp": 0.0, "phase": 0.0, "source_id": ""}, "source_id"),
    ],
)
def test_neural_sample_rejects_invalid_inputs(kw: dict, match: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(BridgeError, match=match):
        NeuralSample(**kw)


def test_market_tick_accepts_nullable_fields() -> None:
    MarketTick(timestamp=1.0, R=None, delta=None, source_id="feed")
    MarketTick(timestamp=1.0, R=0.5, delta=None, source_id="feed")


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"timestamp": math.inf, "R": 0.5, "delta": 0.0, "source_id": "f"}, "timestamp"),
        ({"timestamp": 0.0, "R": math.nan, "delta": 0.0, "source_id": "f"}, "R is non-finite"),
        ({"timestamp": 0.0, "R": 1.5, "delta": 0.0, "source_id": "f"}, "outside"),
        ({"timestamp": 0.0, "R": 0.5, "delta": math.nan, "source_id": "f"}, "delta"),
        ({"timestamp": 0.0, "R": 0.5, "delta": 0.0, "source_id": ""}, "source_id"),
    ],
)
def test_market_tick_rejects_invalid_inputs(kw: dict, match: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(BridgeError, match=match):
        MarketTick(**kw)


# ---------------------------------------------------------------------------
# Ingress adapters
# ---------------------------------------------------------------------------


def _neural_driver_good() -> NeuralSample:
    return NeuralSample(timestamp=1.0, phase=0.5, source_id="Fz")


def _market_driver_good() -> MarketTick:
    return MarketTick(timestamp=1.0, R=0.9, delta=0.01, source_id="feed")


def test_eeg_ingress_happy_path() -> None:
    eeg = EegIngress(_neural_driver_good, source_id="Fz")
    assert eeg.poll().source_id == "Fz"


def test_eeg_ingress_raises_on_driver_exception() -> None:
    def bad() -> NeuralSample:
        raise RuntimeError("sensor detached")

    eeg = EegIngress(bad, source_id="Fz")
    with pytest.raises(BridgeError, match="driver raised"):
        eeg.poll()


def test_eeg_ingress_raises_on_none() -> None:
    eeg = EegIngress(lambda: None, source_id="Fz")
    with pytest.raises(BridgeError, match="returned None"):
        eeg.poll()


def test_eeg_ingress_raises_on_wrong_type() -> None:
    eeg = EegIngress(lambda: "not a sample", source_id="Fz")  # type: ignore[return-value,arg-type]
    with pytest.raises(BridgeError, match="expected NeuralSample"):
        eeg.poll()


def test_eeg_ingress_raises_on_source_id_mismatch() -> None:
    eeg = EegIngress(_neural_driver_good, source_id="Cz")
    with pytest.raises(BridgeError, match="source_id mismatch"):
        eeg.poll()


def test_market_ingress_raises_on_wrong_type() -> None:
    mkt = MarketIngress(lambda: 42, source_id="feed")  # type: ignore[arg-type,return-value]
    with pytest.raises(BridgeError, match="expected MarketTick"):
        mkt.poll()


# ---------------------------------------------------------------------------
# ClockSync
# ---------------------------------------------------------------------------


def test_clock_sync_accepts_aligned_pair() -> None:
    c = ClockSync(max_drift_seconds=0.05)
    ns = NeuralSample(timestamp=10.00, phase=0.0, source_id="Fz")
    mt = MarketTick(timestamp=10.01, R=0.5, delta=0.0, source_id="feed")
    assert c.fuse(ns, mt) == pytest.approx(10.01)


def test_clock_sync_rejects_drift() -> None:
    c = ClockSync(max_drift_seconds=0.001)
    ns = NeuralSample(timestamp=10.00, phase=0.0, source_id="Fz")
    mt = MarketTick(timestamp=11.00, R=0.5, delta=0.0, source_id="feed")
    with pytest.raises(ClockDesyncError, match="clock desync"):
        c.fuse(ns, mt)


def test_clock_sync_rejects_non_positive_tolerance() -> None:
    with pytest.raises(BridgeError):
        ClockSync(max_drift_seconds=0.0)
    with pytest.raises(BridgeError):
        ClockSync(max_drift_seconds=-1.0)


# ---------------------------------------------------------------------------
# FrameMux
# ---------------------------------------------------------------------------


def test_frame_mux_emits_canonical_triple() -> None:
    mux = FrameMux(
        eeg=EegIngress(_neural_driver_good, source_id="Fz"),
        market=MarketIngress(_market_driver_good, source_id="feed"),
        clock=ClockSync(max_drift_seconds=0.05),
    )
    muxed = mux.poll()
    assert muxed.timestamp == pytest.approx(1.0)
    assert muxed.R == 0.9
    assert muxed.delta == 0.01
    assert muxed.neural_source_id == "Fz"
    assert muxed.market_source_id == "feed"


def test_frame_mux_surfaces_ingress_failure() -> None:
    def bad_market() -> MarketTick:
        raise RuntimeError("feed lost")

    mux = FrameMux(
        eeg=EegIngress(_neural_driver_good, source_id="Fz"),
        market=MarketIngress(bad_market, source_id="feed"),
        clock=ClockSync(max_drift_seconds=0.05),
    )
    with pytest.raises(BridgeError, match="feed lost"):
        mux.poll()


def test_frame_mux_surfaces_clock_desync() -> None:
    def drifted_market() -> MarketTick:
        return MarketTick(timestamp=99.0, R=0.5, delta=0.0, source_id="feed")

    mux = FrameMux(
        eeg=EegIngress(_neural_driver_good, source_id="Fz"),
        market=MarketIngress(drifted_market, source_id="feed"),
        clock=ClockSync(max_drift_seconds=0.01),
    )
    with pytest.raises(ClockDesyncError):
        mux.poll()


# ---------------------------------------------------------------------------
# DownstreamAdapter
# ---------------------------------------------------------------------------


def _ready_frame() -> tuple[RuntimeOrchestrator, object]:
    orch = RuntimeOrchestrator(
        OrchestratorConfig(
            pipeline=PipelineConfig(warmup_samples=2, stream_window=4, enable_stillness=False),
            policy=PolicyConfig(),
        )
    )
    # Drive long enough to warm the stream detector and reach READY.
    frame = None
    for i in range(30):
        frame = orch.tick(timestamp=float(i) * 0.01, R=0.9, delta=0.01)
        if frame.pipeline_frame.gate.execution_allowed:
            break
    assert frame is not None and frame.pipeline_frame.gate.execution_allowed, (
        "fixture failed to produce a READY frame"
    )
    return orch, frame


def test_downstream_dispatch_happy_path() -> None:
    sent: list[dict] = []  # type: ignore[type-arg]

    def transport(payload: dict) -> str:  # type: ignore[type-arg]
        sent.append(dict(payload))
        return "ack-1"

    _, frame = _ready_frame()
    adapter = DownstreamAdapter(transport)  # type: ignore[arg-type]
    result = adapter.dispatch(frame)  # type: ignore[arg-type]

    assert result.dispatched is True
    assert result.transport_reply == "ack-1"
    assert result.schema_version
    assert len(sent) == 1
    assert "gate_state" in sent[0]


def test_downstream_refuses_non_ready_frame() -> None:
    orch = RuntimeOrchestrator(
        OrchestratorConfig(
            pipeline=PipelineConfig(warmup_samples=2, enable_stillness=False),
            policy=PolicyConfig(),
        )
    )
    # First tick is DEGRADED (warmup) — execution_allowed is False.
    frame = orch.tick(timestamp=0.0, R=0.5, delta=0.01)

    adapter = DownstreamAdapter(lambda _p: "should not fire")
    with pytest.raises(DownstreamAdapterError, match="gate refused"):
        adapter.dispatch(frame)  # type: ignore[arg-type]


def test_downstream_surfaces_transport_exception() -> None:
    def broken(_payload: dict) -> str:  # type: ignore[type-arg]
        raise ConnectionError("503 service unavailable")

    _, frame = _ready_frame()
    adapter = DownstreamAdapter(broken)  # type: ignore[arg-type]
    with pytest.raises(DownstreamAdapterError, match="transport raised"):
        adapter.dispatch(frame)  # type: ignore[arg-type]


def test_downstream_rejects_non_frame_input() -> None:
    adapter = DownstreamAdapter(lambda _p: None)
    with pytest.raises(DownstreamAdapterError, match="expected OrchestratedFrame"):
        adapter.dispatch("not a frame")  # type: ignore[arg-type]
