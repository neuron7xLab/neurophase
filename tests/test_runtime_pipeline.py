"""Tests for ``neurophase.runtime.pipeline`` (E1 + E2)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from neurophase.audit.decision_ledger import verify_ledger
from neurophase.data.stream_detector import StreamRegime
from neurophase.data.temporal_validator import TimeQuality
from neurophase.gate.execution_gate import GateState
from neurophase.runtime.pipeline import (
    DecisionFrame,
    PipelineConfig,
    StreamingPipeline,
)

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_pipeline_constructs(self) -> None:
        p = StreamingPipeline(PipelineConfig())
        assert p.n_ticks == 0
        assert p.ledger is None

    def test_invalid_threshold_is_surfaced(self) -> None:
        with pytest.raises(ValueError):
            StreamingPipeline(PipelineConfig(threshold=0.0))

    def test_parameter_fingerprint_is_deterministic(self) -> None:
        a = StreamingPipeline(PipelineConfig())
        b = StreamingPipeline(PipelineConfig())
        assert a.parameter_fingerprint == b.parameter_fingerprint

    def test_parameter_fingerprint_distinguishes_configs(self) -> None:
        a = StreamingPipeline(PipelineConfig(threshold=0.65))
        b = StreamingPipeline(PipelineConfig(threshold=0.70))
        assert a.parameter_fingerprint != b.parameter_fingerprint


# ---------------------------------------------------------------------------
# Basic tick
# ---------------------------------------------------------------------------


class TestBasicTick:
    def test_first_tick_is_degraded(self) -> None:
        """The very first tick is WARMUP at B1 → DEGRADED at the gate."""
        p = StreamingPipeline(PipelineConfig())
        frame = p.tick(timestamp=0.0, R=0.9, delta=0.01)
        assert isinstance(frame, DecisionFrame)
        assert frame.tick_index == 0
        assert frame.gate_state is GateState.DEGRADED
        assert frame.execution_allowed is False
        assert frame.temporal.quality is TimeQuality.WARMUP

    def test_steady_healthy_stream_becomes_ready(self) -> None:
        """After warmup, a healthy stream with R ≥ threshold emits READY
        (or UNNECESSARY once stillness settles)."""
        p = StreamingPipeline(
            PipelineConfig(
                warmup_samples=2,
                stream_window=4,
                max_fault_rate=0.25,
                stillness_window=4,
                stillness_delta_min=0.20,
            )
        )
        timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        for ts in timestamps:
            frame = p.tick(timestamp=ts, R=0.9, delta=0.01)
        assert frame.gate_state in {GateState.READY, GateState.UNNECESSARY}
        assert frame.stream_regime is StreamRegime.HEALTHY

    def test_frame_properties(self) -> None:
        p = StreamingPipeline(PipelineConfig())
        frame = p.tick(timestamp=0.0, R=0.9)
        assert frame.gate_state is frame.gate.state
        assert frame.stream_regime is frame.stream.regime
        assert frame.time_quality is frame.temporal.quality
        assert frame.execution_allowed is frame.gate.execution_allowed


# ---------------------------------------------------------------------------
# Stream degradation shuts the gate
# ---------------------------------------------------------------------------


class TestStreamDegradation:
    def test_bad_stream_forces_degraded_gate(self) -> None:
        """A DEGRADED stream regime must force the gate to DEGRADED even
        when R is above threshold."""
        p = StreamingPipeline(
            PipelineConfig(
                max_gap_seconds=0.2,
                warmup_samples=2,
                stream_window=6,
                max_fault_rate=0.25,
                stream_hold_steps=0,
            )
        )
        # Alternate valid / gapped to drive fault_rate > 0.25.
        timestamps = [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 2.0, 2.1, 3.0]
        frames = [p.tick(timestamp=ts, R=0.99, delta=0.01) for ts in timestamps]
        # At least one later frame must have landed in DEGRADED due to
        # the stream regime (not a per-sample B1 failure).
        degraded = [
            f
            for f in frames
            if f.gate_state is GateState.DEGRADED and f.stream_regime is not StreamRegime.HEALTHY
        ]
        assert degraded


# ---------------------------------------------------------------------------
# Ledger integration (F1)
# ---------------------------------------------------------------------------


class TestLedgerIntegration:
    def test_pipeline_without_ledger_emits_none(self) -> None:
        p = StreamingPipeline(PipelineConfig())
        frame = p.tick(timestamp=0.0, R=0.9)
        assert frame.ledger_record is None
        assert p.ledger is None

    def test_pipeline_with_ledger_appends_and_verifies(self, tmp_path: Path) -> None:
        ledger_path = tmp_path / "pipeline.jsonl"
        cfg = PipelineConfig(warmup_samples=2, ledger_path=ledger_path)
        p = StreamingPipeline(cfg)
        for ts in [0.0, 0.1, 0.2, 0.3, 0.4]:
            frame = p.tick(timestamp=ts, R=0.9, delta=0.01)
            assert frame.ledger_record is not None
        assert p.ledger is not None
        v = verify_ledger(ledger_path)
        assert v.ok
        assert v.n_records == 5

    def test_pipeline_ledger_is_byte_identical_across_replays(self, tmp_path: Path) -> None:
        """The strongest reproducibility contract: two pipelines with
        identical config and identical input sequences produce
        byte-identical ledger files. This is E1's direct contribution
        to doctrine #5."""

        def run(name: str) -> Path:
            path = tmp_path / f"{name}.jsonl"
            p = StreamingPipeline(
                PipelineConfig(
                    warmup_samples=2,
                    ledger_path=path,
                )
            )
            for ts in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                p.tick(timestamp=ts, R=0.9, delta=0.01)
            return path

        p1 = run("alpha")
        p2 = run("beta")
        assert p1.read_bytes() == p2.read_bytes()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_restarts_warmup(self) -> None:
        p = StreamingPipeline(PipelineConfig(warmup_samples=2))
        for ts in [0.0, 0.1, 0.2, 0.3]:
            p.tick(timestamp=ts, R=0.9, delta=0.01)
        assert p.n_ticks == 4
        p.reset()
        assert p.n_ticks == 0
        frame = p.tick(timestamp=100.0, R=0.9, delta=0.01)
        # Fresh warmup after reset.
        assert frame.time_quality is TimeQuality.WARMUP


# ---------------------------------------------------------------------------
# Frame serialization
# ---------------------------------------------------------------------------


class TestFrameSerialization:
    def test_frame_is_frozen(self) -> None:
        p = StreamingPipeline(PipelineConfig())
        frame = p.tick(timestamp=0.0, R=0.9)
        with pytest.raises(dataclasses.FrozenInstanceError):
            frame.tick_index = 99  # type: ignore[misc]

    def test_to_json_dict_is_serializable(self) -> None:
        p = StreamingPipeline(PipelineConfig())
        frame = p.tick(timestamp=0.0, R=0.9, delta=0.01)
        payload = frame.to_json_dict()
        # Should round-trip through json.dumps with no default fallback.
        s = json.dumps(payload, sort_keys=True)
        assert "gate_state" in s
        assert "tick_index" in s

    def test_to_json_dict_flat_contract(self) -> None:
        """The frame's JSON projection must be flat — no nested
        dataclasses. This is load-bearing for downstream log
        processors that do not want to pull in neurophase types."""
        p = StreamingPipeline(PipelineConfig())
        frame = p.tick(timestamp=0.0, R=0.9, delta=0.01)
        payload = frame.to_json_dict()
        for key, value in payload.items():
            assert not dataclasses.is_dataclass(value), (
                f"field {key!r} in to_json_dict is a dataclass; flat contract violated"
            )


# ---------------------------------------------------------------------------
# Determinism (F3 extension)
# ---------------------------------------------------------------------------


class TestPipelineDeterminism:
    def test_same_input_same_frames(self) -> None:
        def run() -> list[str]:
            p = StreamingPipeline(PipelineConfig(warmup_samples=2))
            out: list[str] = []
            for ts in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                frame = p.tick(timestamp=ts, R=0.9, delta=0.01)
                out.append(frame.gate_state.name)
            return out

        assert run() == run()
