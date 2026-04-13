"""Tests for neurophase.physio.pipeline — end-to-end replay-to-frames."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurophase.physio.features import MIN_WINDOW_SIZE
from neurophase.physio.gate import PhysioGateState
from neurophase.physio.pipeline import (
    CANONICAL_FRAME_SCHEMA_VERSION,
    PhysioReplayPipeline,
)
from neurophase.physio.replay import ReplayIngestError


@pytest.fixture
def clean_csv(tmp_path: Path) -> Path:
    rows = ["timestamp_s,rr_ms"]
    t = 0.0
    # 40 stable samples -> past warm-up and into EXECUTE_ALLOWED.
    for i in range(40):
        rr = 820.0 + (8.0 if i % 2 == 0 else -8.0)
        t += rr / 1000.0
        rows.append(f"{t:.3f},{rr:.2f}")
    path = tmp_path / "clean.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def impossible_csv(tmp_path: Path) -> Path:
    path = tmp_path / "impossible.csv"
    path.write_text("timestamp_s,rr_ms\n0.0,820.0\n0.8,50.0\n", encoding="utf-8")
    return path


class TestPhysioFrameSchema:
    def test_frame_has_expected_fields(self, clean_csv: Path) -> None:
        pipe = PhysioReplayPipeline()
        frames, _ = pipe.run_csv(clean_csv)
        assert frames, "expected at least one frame"
        first = frames[0]
        d = first.to_json_dict()
        assert set(d) >= {
            "schema_version",
            "tick_index",
            "timestamp_s",
            "rr_ms",
            "features",
            "decision",
            "labels",
        }
        assert d["schema_version"] == CANONICAL_FRAME_SCHEMA_VERSION
        assert d["decision"]["state"] in {s.name for s in PhysioGateState}
        assert d["decision"]["kernel_state"] in {
            "READY",
            "BLOCKED",
            "DEGRADED",
            "SENSOR_ABSENT",
            "UNNECESSARY",
        }

    def test_frame_json_is_serializable(self, clean_csv: Path) -> None:
        pipe = PhysioReplayPipeline()
        frames, summary = pipe.run_csv(clean_csv)
        payload = {
            "summary": summary.to_json_dict(),
            "frames": [f.to_json_dict() for f in frames],
        }
        roundtrip = json.loads(json.dumps(payload))
        assert roundtrip["summary"]["n_frames_emitted"] == len(frames)


class TestPipelineFailClosed:
    def test_impossible_rr_raises(self, impossible_csv: Path) -> None:
        pipe = PhysioReplayPipeline()
        with pytest.raises(ReplayIngestError, match="outside physiological envelope"):
            pipe.run_csv(impossible_csv)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        pipe = PhysioReplayPipeline()
        with pytest.raises(ReplayIngestError, match="not found"):
            pipe.run_csv(tmp_path / "missing.csv")

    def test_warmup_phase_never_execute_allowed(self, clean_csv: Path) -> None:
        pipe = PhysioReplayPipeline()
        frames, _ = pipe.run_csv(clean_csv)
        for frame in frames[: MIN_WINDOW_SIZE - 1]:
            assert frame.decision.state is not PhysioGateState.EXECUTE_ALLOWED
            assert frame.decision.execution_allowed is False


class TestPipelineDeterminism:
    def test_two_pipelines_same_input_same_frames(self, clean_csv: Path) -> None:
        a, _ = PhysioReplayPipeline().run_csv(clean_csv)
        b, _ = PhysioReplayPipeline().run_csv(clean_csv)
        assert len(a) == len(b)
        states_a = [(f.tick_index, f.decision.state.name) for f in a]
        states_b = [(f.tick_index, f.decision.state.name) for f in b]
        assert states_a == states_b


class TestSummary:
    def test_summary_counts_match_frames(self, clean_csv: Path) -> None:
        pipe = PhysioReplayPipeline()
        frames, summary = pipe.run_csv(clean_csv)
        assert summary.n_frames_emitted == len(frames)
        total = sum(summary.state_counts.values())
        assert total == len(frames)

    def test_n_execution_allowed_consistent_with_state_counts(self, clean_csv: Path) -> None:
        pipe = PhysioReplayPipeline()
        _, summary = pipe.run_csv(clean_csv)
        assert summary.n_execution_allowed == summary.state_counts.get(
            PhysioGateState.EXECUTE_ALLOWED.name, 0
        )


def test_frame_is_immutable(clean_csv: Path) -> None:
    pipe = PhysioReplayPipeline()
    frames, _ = pipe.run_csv(clean_csv)
    assert frames
    with pytest.raises(AttributeError):
        frames[0].tick_index = 999  # type: ignore[misc]
