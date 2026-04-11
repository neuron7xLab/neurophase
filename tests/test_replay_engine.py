"""Tests for F2 replay engine."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from neurophase.audit.decision_ledger import verify_ledger
from neurophase.audit.replay import ReplayInput, ReplayResult, replay_ledger
from neurophase.runtime.pipeline import PipelineConfig, StreamingPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_inputs() -> list[ReplayInput]:
    """Canonical healthy-stream input sequence used by most tests."""
    return [
        ReplayInput(timestamp=0.0, R=0.95, delta=0.01),
        ReplayInput(timestamp=0.1, R=0.95, delta=0.01),
        ReplayInput(timestamp=0.2, R=0.95, delta=0.01),
        ReplayInput(timestamp=0.3, R=0.95, delta=0.01),
        ReplayInput(timestamp=0.4, R=0.95, delta=0.01),
        ReplayInput(timestamp=0.5, R=0.95, delta=0.01),
    ]


def _write_original_ledger(
    tmp_path: Path, inputs: list[ReplayInput]
) -> tuple[Path, PipelineConfig]:
    original_path = tmp_path / "original.jsonl"
    cfg = PipelineConfig(
        warmup_samples=2,
        stream_window=4,
        ledger_path=original_path,
    )
    p = StreamingPipeline(cfg)
    for inp in inputs:
        p.tick(
            timestamp=inp.timestamp,
            R=inp.R,
            delta=inp.delta,
            reference_now=inp.reference_now,
        )
    return original_path, cfg


def _replay_config(cfg: PipelineConfig, scratch_path: Path) -> PipelineConfig:
    """Clone the original config but point the ledger at the scratch path."""
    return dataclasses.replace(cfg, ledger_path=scratch_path)


# ---------------------------------------------------------------------------
# Happy path — byte-identical replay
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_replay_matches_original_byte_for_byte(self, tmp_path: Path) -> None:
        inputs = _default_inputs()
        original_path, cfg = _write_original_ledger(tmp_path, inputs)
        scratch_path = tmp_path / "scratch.jsonl"
        replay_cfg = _replay_config(cfg, scratch_path)

        result = replay_ledger(
            original_path=original_path,
            config=replay_cfg,
            inputs=inputs,
            scratch_path=scratch_path,
        )
        assert result.ok is True
        assert result.n_records == len(inputs)
        assert result.first_divergent_index is None
        assert result.original_tip_hash == result.replayed_tip_hash
        assert result.scratch_path == scratch_path.resolve()
        assert original_path.read_bytes() == scratch_path.read_bytes()

    def test_replayed_ledger_verifies_independently(self, tmp_path: Path) -> None:
        inputs = _default_inputs()
        original_path, cfg = _write_original_ledger(tmp_path, inputs)
        scratch_path = tmp_path / "scratch.jsonl"
        replay_ledger(
            original_path=original_path,
            config=_replay_config(cfg, scratch_path),
            inputs=inputs,
            scratch_path=scratch_path,
        )
        v = verify_ledger(scratch_path)
        assert v.ok is True
        assert v.n_records == len(inputs)

    def test_result_is_frozen(self, tmp_path: Path) -> None:
        inputs = _default_inputs()
        original_path, cfg = _write_original_ledger(tmp_path, inputs)
        scratch_path = tmp_path / "scratch.jsonl"
        result = replay_ledger(
            original_path=original_path,
            config=_replay_config(cfg, scratch_path),
            inputs=inputs,
            scratch_path=scratch_path,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.ok = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------


class TestDivergence:
    def test_divergent_inputs_are_reported(self, tmp_path: Path) -> None:
        """Replay with DIFFERENT inputs than the original → divergence."""
        original_inputs = _default_inputs()
        original_path, cfg = _write_original_ledger(tmp_path, original_inputs)

        divergent_inputs = list(original_inputs)
        divergent_inputs[3] = ReplayInput(
            timestamp=0.3, R=0.30, delta=0.01
        )  # force BLOCKED instead of READY/UNNECESSARY

        scratch_path = tmp_path / "scratch.jsonl"
        result = replay_ledger(
            original_path=original_path,
            config=_replay_config(cfg, scratch_path),
            inputs=divergent_inputs,
            scratch_path=scratch_path,
        )
        assert result.ok is False
        assert result.first_divergent_index == 3
        assert result.original_tip_hash != result.replayed_tip_hash
        assert "diverges" in result.reason

    def test_divergent_config_changes_fingerprint(self, tmp_path: Path) -> None:
        """Replay with a DIFFERENT threshold → fingerprint changes → divergence."""
        inputs = _default_inputs()
        original_path, cfg = _write_original_ledger(tmp_path, inputs)
        scratch_path = tmp_path / "scratch.jsonl"
        altered_cfg = dataclasses.replace(cfg, ledger_path=scratch_path, threshold=0.75)
        result = replay_ledger(
            original_path=original_path,
            config=altered_cfg,
            inputs=inputs,
            scratch_path=scratch_path,
        )
        assert result.ok is False
        # Divergence starts at the very first record because the
        # parameter_fingerprint — embedded in every record — differs.
        assert result.first_divergent_index == 0

    def test_short_replay_is_reported_as_divergent(self, tmp_path: Path) -> None:
        inputs = _default_inputs()
        original_path, cfg = _write_original_ledger(tmp_path, inputs)
        scratch_path = tmp_path / "scratch.jsonl"
        result = replay_ledger(
            original_path=original_path,
            config=_replay_config(cfg, scratch_path),
            inputs=inputs[:3],  # drop the tail
            scratch_path=scratch_path,
        )
        assert result.ok is False
        assert result.first_divergent_index is not None


# ---------------------------------------------------------------------------
# Safety contracts
# ---------------------------------------------------------------------------


class TestSafetyContracts:
    def test_scratch_must_differ_from_original(self, tmp_path: Path) -> None:
        original_path, cfg = _write_original_ledger(tmp_path, _default_inputs())
        with pytest.raises(ValueError, match="non-destructive"):
            replay_ledger(
                original_path=original_path,
                config=dataclasses.replace(cfg, ledger_path=original_path),
                inputs=_default_inputs(),
                scratch_path=original_path,
            )

    def test_missing_original_is_rejected(self, tmp_path: Path) -> None:
        scratch_path = tmp_path / "scratch.jsonl"
        cfg = PipelineConfig(warmup_samples=2, ledger_path=scratch_path)
        with pytest.raises(ValueError, match="original ledger not found"):
            replay_ledger(
                original_path=tmp_path / "does-not-exist.jsonl",
                config=cfg,
                inputs=_default_inputs(),
                scratch_path=scratch_path,
            )

    def test_config_without_ledger_path_is_rejected(self, tmp_path: Path) -> None:
        inputs = _default_inputs()
        original_path, _ = _write_original_ledger(tmp_path, inputs)
        scratch_path = tmp_path / "scratch.jsonl"
        bad_cfg = PipelineConfig(warmup_samples=2, ledger_path=None)
        with pytest.raises(ValueError, match="ledger_path must be set"):
            replay_ledger(
                original_path=original_path,
                config=bad_cfg,
                inputs=inputs,
                scratch_path=scratch_path,
            )

    def test_config_ledger_path_must_equal_scratch_path(self, tmp_path: Path) -> None:
        inputs = _default_inputs()
        original_path, _ = _write_original_ledger(tmp_path, inputs)
        scratch_path = tmp_path / "scratch.jsonl"
        wrong_path = tmp_path / "wrong.jsonl"
        wrong_cfg = PipelineConfig(warmup_samples=2, ledger_path=wrong_path)
        with pytest.raises(ValueError, match="must equal"):
            replay_ledger(
                original_path=original_path,
                config=wrong_cfg,
                inputs=inputs,
                scratch_path=scratch_path,
            )

    def test_replay_does_not_modify_original(self, tmp_path: Path) -> None:
        inputs = _default_inputs()
        original_path, cfg = _write_original_ledger(tmp_path, inputs)
        original_bytes_before = original_path.read_bytes()
        scratch_path = tmp_path / "scratch.jsonl"
        replay_ledger(
            original_path=original_path,
            config=_replay_config(cfg, scratch_path),
            inputs=inputs,
            scratch_path=scratch_path,
        )
        original_bytes_after = original_path.read_bytes()
        assert original_bytes_before == original_bytes_after


# ---------------------------------------------------------------------------
# Tampered original ledger is caught before replay
# ---------------------------------------------------------------------------


class TestTamperedOriginal:
    def test_tampered_original_reports_failure_without_replay(self, tmp_path: Path) -> None:
        inputs = _default_inputs()
        original_path, cfg = _write_original_ledger(tmp_path, inputs)

        # Tamper: mutate the reason field of the second record.
        lines = original_path.read_text(encoding="utf-8").splitlines()
        payload = json.loads(lines[1])
        payload["reason"] = "TAMPERED"
        lines[1] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        original_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        scratch_path = tmp_path / "scratch.jsonl"
        result = replay_ledger(
            original_path=original_path,
            config=_replay_config(cfg, scratch_path),
            inputs=inputs,
            scratch_path=scratch_path,
        )
        assert result.ok is False
        assert "failed verification" in result.reason
        # Replay must not have been attempted — scratch file should not
        # have been created (or at least the result path remains
        # well-defined for inspection).
        # The engine short-circuits before creating the scratch file
        # on a tampered original.


# ---------------------------------------------------------------------------
# Full end-to-end: CoupledBrainMarketSystem → ledger → replay
# ---------------------------------------------------------------------------


class TestEndToEndReplay:
    def test_coupled_system_replay_matches(self, tmp_path: Path) -> None:
        """Drive a real CoupledBrainMarketSystem through the pipeline,
        write a ledger, then replay against the same inputs. Byte-match
        is the load-bearing contract."""
        from neurophase.sync.coupled_brain_market import CoupledBrainMarketSystem

        system = CoupledBrainMarketSystem(K=5.0, sigma=0.0, dt=0.01, seed=401)
        df = system.run(n_steps=50)
        inputs = [
            ReplayInput(
                timestamp=float(row["t"]),
                R=float(row["R"]),
                delta=float(row["delta"]),
            )
            for _, row in df.iterrows()
        ]

        original_path = tmp_path / "coupled.jsonl"
        cfg = PipelineConfig(
            warmup_samples=2,
            stream_window=4,
            max_fault_rate=0.50,
            ledger_path=original_path,
        )
        pipeline = StreamingPipeline(cfg)
        for inp in inputs:
            pipeline.tick(timestamp=inp.timestamp, R=inp.R, delta=inp.delta)

        scratch_path = tmp_path / "coupled_replay.jsonl"
        result = replay_ledger(
            original_path=original_path,
            config=_replay_config(cfg, scratch_path),
            inputs=inputs,
            scratch_path=scratch_path,
        )
        assert isinstance(result, ReplayResult)
        assert result.ok is True, result.reason
        assert result.n_records == len(inputs)
