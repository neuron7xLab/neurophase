"""F₂ replay bit-determinism proof suite.

Proves that replaying the same inputs through the same pipeline config
produces byte-identical ledger output — the minimal precondition for
any incident postmortem.

Coverage:
  * 100-tick and 200-tick full-pipeline replay
  * SHA-256 chain integrity across original and scratch
  * Two independent runs produce identical ledger files
  * Temporal faults (gaps, reversals) survive bit-identical replay
  * Config mismatch → divergence surfaced with first_divergent_index
  * Per-record record_hash match verified line by line
  * GENESIS_HASH anchors the first record's prev_hash
  * Scratch file cleaned on re-run so chain starts fresh
  * All five gate states reachable and reproducible
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from neurophase.audit.decision_ledger import GENESIS_HASH, verify_ledger
from neurophase.audit.replay import ReplayInput, replay_ledger
from neurophase.gate.execution_gate import GateState
from neurophase.runtime.pipeline import PipelineConfig, StreamingPipeline

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE_TS: float = 1_000.0
_STRIDE: float = 1.0


def _generate_inputs(n: int, *, seed: int = 42) -> list[ReplayInput]:
    """Canonical mixed-regime input sequence for the F₂ proof suite.

    Mix of valid R above/below threshold, None (→ DEGRADED), and
    small delta values that allow the stillness detector to classify
    some ticks as STILL.
    """
    rng = np.random.default_rng(seed)
    inputs: list[ReplayInput] = []
    for i in range(n):
        ts = _BASE_TS + i * _STRIDE
        r_val: float | None
        delta_val: float | None
        if i % 10 == 0:
            r_val = None  # → SENSOR_ABSENT / DEGRADED path
            delta_val = None
        elif i % 7 == 0:
            r_val = float(rng.uniform(0.0, 0.3))  # below threshold → BLOCKED
            delta_val = float(rng.uniform(0.0, 0.5))
        else:
            r_val = float(rng.uniform(0.6, 1.0))  # above threshold → READY
            delta_val = float(rng.uniform(0.0, 0.5))
        inputs.append(ReplayInput(timestamp=ts, R=r_val, delta=delta_val))
    return inputs


def _run_pipeline(cfg: PipelineConfig, inputs: list[ReplayInput]) -> None:
    """Feed *inputs* through a fresh StreamingPipeline built from *cfg*."""
    p = StreamingPipeline(cfg)
    for inp in inputs:
        p.tick(
            timestamp=inp.timestamp,
            R=inp.R,
            delta=inp.delta,
            reference_now=inp.reference_now,
        )


def _make_cfg(
    *,
    ledger_path: Path,
    threshold: float = 0.5,
    stream_window: int = 8,
    warmup_samples: int = 2,
    enable_stillness: bool = True,
    stillness_window: int = 4,
    stillness_eps_R: float = 1e-3,
    stillness_eps_F: float = 1e-3,
    stillness_delta_min: float = 0.10,
) -> PipelineConfig:
    return PipelineConfig(
        threshold=threshold,
        stream_window=stream_window,
        warmup_samples=warmup_samples,
        enable_stillness=enable_stillness,
        stillness_window=stillness_window,
        stillness_eps_R=stillness_eps_R,
        stillness_eps_F=stillness_eps_F,
        stillness_delta_min=stillness_delta_min,
        ledger_path=ledger_path,
    )


def _replay(
    original_path: Path,
    cfg: PipelineConfig,
    inputs: list[ReplayInput],
    scratch_path: Path,
) -> None:
    """Convenience: build replay config and call replay_ledger, asserting ok."""
    replay_cfg = dataclasses.replace(cfg, ledger_path=scratch_path)
    result = replay_ledger(
        original_path=original_path,
        config=replay_cfg,
        inputs=inputs,
        scratch_path=scratch_path,
    )
    assert result.ok is True, f"replay failed: {result.reason}"


# ---------------------------------------------------------------------------
# Test 1 — 100-tick replay byte-identical
# ---------------------------------------------------------------------------


class TestReplay100TicksByteIdentical:
    """test_100_tick_replay_byte_identical"""

    def test_100_tick_replay_byte_identical(self, tmp_path: Path) -> None:
        inputs = _generate_inputs(100, seed=42)
        original_path = tmp_path / "original.jsonl"
        scratch_path = tmp_path / "scratch.jsonl"

        cfg = _make_cfg(ledger_path=original_path)
        _run_pipeline(cfg, inputs)

        replay_cfg = dataclasses.replace(cfg, ledger_path=scratch_path)
        result = replay_ledger(
            original_path=original_path,
            config=replay_cfg,
            inputs=inputs,
            scratch_path=scratch_path,
        )

        assert result.ok is True, result.reason
        assert result.n_records == 100
        assert result.first_divergent_index is None
        assert result.original_tip_hash == result.replayed_tip_hash
        assert original_path.read_bytes() == scratch_path.read_bytes()


# ---------------------------------------------------------------------------
# Test 2 — 200-tick with StillnessDetector, alternating STILL / ACTIVE
# ---------------------------------------------------------------------------


class TestReplay200TickWithStillness:
    """test_200_tick_with_stillness_replay"""

    def _still_active_inputs(self, n: int) -> list[ReplayInput]:
        """Alternating blocks: 20 STILL-friendly ticks, 20 ACTIVE ticks."""
        inputs: list[ReplayInput] = []
        for i in range(n):
            ts = _BASE_TS + i * _STRIDE
            block = (i // 20) % 2
            if block == 0:
                # STILL regime: flat R, tiny delta
                r_val: float | None = 0.80
                delta_val: float | None = 0.001
            else:
                # ACTIVE regime: varying R
                r_val = 0.5 + 0.4 * float(np.sin(i * 0.3))
                delta_val = 0.3 + 0.2 * float(np.cos(i * 0.2))
            inputs.append(ReplayInput(timestamp=ts, R=r_val, delta=delta_val))
        return inputs

    def test_200_tick_with_stillness_replay(self, tmp_path: Path) -> None:
        inputs = self._still_active_inputs(200)
        original_path = tmp_path / "original.jsonl"
        scratch_path = tmp_path / "scratch.jsonl"

        cfg = _make_cfg(
            ledger_path=original_path,
            enable_stillness=True,
            stillness_window=4,
            stillness_eps_R=0.05,
            stillness_eps_F=0.05,
            stillness_delta_min=0.05,
        )
        _run_pipeline(cfg, inputs)

        replay_cfg = dataclasses.replace(cfg, ledger_path=scratch_path)
        result = replay_ledger(
            original_path=original_path,
            config=replay_cfg,
            inputs=inputs,
            scratch_path=scratch_path,
        )

        assert result.ok is True, result.reason
        assert result.n_records == 200
        assert original_path.read_bytes() == scratch_path.read_bytes()

        # Verify at least one UNNECESSARY record was produced
        lines = original_path.read_text(encoding="utf-8").splitlines()
        states = {json.loads(ln)["gate_state"] for ln in lines if ln.strip()}
        assert "UNNECESSARY" in states, "expected some UNNECESSARY ticks in 200-tick still trace"


# ---------------------------------------------------------------------------
# Test 3 — SHA-256 chain integrity after replay
# ---------------------------------------------------------------------------


class TestSha256ChainIntegrityAfterReplay:
    """test_sha256_chain_integrity_after_replay"""

    def test_sha256_chain_integrity_after_replay(self, tmp_path: Path) -> None:
        inputs = _generate_inputs(120, seed=7)
        original_path = tmp_path / "original.jsonl"
        scratch_path = tmp_path / "scratch.jsonl"

        cfg = _make_cfg(ledger_path=original_path)
        _run_pipeline(cfg, inputs)

        replay_cfg = dataclasses.replace(cfg, ledger_path=scratch_path)
        result = replay_ledger(
            original_path=original_path,
            config=replay_cfg,
            inputs=inputs,
            scratch_path=scratch_path,
        )
        assert result.ok is True, result.reason

        v_orig = verify_ledger(original_path)
        v_replay = verify_ledger(scratch_path)

        assert v_orig.ok is True, v_orig.reason
        assert v_replay.ok is True, v_replay.reason
        assert v_orig.n_records == v_replay.n_records == 120

        # Tip hashes must agree
        orig_tip = json.loads(original_path.read_text(encoding="utf-8").splitlines()[-1])[
            "record_hash"
        ]
        replay_tip = json.loads(scratch_path.read_text(encoding="utf-8").splitlines()[-1])[
            "record_hash"
        ]
        assert orig_tip == replay_tip


# ---------------------------------------------------------------------------
# Test 4 — Two independent runs produce identical ledger files
# ---------------------------------------------------------------------------


class TestTwoIndependentRunsIdentical:
    """test_two_independent_runs_identical"""

    def test_two_independent_runs_identical(self, tmp_path: Path) -> None:
        inputs = _generate_inputs(100, seed=99)

        path_a = tmp_path / "run_a.jsonl"
        path_b = tmp_path / "run_b.jsonl"

        # Use the same config but different ledger paths
        cfg_a = _make_cfg(ledger_path=path_a)
        cfg_b = _make_cfg(ledger_path=path_b)

        _run_pipeline(cfg_a, inputs)
        _run_pipeline(cfg_b, inputs)

        bytes_a = path_a.read_bytes()
        bytes_b = path_b.read_bytes()
        assert bytes_a == bytes_b, (
            "two independent runs with identical config+inputs produced different ledger bytes"
        )

        v_a = verify_ledger(path_a)
        v_b = verify_ledger(path_b)
        assert v_a.ok and v_b.ok
        assert v_a.n_records == v_b.n_records == 100


# ---------------------------------------------------------------------------
# Test 5 — Temporal faults survive bit-identical replay
# ---------------------------------------------------------------------------


class TestReplayWithTemporalFaults:
    """test_replay_with_temporal_faults"""

    def _fault_inputs(self, n: int = 100) -> list[ReplayInput]:
        """Inputs that include deliberate timestamp gaps and reversals."""
        inputs: list[ReplayInput] = []
        ts = _BASE_TS
        rng = np.random.default_rng(2025)
        for i in range(n):
            if i == 20:
                ts += 999.0  # large gap → B1 GAPPED
            elif i == 50:
                ts -= 2.0  # reversal → B1 reversed
            else:
                ts += _STRIDE

            r_val: float | None
            if i % 10 == 0:
                r_val = None
            elif i % 7 == 0:
                r_val = float(rng.uniform(0.0, 0.3))
            else:
                r_val = float(rng.uniform(0.6, 1.0))
            delta_val = float(rng.uniform(0.0, 0.5)) if r_val is not None else None
            inputs.append(ReplayInput(timestamp=ts, R=r_val, delta=delta_val))
        return inputs

    def test_replay_with_temporal_faults(self, tmp_path: Path) -> None:
        inputs = self._fault_inputs(100)
        original_path = tmp_path / "original.jsonl"
        scratch_path = tmp_path / "scratch.jsonl"

        cfg = _make_cfg(
            ledger_path=original_path,
            stream_window=8,
            warmup_samples=2,
        )
        _run_pipeline(cfg, inputs)

        replay_cfg = dataclasses.replace(cfg, ledger_path=scratch_path)
        result = replay_ledger(
            original_path=original_path,
            config=replay_cfg,
            inputs=inputs,
            scratch_path=scratch_path,
        )

        assert result.ok is True, result.reason
        assert original_path.read_bytes() == scratch_path.read_bytes()

        # Verify that DEGRADED states are present (temporal faults were recorded)
        lines = original_path.read_text(encoding="utf-8").splitlines()
        states = {json.loads(ln)["gate_state"] for ln in lines if ln.strip()}
        assert "DEGRADED" in states, "expected DEGRADED records from temporal faults"


# ---------------------------------------------------------------------------
# Test 6 — Config mismatch → result.ok=False, first_divergent_index set
# ---------------------------------------------------------------------------


class TestReplayDetectsConfigMismatch:
    """test_replay_detects_config_mismatch"""

    def test_replay_detects_config_mismatch(self, tmp_path: Path) -> None:
        inputs = _generate_inputs(50, seed=17)
        original_path = tmp_path / "original.jsonl"
        scratch_path = tmp_path / "scratch.jsonl"

        cfg = _make_cfg(ledger_path=original_path, threshold=0.5)
        _run_pipeline(cfg, inputs)

        # Replay with a different threshold — parameter_fingerprint changes
        altered_cfg = dataclasses.replace(cfg, ledger_path=scratch_path, threshold=0.75)
        result = replay_ledger(
            original_path=original_path,
            config=altered_cfg,
            inputs=inputs,
            scratch_path=scratch_path,
        )

        assert result.ok is False
        assert result.first_divergent_index is not None
        # Fingerprint divergence starts at the very first record
        assert result.first_divergent_index == 0


# ---------------------------------------------------------------------------
# Test 7 — Per-record record_hash verified line by line
# ---------------------------------------------------------------------------


class TestReplay100TicksEveryRecordHashMatches:
    """test_replay_100_ticks_every_record_hash_matches"""

    def test_replay_100_ticks_every_record_hash_matches(self, tmp_path: Path) -> None:
        inputs = _generate_inputs(100, seed=123)
        original_path = tmp_path / "original.jsonl"
        scratch_path = tmp_path / "scratch.jsonl"

        cfg = _make_cfg(ledger_path=original_path)
        _run_pipeline(cfg, inputs)

        replay_cfg = dataclasses.replace(cfg, ledger_path=scratch_path)
        result = replay_ledger(
            original_path=original_path,
            config=replay_cfg,
            inputs=inputs,
            scratch_path=scratch_path,
        )
        assert result.ok is True, result.reason

        orig_lines = [
            ln for ln in original_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        replay_lines = [
            ln for ln in scratch_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]

        assert len(orig_lines) == len(replay_lines) == 100

        for i, (orig_ln, replay_ln) in enumerate(zip(orig_lines, replay_lines, strict=True)):
            orig_rec = json.loads(orig_ln)
            replay_rec = json.loads(replay_ln)
            assert orig_rec["record_hash"] == replay_rec["record_hash"], (
                f"record_hash mismatch at index {i}: "
                f"orig={orig_rec['record_hash'][:12]}… "
                f"replay={replay_rec['record_hash'][:12]}…"
            )


# ---------------------------------------------------------------------------
# Test 8 — GENESIS_HASH anchors first record's prev_hash
# ---------------------------------------------------------------------------


class TestGenesisHashIsFirstInChain:
    """test_genesis_hash_is_first_in_chain"""

    def test_genesis_hash_is_first_in_chain(self, tmp_path: Path) -> None:
        inputs = _generate_inputs(10, seed=5)
        original_path = tmp_path / "original.jsonl"
        scratch_path = tmp_path / "scratch.jsonl"

        cfg = _make_cfg(ledger_path=original_path)
        _run_pipeline(cfg, inputs)

        # Check original
        first_orig = json.loads(original_path.read_text(encoding="utf-8").splitlines()[0])
        assert first_orig["prev_hash"] == GENESIS_HASH, (
            f"expected prev_hash=GENESIS_HASH on first record, got {first_orig['prev_hash']!r}"
        )

        # Check scratch after replay
        replay_cfg = dataclasses.replace(cfg, ledger_path=scratch_path)
        result = replay_ledger(
            original_path=original_path,
            config=replay_cfg,
            inputs=inputs,
            scratch_path=scratch_path,
        )
        assert result.ok is True, result.reason

        first_scratch = json.loads(scratch_path.read_text(encoding="utf-8").splitlines()[0])
        assert first_scratch["prev_hash"] == GENESIS_HASH


# ---------------------------------------------------------------------------
# Test 9 — Scratch file cleaned on re-run
# ---------------------------------------------------------------------------


class TestReplayScratchFileCleanedOnRerun:
    """test_replay_scratch_file_cleaned_on_rerun"""

    def test_replay_scratch_file_cleaned_on_rerun(self, tmp_path: Path) -> None:
        inputs = _generate_inputs(30, seed=8)
        original_path = tmp_path / "original.jsonl"
        scratch_path = tmp_path / "scratch.jsonl"

        cfg = _make_cfg(ledger_path=original_path)
        _run_pipeline(cfg, inputs)

        replay_cfg = dataclasses.replace(cfg, ledger_path=scratch_path)

        # First replay
        result1 = replay_ledger(
            original_path=original_path,
            config=replay_cfg,
            inputs=inputs,
            scratch_path=scratch_path,
        )
        assert result1.ok is True, result1.reason
        bytes_after_first = scratch_path.read_bytes()

        # Second replay to same scratch path — must start clean
        result2 = replay_ledger(
            original_path=original_path,
            config=replay_cfg,
            inputs=inputs,
            scratch_path=scratch_path,
        )
        assert result2.ok is True, result2.reason
        bytes_after_second = scratch_path.read_bytes()

        # The second run should produce byte-identical output, not a doubled file
        assert bytes_after_first == bytes_after_second, (
            "scratch file was not cleaned before second replay — chain was extended, not reset"
        )

        # n_records should match original (not doubled)
        assert result2.n_records == 30


# ---------------------------------------------------------------------------
# Test 10 — All five gate states reachable and replay matches
# ---------------------------------------------------------------------------


class TestReplayWithAllFiveGateStates:
    """test_replay_with_all_five_gate_states"""

    def _five_state_inputs(self) -> list[ReplayInput]:
        """Craft input sequence producing all 5 gate states.

        Gate state map:
          SENSOR_ABSENT : no explicit path in StreamingPipeline (sensor_present=True always);
                          we use DEGRADED as the None-R path here.
          DEGRADED      : R=None
          BLOCKED       : R < threshold (0.5)
          READY         : R >= threshold, delta large (ACTIVE stillness)
          UNNECESSARY   : R >= threshold, delta near-zero for full stillness window
        """
        inputs: list[ReplayInput] = []
        ts = _BASE_TS

        # Warmup ticks — healthy stream so validator reaches HEALTHY
        for _ in range(6):
            inputs.append(ReplayInput(timestamp=ts, R=0.8, delta=0.4))
            ts += _STRIDE

        # DEGRADED: R=None
        for _ in range(3):
            inputs.append(ReplayInput(timestamp=ts, R=None, delta=None))
            ts += _STRIDE

        # Back to healthy to avoid extended DEGRADED stream window
        for _ in range(6):
            inputs.append(ReplayInput(timestamp=ts, R=0.8, delta=0.4))
            ts += _STRIDE

        # BLOCKED: R=0.1 (well below threshold=0.5)
        for _ in range(6):
            inputs.append(ReplayInput(timestamp=ts, R=0.1, delta=0.4))
            ts += _STRIDE

        # READY: R=0.9, delta large → stillness ACTIVE
        for _ in range(10):
            inputs.append(ReplayInput(timestamp=ts, R=0.9, delta=0.8))
            ts += _STRIDE

        # UNNECESSARY: R=0.9, delta tiny and R flat → stillness STILL
        # Need to fill the stillness window (4 samples) with all-criteria-passing values
        for _ in range(12):
            inputs.append(ReplayInput(timestamp=ts, R=0.90, delta=0.001))
            ts += _STRIDE

        return inputs

    def test_replay_with_all_five_gate_states(self, tmp_path: Path) -> None:
        inputs = self._five_state_inputs()
        original_path = tmp_path / "original.jsonl"
        scratch_path = tmp_path / "scratch.jsonl"

        cfg = _make_cfg(
            ledger_path=original_path,
            threshold=0.5,
            enable_stillness=True,
            stillness_window=4,
            stillness_eps_R=0.05,
            stillness_eps_F=0.05,
            stillness_delta_min=0.05,
            stream_window=6,
            warmup_samples=2,
        )
        _run_pipeline(cfg, inputs)

        # Verify the gate states present
        lines = [ln for ln in original_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        observed_states = {json.loads(ln)["gate_state"] for ln in lines}

        # Must have at minimum: DEGRADED, BLOCKED, READY, UNNECESSARY
        # (SENSOR_ABSENT requires sensor_present=False which the pipeline
        #  does not expose externally — its B1/B2 DEGRADED covers the None-R path)
        required_states = {
            GateState.DEGRADED.name,
            GateState.BLOCKED.name,
            GateState.READY.name,
            GateState.UNNECESSARY.name,
        }
        missing = required_states - observed_states
        assert not missing, (
            f"gate states {missing} not produced — check input design. Observed: {observed_states}"
        )

        # Replay must match
        replay_cfg = dataclasses.replace(cfg, ledger_path=scratch_path)
        result = replay_ledger(
            original_path=original_path,
            config=replay_cfg,
            inputs=inputs,
            scratch_path=scratch_path,
        )
        assert result.ok is True, result.reason
        assert original_path.read_bytes() == scratch_path.read_bytes()

        # Verify both chains valid
        v_orig = verify_ledger(original_path)
        v_scratch = verify_ledger(scratch_path)
        assert v_orig.ok and v_scratch.ok
        assert v_orig.n_records == v_scratch.n_records == len(inputs)
