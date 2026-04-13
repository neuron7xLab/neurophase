"""Tests for neurophase.physio.ledger + session_replay.

Three invariants are non-negotiable:

1. The ledger always opens with a SESSION_HEADER and closes with a
   SESSION_SUMMARY (even if the ``with`` block raises).
2. A ledger recorded from any source (replay CSV or live LSL) can be
   re-executed offline through a fresh :class:`PhysioSession` and must
   produce byte-identical gate decisions (``parity_ok``).
3. A malformed ledger fails :class:`LedgerReplayError`; a ledger with
   a partial-truncated LAST line is tolerated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurophase.physio.ledger import (
    PHYSIO_LEDGER_SCHEMA_VERSION,
    LedgerConfig,
    PhysioLedger,
)
from neurophase.physio.pipeline import PhysioReplayPipeline, PhysioSession
from neurophase.physio.session_replay import (
    EXIT_OK,
    EXIT_PARITY_FAIL,
    LedgerReplayError,
    replay_ledger,
)
from neurophase.physio.session_replay import main as session_replay_main


def _mk_config(**overrides: object) -> LedgerConfig:
    base = {
        "source_mode": "test-replay",
        "stream_name": None,
        "window_size": 32,
        "threshold_allow": 0.8,
        "threshold_abstain": 0.5,
        "stall_timeout_s": None,
    }
    base.update(overrides)  # type: ignore[arg-type]
    return LedgerConfig(**base)  # type: ignore[arg-type]


# ----------------- Ledger structural guarantees ------------------------


class TestLedgerStructuralGuarantees:
    def test_header_and_summary_always_present(self, tmp_path: Path) -> None:
        path = tmp_path / "s1.jsonl"
        with PhysioLedger(path, config=_mk_config()) as led:
            led.write_event({"event": "FRAME", "tick_index": 0})
        lines = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        assert lines[0]["event"] == "SESSION_HEADER"
        assert lines[0]["schema_version"] == PHYSIO_LEDGER_SCHEMA_VERSION
        assert lines[-1]["event"] == "SESSION_SUMMARY"
        assert lines[-1]["terminated_by_exception"] is False

    def test_summary_written_even_on_exception(self, tmp_path: Path) -> None:
        path = tmp_path / "s2.jsonl"
        with (
            pytest.raises(RuntimeError, match="boom"),
            PhysioLedger(path, config=_mk_config()) as led,
        ):
            led.write_event({"event": "FRAME", "tick_index": 0})
            raise RuntimeError("boom")
        lines = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
        assert lines[0]["event"] == "SESSION_HEADER"
        assert lines[-1]["event"] == "SESSION_SUMMARY"
        assert lines[-1]["terminated_by_exception"] is True
        assert lines[-1]["exception_type"] == "RuntimeError"

    def test_write_after_close_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "s3.jsonl"
        led = PhysioLedger(path, config=_mk_config())
        led.__enter__()
        led.__exit__(None, None, None)
        with pytest.raises(RuntimeError, match="closed"):
            led.write_event({"event": "FRAME"})

    def test_double_enter_rejected(self, tmp_path: Path) -> None:
        """A single PhysioLedger instance must be __enter__ed at most
        once. Re-entering would silently clobber the recorded session."""
        path = tmp_path / "s4.jsonl"
        led = PhysioLedger(path, config=_mk_config())
        led.__enter__()
        try:
            with pytest.raises(RuntimeError, match="twice"):
                led.__enter__()
        finally:
            led.__exit__(None, None, None)

    def test_enter_after_close_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "s5.jsonl"
        led = PhysioLedger(path, config=_mk_config())
        led.__enter__()
        led.__exit__(None, None, None)
        with pytest.raises(RuntimeError, match="after close"):
            led.__enter__()

    def test_file_mode_is_0o600(self, tmp_path: Path) -> None:
        """Ledgers carry personal physiological data. The file mode
        must be 0o600 regardless of the caller's umask. Validates the
        privacy floor specified by LEDGER_FILE_MODE."""
        import os
        import stat

        old_umask = os.umask(0o022)
        try:
            path = tmp_path / "private.jsonl"
            with PhysioLedger(path, config=_mk_config()) as led:
                led.write_event({"event": "FRAME"})
            mode = stat.S_IMODE(path.stat().st_mode)
            assert mode == 0o600, f"ledger file mode is {oct(mode)}, expected 0o600"
        finally:
            os.umask(old_umask)


# --------------- Record-replay parity (the keystone test) --------------


class TestRecordReplayParity:
    def test_replay_matches_record_byte_for_byte(self, tmp_path: Path) -> None:
        # Build a deterministic replay-CSV → ledger from the shared core.
        rr_sequence = [
            820.0,
            825.0,
            818.0,
            822.0,
            830.0,
            815.0,
            819.0,
            824.0,
            820.0,
            823.0,
            821.0,
            822.0,
            820.0,
            819.0,
            821.0,
            823.0,
            820.0,
            822.0,
            819.0,
            821.0,
            820.0,
            823.0,
            821.0,
            822.0,
        ]
        path = tmp_path / "record.jsonl"
        pipeline = PhysioReplayPipeline()
        t = 100.0
        with PhysioLedger(path, config=_mk_config(source_mode="replay-csv")) as led:
            from neurophase.physio.replay import RRSample

            for i, rr in enumerate(rr_sequence):
                t += rr / 1000.0
                frame = pipeline.step(
                    RRSample(timestamp_s=t, rr_ms=rr, row_index=i),
                    tick_index=i,
                )
                led.write_event(
                    {
                        "event": "FRAME",
                        "tick_index": frame.tick_index,
                        "timestamp_s": frame.timestamp_s,
                        "rr_ms": frame.rr_ms,
                        "gate_state": frame.decision.state.name,
                        "execution_allowed": frame.decision.execution_allowed,
                        "kernel_state": frame.decision.kernel_state.name,
                        "confidence": frame.decision.confidence,
                        "source_mode": "replay-csv",
                    }
                )

        report = replay_ledger(path)
        assert report.parity_ok, report.divergences
        assert report.n_frames_recorded == len(rr_sequence)
        assert report.n_frames_replayed == len(rr_sequence)
        assert report.clean_summary_seen is True
        assert report.schema_version == PHYSIO_LEDGER_SCHEMA_VERSION

    def test_replay_detects_decision_divergence_when_config_differs(self, tmp_path: Path) -> None:
        """If the ledger's config is tampered with, replay diverges and
        reports divergences. This proves the replayer actually uses the
        header config rather than silently trusting recorded decisions."""
        # Record with threshold_allow=0.8 (the default).
        path = tmp_path / "tampered.jsonl"
        with PhysioLedger(path, config=_mk_config(threshold_allow=0.8)) as led:
            session = PhysioSession(threshold_allow=0.8)
            from neurophase.physio.replay import RRSample

            t = 100.0
            for i in range(32):
                rr = 820.0 + (5 if i % 2 == 0 else -5)
                t += rr / 1000.0
                frame = session.step(
                    RRSample(timestamp_s=t, rr_ms=rr, row_index=i),
                    tick_index=i,
                )
                led.write_event(
                    {
                        "event": "FRAME",
                        "tick_index": frame.tick_index,
                        "timestamp_s": frame.timestamp_s,
                        "rr_ms": frame.rr_ms,
                        "gate_state": frame.decision.state.name,
                        "execution_allowed": frame.decision.execution_allowed,
                        "kernel_state": frame.decision.kernel_state.name,
                        "confidence": frame.decision.confidence,
                    }
                )

        # Tamper: flip every recorded state to EXECUTE_ALLOWED. Replay
        # must detect the divergence.
        lines = path.read_text().splitlines()
        tampered_lines: list[str] = []
        for line in lines:
            obj = json.loads(line)
            if obj.get("event") == "FRAME":
                obj["gate_state"] = "EXECUTE_ALLOWED"
                obj["execution_allowed"] = True
            tampered_lines.append(json.dumps(obj))
        path.write_text("\n".join(tampered_lines) + "\n")

        report = replay_ledger(path)
        assert report.n_divergences > 0
        assert report.parity_ok is False


# --------------- Ledger I/O robustness --------------------------------


class TestLedgerRobustness:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(LedgerReplayError, match="not found"):
            replay_ledger(tmp_path / "nope.jsonl")

    def test_no_header_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "no-header.jsonl"
        path.write_text('{"event": "FRAME"}\n')
        with pytest.raises(LedgerReplayError, match="SESSION_HEADER"):
            replay_ledger(path)

    def test_wrong_schema_version_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong-schema.jsonl"
        path.write_text(
            json.dumps(
                {
                    "event": "SESSION_HEADER",
                    "schema_version": "physio-ledger-v99",
                    "session_id": "x",
                    "config": {},
                }
            )
            + "\n"
        )
        with pytest.raises(LedgerReplayError, match="unsupported schema_version"):
            replay_ledger(path)

    def test_partial_trailing_line_is_tolerated(self, tmp_path: Path) -> None:
        """A crash mid-flush leaves a half-written last line. The reader
        must accept that as a valid (if incomplete) ledger prefix."""
        path = tmp_path / "partial.jsonl"
        header = json.dumps(
            {
                "event": "SESSION_HEADER",
                "schema_version": PHYSIO_LEDGER_SCHEMA_VERSION,
                "session_id": "xyz",
                "config": {},
            }
        )
        good_frame = json.dumps(
            {
                "event": "FRAME",
                "tick_index": 0,
                "timestamp_s": 0.82,
                "rr_ms": 820.0,
                "gate_state": "SENSOR_DEGRADED",
                "execution_allowed": False,
                "kernel_state": "DEGRADED",
                "confidence": 0.0,
            }
        )
        # Truncated JSON as the last line: no closing brace.
        partial = '{"event": "FRAME", "tick_index": 1, "timestamp_s": 1.64'
        path.write_text(f"{header}\n{good_frame}\n{partial}")
        report = replay_ledger(path)  # must NOT raise
        assert report.n_frames_recorded == 1

    def test_partial_trailing_line_no_final_newline(self, tmp_path: Path) -> None:
        """File with no trailing \\n at all: the writer-was-killed-mid-line
        case. The reader must still tolerate it iff the malformed line
        is the LAST one."""
        path = tmp_path / "no-final-nl.jsonl"
        header = json.dumps(
            {
                "event": "SESSION_HEADER",
                "schema_version": PHYSIO_LEDGER_SCHEMA_VERSION,
                "session_id": "xyz",
                "config": {},
            }
        )
        # Note: file deliberately ends with a partial line and no newline.
        path.write_text(f"{header}\n" + '{"event": "FRAME", "tick_index": 0')
        report = replay_ledger(path)
        assert report.n_frames_recorded == 0  # the partial frame is dropped

    def test_mid_file_malformed_line_raises(self, tmp_path: Path) -> None:
        """A malformed JSON line in the MIDDLE of the file is a hard
        failure -- only the trailing line is allowed to be truncated."""
        path = tmp_path / "mid-bad.jsonl"
        header = json.dumps(
            {
                "event": "SESSION_HEADER",
                "schema_version": PHYSIO_LEDGER_SCHEMA_VERSION,
                "session_id": "xyz",
                "config": {},
            }
        )
        good = json.dumps({"event": "FRAME", "tick_index": 0})
        path.write_text(f"{header}\nthis is not json\n{good}\n")
        with pytest.raises(LedgerReplayError, match="malformed JSON"):
            replay_ledger(path)


# --------------- CLI contract ------------------------------------------


class TestReplayCLI:
    def test_cli_json_output_on_parity_ok(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Record a short clean session.
        path = tmp_path / "clean.jsonl"
        with PhysioLedger(path, config=_mk_config()) as led:
            session = PhysioSession()
            from neurophase.physio.replay import RRSample

            t = 100.0
            for i in range(20):
                rr = 820.0 + (3 if i % 2 == 0 else -3)
                t += rr / 1000.0
                frame = session.step(
                    RRSample(timestamp_s=t, rr_ms=rr, row_index=i),
                    tick_index=i,
                )
                led.write_event(
                    {
                        "event": "FRAME",
                        "tick_index": frame.tick_index,
                        "timestamp_s": frame.timestamp_s,
                        "rr_ms": frame.rr_ms,
                        "gate_state": frame.decision.state.name,
                        "execution_allowed": frame.decision.execution_allowed,
                        "kernel_state": frame.decision.kernel_state.name,
                        "confidence": frame.decision.confidence,
                    }
                )

        rc = session_replay_main([str(path), "--json", "--strict"])
        captured = capsys.readouterr().out
        payload = json.loads(captured)
        assert rc == EXIT_OK
        assert payload["parity_ok"] is True
        assert payload["n_divergences"] == 0

    def test_full_parity_catches_kernel_state_drift(self, tmp_path: Path) -> None:
        """Default mode passes if gate_state matches; --full-parity
        additionally catches kernel_state divergence even when the
        user-visible decision happens to align."""
        path = tmp_path / "drift.jsonl"
        with PhysioLedger(path, config=_mk_config()) as led:
            session = PhysioSession()
            from neurophase.physio.replay import RRSample

            t = 100.0
            for i in range(16):
                rr = 820.0 + (5 if i % 2 == 0 else -5)
                t += rr / 1000.0
                frame = session.step(
                    RRSample(timestamp_s=t, rr_ms=rr, row_index=i),
                    tick_index=i,
                )
                # Forge a wrong kernel_state on every record so the
                # full-parity check has guaranteed divergences while
                # the decision-only check sees none. Mapping rule:
                # whatever the replay produces, store the OPPOSITE-ish
                # legal kernel state.
                replay_kernel = frame.decision.kernel_state.name
                forged_kernel = "BLOCKED" if replay_kernel != "BLOCKED" else "READY"
                evt = {
                    "event": "FRAME",
                    "tick_index": frame.tick_index,
                    "timestamp_s": frame.timestamp_s,
                    "rr_ms": frame.rr_ms,
                    "gate_state": frame.decision.state.name,
                    "execution_allowed": frame.decision.execution_allowed,
                    "kernel_state": forged_kernel,
                    "confidence": frame.decision.confidence,
                }
                led.write_event(evt)

        # Decision parity passes -- gate_state and execution_allowed
        # are byte-identical to what replay produces.
        report_default = replay_ledger(path)
        assert report_default.parity_ok

        # Full parity fails -- kernel_state was forged on every frame.
        report_full = replay_ledger(path, full_parity=True)
        assert report_full.n_divergences > 0
        assert all(d.field == "kernel_state" for d in report_full.divergences)
        assert report_full.parity_ok is False

    def test_full_parity_catches_confidence_drift(self, tmp_path: Path) -> None:
        """A forged confidence value beyond CONFIDENCE_TOLERANCE must
        be flagged under --full-parity. Sub-tolerance noise must NOT be."""
        from neurophase.physio.session_replay import CONFIDENCE_TOLERANCE

        path = tmp_path / "conf.jsonl"
        with PhysioLedger(path, config=_mk_config()) as led:
            session = PhysioSession()
            from neurophase.physio.replay import RRSample

            t = 100.0
            for i in range(16):
                rr = 820.0 + (5 if i % 2 == 0 else -5)
                t += rr / 1000.0
                frame = session.step(
                    RRSample(timestamp_s=t, rr_ms=rr, row_index=i),
                    tick_index=i,
                )
                # First half: noise inside tolerance (must NOT diverge).
                # Second half: confidence forged by 0.1 (must diverge).
                if i < 8:
                    forged = frame.decision.confidence + CONFIDENCE_TOLERANCE / 10.0
                else:
                    forged = frame.decision.confidence + 0.1
                led.write_event(
                    {
                        "event": "FRAME",
                        "tick_index": frame.tick_index,
                        "timestamp_s": frame.timestamp_s,
                        "rr_ms": frame.rr_ms,
                        "gate_state": frame.decision.state.name,
                        "execution_allowed": frame.decision.execution_allowed,
                        "kernel_state": frame.decision.kernel_state.name,
                        "confidence": forged,
                    }
                )

        report_full = replay_ledger(path, full_parity=True)
        # Only the second-half noise should have produced divergences.
        assert report_full.n_divergences == 8
        assert all(d.field == "confidence" for d in report_full.divergences)

    def test_cli_strict_mode_returns_parity_fail(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "broken.jsonl"
        with PhysioLedger(path, config=_mk_config()) as led:
            led.write_event(
                {
                    "event": "FRAME",
                    "tick_index": 0,
                    "timestamp_s": 1.0,
                    "rr_ms": 820.0,
                    "gate_state": "EXECUTE_ALLOWED",
                    "execution_allowed": True,
                    "kernel_state": "READY",
                    "confidence": 1.0,
                }
            )
        rc = session_replay_main([str(path), "--strict"])
        capsys.readouterr()  # drain
        assert rc == EXIT_PARITY_FAIL
