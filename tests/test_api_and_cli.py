"""Tests for ``neurophase.api`` façade and ``neurophase.__main__`` CLI."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from neurophase import __version__
from neurophase.__main__ import main as cli_main

# ---------------------------------------------------------------------------
# Façade contract.
# ---------------------------------------------------------------------------


class TestFacadeContract:
    def test_facade_exports_minimal_set(self) -> None:
        import neurophase.api as api

        expected = {
            "DEFAULT_THRESHOLD",
            "ActionDecision",
            "ActionIntent",
            "ActionPolicy",
            "Contract",
            "DecisionExplanation",
            "DecisionFrame",
            "ExecutionGate",
            "ExplanationStep",
            "GateDecision",
            "GateState",
            "OrchestratedFrame",
            "OrchestratorConfig",
            "PipelineConfig",
            "PolicyConfig",
            "RegimeClassifier",
            "RegimeLabel",
            "RegimeState",
            "RegimeThresholds",
            "RegimeTransitionMatrix",
            "RegimeTransitionTracker",
            "RuntimeOrchestrator",
            "StillnessDetector",
            "StillnessState",
            "StreamingPipeline",
            "TimeQuality",
            "TransitionEvent",
            "Verdict",
            "__version__",
            "create_pipeline",
            "explain_decision",
            "explain_gate",
        }
        assert set(api.__all__) == expected

    def test_facade_version_matches_package(self) -> None:
        from neurophase.api import __version__ as api_version

        assert api_version == __version__

    def test_create_pipeline_returns_streaming_pipeline(self) -> None:
        from neurophase.api import PipelineConfig, StreamingPipeline, create_pipeline

        p = create_pipeline(threshold=0.65, warmup_samples=2)
        assert isinstance(p, StreamingPipeline)
        assert isinstance(p.config, PipelineConfig)
        assert p.config.threshold == 0.65
        assert p.config.warmup_samples == 2

    def test_facade_symbols_are_the_real_classes(self) -> None:
        """Every re-export must be identity-equal to its canonical module."""
        from neurophase import api
        from neurophase.explain import DecisionExplanation as _CanonExplain
        from neurophase.gate.execution_gate import GateState as _CanonGateState
        from neurophase.runtime.pipeline import (
            StreamingPipeline as _CanonPipeline,
        )

        assert api.DecisionExplanation is _CanonExplain
        assert api.GateState is _CanonGateState
        assert api.StreamingPipeline is _CanonPipeline


# ---------------------------------------------------------------------------
# CLI smoke tests.
# ---------------------------------------------------------------------------


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            rc = cli_main(argv)
        except SystemExit as exc:
            rc = int(exc.code if exc.code is not None else 0)
    return rc, stdout.getvalue(), stderr.getvalue()


class TestCLIVersion:
    def test_version_command(self) -> None:
        rc, out, _ = _run_cli(["version"])
        assert rc == 0
        assert __version__ in out


class TestCLIDemo:
    def test_demo_runs_and_prints_header(self) -> None:
        rc, out, _ = _run_cli(["demo", "--ticks", "8"])
        assert rc == 0
        assert "neurophase demo" in out
        assert "state" in out
        # 8 ticks → 8 data rows + header + separator
        lines = [line for line in out.splitlines() if line.strip() and not line.startswith("#")]
        # Header + separator + 8 data lines.
        assert len(lines) >= 9

    def test_demo_respects_tick_count(self) -> None:
        rc, out_small, _ = _run_cli(["demo", "--ticks", "4"])
        rc_large, out_large, _ = _run_cli(["demo", "--ticks", "32"])
        assert rc == 0 and rc_large == 0
        assert len(out_large.splitlines()) > len(out_small.splitlines())


class TestCLIVerifyLedger:
    def test_verify_clean_ledger(self, tmp_path: Path) -> None:
        from neurophase.api import create_pipeline

        ledger_path = tmp_path / "clean.jsonl"
        p = create_pipeline(warmup_samples=2, stream_window=4, ledger_path=ledger_path)
        for i in range(6):
            p.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.01)

        rc, out, _ = _run_cli(["verify-ledger", str(ledger_path)])
        assert rc == 0
        assert "verified" in out

    def test_verify_tampered_ledger(self, tmp_path: Path) -> None:
        from neurophase.api import create_pipeline

        ledger_path = tmp_path / "tampered.jsonl"
        p = create_pipeline(warmup_samples=2, stream_window=4, ledger_path=ledger_path)
        for i in range(4):
            p.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.01)

        # Tamper.
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
        payload = json.loads(lines[1])
        payload["reason"] = "TAMPERED"
        lines[1] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        ledger_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        rc, out, _ = _run_cli(["verify-ledger", str(ledger_path)])
        assert rc == 1
        assert "broken" in out


class TestCLIExplainLedger:
    def test_explain_ledger_outputs_jsonl(self, tmp_path: Path) -> None:
        from neurophase.api import create_pipeline

        ledger_path = tmp_path / "explain.jsonl"
        p = create_pipeline(warmup_samples=2, stream_window=4, ledger_path=ledger_path)
        for i in range(6):
            R = 0.9 if i < 4 else 0.30  # last two drop below threshold
            p.tick(timestamp=float(i) * 0.1, R=R, delta=0.01)

        rc, out, _ = _run_cli(["explain-ledger", str(ledger_path)])
        assert rc == 0
        lines = [line for line in out.splitlines() if line.strip()]
        assert len(lines) == 6
        # Every line is valid JSON with the expected fields.
        for line in lines:
            payload = json.loads(line)
            assert "final_state" in payload
            assert "causal_contract" in payload
            assert "chain" in payload

    def test_explain_ledger_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope.jsonl"
        rc, _, err = _run_cli(["explain-ledger", str(missing)])
        assert rc == 2
        assert "not found" in err
