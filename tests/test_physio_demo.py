"""Test that the one-command physio demo runs end-to-end against the bundled CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurophase.physio.demo import main as demo_main

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUNDLED_CSV = _REPO_ROOT / "examples" / "data" / "physio_replay_sample.csv"


def test_bundled_csv_exists() -> None:
    assert _BUNDLED_CSV.exists(), "bundled replay sample missing"


def test_demo_runs_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    rc = demo_main([])
    captured = capsys.readouterr().out
    assert rc == 0
    # Banner
    assert "NEUROPHASE" in captured
    assert "replay-only" in captured.lower()
    # All four physio states must be reachable by the bundled CSV.
    for state_name in (
        "SENSOR_DEGRADED",
        "EXECUTE_ALLOWED",
        "EXECUTE_REDUCED",
        "ABSTAIN",
    ):
        assert state_name in captured, f"state {state_name} not exercised by demo"


def test_demo_json_out(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "dump.json"
    rc = demo_main(["--json-out", str(out)])
    capsys.readouterr()  # drop stdout
    assert rc == 0
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "frames" in payload
    assert payload["summary"]["n_frames_emitted"] > 0
    assert payload["summary"]["n_frames_emitted"] == len(payload["frames"])


def test_demo_fails_closed_on_missing_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = demo_main(["--csv", str(tmp_path / "nope.csv")])
    captured = capsys.readouterr()
    # The demo must exit non-zero and print the fail-closed reason to stderr.
    assert rc != 0
    assert "fail-closed" in captured.err.lower() or "not found" in captured.err.lower()
