"""Consolidated parametrised parity matrix for session_replay.

session_replay has two parity modes and three tolerance / tolerance-
adjacent dimensions. Individual invariants are still tested in
``test_physio_ledger.py``; this file is the **matrix** -- one parametrised
axis for each dimension, so a whole column of behaviour is either green
or red at a glance.

Axes:

  * mode    : {"decision_only", "full_parity"}
  * forge   : {"none", "kernel_state", "confidence_within_tol",
               "confidence_above_tol"}
  * file    : {"clean", "partial_last_line", "no_final_newline",
               "mid_file_malformed", "only_header"}

Expected truth table (one row per interesting cell):

  +-------+------+----------------+------------------+---------+
  | mode  | forge| file           | expected         | n_div   |
  +-------+------+----------------+------------------+---------+
  | any   | none | clean          | parity_ok        | 0       |
  | any   | none | partial_last   | parity_ok        | 0       |
  | any   | none | no_final_nl    | parity_ok        | 0       |
  | any   | none | only_header    | empty ledger raise |       |
  | any   | none | mid_file_bad   | raise            |         |
  | d_only| kernel_forged | clean | parity_ok        | 0       |
  | full  | kernel_forged | clean | NOT parity_ok    | >= 1    |
  | d_only| conf_above    | clean | parity_ok        | 0       |
  | full  | conf_within   | clean | parity_ok        | 0       |
  | full  | conf_above    | clean | NOT parity_ok    | >= 1    |
  +-------+------+----------------+------------------+---------+

Ledger header + writing helpers are shared; only the matrix parameters
differ across tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurophase.physio.ledger import (
    PHYSIO_LEDGER_SCHEMA_VERSION,
    LedgerConfig,
    PhysioLedger,
)
from neurophase.physio.pipeline import PhysioSession
from neurophase.physio.replay import RRSample
from neurophase.physio.session_replay import (
    CONFIDENCE_TOLERANCE,
    LedgerReplayError,
    replay_ledger,
)


def _mk_config() -> LedgerConfig:
    return LedgerConfig(
        source_mode="matrix-test",
        stream_name=None,
        window_size=32,
        threshold_allow=0.8,
        threshold_abstain=0.5,
        stall_timeout_s=None,
    )


def _write_clean_ledger(path: Path, *, forge: str) -> None:
    """Write a well-formed ledger optionally with a named 'forge' applied
    to every FRAME event, which the session_replay reader should catch
    under the appropriate parity mode."""
    with PhysioLedger(path, config=_mk_config()) as led:
        session = PhysioSession()
        t = 100.0
        for i in range(16):
            rr = 820.0 + (5 if i % 2 == 0 else -5)
            t += rr / 1000.0
            frame = session.step(
                RRSample(timestamp_s=t, rr_ms=rr, row_index=i),
                tick_index=i,
            )
            evt = {
                "event": "FRAME",
                "tick_index": frame.tick_index,
                "timestamp_s": frame.timestamp_s,
                "rr_ms": frame.rr_ms,
                "gate_state": frame.decision.state.name,
                "execution_allowed": frame.decision.execution_allowed,
                "kernel_state": frame.decision.kernel_state.name,
                "confidence": frame.decision.confidence,
            }
            if forge == "kernel_state":
                replay_kernel = frame.decision.kernel_state.name
                evt["kernel_state"] = "BLOCKED" if replay_kernel != "BLOCKED" else "READY"
            elif forge == "confidence_within_tol":
                evt["confidence"] = frame.decision.confidence + CONFIDENCE_TOLERANCE / 10.0
            elif forge == "confidence_above_tol":
                evt["confidence"] = frame.decision.confidence + 0.1
            led.write_event(evt)


def _truncate_last_byte(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    # Drop the last closing brace of the final JSON line so it becomes
    # syntactically malformed.
    assert raw.endswith("}\n")
    path.write_text(raw[:-2] + "\n", encoding="utf-8")


def _strip_final_newline(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    path.write_text(raw.rstrip("\n"), encoding="utf-8")


def _inject_mid_file_garbage(path: Path) -> None:
    """Insert 'this is not json' as line 2 of the ledger."""
    lines = path.read_text(encoding="utf-8").splitlines()
    # line 0 is SESSION_HEADER; inject garbage as line 1 (before the first FRAME).
    lines.insert(1, "this is not json")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _only_header(path: Path) -> None:
    """Rewrite the file to contain only the SESSION_HEADER line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    # Find and keep only the SESSION_HEADER line.
    header_line = next(ln for ln in lines if '"SESSION_HEADER"' in ln)
    path.write_text(header_line + "\n", encoding="utf-8")


# =======================================================================
#   MATRIX: forge x mode on a clean file
# =======================================================================


FORGE_MODE_MATRIX = [
    # forge,                      full_parity,  expected_parity_ok
    ("none", False, True),
    ("none", True, True),
    ("kernel_state", False, True),  # decision-only still ok
    ("kernel_state", True, False),  # full catches it
    ("confidence_within_tol", False, True),
    ("confidence_within_tol", True, True),  # sub-tolerance noise tolerated
    ("confidence_above_tol", False, True),  # decision-only doesn't look at conf
    ("confidence_above_tol", True, False),  # above tolerance -> diverges
]


@pytest.mark.parametrize("forge, full_parity, expected_parity_ok", FORGE_MODE_MATRIX)
def test_forge_mode_matrix(
    tmp_path: Path, forge: str, full_parity: bool, expected_parity_ok: bool
) -> None:
    path = tmp_path / f"matrix-{forge}-{full_parity}.jsonl"
    _write_clean_ledger(path, forge=forge)
    report = replay_ledger(path, full_parity=full_parity)
    assert report.parity_ok is expected_parity_ok, (
        f"forge={forge!r} full_parity={full_parity!r} -> parity_ok={report.parity_ok!r} "
        f"(expected {expected_parity_ok!r}); divergences={report.divergences!r}"
    )
    # If parity FAILS, we must have at least one divergence (not a count bug).
    if not expected_parity_ok:
        assert report.n_divergences >= 1


# =======================================================================
#   MATRIX: file shape (clean / partial / no-final-nl / mid-bad / header-only)
# =======================================================================


def test_file_clean_ok(tmp_path: Path) -> None:
    path = tmp_path / "clean.jsonl"
    _write_clean_ledger(path, forge="none")
    report = replay_ledger(path)
    assert report.parity_ok is True
    assert report.n_frames_recorded == 16
    assert report.clean_summary_seen is True


def test_file_partial_last_line_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    _write_clean_ledger(path, forge="none")
    _truncate_last_byte(path)
    report = replay_ledger(path)
    # Last line truncated -- reader tolerates the prefix.
    assert report.parity_ok is True
    # The SUMMARY was the last line; truncating it drops it.
    assert report.clean_summary_seen is False


def test_file_no_final_newline_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "no-nl.jsonl"
    _write_clean_ledger(path, forge="none")
    _strip_final_newline(path)
    report = replay_ledger(path)
    assert report.parity_ok is True


def test_file_mid_file_malformed_raises(tmp_path: Path) -> None:
    path = tmp_path / "mid-bad.jsonl"
    _write_clean_ledger(path, forge="none")
    _inject_mid_file_garbage(path)
    with pytest.raises(LedgerReplayError, match="malformed JSON"):
        replay_ledger(path)


def test_file_header_only_raises(tmp_path: Path) -> None:
    path = tmp_path / "header-only.jsonl"
    _write_clean_ledger(path, forge="none")
    _only_header(path)
    report = replay_ledger(path)
    # A header-only ledger has no frames -- nothing to replay, parity
    # is trivially ok, but clean_summary_seen is False.
    assert report.n_frames_recorded == 0
    assert report.clean_summary_seen is False
    assert report.schema_version == PHYSIO_LEDGER_SCHEMA_VERSION


# =======================================================================
#   CLI --full-parity matrix
# =======================================================================


@pytest.mark.parametrize(
    "forge, full, expect_rc",
    [
        ("none", True, 0),
        ("none", False, 0),
        ("kernel_state", True, 1),  # parity fail
        ("kernel_state", False, 0),  # decision-only ok
        ("confidence_within_tol", True, 0),
        ("confidence_above_tol", True, 1),  # parity fail
    ],
)
def test_cli_strict_full_parity_matrix(
    tmp_path: Path, forge: str, full: bool, expect_rc: int
) -> None:
    from neurophase.physio.session_replay import main as session_replay_main

    path = tmp_path / f"cli-{forge}-{full}.jsonl"
    _write_clean_ledger(path, forge=forge)
    argv = [str(path), "--strict"]
    if full:
        argv.append("--full-parity")
    rc = session_replay_main(argv)
    assert rc == expect_rc, f"forge={forge!r} full={full!r} expected rc={expect_rc}, got {rc}"
