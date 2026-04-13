"""Fresh-clone acceptance as a testable artifact.

The QUICKSTART + release notes claim a specific operator chain works
on a fresh checkout:

    python -m neurophase.physio.demo          # replay-from-CSV demo
    python -m neurophase.physio.live + producer + --ledger-out
    python -m neurophase.physio.session_replay --strict

Previously this was verified by hand (commit log + manual fresh clone).
This file turns that verification into a **pytest artifact** so the
repo mechanically re-proves its own liftability on every CI run.

The test is marked ``@pytest.mark.acceptance`` and excluded from the
default suite (see ``addopts`` in pyproject.toml). CI runs it as a
separate step; locally, use::

    pytest tests/test_acceptance_chain.py -q -m acceptance

Strategy (no network, no kernel re-install):

  1. ``git clone`` from the local working tree into a tmp dir — the
     freshest possible fresh clone.
  2. Run each chain step as a subprocess using the *parent* process's
     interpreter + ``PYTHONPATH=<clone-path>`` so we exercise the
     clone's source, not the installed kernel.
  3. Assert exit codes and key events on each step.

If any step exits non-zero or its stdout does not match the expected
shape, the acceptance chain is broken and CI goes red **before** a
release tag is cut.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.acceptance

pytest.importorskip("pylsl")

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _clone_into(tmp_path: Path) -> Path:
    """git clone the current working tree into tmp_path / 'clone'."""
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "--quiet", str(_REPO_ROOT), str(clone)],
        check=True,
        cwd=str(tmp_path),
    )
    return clone


def _env_for_clone(clone: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(clone) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _wait_for_event(proc: subprocess.Popen[str], event: str, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise AssertionError(f"proc exited before {event!r}")
            continue
        try:
            if json.loads(line.strip()).get("event") == event:
                return
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"timed out waiting for {event!r}")


# =======================================================================
#   Step A: replay demo
# =======================================================================


def test_acceptance_replay_demo(tmp_path: Path) -> None:
    clone = _clone_into(tmp_path)
    env = _env_for_clone(clone)
    rc = subprocess.call(
        [sys.executable, "-m", "neurophase.physio.demo"],
        cwd=str(clone),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert rc == 0


# =======================================================================
#   Step B: live loopback + ledger round-trip
# =======================================================================


def test_acceptance_live_loopback_with_ledger_and_replay(tmp_path: Path) -> None:
    clone = _clone_into(tmp_path)
    env = _env_for_clone(clone)
    stream = f"acceptance-{uuid.uuid4().hex[:8]}"
    ledger = tmp_path / "acceptance.jsonl"

    consumer = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "neurophase.physio.live",
            "--stream-name",
            stream,
            "--max-frames",
            "24",
            "--stall-timeout-s",
            "4.0",
            "--ledger-out",
            str(ledger),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(clone),
        env=env,
    )
    try:
        _wait_for_event(consumer, "LISTENING", timeout_s=20.0)
        producer = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "neurophase.physio.live_producer",
                "--stream-name",
                stream,
                "--inter-sample-s",
                "0.02",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(clone),
            env=env,
        )
        producer.wait(timeout=45.0)
        consumer.wait(timeout=30.0)
        assert producer.returncode == 0
        assert consumer.returncode == 0
    finally:
        if consumer.poll() is None:
            consumer.terminate()

    # Step C: replay --strict on the produced ledger.
    rc = subprocess.call(
        [
            sys.executable,
            "-m",
            "neurophase.physio.session_replay",
            str(ledger),
            "--strict",
        ],
        cwd=str(clone),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert rc == 0
    # Full-parity strict must also pass on a fresh live session.
    rc_full = subprocess.call(
        [
            sys.executable,
            "-m",
            "neurophase.physio.session_replay",
            str(ledger),
            "--strict",
            "--full-parity",
        ],
        cwd=str(clone),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert rc_full == 0


# =======================================================================
#   Step D: contract check
# =======================================================================


def test_acceptance_polar_contract_check(tmp_path: Path) -> None:
    clone = _clone_into(tmp_path)
    env = _env_for_clone(clone)
    rc = subprocess.call(
        [sys.executable, str(clone / "tools" / "check_contract.py")],
        cwd=str(clone),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert rc == 0
