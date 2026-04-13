"""Hardware-gated Layer-C / Layer-D formalisation for Polar H10.

**This file is never run in the default pytest suite.** It is the
pytest-side formalisation of what an operator MUST verify once a real
Polar H10 + BT adapter is available. The marker ``@pytest.mark.hardware``
keeps the tests addressable (discoverable, runnable individually) but
excluded from CI by the ``addopts`` in ``pyproject.toml``.

To run on a real device::

    pytest tests/test_hardware_polar.py -q -m hardware

Each test maps to one layer-C/D contract from
``docs/LIVE_SESSION_PROTOCOL.md`` and the v1.1 Polar producer brief.
If hardware is unavailable, each test fails fast with a clear reason
— it does not hang.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.hardware

pytest.importorskip("pylsl")
pytest.importorskip("bleak")


_REPO_ROOT = Path(__file__).resolve().parents[1]
_POLAR = _REPO_ROOT / "tools" / "polar_producer.py"


def _load_polar() -> object:
    spec = importlib.util.spec_from_file_location("polar_producer_hw", _POLAR)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["polar_producer_hw"] = mod
    spec.loader.exec_module(mod)
    return mod


# =======================================================================
#   Layer C: BLE discovery + connect + characteristic check
# =======================================================================


class TestLayerCDiscovery:
    """Requires a powered Bluetooth adapter on the host."""

    def test_scan_does_not_raise(self) -> None:
        """Scanning must resolve cleanly to either a device list or a
        concrete SCAN_FAILED / NO_DEVICE signal. It must never raise
        a bare Python exception up the stack."""
        import asyncio

        pp = _load_polar()
        # Run the tool's discovery path; expect a DeviceTarget or None,
        # never a traceback. Short timeout so the test is bounded.
        result = asyncio.run(
            pp.discover_polar_h10(  # type: ignore[attr-defined]
                address=None, name_filter="Polar H10", scan_timeout_s=6.0
            )
        )
        # Either a concrete target was returned (hardware present) or
        # None was returned (documented unavailable path). Both are ok.
        assert result is None or (hasattr(result, "address") and hasattr(result, "name"))


# =======================================================================
#   Layer D: full producer subprocess + live consumer + ledger + replay
# =======================================================================


class TestLayerDLiveChain:
    """Requires a real Polar H10 + BT adapter. Verifies the entire
    real-hardware live chain end-to-end."""

    def test_producer_subprocess_connects_and_emits_rr(self, tmp_path: Path) -> None:
        """Spawn tools/polar_producer.py as a subprocess and the kernel
        live consumer in another. Require at least one RR_EMIT event
        on the producer's stdout AND one FRAME event on the consumer's
        stdout within 60 s of start-up."""
        stream = f"polar-h10-{uuid.uuid4().hex[:6]}"
        ledger = tmp_path / "hw.jsonl"
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"

        consumer = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "neurophase.physio.live",
                "--stream-name",
                stream,
                "--stall-timeout-s",
                "10.0",
                "--ledger-out",
                str(ledger),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(_REPO_ROOT),
        )
        try:
            # Wait for LISTENING.
            deadline = time.monotonic() + 20.0
            assert consumer.stdout is not None
            while time.monotonic() < deadline:
                line = consumer.stdout.readline()
                if not line:
                    continue
                try:
                    evt = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if evt.get("event") == "LISTENING":
                    break
            else:
                raise AssertionError("consumer never emitted LISTENING")

            producer = subprocess.Popen(
                [
                    sys.executable,
                    str(_POLAR),
                    "--stream-name",
                    stream,
                    "--scan-timeout",
                    "15.0",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=str(_REPO_ROOT),
            )

            # Observe both streams for up to 60 s, looking for evidence
            # of real-hardware RR flow.
            saw_rr_emit = False
            saw_frame = False
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                for proc, flag_attr in (
                    (producer, "saw_rr_emit"),
                    (consumer, "saw_frame"),
                ):
                    assert proc.stdout is not None
                    line = proc.stdout.readline()
                    if not line:
                        continue
                    try:
                        evt = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    if flag_attr == "saw_rr_emit" and evt.get("event") == "RR_EMIT":
                        saw_rr_emit = True
                    if flag_attr == "saw_frame" and evt.get("event") == "FRAME":
                        saw_frame = True
                if saw_rr_emit and saw_frame:
                    break

            assert saw_rr_emit, "producer emitted no RR_EMIT events within 60 s"
            assert saw_frame, "consumer emitted no FRAME events within 60 s"

            # Clean shutdown of the producer via SIGINT / terminate.
            producer.terminate()
            producer.wait(timeout=10.0)
        finally:
            if consumer.poll() is None:
                consumer.terminate()
                consumer.wait(timeout=10.0)

        # Ledger must exist and be replayable.
        assert ledger.exists(), "consumer did not write the ledger"
        from neurophase.physio.session_replay import replay_ledger

        report = replay_ledger(ledger)
        assert report.parity_ok


# =======================================================================
#   Parser self-test (no hardware) — runs under `-m hardware` too as a
#   fail-fast sanity precondition
# =======================================================================


class TestPolarSelfTest:
    """Runs the tool's own self-test as a precondition sanity check.
    If this fails under ``-m hardware``, there is no point continuing
    to the BLE tests."""

    def test_polar_producer_selftest_passes(self) -> None:
        rc = subprocess.call(
            [sys.executable, str(_POLAR), "--self-test"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert rc == 0
