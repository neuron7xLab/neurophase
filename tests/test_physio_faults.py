"""Adversarial fault suite for the live physio path.

Every test spawns the consumer + ``tools/fault_producer.py`` as two
independent processes under a unique LSL stream name. The core
invariant being defended, across every fault class, is::

    NO FRAME event produced under a fault must EVER carry
    execution_allowed=True while the corresponding state is anything
    other than EXECUTE_ALLOWED.

Stricter still: several faults are expected to never admit an
EXECUTE_ALLOWED frame at all, and those tests assert that explicitly.

This suite is the adversarial counterpart to tests/test_physio_live.py:
that file proves the happy path works; this file proves the unhappy
paths fail closed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

import pytest

pytest.importorskip("pylsl")

_REPO_ROOT = Path(__file__).resolve().parents[1]

_READY_TIMEOUT_S = 20.0
_PROC_JOIN_TIMEOUT_S = 60.0


def _unique_stream_name(tag: str) -> str:
    return f"neurophase-rr-fault-{tag}-{uuid.uuid4().hex[:8]}"


def _spawn_module(module: str, args: Iterable[str]) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [sys.executable, "-m", module, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
    )


def _spawn_script(path: Path, args: Iterable[str]) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [sys.executable, str(path), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
    )


def _wait_for_event(
    proc: subprocess.Popen[str], event_name: str, *, timeout_s: float
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise AssertionError(f"process ended before {event_name!r}: exit={proc.returncode}")
            continue
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("event") == event_name:
            return evt
    raise AssertionError(f"timed out waiting for {event_name!r}")


def _drain_events(proc: subprocess.Popen[str], *, timeout_s: float) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    deadline = time.monotonic() + timeout_s
    assert proc.stdout is not None
    while proc.poll() is None and time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            continue
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise AssertionError(f"proc {proc.pid} did not exit within {timeout_s}s")
    tail = proc.stdout.read() if proc.stdout else ""
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


_FAULT_TOOL = _REPO_ROOT / "tools" / "fault_producer.py"


# =======================================================================
#   Cross-fault invariants (helper)
# =======================================================================


def _no_false_execute_allowed(events: list[dict[str, object]]) -> None:
    """The load-bearing invariant of the whole suite."""
    for e in events:
        if e.get("event") != "FRAME":
            continue
        gate_state = e.get("gate_state")
        allowed = e.get("execution_allowed")
        if allowed and gate_state != "EXECUTE_ALLOWED":
            raise AssertionError(
                f"fail-closed violation: execution_allowed=True with "
                f"gate_state={gate_state!r}; full event: {e!r}"
            )


# =======================================================================
#   Fault: flatline (identical RRs -> RMSSD = 0 -> SENSOR_DEGRADED)
# =======================================================================


class TestFlatline:
    def test_flatline_never_executes(self) -> None:
        stream = _unique_stream_name("flatline")
        consumer = _spawn_module(
            "neurophase.physio.live",
            ["--stream-name", stream, "--max-frames", "24", "--stall-timeout-s", "4.0"],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn_script(
                _FAULT_TOOL,
                ["--stream-name", stream, "--fault", "flatline", "--inter-sample-s", "0.02"],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            events = _drain_events(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        _no_false_execute_allowed(events)
        frames = [e for e in events if e.get("event") == "FRAME"]
        # Flatline should dominate the gate state after the warm-up.
        # No frame must be EXECUTE_ALLOWED at all.
        assert all(f.get("gate_state") != "EXECUTE_ALLOWED" for f in frames), (
            "flatline must never produce EXECUTE_ALLOWED"
        )
        # At least one SENSOR_DEGRADED frame (or all of them; either is fine).
        assert any(f.get("gate_state") == "SENSOR_DEGRADED" for f in frames)


# =======================================================================
#   Fault: spike_burst (impossible RRs mid-stream)
# =======================================================================


class TestSpikeBurst:
    def test_impossible_rrs_all_rejected(self) -> None:
        stream = _unique_stream_name("spike")
        consumer = _spawn_module(
            "neurophase.physio.live",
            ["--stream-name", stream, "--max-frames", "24", "--stall-timeout-s", "4.0"],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn_script(
                _FAULT_TOOL,
                [
                    "--stream-name",
                    stream,
                    "--fault",
                    "spike_burst",
                    "--inter-sample-s",
                    "0.02",
                    "--spike-count",
                    "5",
                ],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            events = _drain_events(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        _no_false_execute_allowed(events)
        rejected = [e for e in events if e.get("event") == "INGEST_REJECTED"]
        assert len(rejected) >= 5, (
            f"expected at least 5 INGEST_REJECTED events for a 5-sample spike burst; "
            f"got {len(rejected)}"
        )
        for r in rejected:
            reason = str(r.get("reason", ""))
            assert "envelope" in reason, reason


# =======================================================================
#   Fault: dup_timestamps (non-monotonic -> reject)
# =======================================================================


class TestDupTimestamps:
    def test_duplicates_force_rejections(self) -> None:
        stream = _unique_stream_name("dup")
        consumer = _spawn_module(
            "neurophase.physio.live",
            ["--stream-name", stream, "--max-frames", "32", "--stall-timeout-s", "4.0"],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn_script(
                _FAULT_TOOL,
                [
                    "--stream-name",
                    stream,
                    "--fault",
                    "dup_timestamps",
                    "--inter-sample-s",
                    "0.02",
                ],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            events = _drain_events(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        _no_false_execute_allowed(events)
        rejected = [e for e in events if e.get("event") == "INGEST_REJECTED"]
        assert any("non-monotonic" in str(e.get("reason", "")) for e in rejected), (
            f"expected a non-monotonic rejection, got {rejected!r}"
        )


# =======================================================================
#   Fault: dropped_packets (long stall)
# =======================================================================


class TestDroppedPackets:
    def test_stall_triggers_sensor_degraded(self) -> None:
        stream = _unique_stream_name("drop")
        consumer = _spawn_module(
            "neurophase.physio.live",
            [
                "--stream-name",
                stream,
                "--max-frames",
                "24",
                "--stall-timeout-s",
                "2.0",
                "--read-timeout-s",
                "0.25",
            ],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn_script(
                _FAULT_TOOL,
                [
                    "--stream-name",
                    stream,
                    "--fault",
                    "dropped_packets",
                    "--inter-sample-s",
                    "0.02",
                    "--stall-hold-s",
                    "3.0",
                ],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            events = _drain_events(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        _no_false_execute_allowed(events)
        stalls = [e for e in events if e.get("event") == "STALL"]
        assert stalls, "expected at least one STALL event during a 3 s pause"
        for s in stalls:
            assert s.get("gate_state") == "SENSOR_DEGRADED"
            assert s.get("execution_allowed") is False


# =======================================================================
#   Fault: delayed_outlet (stream never shows up in time)
# =======================================================================


class TestDelayedOutlet:
    def test_resolve_timeout_exits_fatal(self) -> None:
        stream = _unique_stream_name("delayed")
        # Tight resolve timeout so the consumer gives up BEFORE the
        # producer creates its outlet.
        consumer = _spawn_module(
            "neurophase.physio.live",
            [
                "--stream-name",
                stream,
                "--resolve-timeout-s",
                "1.5",
                "--stall-timeout-s",
                "3.0",
            ],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            # Producer sleeps 3 s before creating the outlet.
            producer = _spawn_script(
                _FAULT_TOOL,
                [
                    "--stream-name",
                    stream,
                    "--fault",
                    "delayed_outlet",
                    "--delay-s",
                    "3.0",
                ],
            )
            consumer.wait(timeout=15.0)
            assert consumer.returncode != 0, "consumer should have exited non-zero"
            # Clean up the producer so it doesn't linger.
            try:
                producer.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                producer.terminate()
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        tail = consumer.stdout.read() if consumer.stdout else ""
        assert "FATAL" in tail


# =======================================================================
#   Fault: abrupt_disconnect (no sentinel, peer exits)
# =======================================================================


class TestAbruptDisconnect:
    def test_abrupt_disconnect_never_executes_after_disconnect(self) -> None:
        """When the producer exits without a sentinel, the consumer
        must not emit EXECUTE_ALLOWED during the disconnect window.

        The abrupt producer pushes exactly 16 clean samples and then
        exits. The consumer is bounded with --max-frames 16 so it
        wraps cleanly after receiving the last sample; post-buffer
        stalls would otherwise loop forever (LSL has no "outlet gone"
        signal to the inlet)."""
        stream = _unique_stream_name("abrupt")
        consumer = _spawn_module(
            "neurophase.physio.live",
            [
                "--stream-name",
                stream,
                "--max-frames",
                "16",
                "--stall-timeout-s",
                "2.0",
                "--read-timeout-s",
                "0.25",
            ],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn_script(
                _FAULT_TOOL,
                [
                    "--stream-name",
                    stream,
                    "--fault",
                    "abrupt_disconnect",
                    "--inter-sample-s",
                    "0.02",
                ],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            events = _drain_events(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        _no_false_execute_allowed(events)
        # Producer exits 4 (EXIT_ABRUPT_REQUESTED); consumer exits 0
        # on its own after --max-frames. What matters for the fail-
        # closed contract is solely that no EXECUTE_ALLOWED leaked
        # under the fault condition.
        assert producer.returncode == 4
        frames = [e for e in events if e.get("event") == "FRAME"]
        # We only had 16 warm-up samples -> buffer fills on the last
        # frame; no EXECUTE_ALLOWED is reachable in that window.
        assert all(f.get("gate_state") != "EXECUTE_ALLOWED" for f in frames)


# =======================================================================
#   Global (clean) reference: fault_producer in "clean" mode passes.
# =======================================================================


class TestMalformedBurst:
    """Malformed payloads that pass LSL transport but explode the
    consumer's ingest validation. Distinct from spike_burst (which
    only injects out-of-envelope but otherwise well-formed RRs)."""

    def test_malformed_payloads_all_rejected(self) -> None:
        stream = _unique_stream_name("malformed")
        consumer = _spawn_module(
            "neurophase.physio.live",
            ["--stream-name", stream, "--max-frames", "30", "--stall-timeout-s", "4.0"],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn_script(
                _FAULT_TOOL,
                [
                    "--stream-name",
                    stream,
                    "--fault",
                    "malformed_burst",
                    "--inter-sample-s",
                    "0.02",
                ],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            events = _drain_events(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        _no_false_execute_allowed(events)
        rejected = [e for e in events if e.get("event") == "INGEST_REJECTED"]
        # Three structurally-malformed shapes injected: ts_inf, ts_negative,
        # rr_inf. Each must surface as exactly one INGEST_REJECTED.
        # ts_negative arrives via the monotonicity check; ts_inf and rr_inf
        # via the RRSample finite-checks. All three must reject.
        assert len(rejected) >= 3, (
            f"expected >= 3 INGEST_REJECTED events for malformed_burst; got {len(rejected)}"
        )
        reasons = " | ".join(str(r.get("reason", "")) for r in rejected)
        assert "not finite" in reasons or "non-monotonic" in reasons or "envelope" in reasons


class TestCleanReference:
    def test_clean_mode_matches_happy_path(self) -> None:
        """Sanity check: fault_producer --fault clean should behave
        identically to the happy-path live_producer. This anchors the
        fault tests against a known-good baseline produced by the same
        tool under test."""
        stream = _unique_stream_name("clean")
        consumer = _spawn_module(
            "neurophase.physio.live",
            ["--stream-name", stream, "--max-frames", "24", "--stall-timeout-s", "4.0"],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn_script(
                _FAULT_TOOL,
                ["--stream-name", stream, "--fault", "clean", "--inter-sample-s", "0.02"],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            events = _drain_events(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        _no_false_execute_allowed(events)
        assert consumer.returncode == 0
        frames = [e for e in events if e.get("event") == "FRAME"]
        assert len(frames) == 24
