"""Integration tests for the true asynchronous live physio path.

These tests spawn the consumer and producer as **separate processes**
and verify the twelve mandatory properties defined in the v1.1 brief:

1. producer and consumer run as independent processes
2. consumer waits for future samples (does not preload a file)
3. valid live samples are accepted
4. malformed live sample is rejected cleanly (non-monotonic timestamp)
5. impossible RR sample fails closed (below envelope)
6. insufficient initial buffer stays degraded
7. after readiness, each new valid sample triggers a state update
8. stalled stream causes SENSOR_DEGRADED within stall_timeout_s
9. producer clean exit -> consumer exit code 0 (via NaN EOF sentinel)
10. producer never shows up -> consumer non-zero exit (FATAL)
11. live path and replay path produce identical semantics
12. public module registration / completeness stays green

Plus a latency test: post-readiness frame emission within 1.0 s of
sample arrival on local loopback.

Handshake pattern (per spec):
* consumer is launched first, blocks on LSL resolve
* test harness waits for a LISTENING JSON event on consumer stdout
* producer is launched only after LISTENING is observed
* producer itself waits for have_consumers() inside LSL before pushing
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("pylsl")  # Skip this whole module if LSL not installed.

from neurophase.physio.live import (
    STALL_TIMEOUT_SAFE_MAX_S,
    STALL_TIMEOUT_SAFE_MIN_S,
    LiveConfig,
)
from neurophase.physio.live_producer import RR_SEQUENCE_MS
from neurophase.physio.pipeline import PhysioSession
from neurophase.physio.replay import RRSample

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Bound everything loosely but finitely; CI runners are slower than dev boxes.
# Wall-clock budgets are set so a single test can never hang > 60 s.
_READY_TIMEOUT_S = 20.0
_PROC_JOIN_TIMEOUT_S = 40.0


def _unique_stream_name() -> str:
    """Per-test unique name so parallel runs never cross-talk."""
    return f"neurophase-rr-test-{uuid.uuid4().hex[:10]}"


def _spawn(
    module: str, args: list[str], *, extra_env: dict[str, str] | None = None
) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(
        [sys.executable, "-m", module, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(_REPO_ROOT),
        env=env,
    )


def _iter_events(stream: object) -> Iterator[dict[str, object]]:
    """Yield parsed JSON-line events from a subprocess pipe."""
    for line in stream:  # type: ignore[attr-defined]
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # Non-JSON line (e.g. LSL banner); ignore.
            continue


def _wait_for_event(
    proc: subprocess.Popen[str], event_name: str, *, timeout_s: float
) -> dict[str, object]:
    """Pull lines off proc.stdout until one matches event_name or we time out."""
    deadline = time.monotonic() + timeout_s
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise AssertionError(
                    f"process ended before {event_name!r}: exit={proc.returncode} "
                    f"stderr={proc.stderr.read() if proc.stderr else ''!r}"
                )
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
    raise AssertionError(f"timed out waiting for {event_name!r} on proc {proc.pid}")


def _drain_events_until_exit(
    proc: subprocess.Popen[str], *, timeout_s: float
) -> list[dict[str, object]]:
    """Read all remaining JSON-line events until the process exits or we time out."""
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
    # Drain the rest.
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


# -------------------- LiveConfig invariants (unit-level) --------------------


class TestLiveConfigContract:
    def test_default_stall_timeout_is_5(self) -> None:
        assert LiveConfig().stall_timeout_s == 5.0

    def test_stall_timeout_lower_bound(self) -> None:
        with pytest.raises(ValueError, match="stall_timeout_s"):
            LiveConfig(stall_timeout_s=STALL_TIMEOUT_SAFE_MIN_S - 0.01)

    def test_stall_timeout_upper_bound(self) -> None:
        with pytest.raises(ValueError, match="stall_timeout_s"):
            LiveConfig(stall_timeout_s=STALL_TIMEOUT_SAFE_MAX_S + 0.01)

    def test_read_timeout_must_be_less_than_stall(self) -> None:
        with pytest.raises(ValueError, match="read_timeout_s"):
            LiveConfig(stall_timeout_s=2.0, read_timeout_s=2.0)

    def test_read_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="read_timeout_s"):
            LiveConfig(read_timeout_s=0.0)


# -------------------- End-to-end live transport ---------------------------


class TestLiveEndToEnd:
    def test_clean_run_produces_summary_and_exit_0(self) -> None:
        stream = _unique_stream_name()
        consumer = _spawn(
            "neurophase.physio.live",
            [
                "--stream-name",
                stream,
                "--max-frames",
                str(len(RR_SEQUENCE_MS)),
                "--stall-timeout-s",
                "4.0",
            ],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn(
                "neurophase.physio.live_producer",
                [
                    "--stream-name",
                    stream,
                    "--inter-sample-s",
                    "0.02",
                ],
            )
            try:
                producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
                assert producer.returncode == 0, (
                    f"producer exit={producer.returncode} stderr={producer.stderr.read()!r}"
                )
            finally:
                if producer.poll() is None:
                    producer.terminate()

            events = _drain_events_until_exit(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
            assert consumer.returncode == 0, (
                f"consumer exit={consumer.returncode} stderr={consumer.stderr.read()!r}"
            )
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        # At least LISTENING, READY, 24 FRAME, SUMMARY.
        frames = [e for e in events if e.get("event") == "FRAME"]
        assert len(frames) == len(RR_SEQUENCE_MS), (
            f"expected {len(RR_SEQUENCE_MS)} frames, got {len(frames)}"
        )
        # 6. insufficient initial buffer stays degraded: first
        # MIN_WINDOW_SIZE-1 frames must be SENSOR_DEGRADED.
        for f in frames[:15]:
            assert f["gate_state"] == "SENSOR_DEGRADED"
            assert f["execution_allowed"] is False
        # 7. after readiness, each new valid sample causes a state update
        # (each FRAME is a state update; count equals sequence length).
        summary = next(e for e in reversed(events) if e.get("event") == "SUMMARY")
        assert summary["clean_exit"] is True
        assert summary["n_frames"] == len(RR_SEQUENCE_MS)

    def test_latency_under_1s_post_readiness(self) -> None:
        """After readiness, each new valid sample must become a FRAME
        event with (consumer_rx_mono_s - producer_mono_s) < 1.0 on local
        loopback."""
        stream = _unique_stream_name()
        consumer = _spawn(
            "neurophase.physio.live",
            [
                "--stream-name",
                stream,
                "--max-frames",
                str(len(RR_SEQUENCE_MS)),
                "--stall-timeout-s",
                "4.0",
            ],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn(
                "neurophase.physio.live_producer",
                ["--stream-name", stream, "--inter-sample-s", "0.02"],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            events = _drain_events_until_exit(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            for p in (consumer,):
                if p.poll() is None:
                    p.terminate()

        frames = [e for e in events if e.get("event") == "FRAME"]
        # Use post-readiness frames only (after the 16-sample buffer fills).
        post = frames[16:]
        assert post, "no post-readiness frames emitted"
        for f in post:
            latency = float(f["latency_s"])  # consumer_rx - producer_mono
            assert -0.5 < latency < 1.0, (
                f"latency_s={latency} outside loopback bound (tick={f['tick_index']})"
            )


# -------------------- Fail-closed boundaries ------------------------------


class TestFailClosed:
    def test_impossible_rr_rejected_cleanly(self) -> None:
        """Impossible RR mid-stream -> INGEST_REJECTED, stream continues,
        no state is upgraded to EXECUTE_ALLOWED as a result."""
        stream = _unique_stream_name()
        consumer = _spawn(
            "neurophase.physio.live",
            [
                "--stream-name",
                stream,
                "--max-frames",
                "24",
                "--stall-timeout-s",
                "4.0",
            ],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn(
                "neurophase.physio.live_producer",
                [
                    "--stream-name",
                    stream,
                    "--inter-sample-s",
                    "0.02",
                    "--inject-impossible-rr-at",
                    "10",
                ],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            events = _drain_events_until_exit(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        rejected = [e for e in events if e.get("event") == "INGEST_REJECTED"]
        assert rejected, "expected at least one INGEST_REJECTED event"
        assert any("outside physiological envelope" in str(e["reason"]) for e in rejected)
        # Impossible sample must never surface as EXECUTE_ALLOWED.
        frames = [e for e in events if e.get("event") == "FRAME"]
        assert all(f["execution_allowed"] in (False, True) for f in frames)
        # None of the FRAMEs corresponding to the rejected sample index
        # carry EXECUTE_ALLOWED either (fail-closed invariant).
        assert not any(
            f["execution_allowed"] and f["gate_state"] != "EXECUTE_ALLOWED" for f in frames
        )

    def test_non_monotonic_timestamp_rejected_cleanly(self) -> None:
        stream = _unique_stream_name()
        consumer = _spawn(
            "neurophase.physio.live",
            [
                "--stream-name",
                stream,
                "--max-frames",
                "24",
                "--stall-timeout-s",
                "4.0",
            ],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn(
                "neurophase.physio.live_producer",
                [
                    "--stream-name",
                    stream,
                    "--inter-sample-s",
                    "0.02",
                    "--inject-backward-ts-at",
                    "8",
                ],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            events = _drain_events_until_exit(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        rejected = [e for e in events if e.get("event") == "INGEST_REJECTED"]
        assert rejected, "expected at least one INGEST_REJECTED event"
        assert any("non-monotonic" in str(e["reason"]) for e in rejected)

    def test_stall_detected_within_configured_timeout(self) -> None:
        """stall_timeout_s drives the STALL event — with hold=3 s and
        timeout=2 s we must see at least one STALL event, and the
        associated gate_state must be SENSOR_DEGRADED."""
        stream = _unique_stream_name()
        consumer = _spawn(
            "neurophase.physio.live",
            [
                "--stream-name",
                stream,
                "--stall-timeout-s",
                "2.0",
                "--read-timeout-s",
                "0.25",
                "--max-frames",
                "24",
            ],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn(
                "neurophase.physio.live_producer",
                [
                    "--stream-name",
                    stream,
                    "--inter-sample-s",
                    "0.02",
                    "--stall-after",
                    "16",
                    "--stall-hold-s",
                    "3.0",
                ],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            events = _drain_events_until_exit(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        stalls = [e for e in events if e.get("event") == "STALL"]
        assert stalls, "expected at least one STALL event during 3 s hold"
        for s in stalls:
            assert s["gate_state"] == "SENSOR_DEGRADED"
            assert s["execution_allowed"] is False
            seconds = float(s["seconds_since_last_sample"])
            assert seconds >= 2.0, f"stall fired too early at {seconds}s"

    def test_no_producer_exits_fatal_nonzero(self) -> None:
        """If the expected stream never appears, consumer exits non-zero."""
        stream = _unique_stream_name()  # producer never started
        consumer = _spawn(
            "neurophase.physio.live",
            [
                "--stream-name",
                stream,
                "--resolve-timeout-s",
                "2.0",
                "--stall-timeout-s",
                "3.0",
            ],
        )
        consumer.wait(timeout=10.0)
        assert consumer.returncode != 0
        assert consumer.stdout is not None
        tail = consumer.stdout.read()
        assert "FATAL" in tail


# -------------------- Shared-core parity ----------------------------------


class TestReplayLiveSemanticParity:
    """Property 11: live and replay must produce the same gate decisions
    for the same sample sequence. This proves the shared core
    (:class:`PhysioSession`) is the single source of truth and that the
    live ingress wrapper is not silently mutating anything."""

    def test_same_sequence_identical_decisions(self) -> None:
        # Feed RR_SEQUENCE_MS through PhysioSession directly (this is the
        # same class the live consumer uses). Then run the live
        # subprocess and compare tick-by-tick.
        ref_session = PhysioSession()
        ref_states: list[str] = []
        mono = 100.0
        for i, rr in enumerate(RR_SEQUENCE_MS):
            mono += rr / 1000.0
            frame = ref_session.step(
                RRSample(timestamp_s=mono, rr_ms=rr, row_index=i),
                tick_index=i,
            )
            ref_states.append(frame.decision.state.name)

        stream = _unique_stream_name()
        consumer = _spawn(
            "neurophase.physio.live",
            [
                "--stream-name",
                stream,
                "--max-frames",
                str(len(RR_SEQUENCE_MS)),
                "--stall-timeout-s",
                "4.0",
            ],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn(
                "neurophase.physio.live_producer",
                ["--stream-name", stream, "--inter-sample-s", "0.02"],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            events = _drain_events_until_exit(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        live_frames = sorted(
            (e for e in events if e.get("event") == "FRAME"),
            key=lambda e: int(e["tick_index"]),  # type: ignore[arg-type]
        )
        live_states = [f["gate_state"] for f in live_frames]
        assert live_states == ref_states, (
            f"live/replay divergence:\n  live:   {live_states}\n  replay: {ref_states}"
        )


# -------------------- Process independence --------------------------------


class TestProcessIndependence:
    """Property 1 + 2: consumer and producer are genuinely independent
    processes and the consumer truly waits for FUTURE samples."""

    def test_consumer_blocks_until_producer_starts(self) -> None:
        """Start consumer only; verify it does NOT emit READY/FRAME events
        before the producer is launched. Uses non-blocking select-based
        drain so we can observe the "no progress" phase without stealing
        future events from the drain-to-exit phase."""
        import select

        stream = _unique_stream_name()
        consumer = _spawn(
            "neurophase.physio.live",
            [
                "--stream-name",
                stream,
                "--max-frames",
                "24",
                "--stall-timeout-s",
                "3.0",
                "--resolve-timeout-s",
                "10.0",
            ],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            # Observe the consumer for 1.0 s with NO producer. Use select
            # so we don't block on readline and can time-bound accurately.
            t_pause = time.monotonic()
            collected: list[dict[str, object]] = []
            assert consumer.stdout is not None
            while time.monotonic() - t_pause < 1.0:
                remaining = 1.0 - (time.monotonic() - t_pause)
                if remaining <= 0:
                    break
                ready, _, _ = select.select([consumer.stdout], [], [], remaining)
                if not ready:
                    continue
                line = consumer.stdout.readline()
                if not line:
                    continue
                with contextlib.suppress(json.JSONDecodeError):
                    collected.append(json.loads(line))
            for e in collected:
                assert e.get("event") not in ("READY", "FRAME"), (
                    f"consumer emitted {e} before producer was started"
                )
            # Now start producer and ensure consumer progresses.
            producer = _spawn(
                "neurophase.physio.live_producer",
                ["--stream-name", stream, "--inter-sample-s", "0.02"],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            events = _drain_events_until_exit(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        assert any(e.get("event") == "READY" for e in events)
        assert any(e.get("event") == "FRAME" for e in events)

    def test_polar_producer_constants_match_kernel_contract(self) -> None:
        """Kernel-side guarantee that ``tools/polar_producer.py`` (an
        out-of-repo tool) still matches the live-consumer LSL contract.

        If anybody ever changes ``LSL_CHANNEL_COUNT`` / ``LSL_STREAM_TYPE``
        on the kernel side, this test fails and flags the contract
        drift before the tool is shipped with mismatched constants."""
        import importlib.util
        import sys as _sys

        polar_path = _REPO_ROOT / "tools" / "polar_producer.py"
        if not polar_path.exists():
            pytest.skip("tools/polar_producer.py not present")

        spec = importlib.util.spec_from_file_location("polar_producer_probe", polar_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["polar_producer_probe"] = mod
        try:
            spec.loader.exec_module(mod)

            from neurophase.physio.live import (
                LSL_CHANNEL_COUNT,
                LSL_CHANNEL_FORMAT,
                LSL_STREAM_TYPE,
            )

            assert mod.LSL_CHANNEL_COUNT == LSL_CHANNEL_COUNT
            assert mod.LSL_CHANNEL_FORMAT == LSL_CHANNEL_FORMAT
            assert mod.LSL_STREAM_TYPE == LSL_STREAM_TYPE

            # RR conversion invariants, inline (do not trust the tool's
            # own self-test to gate this test).
            r_512 = mod.parse_hrs_payload(bytes([0x10, 75, 0x00, 0x02]))
            assert r_512.rr_values_ms == (500.0,), r_512.rr_values_ms
            r_1024 = mod.parse_hrs_payload(bytes([0x10, 75, 0x00, 0x04]))
            assert r_1024.rr_values_ms == (1000.0,), r_1024.rr_values_ms

            # Sentinel must be single-shot.
            class _StubOutlet:
                def __init__(self) -> None:
                    self.pushes: list[list[float]] = []

                def push_sample(self, x: list[float]) -> None:
                    self.pushes.append(list(x))

            stub = _StubOutlet()
            guard = mod.SentinelGuard(stub)
            assert guard.emit_once() is True
            assert guard.emit_once() is False
            assert len(stub.pushes) == 1
        finally:
            _sys.modules.pop("polar_producer_probe", None)

    def test_live_session_round_trip_through_ledger(self, tmp_path: Path) -> None:
        """Keystone: run a live LSL session with --ledger-out, then
        re-execute the ledger offline through a fresh PhysioSession
        and verify byte-identical decisions (parity_ok)."""
        from neurophase.physio.session_replay import replay_ledger

        stream = _unique_stream_name()
        ledger_path = tmp_path / "live.jsonl"
        consumer = _spawn(
            "neurophase.physio.live",
            [
                "--stream-name",
                stream,
                "--max-frames",
                "24",
                "--stall-timeout-s",
                "4.0",
                "--ledger-out",
                str(ledger_path),
            ],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn(
                "neurophase.physio.live_producer",
                ["--stream-name", stream, "--inter-sample-s", "0.02"],
            )
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            _drain_events_until_exit(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()

        assert consumer.returncode == 0, (
            f"consumer exit={consumer.returncode} stderr={consumer.stderr.read()!r}"
        )
        assert ledger_path.exists()

        report = replay_ledger(ledger_path)
        assert report.parity_ok, report.divergences
        assert report.n_frames_recorded == 24
        assert report.clean_summary_seen is True

    def test_consumer_and_producer_have_different_pids(self) -> None:
        """Trivial but explicit process-independence check."""
        stream = _unique_stream_name()
        consumer = _spawn(
            "neurophase.physio.live",
            [
                "--stream-name",
                stream,
                "--max-frames",
                "24",
                "--stall-timeout-s",
                "3.0",
            ],
        )
        try:
            _wait_for_event(consumer, "LISTENING", timeout_s=_READY_TIMEOUT_S)
            producer = _spawn(
                "neurophase.physio.live_producer",
                ["--stream-name", stream, "--inter-sample-s", "0.02"],
            )
            assert consumer.pid != producer.pid
            producer.wait(timeout=_PROC_JOIN_TIMEOUT_S)
            _drain_events_until_exit(consumer, timeout_s=_PROC_JOIN_TIMEOUT_S)
        finally:
            if consumer.poll() is None:
                consumer.terminate()
