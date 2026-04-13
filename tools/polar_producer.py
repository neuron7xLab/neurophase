#!/usr/bin/env python3
"""polar_producer.py -- Polar H10 RR-interval bridge to the NeuroPhase LSL stream.

This script is **OUT OF REPO KERNEL**. It does not import anything from
``neurophase``. It speaks only the public contract that
``neurophase/physio/live.py`` expects:

* LSL stream name   ``neurophase-rr`` (CLI override)
* channel_count     2
* channel_format    float32
* type              "RR"
* sample            ``[timestamp_s, rr_ms]`` per RR interval
* EOF sentinel      ``[NaN, NaN]`` pushed exactly once on shutdown

Pipeline: Polar H10 BLE Heart Rate Service (0x180D) -> Heart Rate
Measurement characteristic (0x2A37) notifications -> strict HRS payload
parser -> one LSL sample per RR interval.

Standard BLE HRS is the only path. Polar SDK is not used. No reconnect
logic. No multi-device support.

Usage::

    # parser self-test (no BLE, no LSL; exits 0/6)
    python polar_producer.py --self-test

    # scan + connect + stream (requires real Polar H10)
    python polar_producer.py --stream-name neurophase-rr

    # targeted connect by BLE address (skips the scan)
    python polar_producer.py --address AA:BB:CC:DD:EE:FF

Exit codes:

    0  orderly shutdown (interrupt, clean disconnect, EOF sentinel sent)
    1  device not found, ambiguous selection
    2  BLE connect failure
    3  required HRS characteristic missing
    4  unexpected disconnect after successful streaming
    5  fatal LSL outlet / streaming failure
    6  --self-test failure
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import signal
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

__version__ = "1.0.0"

# --- BLE Heart Rate Service UUIDs (Bluetooth SIG standard) -------------
HRS_SERVICE_UUID: str = "0000180d-0000-1000-8000-00805f9b34fb"
HRS_MEASUREMENT_CHAR_UUID: str = "00002a37-0000-1000-8000-00805f9b34fb"

# --- LSL contract (mirrors neurophase/physio/live.py exactly) ----------
LSL_STREAM_TYPE: str = "RR"
LSL_CHANNEL_COUNT: int = 2
LSL_CHANNEL_FORMAT: str = "float32"

# --- RR plausibility envelope (mirrors neurophase/physio/replay.py) ----
RR_MIN_MS: float = 300.0
RR_MAX_MS: float = 2000.0

# --- Exit codes --------------------------------------------------------
EXIT_OK: int = 0
EXIT_NO_DEVICE: int = 1
EXIT_CONNECT_FAIL: int = 2
EXIT_NO_CHAR: int = 3
EXIT_UNEXPECTED_DISCONNECT: int = 4
EXIT_LSL_FATAL: int = 5
EXIT_SELF_TEST_FAIL: int = 6

# =======================================================================
#   Structured logging
# =======================================================================


def log_event(event: str, **fields: Any) -> None:
    """Write one structured JSON-lines event to stdout, flushed immediately."""
    payload = {"event": event, **fields}
    sys.stdout.write(json.dumps(payload, default=str))
    sys.stdout.write("\n")
    sys.stdout.flush()


# =======================================================================
#   Heart Rate Measurement payload parser (strict, deterministic, pure)
# =======================================================================


class HRSParseError(ValueError):
    """Raised when the HRS payload is malformed / truncated."""


@dataclass(frozen=True)
class HRSParseResult:
    """Parsed Heart Rate Measurement characteristic payload."""

    heart_rate_bpm: int | None
    rr_values_ms: tuple[float, ...]
    energy_expended: int | None


def parse_hrs_payload(data: bytes) -> HRSParseResult:
    """Parse a Bluetooth Heart Rate Measurement characteristic payload.

    Payload layout per Bluetooth SIG "Heart Rate Measurement" specification:

        byte 0       Flags
          bit 0        HR value format (0 = uint8, 1 = uint16)
          bit 3        Energy Expended Status (1 = present)
          bit 4        RR-Interval bit (1 = present)
        bytes 1..     HR value (1 byte if flags.0=0 else 2 bytes LE)
        (optional)    Energy Expended (uint16 LE, if flags.3=1)
        (optional)    RR intervals (uint16 LE each, 1/1024-second units,
                      if flags.4=1 and remaining bytes are pairs)

    RR conversion rule (NON-NEGOTIABLE):  ``rr_ms = rr_raw * 1000 / 1024``

    Raises :class:`HRSParseError` on truncated payloads. Does NOT raise
    on "HR only / no RR" payloads -- it simply returns an empty
    ``rr_values_ms`` tuple.
    """
    if len(data) < 2:
        raise HRSParseError(f"payload too short: {len(data)} byte(s)")

    flags = data[0]
    hr_uint16 = bool(flags & 0x01)
    ee_present = bool(flags & 0x08)
    rr_present = bool(flags & 0x10)

    offset = 1
    if hr_uint16:
        if len(data) < offset + 2:
            raise HRSParseError("truncated at 16-bit heart-rate value")
        hr = int.from_bytes(data[offset : offset + 2], "little", signed=False)
        offset += 2
    else:
        hr = int(data[offset])
        offset += 1

    ee: int | None = None
    if ee_present:
        if len(data) < offset + 2:
            raise HRSParseError("truncated at energy-expended field")
        ee = int.from_bytes(data[offset : offset + 2], "little", signed=False)
        offset += 2

    rr_values: list[float] = []
    if rr_present:
        remaining = data[offset:]
        if len(remaining) % 2 != 0:
            raise HRSParseError(
                f"RR region has odd byte count: {len(remaining)}"
            )
        for i in range(0, len(remaining), 2):
            rr_raw = int.from_bytes(remaining[i : i + 2], "little", signed=False)
            # Protected conversion: divide by 1024 (not 1000).
            rr_ms = rr_raw * 1000.0 / 1024.0
            rr_values.append(rr_ms)

    return HRSParseResult(
        heart_rate_bpm=hr,
        rr_values_ms=tuple(rr_values),
        energy_expended=ee,
    )


# =======================================================================
#   Sentinel guard (single-shot emitter of the [NaN, NaN] LSL sample)
# =======================================================================


class SentinelGuard:
    """Single-emission guard for the NeuroPhase EOF sentinel.

    The NaN/NaN sample is the consumer's cue to exit 0 cleanly. Emitting
    it more than once per process lifetime would be a contract violation
    -- the consumer would exit on the first one and the second would
    land on a dead socket. This class enforces the one-shot invariant.
    """

    __slots__ = ("_outlet", "_sent", "_on_emit")

    def __init__(
        self,
        outlet: Any,
        *,
        on_emit: Any = None,
    ) -> None:
        self._outlet = outlet
        self._sent: bool = False
        self._on_emit = on_emit

    @property
    def sent(self) -> bool:
        return self._sent

    def emit_once(self) -> bool:
        """Push ``[NaN, NaN]`` exactly once. Returns True iff actually emitted."""
        if self._sent:
            return False
        self._sent = True  # mark first so a crashing push cannot double-fire
        try:
            self._outlet.push_sample([math.nan, math.nan])
        except Exception as exc:
            log_event("SENTINEL_PUSH_FAILED", error=str(exc))
            return False
        if self._on_emit is not None:
            try:
                self._on_emit()
            except Exception as exc:  # belt-and-braces
                log_event("SENTINEL_CALLBACK_FAILED", error=str(exc))
        return True


# =======================================================================
#   LSL outlet
# =======================================================================


def create_lsl_outlet(stream_name: str, source_id: str) -> Any:
    """Create one LSL outlet matching the neurophase live-consumer contract."""
    from pylsl import StreamInfo, StreamOutlet

    info = StreamInfo(
        name=stream_name,
        type=LSL_STREAM_TYPE,
        channel_count=LSL_CHANNEL_COUNT,
        nominal_srate=0.0,
        channel_format=LSL_CHANNEL_FORMAT,
        source_id=source_id,
    )
    desc = info.desc()
    desc.append_child_value("origin", "polar-h10-rr-producer")
    desc.append_child_value(
        "note",
        "REAL RR INTERVALS from Polar H10 over BLE HRS 0x2A37",
    )
    desc.append_child_value("sample_contract", "ch0=timestamp_s, ch1=rr_ms")
    return StreamOutlet(info)


# =======================================================================
#   Device discovery
# =======================================================================


@dataclass(frozen=True)
class DeviceTarget:
    """Minimal target description for BleakClient."""

    address: str
    name: str


async def discover_polar_h10(
    *,
    address: str | None,
    name_filter: str | None,
    scan_timeout_s: float,
) -> DeviceTarget | None:
    """Discover a Polar H10 or fail clearly.

    Selection rules (in order):
      1. If ``address`` is given, return it directly (no scan).
      2. Else scan for ``scan_timeout_s`` and keep devices whose name
         contains ``name_filter`` (default: "Polar H10", case-insensitive).
      3. If exactly one matches, select it.
      4. If the user gave an explicit ``--name`` and exactly one device
         matches case-sensitively, select it.
      5. Otherwise fail (ambiguous or empty).
    """
    if address:
        log_event("DEVICE_SELECTED_BY_ADDRESS", address=address)
        return DeviceTarget(address=address, name=name_filter or "<user-specified>")

    from bleak import BleakScanner
    from bleak.exc import BleakError

    effective_filter = name_filter or "Polar H10"
    log_event("SCAN_START", scan_timeout_s=scan_timeout_s, filter=effective_filter)
    try:
        devices = await BleakScanner.discover(timeout=scan_timeout_s)
    except BleakError as exc:
        # No powered adapter / backend unavailable -> treat as "device
        # not found" from the operator's perspective. Fail-closed, no
        # traceback noise, concrete event + EXIT_NO_DEVICE upstream.
        log_event("SCAN_FAILED", error=f"{type(exc).__name__}: {exc}")
        return None

    candidates: list[Any] = []
    for d in devices:
        d_name = getattr(d, "name", None) or ""
        if effective_filter.lower() in d_name.lower():
            candidates.append(d)
            log_event("DEVICE_FOUND", address=d.address, name=d_name)

    if not candidates:
        log_event("NO_DEVICE", filter=effective_filter)
        return None

    if len(candidates) == 1:
        d = candidates[0]
        return DeviceTarget(address=d.address, name=d.name or "")

    # Multiple matches -- require an exact-name tie-breaker.
    if name_filter:
        exact = [c for c in candidates if (c.name or "") == name_filter]
        if len(exact) == 1:
            d = exact[0]
            return DeviceTarget(address=d.address, name=d.name or "")

    log_event(
        "AMBIGUOUS_DEVICE",
        matches=[{"address": c.address, "name": c.name or ""} for c in candidates],
    )
    return None


# =======================================================================
#   Async main: BLE -> notifications -> LSL
# =======================================================================


@dataclass
class DisconnectState:
    happened: bool = False
    initiated_by_us: bool = False
    reason: str = ""
    event: asyncio.Event = field(default_factory=asyncio.Event)


async def run_producer(args: argparse.Namespace) -> int:
    target = await discover_polar_h10(
        address=args.address,
        name_filter=args.name,
        scan_timeout_s=args.scan_timeout,
    )
    if target is None:
        return EXIT_NO_DEVICE

    log_event("DEVICE_SELECTED", address=target.address, name=target.name)

    # Build the LSL outlet BEFORE connecting so sentinel can fire on any
    # later failure path where we managed to reach this point.
    source_id = args.source_id or f"polar-h10-rr-{target.address}"
    try:
        outlet = create_lsl_outlet(args.stream_name, source_id)
    except Exception as exc:
        log_event("LSL_OUTLET_FAILED", error=str(exc))
        return EXIT_LSL_FATAL
    log_event(
        "LSL_OUTLET_CREATED",
        stream_name=args.stream_name,
        channel_count=LSL_CHANNEL_COUNT,
        channel_format=LSL_CHANNEL_FORMAT,
        source_id=source_id,
    )

    guard = SentinelGuard(
        outlet, on_emit=lambda: log_event("SENTINEL_SENT", stream_name=args.stream_name)
    )

    disconnect = DisconnectState()

    def _on_disconnected(_client: Any) -> None:
        disconnect.happened = True
        disconnect.reason = disconnect.reason or "peer-initiated"
        disconnect.event.set()
        log_event("DISCONNECTED", reason=disconnect.reason)

    def _on_notification(_sender: Any, data: bytearray) -> None:
        now = time.monotonic()
        raw = bytes(data)
        try:
            result = parse_hrs_payload(raw)
        except HRSParseError as exc:
            log_event(
                "PACKET_REJECTED",
                reason=str(exc),
                hex=(raw.hex() if args.debug else None),
            )
            return
        for rr_ms in result.rr_values_ms:
            if not (RR_MIN_MS <= rr_ms <= RR_MAX_MS):
                log_event(
                    "PACKET_REJECTED",
                    reason=f"rr_ms={rr_ms!r} outside envelope "
                    f"[{RR_MIN_MS}, {RR_MAX_MS}]",
                )
                continue
            try:
                outlet.push_sample([now, rr_ms])
            except Exception as exc:
                log_event("LSL_PUSH_FAILED", error=str(exc))
                disconnect.reason = "lsl-push-failed"
                disconnect.event.set()
                return
            log_event(
                "RR_EMIT",
                timestamp_s=now,
                rr_ms=rr_ms,
                stream_name=args.stream_name,
            )

    # Install SIGTERM handler to convert into a CancelledError via loop signal.
    loop = asyncio.get_running_loop()

    def _request_shutdown(signame: str) -> None:
        if not disconnect.event.is_set():
            disconnect.initiated_by_us = True
            disconnect.reason = f"signal:{signame}"
            disconnect.event.set()
            log_event("SHUTDOWN_REQUESTED", source=signame)

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError):
            # Some platforms (Windows) don't support add_signal_handler on the
            # ProactorEventLoop; fall back to default Python SIGINT KeyboardInterrupt.
            loop.add_signal_handler(sig, _request_shutdown, sig.name)

    # Connect + stream.
    from bleak import BleakClient
    from bleak.exc import BleakError

    log_event("CONNECTING", address=target.address)
    try:
        async with BleakClient(
            target.address, disconnected_callback=_on_disconnected
        ) as client:
            log_event("CONNECTED", address=target.address)

            # Locate the HRS measurement characteristic explicitly.
            hrs_char = None
            for service in client.services:
                for char in service.characteristics:
                    if char.uuid.lower() == HRS_MEASUREMENT_CHAR_UUID:
                        hrs_char = char
                        break
                if hrs_char is not None:
                    break
            if hrs_char is None:
                log_event("CHAR_MISSING", uuid=HRS_MEASUREMENT_CHAR_UUID)
                return EXIT_NO_CHAR

            await client.start_notify(hrs_char, _on_notification)
            log_event("NOTIFY_STARTED", char_uuid=HRS_MEASUREMENT_CHAR_UUID)

            # Run until either bleak reports disconnect or we receive a signal.
            await disconnect.event.wait()

            # Best-effort stop_notify (may fail if peer already gone).
            with contextlib.suppress(Exception):
                await client.stop_notify(hrs_char)
    except BleakError as exc:
        log_event("CONNECT_FAILED", error=str(exc))
        # Outlet already exists: attempt to land one sentinel before exit.
        guard.emit_once()
        log_event("EXIT", code=EXIT_CONNECT_FAIL)
        return EXIT_CONNECT_FAIL
    except Exception as exc:
        log_event("FATAL", error=f"{type(exc).__name__}: {exc}")
        guard.emit_once()
        log_event("EXIT", code=EXIT_LSL_FATAL)
        return EXIT_LSL_FATAL

    # Reached here only if the disconnect event fired.
    guard.emit_once()

    # Exit-code selection: orderly paths map to 0; peer-initiated during
    # a supposedly healthy stream maps to EXIT_UNEXPECTED_DISCONNECT.
    if disconnect.initiated_by_us or disconnect.reason.startswith("signal:"):
        log_event("EXIT", code=EXIT_OK, reason=disconnect.reason)
        return EXIT_OK
    if disconnect.reason == "peer-initiated":
        log_event("EXIT", code=EXIT_UNEXPECTED_DISCONNECT, reason=disconnect.reason)
        return EXIT_UNEXPECTED_DISCONNECT
    # Any other reason (e.g. lsl-push-failed) is fatal.
    log_event("EXIT", code=EXIT_LSL_FATAL, reason=disconnect.reason)
    return EXIT_LSL_FATAL


# =======================================================================
#   Self-test (parser fixtures + RR conversion invariants). No BLE, no LSL.
# =======================================================================


def _self_test_cases() -> Iterable[tuple[str, bool, Any]]:
    """Yield (description, expected_pass, check) tuples for the self-test runner.

    Each ``check`` is a zero-arg callable; it raises AssertionError on
    failure. The second tuple element is ignored (kept for symmetry).
    """
    # Fixture 1: HR only, no RR
    def f1() -> None:
        r = parse_hrs_payload(bytes([0x00, 75]))
        assert r.heart_rate_bpm == 75, r
        assert r.rr_values_ms == ()
        assert r.energy_expended is None

    # Fixture 2: one RR interval (rr_raw=512 -> 500 ms)
    def f2() -> None:
        r = parse_hrs_payload(bytes([0x10, 75, 0x00, 0x02]))
        assert r.heart_rate_bpm == 75
        assert r.rr_values_ms == (500.0,), r.rr_values_ms

    # Fixture 3: multiple RR values (rr_raw=1024 -> 1000ms, rr_raw=820 -> ~800.78125ms)
    def f3() -> None:
        r = parse_hrs_payload(bytes([0x10, 75, 0x00, 0x04, 0x34, 0x03]))
        assert r.rr_values_ms == (1000.0, 820.0 * 1000.0 / 1024.0), r.rr_values_ms

    # Fixture 4: EE + RR (flags=0x18)
    def f4() -> None:
        r = parse_hrs_payload(bytes([0x18, 75, 0x64, 0x00, 0x00, 0x02]))
        assert r.heart_rate_bpm == 75
        assert r.energy_expended == 100
        assert r.rr_values_ms == (500.0,), r.rr_values_ms

    # Fixture 5: truncated RR region (odd remaining bytes)
    def f5() -> None:
        raised = False
        try:
            parse_hrs_payload(bytes([0x10, 75, 0x00]))
        except HRSParseError:
            raised = True
        assert raised, "expected HRSParseError on truncated RR region"

    # RR conversion invariants (NON-NEGOTIABLE)
    def conv_512() -> None:
        r = parse_hrs_payload(bytes([0x10, 75, 0x00, 0x02]))
        assert r.rr_values_ms == (500.0,), r.rr_values_ms

    def conv_1024() -> None:
        r = parse_hrs_payload(bytes([0x10, 75, 0x00, 0x04]))
        assert r.rr_values_ms == (1000.0,), r.rr_values_ms

    yield ("F1: HR only, no RR", True, f1)
    yield ("F2: single RR (512 -> 500.0 ms)", True, f2)
    yield ("F3: multiple RR values", True, f3)
    yield ("F4: EE present + RR", True, f4)
    yield ("F5: truncated payload rejected", True, f5)
    yield ("RR conversion: 512 -> 500.0 ms", True, conv_512)
    yield ("RR conversion: 1024 -> 1000.0 ms", True, conv_1024)

    # Sentinel guard single-emit invariant (no LSL required: use a stub).
    def guard_invariant() -> None:
        class _StubOutlet:
            def __init__(self) -> None:
                self.pushes: list[list[float]] = []

            def push_sample(self, x: list[float]) -> None:  # pragma: no cover
                self.pushes.append(list(x))

        stub = _StubOutlet()
        guard = SentinelGuard(stub)
        assert guard.emit_once() is True, "first emit should return True"
        assert guard.emit_once() is False, "second emit must be a no-op"
        assert guard.emit_once() is False, "third emit must be a no-op"
        assert len(stub.pushes) == 1, f"stub got {stub.pushes!r}"
        assert all(math.isnan(v) for v in stub.pushes[0]), stub.pushes

    yield ("Sentinel single-shot invariant", True, guard_invariant)


def run_self_test() -> int:
    passed = 0
    failed = 0
    failures: list[str] = []
    for description, _expected_pass, check in _self_test_cases():
        try:
            check()
        except AssertionError as exc:
            failed += 1
            failures.append(f"FAIL  {description}: {exc}")
            continue
        except Exception as exc:
            failed += 1
            failures.append(f"ERROR {description}: {type(exc).__name__}: {exc}")
            continue
        passed += 1

    total = passed + failed
    for line in failures:
        print(line, file=sys.stderr)
    print(f"self-test: {passed}/{total} passed", file=sys.stderr)
    return EXIT_OK if failed == 0 else EXIT_SELF_TEST_FAIL


# =======================================================================
#   CLI
# =======================================================================


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="polar_producer",
        description=(
            "Polar H10 RR -> LSL producer. Standalone, out-of-repo, "
            "matches the neurophase live-consumer contract verbatim."
        ),
        epilog=(
            "exit 0 = orderly;  1 = no/ambiguous device;  2 = connect failure; "
            "3 = HRS char missing;  4 = unexpected disconnect;  "
            "5 = fatal LSL/stream;  6 = self-test failure."
        ),
    )
    p.add_argument("--stream-name", default="neurophase-rr")
    p.add_argument("--address", default=None, help="Explicit BLE address (wins over --name).")
    p.add_argument("--name", default=None, help="Name filter; default 'Polar H10'.")
    p.add_argument("--scan-timeout", type=float, default=10.0)
    p.add_argument("--source-id", default=None, help="LSL source_id override.")
    p.add_argument("--debug", action="store_true", help="Include raw bytearray hex on rejected packets.")
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run parser / sentinel self-tests without touching BLE or LSL; exits 0 or 6.",
    )
    p.add_argument("--version", action="version", version=f"polar_producer {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.self_test:
        return run_self_test()

    try:
        return asyncio.run(run_producer(args))
    except KeyboardInterrupt:
        # KeyboardInterrupt at this level means asyncio.run caught SIGINT
        # before our loop handler had a chance. The producer state-machine
        # already handles signal.SIGINT internally in the happy path; this
        # is the ultra-edge fallback. We do not hold the outlet here, so
        # the sentinel may be missed. This is explicitly documented.
        log_event("KEYBOARD_INTERRUPT_OUTSIDE_LOOP")
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
