#!/usr/bin/env python3
"""tools/fault_producer.py -- adversarial LSL fault producer for the physio live path.

Lives alongside ``tools/polar_producer.py`` and
``neurophase/physio/live_producer.py`` but is intentionally NOT an
internal module -- it exists to WEAPONISE the live path against
malformed, degraded, and disconnect-style inputs. Every test in
``tests/test_physio_faults.py`` spawns this tool as a real subprocess.

Sample schema is the same as the happy-path producer:
    [timestamp_s, rr_ms]  (LSL float32, channel_count = 2)
    (NaN, NaN)            = EOF sentinel
    ch1 = nan elsewhere   = consumer fail-closed EOF

Supported fault modes (pick ONE via --fault):

    clean               happy path reference (24 stable samples + sentinel)
    flatline            all samples share an identical RR value
                        -> RMSSD == 0 -> SENSOR_DEGRADED
    spike_burst         N impossible-RR (100 ms) samples in a row
                        -> each is INGEST_REJECTED, no frame advances
    dup_timestamps      consecutive samples share their timestamp
                        -> non-monotonic -> INGEST_REJECTED
    dropped_packets     producer pauses for --stall-hold-s after a few
                        samples -> consumer STALL event fires
    delayed_outlet      producer sleeps --delay-s BEFORE creating the
                        outlet -> consumer resolve_timeout exercised
    abrupt_disconnect   producer pushes 16 samples then exits WITHOUT
                        a sentinel -> consumer sees prolonged silence
                        then a DISCONNECTED (peer-initiated) trigger
                        and exits EXIT_UNEXPECTED_DISCONNECT on the
                        kernel side (or transitions to STALL on LSL-
                        only graceful close).

Exit codes match ``live_producer.py`` conventions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Make the kernel reachable on a fresh checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from neurophase.physio.live import (  # noqa: E402
    LSL_CHANNEL_COUNT,
    LSL_CHANNEL_FORMAT,
    LSL_STREAM_TYPE,
)

_STABLE_RR: tuple[float, ...] = (
    820.0, 825.0, 818.0, 822.0, 830.0, 815.0, 819.0, 824.0,
    820.0, 823.0, 821.0, 822.0, 820.0, 819.0, 821.0, 823.0,
    820.0, 822.0, 819.0, 821.0, 820.0, 823.0, 821.0, 822.0,
)


EXIT_OK: int = 0
EXIT_USAGE: int = 2
EXIT_ABRUPT_REQUESTED: int = 4


@dataclass(frozen=True)
class FaultConfig:
    fault: str
    stream_name: str
    consumer_wait_s: float = 10.0
    inter_sample_s: float = 0.05
    stall_hold_s: float = 3.0
    delay_s: float = 3.0
    spike_count: int = 5
    source_id: str = "neurophase-fault-producer-v1"

    def __post_init__(self) -> None:
        if self.consumer_wait_s <= 0:
            raise ValueError("consumer_wait_s must be > 0")
        if self.stall_hold_s <= 0:
            raise ValueError("stall_hold_s must be > 0")
        if self.spike_count < 1:
            raise ValueError("spike_count must be >= 1")


@dataclass
class _State:
    last_mono: float = 0.0
    events: list[str] = field(default_factory=list)


def _emit(event: dict[str, object]) -> None:
    print(json.dumps(event, default=str), flush=True)


def _make_info(config: FaultConfig) -> object:
    from pylsl import StreamInfo

    info = StreamInfo(
        name=config.stream_name,
        type=LSL_STREAM_TYPE,
        channel_count=LSL_CHANNEL_COUNT,
        nominal_srate=0.0,
        channel_format=LSL_CHANNEL_FORMAT,
        source_id=config.source_id,
    )
    desc = info.desc()
    desc.append_child_value("origin", "neurophase-fault-producer")
    desc.append_child_value(
        "note",
        "SYNTHETIC ADVERSARIAL SAMPLES - NOT REAL DATA, fault injection tool",
    )
    return info


def _open_outlet(config: FaultConfig) -> object:
    from pylsl import StreamOutlet

    info = _make_info(config)
    outlet = StreamOutlet(info)
    _emit({"event": "OUTLET_CREATED", "stream_name": config.stream_name})

    deadline = time.monotonic() + config.consumer_wait_s
    while time.monotonic() < deadline and not outlet.have_consumers():
        time.sleep(0.05)
    if not outlet.have_consumers():
        _emit({"event": "NO_CONSUMER"})
        return None
    _emit({"event": "CONSUMER_SEEN"})
    return outlet


def _push(outlet: object, ts_s: float, rr_ms: float) -> None:
    outlet.push_sample([ts_s, rr_ms])  # type: ignore[attr-defined]


def _push_sentinel(outlet: object) -> None:
    _push(outlet, math.nan, math.nan)
    time.sleep(0.2)


# ------------------- Fault drivers ------------------------------------


def _run_clean(outlet: object, config: FaultConfig) -> int:
    for i, rr in enumerate(_STABLE_RR):
        _push(outlet, time.monotonic(), rr)
        _emit({"event": "SENT", "index": i, "rr_ms": rr, "fault": "clean"})
        if config.inter_sample_s > 0:
            time.sleep(config.inter_sample_s)
    _push_sentinel(outlet)
    _emit({"event": "DONE", "fault": "clean"})
    return EXIT_OK


def _run_flatline(outlet: object, config: FaultConfig) -> int:
    # All-identical RR -> RMSSD collapses to 0 -> fails plausibility
    # envelope -> SENSOR_DEGRADED on every frame inside the filled window.
    for i in range(len(_STABLE_RR)):
        _push(outlet, time.monotonic(), 820.0)
        _emit({"event": "SENT", "index": i, "rr_ms": 820.0, "fault": "flatline"})
        if config.inter_sample_s > 0:
            time.sleep(config.inter_sample_s)
    _push_sentinel(outlet)
    _emit({"event": "DONE", "fault": "flatline"})
    return EXIT_OK


def _run_spike_burst(outlet: object, config: FaultConfig) -> int:
    # Warm up, then a contiguous run of impossible RRs, then back to normal.
    warmup = 16
    for i in range(warmup):
        _push(outlet, time.monotonic(), _STABLE_RR[i])
        _emit({"event": "SENT", "index": i, "rr_ms": _STABLE_RR[i], "fault": "spike_burst"})
        if config.inter_sample_s > 0:
            time.sleep(config.inter_sample_s)
    for j in range(config.spike_count):
        _push(outlet, time.monotonic(), 100.0)  # below envelope
        _emit(
            {
                "event": "SENT",
                "index": warmup + j,
                "rr_ms": 100.0,
                "fault": "spike_burst",
                "spike": True,
            }
        )
        if config.inter_sample_s > 0:
            time.sleep(config.inter_sample_s)
    # Continue with a few more clean samples to give the consumer a
    # chance to recover.
    tail_start = warmup + config.spike_count
    for k in range(5):
        rr = _STABLE_RR[(tail_start + k) % len(_STABLE_RR)]
        _push(outlet, time.monotonic(), rr)
        _emit({"event": "SENT", "index": tail_start + k, "rr_ms": rr, "fault": "spike_burst"})
        if config.inter_sample_s > 0:
            time.sleep(config.inter_sample_s)
    _push_sentinel(outlet)
    _emit({"event": "DONE", "fault": "spike_burst"})
    return EXIT_OK


def _run_dup_timestamps(outlet: object, config: FaultConfig) -> int:
    # Send each sample TWICE at exactly the same timestamp.
    for i, rr in enumerate(_STABLE_RR[:16]):
        ts = time.monotonic()
        _push(outlet, ts, rr)
        _emit({"event": "SENT", "index": i * 2, "ts": ts, "rr_ms": rr, "fault": "dup_timestamps"})
        _push(outlet, ts, rr)  # duplicate ts -> non-monotonic on consumer
        _emit(
            {
                "event": "SENT_DUP",
                "index": i * 2 + 1,
                "ts": ts,
                "rr_ms": rr,
                "fault": "dup_timestamps",
            }
        )
        if config.inter_sample_s > 0:
            time.sleep(config.inter_sample_s)
    _push_sentinel(outlet)
    _emit({"event": "DONE", "fault": "dup_timestamps"})
    return EXIT_OK


def _run_dropped_packets(outlet: object, config: FaultConfig) -> int:
    # A few warm-up samples, then a long pause (stall), then resume.
    for i in range(8):
        _push(outlet, time.monotonic(), _STABLE_RR[i])
        _emit({"event": "SENT", "index": i, "rr_ms": _STABLE_RR[i], "fault": "dropped_packets"})
        if config.inter_sample_s > 0:
            time.sleep(config.inter_sample_s)
    _emit({"event": "STALL_HOLD_BEGIN", "hold_s": config.stall_hold_s})
    time.sleep(config.stall_hold_s)
    _emit({"event": "STALL_HOLD_END"})
    # Resume with more samples.
    for i in range(8, 16):
        _push(outlet, time.monotonic(), _STABLE_RR[i])
        _emit({"event": "SENT", "index": i, "rr_ms": _STABLE_RR[i], "fault": "dropped_packets"})
        if config.inter_sample_s > 0:
            time.sleep(config.inter_sample_s)
    _push_sentinel(outlet)
    _emit({"event": "DONE", "fault": "dropped_packets"})
    return EXIT_OK


def _run_delayed_outlet(_outlet: object, config: FaultConfig) -> int:
    # In this mode the "outlet" passed in is unused: we don't create it
    # until the delay has elapsed, at which point the consumer's
    # resolve_timeout will already have fired.
    raise RuntimeError("_run_delayed_outlet must be handled before outlet creation")


def _run_abrupt_disconnect(outlet: object, config: FaultConfig) -> int:
    # Push 16 clean samples then exit WITHOUT a sentinel and without a
    # graceful outlet.__del__ -> consumer stalls then sees DISCONNECTED.
    for i in range(16):
        _push(outlet, time.monotonic(), _STABLE_RR[i])
        _emit({"event": "SENT", "index": i, "rr_ms": _STABLE_RR[i], "fault": "abrupt_disconnect"})
        if config.inter_sample_s > 0:
            time.sleep(config.inter_sample_s)
    _emit({"event": "PRODUCER_ABRUPT_EXIT", "fault": "abrupt_disconnect"})
    return EXIT_ABRUPT_REQUESTED  # producer intentionally exits non-zero


_FAULT_DRIVERS = {
    "clean": _run_clean,
    "flatline": _run_flatline,
    "spike_burst": _run_spike_burst,
    "dup_timestamps": _run_dup_timestamps,
    "dropped_packets": _run_dropped_packets,
    "abrupt_disconnect": _run_abrupt_disconnect,
}


# ------------------- CLI -----------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fault_producer",
        description="Adversarial LSL fault producer for physio live-path hardening.",
    )
    p.add_argument("--stream-name", default="neurophase-rr")
    p.add_argument(
        "--fault",
        required=True,
        choices=sorted([*_FAULT_DRIVERS.keys(), "delayed_outlet"]),
        help="Fault class to inject.",
    )
    p.add_argument("--consumer-wait-s", type=float, default=10.0)
    p.add_argument("--inter-sample-s", type=float, default=0.05)
    p.add_argument("--stall-hold-s", type=float, default=3.0)
    p.add_argument(
        "--delay-s",
        type=float,
        default=3.0,
        help="delayed_outlet: sleep this long before creating the outlet.",
    )
    p.add_argument(
        "--spike-count",
        type=int,
        default=5,
        help="spike_burst: number of consecutive impossible-RR samples.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    try:
        config = FaultConfig(
            fault=args.fault,
            stream_name=args.stream_name,
            consumer_wait_s=args.consumer_wait_s,
            inter_sample_s=args.inter_sample_s,
            stall_hold_s=args.stall_hold_s,
            delay_s=args.delay_s,
            spike_count=args.spike_count,
        )
    except ValueError as exc:
        _emit({"event": "FATAL", "reason": f"bad config: {exc}"})
        return EXIT_USAGE

    # delayed_outlet is special: the delay happens BEFORE the outlet
    # is created. The consumer is expected to resolve-timeout during
    # this window (EXIT_NO_DEVICE / EXIT_FATAL on the consumer side).
    if args.fault == "delayed_outlet":
        _emit({"event": "DELAY_BEFORE_OUTLET", "delay_s": args.delay_s})
        time.sleep(args.delay_s)
        outlet = _open_outlet(config)
        if outlet is None:
            # Consumer already gave up; exit cleanly.
            _emit({"event": "DONE", "fault": "delayed_outlet", "note": "no consumer"})
            return EXIT_OK
        # Push a short clean trailer so the session has SOMETHING in
        # the ledger, then sentinel out.
        for i in range(8):
            _push(outlet, time.monotonic(), _STABLE_RR[i])
        _push_sentinel(outlet)
        _emit({"event": "DONE", "fault": "delayed_outlet"})
        return EXIT_OK

    outlet = _open_outlet(config)
    if outlet is None:
        return EXIT_OK
    driver = _FAULT_DRIVERS[args.fault]
    return driver(outlet, config)


if __name__ == "__main__":
    raise SystemExit(main())
