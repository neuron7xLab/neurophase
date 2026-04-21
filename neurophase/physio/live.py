"""True asynchronous live RR ingress over LSL, feeding the shared physio core.

Scope and honest limits
-----------------------

* This module ships a single live transport: **LSL** (Lab Streaming Layer,
  ``pylsl``). LSL was chosen after the objective viability probe defined
  in the v1.1 task brief (``pip install pylsl`` + ``import pylsl`` +
  loopback producer / consumer smoke test, all exit 0).
* This is a **live physiological transport path**, not a neural gate
  and not a medical device. HRV-style features remain signal-quality
  indicators only.
* Shared core: every valid sample is fed into :class:`PhysioSession`,
  the same class that powers :class:`PhysioReplayPipeline`. There is
  no duplicated feature or gate logic. Replay and live diverge only at
  the ingress boundary.

Sample schema
-------------

The LSL stream is double64 with two channels::

    channel 0  timestamp_s  (seconds, monotonic-clock source)
    channel 1  rr_ms        (R-R interval in milliseconds)

``double64`` is required because producers push absolute
``time.monotonic()`` values. On a host with uptime ≳ 3 h the monotonic
value exceeds 10⁴ s, at which magnitude float32 ULP is ≥ 10⁻³ s — larger
than the 20 ms producer cadence — so consecutive samples would collide to
the same float32 timestamp and every second sample would be rejected by
the consumer's monotonicity guard. See CHANGELOG for the 2026-04-21
incident write-up.

A special EOF sentinel is ``(NaN, NaN)``: on receipt, the consumer
emits a final summary and exits ``0``.

Run
---

Consumer::

    python -m neurophase.physio.live --stream-name neurophase-rr

Producer (for validation / smoke tests)::

    python -m neurophase.physio.live_producer --stream-name neurophase-rr

See ``tests/test_physio_live.py`` for the handshake-based integration
tests proving this is a truly live path (independent producer and
consumer processes, post-readiness sample triggers frame within 1.0 s,
stall detection, clean EOF, fatal errors yield non-zero exit).
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TextIO

from neurophase.physio.features import DEFAULT_WINDOW_SIZE
from neurophase.physio.gate import (
    DEFAULT_THRESHOLD_ABSTAIN,
    DEFAULT_THRESHOLD_ALLOW,
    PhysioGateState,
)
from neurophase.physio.ledger import LedgerConfig, PhysioLedger
from neurophase.physio.pipeline import CANONICAL_FRAME_SCHEMA_VERSION, PhysioSession
from neurophase.physio.profile import ProfileValidationError, load_profile
from neurophase.physio.replay import ReplayIngestError, RRSample

if TYPE_CHECKING:
    from pylsl import StreamInlet

LSL_CHANNEL_COUNT: int = 2
LSL_CHANNEL_FORMAT: str = "double64"
LSL_STREAM_TYPE: str = "RR"
STALL_TIMEOUT_SAFE_MIN_S: float = 2.0
STALL_TIMEOUT_SAFE_MAX_S: float = 30.0


@dataclass(frozen=True)
class LiveConfig:
    """SINGLE config location for live-path timing + transport semantics.

    No hidden magic numbers. Each field is documented. ``stall_timeout_s``
    defaults to ``5.0`` and can be overridden only within the safe
    range ``[2.0, 30.0]`` per the v1.1 spec.
    """

    stream_name: str = "neurophase-rr"
    stream_type: str = LSL_STREAM_TYPE

    #: If no valid sample arrives within this many seconds, emit a
    #: SENSOR_DEGRADED stall frame. Must stay in [2.0, 30.0].
    stall_timeout_s: float = 5.0

    #: Per-read LSL pull timeout. Must be > 0 and < stall_timeout_s
    #: so the loop wakes up often enough to detect stalls.
    read_timeout_s: float = 0.25

    #: How long to wait for the LSL stream to appear on start-up.
    resolve_timeout_s: float = 15.0

    #: 0 = unbounded; non-zero caps the number of frames emitted before
    #: a clean shutdown (useful for deterministic tests).
    max_frames: int = 0

    #: Rolling window / admission-threshold knobs (mirror PhysioSession).
    window_size: int = DEFAULT_WINDOW_SIZE
    threshold_allow: float = DEFAULT_THRESHOLD_ALLOW
    threshold_abstain: float = DEFAULT_THRESHOLD_ABSTAIN

    def __post_init__(self) -> None:
        if not (STALL_TIMEOUT_SAFE_MIN_S <= self.stall_timeout_s <= STALL_TIMEOUT_SAFE_MAX_S):
            raise ValueError(
                f"stall_timeout_s={self.stall_timeout_s!r} outside safe range "
                f"[{STALL_TIMEOUT_SAFE_MIN_S}, {STALL_TIMEOUT_SAFE_MAX_S}]"
            )
        if not (0.0 < self.read_timeout_s < self.stall_timeout_s):
            raise ValueError(
                f"read_timeout_s={self.read_timeout_s!r} must be > 0 and "
                f"< stall_timeout_s={self.stall_timeout_s!r}"
            )
        if self.resolve_timeout_s <= 0:
            raise ValueError(f"resolve_timeout_s={self.resolve_timeout_s!r} must be > 0")
        if self.max_frames < 0:
            raise ValueError(f"max_frames={self.max_frames!r} must be >= 0")


# --- Event emission (JSON-lines) -----------------------------------------


def _emit(
    event: dict[str, Any],
    out: TextIO,
    *,
    ledger: PhysioLedger | None = None,
) -> None:
    """Write one JSON-lines event to *out* and (if set) tee to *ledger*.

    The ledger receives the raw event dict unchanged, so that the
    persisted trace is byte-identical with the stdout trace modulo
    ordering (ledger prepends SESSION_HEADER + appends SESSION_SUMMARY
    on context-manager entry/exit).
    """
    out.write(json.dumps(event, default=str))
    out.write("\n")
    out.flush()
    if ledger is not None:
        ledger.write_event(event)


# --- Stream resolution ---------------------------------------------------


def _resolve_inlet(
    config: LiveConfig,
    *,
    out: TextIO,
    ledger: PhysioLedger | None = None,
) -> StreamInlet | None:
    """Poll-based LSL stream resolution with a bounded total timeout.

    Returns ``None`` if the stream never appears inside
    ``config.resolve_timeout_s``. The caller is expected to emit a
    final summary and exit non-zero.
    """
    from pylsl import StreamInlet, resolve_byprop

    deadline = time.monotonic() + config.resolve_timeout_s
    while time.monotonic() < deadline:
        streams = resolve_byprop("name", config.stream_name, 1, 0.5)
        if streams:
            info = streams[0]
            if info.channel_count() != LSL_CHANNEL_COUNT:
                _emit(
                    {
                        "event": "FATAL",
                        "reason": (
                            f"stream channel_count={info.channel_count()} "
                            f"!= expected {LSL_CHANNEL_COUNT}"
                        ),
                    },
                    out,
                    ledger=ledger,
                )
                return None
            return StreamInlet(info)
    return None


# --- Consumer loop -------------------------------------------------------


def _run_consumer(
    config: LiveConfig,
    *,
    out: TextIO,
    readiness_out: TextIO | None = None,
    ledger: PhysioLedger | None = None,
) -> int:
    """Core consumer loop. Returns the process exit code."""
    _emit(
        {
            "event": "LISTENING",
            "stream_name": config.stream_name,
            "resolve_timeout_s": config.resolve_timeout_s,
            "stall_timeout_s": config.stall_timeout_s,
            "schema_version": CANONICAL_FRAME_SCHEMA_VERSION,
        },
        readiness_out or out,
        ledger=ledger,
    )

    inlet = _resolve_inlet(config, out=out, ledger=ledger)
    if inlet is None:
        _emit(
            {
                "event": "FATAL",
                "reason": f"stream {config.stream_name!r} not found within "
                f"{config.resolve_timeout_s}s",
            },
            out,
            ledger=ledger,
        )
        return 2

    _emit(
        {"event": "READY", "stream_name": config.stream_name},
        readiness_out or out,
        ledger=ledger,
    )

    session = PhysioSession(
        window_size=config.window_size,
        threshold_allow=config.threshold_allow,
        threshold_abstain=config.threshold_abstain,
    )

    state_counts: Counter[str] = Counter()
    n_ingest_errors = 0
    n_stalls_emitted = 0
    last_valid_rx_mono: float = time.monotonic()
    last_ts_s: float | None = None
    tick_index = 0
    clean_exit = False

    try:
        while True:
            if config.max_frames and tick_index >= config.max_frames:
                clean_exit = True
                break

            sample, lsl_ts = inlet.pull_sample(timeout=config.read_timeout_s)
            now_mono = time.monotonic()

            if sample is None:
                # No sample this window; check stall.
                if now_mono - last_valid_rx_mono > config.stall_timeout_s:
                    _emit(
                        {
                            "event": "STALL",
                            "seconds_since_last_sample": round(now_mono - last_valid_rx_mono, 3),
                            "gate_state": PhysioGateState.SENSOR_DEGRADED.name,
                            "execution_allowed": False,
                            "reason": "stall timeout exceeded",
                        },
                        out,
                        ledger=ledger,
                    )
                    n_stalls_emitted += 1
                    # Reset the clock so we don't flood STALL events.
                    last_valid_rx_mono = now_mono
                continue

            # EOF sentinel: either channel NaN -> clean shutdown.
            ts_s = float(sample[0])
            rr_ms = float(sample[1])
            if math.isnan(ts_s) or math.isnan(rr_ms):
                clean_exit = True
                break

            # Monotonicity check at the live ingress boundary.
            if last_ts_s is not None and ts_s <= last_ts_s:
                n_ingest_errors += 1
                _emit(
                    {
                        "event": "INGEST_REJECTED",
                        "reason": (f"non-monotonic timestamp_s={ts_s!r} <= previous {last_ts_s!r}"),
                        "gate_state": PhysioGateState.SENSOR_DEGRADED.name,
                        "execution_allowed": False,
                    },
                    out,
                    ledger=ledger,
                )
                continue

            # Shape / envelope validation via the existing RRSample contract.
            try:
                rr_sample = RRSample(timestamp_s=ts_s, rr_ms=rr_ms, row_index=tick_index)
            except ReplayIngestError as exc:
                n_ingest_errors += 1
                _emit(
                    {
                        "event": "INGEST_REJECTED",
                        "reason": str(exc),
                        "gate_state": PhysioGateState.SENSOR_DEGRADED.name,
                        "execution_allowed": False,
                    },
                    out,
                    ledger=ledger,
                )
                continue

            # Shared core: replay and live use the exact same step().
            frame = session.step(rr_sample, tick_index=tick_index)
            consumer_rx_mono = now_mono
            _emit(
                {
                    "event": "FRAME",
                    "tick_index": frame.tick_index,
                    "timestamp_s": frame.timestamp_s,
                    "rr_ms": frame.rr_ms,
                    "confidence": frame.decision.confidence,
                    "gate_state": frame.decision.state.name,
                    "execution_allowed": frame.decision.execution_allowed,
                    "kernel_state": frame.decision.kernel_state.name,
                    "producer_mono_s": ts_s,
                    "consumer_rx_mono_s": consumer_rx_mono,
                    "latency_s": round(consumer_rx_mono - ts_s, 6),
                    "lsl_ts": lsl_ts,
                    "source_mode": "live-lsl",
                },
                out,
                ledger=ledger,
            )
            state_counts[frame.decision.state.name] += 1
            tick_index += 1
            last_ts_s = ts_s
            last_valid_rx_mono = now_mono
    except KeyboardInterrupt:
        clean_exit = True
    except Exception as exc:
        _emit(
            {
                "event": "FATAL",
                "reason": f"{type(exc).__name__}: {exc}",
            },
            out,
            ledger=ledger,
        )
        return 3

    _emit(
        {
            "event": "SUMMARY",
            "n_frames": tick_index,
            "n_ingest_errors": n_ingest_errors,
            "n_stalls_emitted": n_stalls_emitted,
            "state_counts": dict(state_counts),
            "clean_exit": clean_exit,
        },
        out,
        ledger=ledger,
    )
    return 0


# --- CLI -----------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurophase.physio.live",
        description=(
            "True asynchronous live RR ingress over LSL, feeding the shared "
            "physio gate. Replay-compatible core. Not a neural gate and "
            "not a medical device."
        ),
    )
    parser.add_argument(
        "--stream-name",
        default="neurophase-rr",
        help="LSL stream name to resolve (default: neurophase-rr).",
    )
    parser.add_argument(
        "--stall-timeout-s",
        type=float,
        default=5.0,
        help=(
            f"Seconds without a valid sample before a STALL is emitted. "
            f"Must be in [{STALL_TIMEOUT_SAFE_MIN_S}, {STALL_TIMEOUT_SAFE_MAX_S}]."
        ),
    )
    parser.add_argument(
        "--read-timeout-s",
        type=float,
        default=0.25,
        help="Per-pull LSL timeout (must be > 0 and < stall_timeout_s).",
    )
    parser.add_argument(
        "--resolve-timeout-s",
        type=float,
        default=15.0,
        help="Seconds to wait for the LSL stream to appear on start-up.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="0 = unbounded; otherwise cap frames for deterministic tests.",
    )
    parser.add_argument(
        "--ledger-out",
        type=str,
        default=None,
        help=(
            "Optional path for an append-only JSONL session ledger. "
            "Every stdout event is tee'd to this file with a "
            "SESSION_HEADER prepended and a SESSION_SUMMARY appended. "
            "Replayable offline via neurophase.physio.session_replay."
        ),
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=(
            "Path to a per-user PhysioProfile JSON produced by "
            "tools/calibrate_physio.py. When set, the gate runs in "
            "calibrated mode: thresholds come from the profile, not "
            "from the illustrative defaults. Without --profile the "
            "gate stays in default mode."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    # Profile loading (fail-closed: a broken profile aborts before LSL).
    profile_overrides: dict[str, Any] = {}
    profile_user_id: str | None = None
    mode = "default"
    if args.profile:
        try:
            profile = load_profile(args.profile)
        except ProfileValidationError as exc:
            _emit(
                {"event": "FATAL", "reason": f"profile load failed: {exc}"},
                sys.stderr,
            )
            return 2
        profile_overrides["threshold_allow"] = profile.threshold_allow
        profile_overrides["threshold_abstain"] = profile.threshold_abstain
        profile_overrides["window_size"] = profile.window_size
        profile_user_id = profile.user_id
        mode = "calibrated"

    try:
        config = LiveConfig(
            stream_name=args.stream_name,
            stall_timeout_s=args.stall_timeout_s,
            read_timeout_s=args.read_timeout_s,
            resolve_timeout_s=args.resolve_timeout_s,
            max_frames=args.max_frames,
            **profile_overrides,
        )
    except ValueError as exc:
        _emit({"event": "FATAL", "reason": f"bad config: {exc}"}, sys.stderr)
        return 2

    _emit(
        {
            "event": "GATE_MODE",
            "mode": mode,
            "profile_user_id": profile_user_id,
            "threshold_allow": config.threshold_allow,
            "threshold_abstain": config.threshold_abstain,
        },
        sys.stdout,
    )

    # Ensure we respond to SIGTERM cleanly on typical Linux test runners.
    def _sig_handler(_sig: int, _frm: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sig_handler)

    if args.ledger_out:
        ledger_config = LedgerConfig(
            source_mode="live-lsl",
            stream_name=config.stream_name,
            window_size=config.window_size,
            threshold_allow=config.threshold_allow,
            threshold_abstain=config.threshold_abstain,
            stall_timeout_s=config.stall_timeout_s,
        )
        with PhysioLedger(args.ledger_out, config=ledger_config) as ledger:
            return _run_consumer(config, out=sys.stdout, ledger=ledger)
    return _run_consumer(config, out=sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
