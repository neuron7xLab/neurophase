"""Deterministic LSL validation producer for the live physio path.

This is a minimal validation tool, not a device driver and not a fake
physiology generator. It:

* creates one LSL outlet matching the contract expected by
  :mod:`neurophase.physio.live`;
* pushes a hard-coded sequence of 24 RR samples followed by a
  ``(NaN, NaN)`` EOF sentinel;
* exits ``0`` on normal completion.

Samples are labelled as synthetic validation samples in the LSL stream
metadata so downstream audit can never confuse them with real data.

Run::

    python -m neurophase.physio.live_producer --stream-name neurophase-rr

Flags:

    --stream-name STR    LSL stream name (must match the consumer's).
    --consumer-wait-s F  Max seconds to wait for a consumer to subscribe
                         before pushing samples (default 10).
    --inter-sample-s F   Delay between consecutive pushes (default 0.05).
    --stall-after INT    Optional: after N samples pause for --stall-hold-s
                         seconds, then resume. Used by the consumer's stall
                         detection integration test.
    --stall-hold-s F     Pause duration for --stall-after (default 3.0).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass

from neurophase.physio.live import LSL_CHANNEL_COUNT, LSL_CHANNEL_FORMAT, LSL_STREAM_TYPE

# Hard-coded 24-sample deterministic sequence, exported for tests.
# Regimes:
#   * 0..15  (16 samples)  warm-up + buffer fill,  RR ~ 820 ms stable
#   * 16..23 (8  samples)  post-readiness stable regime
# All values are in the physiological envelope (300 ms, 2000 ms).
# NOT a realistic HRV generator -- purely enough structure to validate
# transport + state transitions.
RR_SEQUENCE_MS: tuple[float, ...] = (
    820.0,
    825.0,
    818.0,
    822.0,
    830.0,
    815.0,
    819.0,
    824.0,
    820.0,
    823.0,
    821.0,
    822.0,
    820.0,
    819.0,
    821.0,
    823.0,
    820.0,
    822.0,
    819.0,
    821.0,
    820.0,
    823.0,
    821.0,
    822.0,
)
assert len(RR_SEQUENCE_MS) == 24


@dataclass(frozen=True)
class ProducerConfig:
    stream_name: str = "neurophase-rr"
    consumer_wait_s: float = 10.0
    inter_sample_s: float = 0.05
    stall_after: int = 0  # 0 = disabled
    stall_hold_s: float = 3.0
    #: If >= 0, replace the sample at this index with an impossible RR
    #: (100 ms) to exercise the consumer's fail-closed ingest check.
    inject_impossible_rr_at: int = -1
    #: If >= 0, send a timestamp 1 s BELOW the previous one at this
    #: index to exercise monotonicity rejection.
    inject_backward_ts_at: int = -1
    source_id: str = "neurophase-live-producer-v1"

    def __post_init__(self) -> None:
        if self.consumer_wait_s <= 0:
            raise ValueError("consumer_wait_s must be > 0")
        if self.inter_sample_s < 0:
            raise ValueError("inter_sample_s must be >= 0")
        if self.stall_after < 0:
            raise ValueError("stall_after must be >= 0")
        if self.stall_hold_s <= 0:
            raise ValueError("stall_hold_s must be > 0")
        if self.inject_impossible_rr_at >= len(RR_SEQUENCE_MS):
            raise ValueError(
                f"inject_impossible_rr_at={self.inject_impossible_rr_at!r} "
                f">= sequence length {len(RR_SEQUENCE_MS)}"
            )
        if self.inject_backward_ts_at >= len(RR_SEQUENCE_MS):
            raise ValueError(
                f"inject_backward_ts_at={self.inject_backward_ts_at!r} "
                f">= sequence length {len(RR_SEQUENCE_MS)}"
            )


def _emit_event(event: dict[str, object]) -> None:
    print(json.dumps(event, default=str), flush=True)


def run_producer(config: ProducerConfig) -> int:
    from pylsl import StreamInfo, StreamOutlet

    info = StreamInfo(
        name=config.stream_name,
        type=LSL_STREAM_TYPE,
        channel_count=LSL_CHANNEL_COUNT,
        nominal_srate=0.0,  # irregular stream
        channel_format=LSL_CHANNEL_FORMAT,
        source_id=config.source_id,
    )
    # Explicit provenance in LSL XML metadata.
    desc = info.desc()
    desc.append_child_value("origin", "neurophase-live-producer")
    desc.append_child_value("note", "SYNTHETIC VALIDATION SAMPLES - NOT REAL DATA")
    desc.append_child_value("sample_contract", "ch0=timestamp_s, ch1=rr_ms")
    outlet = StreamOutlet(info)
    _emit_event(
        {
            "event": "OUTLET_CREATED",
            "stream_name": config.stream_name,
            "channel_count": LSL_CHANNEL_COUNT,
            "sequence_length": len(RR_SEQUENCE_MS),
        }
    )

    # Wait for consumer to subscribe (deterministic readiness handshake).
    deadline = time.monotonic() + config.consumer_wait_s
    while time.monotonic() < deadline and not outlet.have_consumers():
        time.sleep(0.05)
    if not outlet.have_consumers():
        _emit_event(
            {
                "event": "NO_CONSUMER",
                "reason": f"no subscriber within {config.consumer_wait_s}s",
            }
        )
        return 4
    _emit_event({"event": "CONSUMER_SEEN"})

    # Push samples. Injections (if configured) override the normal value
    # for a specific index to exercise the consumer's fail-closed paths.
    last_mono: float = 0.0
    for i, rr_ms in enumerate(RR_SEQUENCE_MS):
        mono = time.monotonic()
        if i == config.inject_backward_ts_at and last_mono > 0:
            mono = last_mono - 1.0  # deliberate non-monotonic
        if i == config.inject_impossible_rr_at:
            rr_ms = 100.0  # below physiological envelope of 300 ms
        outlet.push_sample([mono, rr_ms])
        last_mono = mono
        _emit_event(
            {
                "event": "SENT",
                "index": i,
                "rr_ms": rr_ms,
                "mono_s": mono,
            }
        )
        # Optional induced stall for integration testing.
        if config.stall_after and i + 1 == config.stall_after:
            _emit_event(
                {
                    "event": "STALL_HOLD_BEGIN",
                    "hold_s": config.stall_hold_s,
                }
            )
            time.sleep(config.stall_hold_s)
            _emit_event({"event": "STALL_HOLD_END"})
        elif config.inter_sample_s > 0:
            time.sleep(config.inter_sample_s)

    # EOF sentinel: (NaN, NaN) -> consumer exits cleanly.
    outlet.push_sample([math.nan, math.nan])
    # Small grace so the sentinel traverses LSL before outlet closure.
    time.sleep(0.2)
    _emit_event(
        {
            "event": "DONE",
            "n_samples_sent": len(RR_SEQUENCE_MS),
        }
    )
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neurophase.physio.live_producer",
        description=(
            "Deterministic LSL validation producer. Emits 24 SYNTHETIC RR "
            "samples + EOF sentinel. Not a device driver; not real data."
        ),
    )
    p.add_argument("--stream-name", default="neurophase-rr")
    p.add_argument("--consumer-wait-s", type=float, default=10.0)
    p.add_argument("--inter-sample-s", type=float, default=0.05)
    p.add_argument("--stall-after", type=int, default=0)
    p.add_argument("--stall-hold-s", type=float, default=3.0)
    p.add_argument(
        "--inject-impossible-rr-at",
        type=int,
        default=-1,
        help="Replace RR at this index with 100 ms (below envelope) for fail-closed testing.",
    )
    p.add_argument(
        "--inject-backward-ts-at",
        type=int,
        default=-1,
        help="Send a backward timestamp at this index for monotonicity-rejection testing.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    try:
        config = ProducerConfig(
            stream_name=args.stream_name,
            consumer_wait_s=args.consumer_wait_s,
            inter_sample_s=args.inter_sample_s,
            stall_after=args.stall_after,
            stall_hold_s=args.stall_hold_s,
            inject_impossible_rr_at=args.inject_impossible_rr_at,
            inject_backward_ts_at=args.inject_backward_ts_at,
        )
    except ValueError as exc:
        _emit_event({"event": "FATAL", "reason": f"bad config: {exc}"})
        return 2
    return run_producer(config)


if __name__ == "__main__":
    raise SystemExit(main())
