"""One-command replay-to-gate demo.

Run::

    python -m neurophase.physio.demo

or point at a custom CSV::

    python -m neurophase.physio.demo --csv path/to/rr.csv

The demo:

1. Reads a short bundled replay CSV
   (``examples/data/physio_replay_sample.csv``).
2. Feeds it through :class:`PhysioReplayPipeline` with default settings.
3. Prints per-tick decisions + state transitions.
4. Prints a final state-count summary.
5. Optionally writes the full frame sequence as JSON.

Time-to-first-result on a clean setup: under ten minutes.
No hardware is accessed; no external network is accessed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from neurophase.physio.pipeline import PhysioFrame, PhysioReplayPipeline, PhysioRunSummary
from neurophase.physio.replay import ReplayIngestError

_SAMPLE_CSV_PATH: Path = (
    Path(__file__).resolve().parents[2] / "examples" / "data" / "physio_replay_sample.csv"
)


def _print_banner() -> None:
    bar = "=" * 72
    print(bar)
    print("  NEUROPHASE  -  Physio replay-to-gate demo  (v1)")
    print(bar)
    print(
        "  Honest scope: replay-only. No live device. HRV features are used as\n"
        "  signal-quality indicators, not clinical or trading primitives."
    )
    print(bar)
    print()


def _print_transitions(frames: list[PhysioFrame]) -> None:
    last_state: str | None = None
    for frame in frames:
        name = frame.decision.state.name
        if name != last_state:
            print(
                f"  tick {frame.tick_index:>4}  t={frame.timestamp_s:7.3f}s  "
                f"rr={frame.rr_ms:7.2f}ms  "
                f"conf={frame.decision.confidence:.3f}  "
                f"{name}  -  {frame.decision.reason}"
            )
            last_state = name


def _print_summary(summary: PhysioRunSummary) -> None:
    print()
    print("-" * 72)
    print("  Summary")
    print("-" * 72)
    print(f"  frames emitted:        {summary.n_frames_emitted}")
    print(f"  execution allowed:     {summary.n_execution_allowed}")
    print()
    print("  state counts:")
    for name, count in sorted(summary.state_counts.items()):
        tick = summary.first_state_tick.get(name, -1)
        print(f"    {name:<18} {count:>5}   first seen at tick {tick}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="neurophase.physio.demo",
        description=(
            "Replay-only physio-quality gate demo. "
            "Not a live device path, not a medical instrument, "
            "not a trading-alpha signal."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_SAMPLE_CSV_PATH,
        help=f"replay CSV path (default: {_SAMPLE_CSV_PATH})",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional JSON dump of the full frame sequence",
    )
    args = parser.parse_args(argv)

    _print_banner()
    print(f"  input CSV: {args.csv}")
    print()

    pipeline = PhysioReplayPipeline()
    try:
        frames, summary = pipeline.run_csv(args.csv)
    except ReplayIngestError as exc:
        print(f"  REPLAY INGEST ERROR (fail-closed): {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"  FILE NOT FOUND (fail-closed): {exc}", file=sys.stderr)
        return 2

    if not frames:
        print("  no samples ingested - nothing to gate.")
        return 1

    _print_transitions(frames)
    _print_summary(summary)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        dump = {
            "summary": summary.to_json_dict(),
            "frames": [f.to_json_dict() for f in frames],
        }
        args.json_out.write_text(json.dumps(dump, indent=2), encoding="utf-8")
        print()
        print(f"  full frame dump written to: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
