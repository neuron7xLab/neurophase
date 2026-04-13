#!/usr/bin/env python3
"""tools/calibrate_physio.py -- build a user's PhysioProfile from baseline sessions.

Standalone CLI. Lives under ``tools/`` because it is an operator tool,
not a runtime kernel module, but it exercises only public
:mod:`neurophase.physio` APIs and writes a JSON profile that the kernel
can load in ``calibrated`` mode.

Usage::

    python tools/calibrate_physio.py \
        --user-id alex-2026-04 \
        --out profiles/alex-2026-04.json \
        --csv baselines/morning.csv \
        --csv baselines/post-load.csv \
        --ledger baselines/focus.jsonl \
        --ledger baselines/recovery.jsonl

Rules (all from :mod:`neurophase.physio.calibration`):

* at least 3 baseline sessions,
* at least 32 healthy frames per session,
* at least 128 healthy frames in total.

Any shortfall raises :class:`CalibrationError` and exits non-zero with
a concrete reason. The tool never writes a profile that fails the
post-init validation of :class:`PhysioProfile`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the kernel importable when running from a fresh checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from neurophase.physio.calibration import (  # noqa: E402
    CalibrationError,
    SessionSource,
    calibrate_profile,
    calibration_report,
)
from neurophase.physio.features import DEFAULT_WINDOW_SIZE  # noqa: E402
from neurophase.physio.profile import save_profile  # noqa: E402

EXIT_OK: int = 0
EXIT_CALIBRATION_FAIL: int = 1
EXIT_USAGE: int = 2


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="calibrate_physio",
        description=(
            "Build a per-user PhysioProfile (JSON) from baseline sessions. "
            "Baseline sessions can be replay CSVs (tools/rr.csv) or live "
            "session JSONL ledgers."
        ),
    )
    p.add_argument("--user-id", required=True, help="Opaque user identifier.")
    p.add_argument("--out", type=Path, required=True, help="Output profile JSON path.")
    p.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=[],
        help="Baseline session via CSV (repeatable).",
    )
    p.add_argument(
        "--ledger",
        type=Path,
        action="append",
        default=[],
        help="Baseline session via JSONL ledger (repeatable).",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=f"Rolling window depth (default: {DEFAULT_WINDOW_SIZE}).",
    )
    p.add_argument(
        "--note",
        action="append",
        default=[],
        help="Free-form operator note to attach to the profile (repeatable).",
    )
    p.add_argument(
        "--json-report",
        action="store_true",
        help="Also print a JSON calibration report to stdout on success.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    sources: list[SessionSource] = []
    for csv in args.csv:
        sources.append(SessionSource(csv_path=csv))
    for ledger in args.ledger:
        sources.append(SessionSource(ledger_path=ledger))

    if not sources:
        sys.stderr.write("no baseline sources provided (use --csv / --ledger)\n")
        return EXIT_USAGE

    try:
        profile = calibrate_profile(
            sources,
            user_id=args.user_id,
            window_size=args.window_size,
            notes=tuple(args.note),
        )
    except CalibrationError as exc:
        sys.stderr.write(f"calibration failed: {exc}\n")
        return EXIT_CALIBRATION_FAIL

    out_path = save_profile(profile, args.out)
    sys.stderr.write(f"profile written: {out_path}\n")
    if args.json_report:
        sys.stdout.write(json.dumps(calibration_report(profile), indent=2, default=str) + "\n")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
