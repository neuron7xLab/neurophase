#!/usr/bin/env python3
"""Ops-level telemetry for the three metrics that matter:

  * recovery_latency   — how long from first CI red to next all-green push
  * fix_diagnosis_time — from fail_detected to fix_push (think time)
  * fix_validation_time — from fix_push to all-green (CI confirmation)
  * first_push_green_rate — fraction of pushes that land all-green on first try

Reads ``gh run list`` output (JSON) for a repo's default branch and
computes the metrics over the last N push-triggered runs. Intended for
weekly review, not continuous monitoring — this is a *baseline drift*
detector, not a pager.

Usage:
    python scripts/ops_metrics.py --repo owner/name [--limit 50]

Requires ``gh`` CLI authenticated with read access to the repo.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta


def _iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@dataclass(frozen=True)
class PushState:
    sha: str
    created_at: datetime
    last_update: datetime
    green: bool  # all observed workflows succeeded


def _fetch_runs(repo: str, limit: int) -> list[dict[str, str]]:
    cmd = [
        "gh",
        "run",
        "list",
        "--repo",
        repo,
        "--branch",
        "main",
        "--limit",
        str(limit),
        "--json",
        "databaseId,name,conclusion,headSha,createdAt,updatedAt,event",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw: list[dict[str, str]] = json.loads(proc.stdout)
    return [r for r in raw if r.get("event") == "push"]


def _group_by_push(runs: Iterable[dict[str, str]]) -> list[PushState]:
    by_sha: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in runs:
        by_sha[r["headSha"]].append(r)

    pushes: list[PushState] = []
    for sha, jobs in by_sha.items():
        created = min(_iso(j["createdAt"]) for j in jobs)
        updated = max(_iso(j["updatedAt"]) for j in jobs)
        # A push is "green" only if every observed workflow succeeded.
        # ``skipped`` and ``neutral`` are treated as non-green conservatively.
        green = all(j.get("conclusion") == "success" for j in jobs)
        pushes.append(PushState(sha=sha[:8], created_at=created, last_update=updated, green=green))
    pushes.sort(key=lambda p: p.created_at)
    return pushes


def _fmt(td: timedelta) -> str:
    s = int(td.total_seconds())
    return f"{s // 60}m{s % 60:02d}s"


def _report(pushes: list[PushState]) -> None:
    if not pushes:
        print("no push-triggered runs found", file=sys.stderr)
        sys.exit(1)

    total = len(pushes)
    green_first = sum(1 for p in pushes if p.green)
    rate = green_first / total

    # Detect red→green cycles. A cycle starts at the first red push in a
    # run of consecutive reds and ends at the next green push.
    cycles: list[tuple[PushState, PushState]] = []
    i = 0
    while i < len(pushes):
        if not pushes[i].green:
            j = i
            while j < len(pushes) and not pushes[j].green:
                j += 1
            if j < len(pushes):
                cycles.append((pushes[i], pushes[j]))
            i = j + 1
        else:
            i += 1

    print(
        f"# ops metrics — window: {pushes[0].created_at:%Y-%m-%d %H:%M}"
        f" → {pushes[-1].last_update:%Y-%m-%d %H:%M} UTC"
    )
    print(f"# pushes analyzed: {total}")
    print(f"# first_push_green_rate: {rate:.0%} ({green_first}/{total})")
    print()
    if not cycles:
        print("no red→green cycles in window (either all green or no recovery yet)")
        return

    print(f"# recovery cycles: {len(cycles)}")
    print()
    for idx, (red, green) in enumerate(cycles, 1):
        diagnosis = green.created_at - red.last_update
        validation = green.last_update - green.created_at
        total_td = green.last_update - red.last_update
        print(f"  cycle {idx}: {red.sha} (red) → {green.sha} (green)")
        print(f"    diagnosis (fail → fix push)  : {_fmt(diagnosis)}")
        print(f"    validation (fix push → green): {_fmt(validation)}")
        print(f"    total recovery               : {_fmt(total_td)}")
        print()

    diag = [g.created_at - r.last_update for r, g in cycles]
    val = [g.last_update - g.created_at for _, g in cycles]
    print("# medians")
    diag_m = sorted(diag)[len(diag) // 2]
    val_m = sorted(val)[len(val) // 2]
    print(f"  diagnosis : {_fmt(diag_m)}")
    print(f"  validation: {_fmt(val_m)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="e.g. owner/name")
    ap.add_argument(
        "--limit", type=int, default=50, help="how many recent runs to pull (default 50)"
    )
    args = ap.parse_args()

    runs = _fetch_runs(args.repo, args.limit)
    pushes = _group_by_push(runs)
    _report(pushes)


if __name__ == "__main__":
    main()
