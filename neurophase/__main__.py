"""Command-line interface for ``neurophase``.

Usage
-----

.. code-block:: console

    $ python -m neurophase version
    $ python -m neurophase demo
    $ python -m neurophase verify-ledger <path>
    $ python -m neurophase explain-ledger <path>

Subcommands are deliberately minimal. The CLI is a **hands-on
inspection surface**, not an orchestration layer — production
runtimes should import from :mod:`neurophase.api` and drive the
pipeline themselves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from neurophase import __version__
from neurophase.api import create_pipeline, explain_decision
from neurophase.audit.decision_ledger import verify_ledger
from neurophase.runtime.pipeline import DecisionFrame


def _cmd_version(_args: argparse.Namespace) -> int:
    print(f"neurophase {__version__}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run a short synthetic pipeline and print gate states."""
    pipeline = create_pipeline(
        threshold=0.65,
        warmup_samples=2,
        stream_window=4,
        max_fault_rate=0.50,
        enable_stillness=True,
        stillness_window=4,
        stillness_delta_min=0.20,
    )
    n = max(1, int(args.ticks))
    print(f"# neurophase demo — {n} ticks")
    print(f"{'tick':>4}  {'R':>6}  {'δ':>6}  {'state':<14}  reason")
    print("-" * 80)
    for i in range(n):
        R = 0.95 if i < n // 2 else 0.30  # second half drops below threshold
        delta = 0.01
        frame = pipeline.tick(timestamp=float(i) * 0.1, R=R, delta=delta)
        state = frame.gate_state.name
        print(
            f"{frame.tick_index:>4}  {R:>6.3f}  {delta:>6.3f}  "
            f"{state:<14}  {frame.gate.reason[:70]}"
        )
    return 0


def _cmd_verify_ledger(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser().resolve()
    verification = verify_ledger(path)
    if verification.ok:
        print(f"✓ ledger verified: {verification.n_records} records at {path}")
        return 0
    print(f"✗ ledger broken at record {verification.first_broken_index}: {verification.reason}")
    return 1


def _cmd_explain_ledger(args: argparse.Namespace) -> int:
    """Print one ``DecisionExplanation`` per record in a ledger file.

    This is the postmortem-friendly rendering: for each stored
    decision, reconstruct a minimal frame and run
    :func:`explain_decision` against it. The output is a JSONL
    stream so downstream tools can consume it.

    Note: the explanation is reconstructed from the ledger record's
    materialised fields only. Stillness state and temporal quality
    are stored in the ``extras`` dict by the pipeline, so the
    explanation is lossy for the stream regime layer (it is
    collapsed into the time quality field). This is by design: the
    CLI is for *hands-on inspection*, not for reproducing the full
    pipeline — use :func:`neurophase.audit.replay.replay_ledger` for
    byte-exact replay.
    """
    from dataclasses import dataclass as _dc
    from dataclasses import field as _field

    from neurophase.data.stream_detector import StreamQualityStats, StreamRegime
    from neurophase.data.temporal_validator import (
        TimeQuality,
    )
    from neurophase.gate.execution_gate import (
        GateDecision,
        GateState,
    )
    from neurophase.gate.stillness_detector import StillnessState

    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        print(f"✗ ledger not found: {path}", file=sys.stderr)
        return 2

    # Re-use the minimal frame idea from explain_gate but
    # hydrated from the ledger record fields.
    @_dc(frozen=True)
    class _MinimalTemporal:
        quality: TimeQuality
        reason: str
        ts: float = 0.0
        last_ts: float | None = None
        gap_seconds: float | None = None
        staleness_seconds: float | None = None
        warmup_remaining: int = 0

    @_dc(frozen=True)
    class _MinimalStream:
        regime: StreamRegime
        reason: str
        stats: StreamQualityStats = _field(
            default_factory=lambda: StreamQualityStats(
                total=0,
                valid=0,
                gapped=0,
                stale=0,
                reversed=0,
                duplicate=0,
                invalid=0,
                warmup=0,
                fault_rate=0.0,
            )
        )

    @_dc(frozen=True)
    class _MinimalFrame:
        tick_index: int
        timestamp: float
        R: float | None
        delta: float | None
        temporal: _MinimalTemporal
        stream: _MinimalStream
        gate: GateDecision

    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            extras = rec.get("extras", {}) or {}
            quality_name = extras.get("time_quality", "VALID")
            regime_name = extras.get("stream_regime", "HEALTHY")
            tq = TimeQuality[quality_name]
            reg = StreamRegime[regime_name]
            gs = GateState[rec["gate_state"]]
            stillness_name = extras.get("stillness_state")
            stillness_value = StillnessState[stillness_name] if stillness_name else None
            # Reconstruct a GateDecision with the invariant-compatible
            # state/execution combination we already know holds.
            gate_decision = GateDecision(
                state=gs,
                execution_allowed=bool(rec["execution_allowed"]),
                R=rec["R"],
                threshold=float(rec["threshold"]),
                reason=str(rec["reason"]),
                stillness_state=stillness_value,
            )
            frame = _MinimalFrame(
                tick_index=int(extras.get("tick_index", rec["index"])),
                timestamp=float(rec["timestamp"]),
                R=rec["R"],
                delta=extras.get("delta"),
                temporal=_MinimalTemporal(quality=tq, reason=f"{tq.name.lower()}: from ledger"),
                stream=_MinimalStream(regime=reg, reason=f"{reg.name.lower()}: from ledger"),
                gate=gate_decision,
            )
            # The minimal frame is structurally compatible with
            # explain_decision's input contract; mypy can't infer
            # this across local dataclass factories so we cast.
            explanation = explain_decision(
                cast(DecisionFrame, frame), threshold=gate_decision.threshold
            )
            print(json.dumps(explanation.to_dict(), sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="neurophase",
        description="neurophase — disciplined phase-synchronization decision system",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version", help="print the installed package version").set_defaults(
        func=_cmd_version
    )

    demo = sub.add_parser("demo", help="run a short synthetic pipeline and print gate states")
    demo.add_argument("--ticks", type=int, default=16, help="number of ticks")
    demo.set_defaults(func=_cmd_demo)

    verify = sub.add_parser("verify-ledger", help="verify the SHA256 chain of a ledger file")
    verify.add_argument("path", type=str, help="path to the ledger JSONL file")
    verify.set_defaults(func=_cmd_verify_ledger)

    explain = sub.add_parser(
        "explain-ledger",
        help="emit one DecisionExplanation per ledger record as JSONL",
    )
    explain.add_argument("path", type=str, help="path to the ledger JSONL file")
    explain.set_defaults(func=_cmd_explain_ledger)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover — entry point
    sys.exit(main())
