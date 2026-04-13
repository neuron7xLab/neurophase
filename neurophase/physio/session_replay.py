"""Offline replay of a :class:`PhysioLedger` file.

Reads a ``physio-ledger-v1`` JSONL file and reconstructs the session
frame-by-frame through a fresh :class:`PhysioSession` configured from
the ledger's ``SESSION_HEADER``. The goal is byte-identical replay:
every FRAME event in the ledger must reproduce the exact same gate
decision when fed through the same session core.

This is the keystone guarantee for the physio audit story:

    any execute/degrade can be explained post-hoc, deterministically,
    without touching live hardware.

Run as a CLI::

    python -m neurophase.physio.session_replay  ledger.jsonl
    python -m neurophase.physio.session_replay  ledger.jsonl --json
    python -m neurophase.physio.session_replay  ledger.jsonl --strict

``--strict`` returns a non-zero exit code on any decision divergence;
without it the tool still diagnoses divergences but returns zero if
the ledger is only structurally well-formed. The strict mode is what
CI should run.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from neurophase.physio.features import DEFAULT_WINDOW_SIZE
from neurophase.physio.gate import (
    DEFAULT_THRESHOLD_ABSTAIN,
    DEFAULT_THRESHOLD_ALLOW,
)
from neurophase.physio.ledger import PHYSIO_LEDGER_SCHEMA_VERSION
from neurophase.physio.pipeline import PhysioFrame, PhysioSession
from neurophase.physio.replay import RRSample

# Exit codes.
EXIT_OK: int = 0
EXIT_PARITY_FAIL: int = 1
EXIT_LEDGER_BAD: int = 2


class LedgerReplayError(ValueError):
    """Raised when a ledger is structurally invalid beyond recovery."""


CONFIDENCE_TOLERANCE: float = 1e-9


@dataclass(frozen=True)
class ReplayDivergence:
    tick_index: int
    recorded_state: str
    replay_state: str
    recorded_execution_allowed: bool
    replay_execution_allowed: bool
    timestamp_s: float
    rr_ms: float
    #: Set only when --full-parity uncovers a kernel_state or
    #: confidence mismatch ON TOP OF a state match. ``None`` for
    #: divergences that already fail the decision-only check.
    field: str | None = None
    recorded_value: str | None = None
    replay_value: str | None = None


@dataclass(frozen=True)
class ReplayReport:
    path: Path
    session_id: str
    schema_version: str
    n_frames_recorded: int
    n_frames_replayed: int
    n_divergences: int
    divergences: tuple[ReplayDivergence, ...]
    event_counts: dict[str, int]
    clean_summary_seen: bool

    @property
    def parity_ok(self) -> bool:
        return self.n_divergences == 0 and self.n_frames_recorded == self.n_frames_replayed

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "session_id": self.session_id,
            "schema_version": self.schema_version,
            "n_frames_recorded": self.n_frames_recorded,
            "n_frames_replayed": self.n_frames_replayed,
            "n_divergences": self.n_divergences,
            "divergences": [asdict(d) for d in self.divergences],
            "event_counts": dict(self.event_counts),
            "clean_summary_seen": self.clean_summary_seen,
            "parity_ok": self.parity_ok,
        }


def _iter_ledger_events(path: Path) -> list[dict[str, Any]]:
    """Read all JSONL events from path, tolerating a partial trailing line.

    Robustness rule (single source of truth): a malformed JSON line
    is tolerated **iff it is the LAST non-blank line** of the file.
    Anything else is a hard :class:`LedgerReplayError`. This matches
    the ledger writer's flush-after-each-line semantics (a crash can
    only truncate the line currently being written).
    """
    if not path.exists():
        raise LedgerReplayError(f"ledger file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    # Index of the last non-blank line, if any. Lines that are
    # whitespace-only are ignored from both the count and the parse.
    non_blank: list[tuple[int, str]] = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if line.strip():
            non_blank.append((lineno, line.strip()))
    last_idx = len(non_blank) - 1

    events: list[dict[str, Any]] = []
    for i, (lineno, line) in enumerate(non_blank):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if i == last_idx:
                # Truncated trailing line: tolerated.
                break
            raise LedgerReplayError(
                f"{path}: malformed JSON at line {lineno}: {line[:80]!r}"
            ) from None
    return events


def _extract_session_core(header: dict[str, Any]) -> PhysioSession:
    """Build a PhysioSession from the SESSION_HEADER's config snapshot."""
    cfg = header.get("config") or {}
    return PhysioSession(
        window_size=int(cfg.get("window_size", DEFAULT_WINDOW_SIZE)),
        threshold_allow=float(cfg.get("threshold_allow", DEFAULT_THRESHOLD_ALLOW)),
        threshold_abstain=float(cfg.get("threshold_abstain", DEFAULT_THRESHOLD_ABSTAIN)),
    )


def replay_ledger(
    path: str | Path,
    *,
    full_parity: bool = False,
) -> ReplayReport:
    """Reconstruct a session from a ledger file, frame-by-frame.

    Default mode (``full_parity=False``) compares ``gate_state`` and
    ``execution_allowed`` only — the **decision parity** check that
    proves the gate's user-visible output is reproducible.

    With ``full_parity=True`` the replayer additionally checks
    ``kernel_state`` and ``confidence`` (within
    :data:`CONFIDENCE_TOLERANCE`). This catches subtle drift in the
    underlying feature pipeline that would not change the decision
    today but could on a different sample sequence.

    Returns a :class:`ReplayReport`. Malformed ledgers raise
    :class:`LedgerReplayError`.
    """
    path = Path(path)
    events = _iter_ledger_events(path)
    if not events:
        raise LedgerReplayError(f"{path}: ledger has no events")

    header = events[0]
    if header.get("event") != "SESSION_HEADER":
        raise LedgerReplayError(f"{path}: first event is not SESSION_HEADER")
    schema_version = str(header.get("schema_version", ""))
    if schema_version != PHYSIO_LEDGER_SCHEMA_VERSION:
        raise LedgerReplayError(
            f"{path}: unsupported schema_version={schema_version!r} "
            f"(expected {PHYSIO_LEDGER_SCHEMA_VERSION})"
        )
    session_id = str(header.get("session_id", "")) or "<missing>"

    session = _extract_session_core(header)

    divergences: list[ReplayDivergence] = []
    n_frames_recorded = 0
    n_frames_replayed = 0
    event_counts: dict[str, int] = {}
    clean_summary_seen = False

    for evt in events[1:]:
        name = str(evt.get("event", ""))
        event_counts[name] = event_counts.get(name, 0) + 1

        if name == "SESSION_SUMMARY":
            clean_summary_seen = True
            continue

        if name != "FRAME":
            # INGEST_REJECTED / STALL / READY / LISTENING etc. are
            # informational; replay does not reconstruct them because
            # they do not advance the session core.
            continue

        n_frames_recorded += 1

        tick_index = int(evt.get("tick_index", n_frames_replayed))
        timestamp_s = float(evt["timestamp_s"])
        rr_ms = float(evt["rr_ms"])

        sample = RRSample(timestamp_s=timestamp_s, rr_ms=rr_ms, row_index=tick_index)
        frame: PhysioFrame = session.step(sample, tick_index=tick_index)
        n_frames_replayed += 1

        recorded_state = str(evt["gate_state"])
        recorded_allowed = bool(evt["execution_allowed"])
        replay_state = frame.decision.state.name
        replay_allowed = frame.decision.execution_allowed

        if recorded_state != replay_state or recorded_allowed != replay_allowed:
            divergences.append(
                ReplayDivergence(
                    tick_index=tick_index,
                    recorded_state=recorded_state,
                    replay_state=replay_state,
                    recorded_execution_allowed=recorded_allowed,
                    replay_execution_allowed=replay_allowed,
                    timestamp_s=timestamp_s,
                    rr_ms=rr_ms,
                )
            )
            continue

        if not full_parity:
            continue

        # Full-parity additional checks. Only run when the decision
        # itself matched -- a decision divergence is the headline
        # finding and is reported above without piling on.
        recorded_kernel = evt.get("kernel_state")
        if recorded_kernel is not None and str(recorded_kernel) != frame.decision.kernel_state.name:
            divergences.append(
                ReplayDivergence(
                    tick_index=tick_index,
                    recorded_state=recorded_state,
                    replay_state=replay_state,
                    recorded_execution_allowed=recorded_allowed,
                    replay_execution_allowed=replay_allowed,
                    timestamp_s=timestamp_s,
                    rr_ms=rr_ms,
                    field="kernel_state",
                    recorded_value=str(recorded_kernel),
                    replay_value=frame.decision.kernel_state.name,
                )
            )
            continue

        recorded_conf = evt.get("confidence")
        if recorded_conf is not None:
            replay_conf = frame.decision.confidence
            if abs(float(recorded_conf) - float(replay_conf)) > CONFIDENCE_TOLERANCE:
                divergences.append(
                    ReplayDivergence(
                        tick_index=tick_index,
                        recorded_state=recorded_state,
                        replay_state=replay_state,
                        recorded_execution_allowed=recorded_allowed,
                        replay_execution_allowed=replay_allowed,
                        timestamp_s=timestamp_s,
                        rr_ms=rr_ms,
                        field="confidence",
                        recorded_value=f"{float(recorded_conf):.12f}",
                        replay_value=f"{float(replay_conf):.12f}",
                    )
                )

    return ReplayReport(
        path=path,
        session_id=session_id,
        schema_version=schema_version,
        n_frames_recorded=n_frames_recorded,
        n_frames_replayed=n_frames_replayed,
        n_divergences=len(divergences),
        divergences=tuple(divergences),
        event_counts=event_counts,
        clean_summary_seen=clean_summary_seen,
    )


# =======================================================================
#   CLI
# =======================================================================


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neurophase.physio.session_replay",
        description=(
            "Offline, frame-by-frame replay of a physio session ledger. "
            "Proves that every recorded gate decision is reproducible "
            "without live hardware."
        ),
    )
    p.add_argument("ledger", type=Path, help="Path to the JSONL ledger file.")
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit the replay report as one JSON object (for CI pipelines).",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit non-zero on ANY decision divergence or frame-count "
            "mismatch. Recommended for CI. Without --strict the tool "
            "still reports divergences but returns 0 on a structurally "
            "valid ledger."
        ),
    )
    p.add_argument(
        "--full-parity",
        action="store_true",
        help=(
            "Additionally check kernel_state and confidence (within "
            f"{CONFIDENCE_TOLERANCE:g}). Default mode checks gate_state "
            "and execution_allowed only -- the user-visible decision."
        ),
    )
    return p


def _format_human(report: ReplayReport) -> str:
    lines = [
        f"Ledger:           {report.path}",
        f"Session ID:       {report.session_id}",
        f"Schema version:   {report.schema_version}",
        f"Frames recorded:  {report.n_frames_recorded}",
        f"Frames replayed:  {report.n_frames_replayed}",
        f"Divergences:      {report.n_divergences}",
        f"Clean summary:    {report.clean_summary_seen}",
        f"Event counts:     {dict(report.event_counts)}",
        f"Parity:           {'OK' if report.parity_ok else 'BROKEN'}",
    ]
    if report.divergences:
        lines.append("")
        lines.append("Divergences:")
        for d in report.divergences[:10]:
            lines.append(
                f"  tick={d.tick_index:<4} "
                f"recorded={d.recorded_state:<16} replay={d.replay_state:<16} "
                f"rec_allowed={int(d.recorded_execution_allowed)} "
                f"rep_allowed={int(d.replay_execution_allowed)}"
            )
        if len(report.divergences) > 10:
            lines.append(f"  ... ({len(report.divergences) - 10} more)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    try:
        report = replay_ledger(args.ledger, full_parity=args.full_parity)
    except LedgerReplayError as exc:
        if args.json:
            sys.stdout.write(json.dumps({"error": str(exc)}) + "\n")
        else:
            sys.stderr.write(f"ledger error: {exc}\n")
        return EXIT_LEDGER_BAD

    if args.json:
        sys.stdout.write(json.dumps(report.to_json_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(_format_human(report) + "\n")

    if args.strict and not report.parity_ok:
        return EXIT_PARITY_FAIL
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
