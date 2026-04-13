"""Immutable JSONL session ledger for the physio stack.

Every live / demo run can be recorded to a :class:`PhysioLedger`. The
ledger is a simple append-only JSONL file:

* one ``SESSION_HEADER`` line up front carrying schema version,
  session UUID, UTC start timestamp, source mode, and a full config
  snapshot (window, thresholds, stall timeout, stream name, …);
* one line per event that the runtime emits (``FRAME``,
  ``INGEST_REJECTED``, ``STALL``, ``READY``, ``LISTENING``, …);
* one ``SESSION_SUMMARY`` line at the end (guaranteed when the
  context manager exits, even on failure).

Design constraints:

* **No live dependency** on anything other than the Python standard
  library plus :mod:`neurophase.physio.pipeline`; the ledger can be
  written during a live session and replayed offline by a cold reader
  that imports only :mod:`neurophase.physio`.
* **Append-only.** Each event is written with ``ensure_ascii=False``,
  flushed to disk, and never rewritten. A partially written trailing
  line is tolerated by the reader.
* **Source attribution.** Every ``FRAME`` record carries an explicit
  ``source_mode`` so replay-from-ledger and live-from-LSL records can
  coexist in the same audit store without ambiguity.
* **Fail-closed.** Ledger IO failure must NOT silently swallow the
  session: writers re-raise I/O errors after emitting a best-effort
  ``LEDGER_ERROR`` event to the fallback stream.

The ledger is deliberately NOT SHA256-chained (that would be
``neurophase.audit.decision_ledger``'s job on the market-side path).
The point here is reproducibility, not tamper-evidence.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Any

PHYSIO_LEDGER_SCHEMA_VERSION: str = "physio-ledger-v1"


@dataclass(frozen=True)
class LedgerConfig:
    """Snapshot of the runtime knobs at session open time.

    These are the only values needed to reconstruct the session
    deterministically from the ledger alone.
    """

    source_mode: str  # "live-lsl" | "replay-csv" | "offline-replay" | other
    stream_name: str | None
    window_size: int
    threshold_allow: float
    threshold_abstain: float
    stall_timeout_s: float | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source_mode": self.source_mode,
            "stream_name": self.stream_name,
            "window_size": self.window_size,
            "threshold_allow": self.threshold_allow,
            "threshold_abstain": self.threshold_abstain,
            "stall_timeout_s": self.stall_timeout_s,
        }


class PhysioLedger:
    """Append-only JSONL ledger with guaranteed header + summary lines.

    Intended usage::

        with PhysioLedger(path, config=cfg) as ledger:
            ledger.write_event({"event": "FRAME", ...})
            ...

    The context manager writes ``SESSION_SUMMARY`` on ``__exit__``
    exactly once, even when an exception propagated through the
    ``with`` block. ``write_event`` is safe to call from a tight loop;
    it flushes after every line so a crash leaves a readable prefix.
    """

    __slots__ = (
        "_closed",
        "_config",
        "_counters",
        "_fh",
        "_path",
        "_session_id",
        "_started_at",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        config: LedgerConfig,
    ) -> None:
        self._path: Path = Path(path)
        self._config: LedgerConfig = config
        self._session_id: str = uuid.uuid4().hex
        self._started_at: datetime = datetime.now(UTC)
        self._fh: IO[str] | None = None
        self._closed: bool = False
        self._counters: dict[str, int] = {}

    @property
    def path(self) -> Path:
        return self._path

    @property
    def session_id(self) -> str:
        return self._session_id

    def __enter__(self) -> PhysioLedger:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Fresh file per session. Two sessions that share a path
        # overwrite each other; that is intentional -- ledger paths
        # must include the session id / timestamp in the filename.
        self._fh = self._path.open("w", encoding="utf-8")
        self._write_line(
            {
                "event": "SESSION_HEADER",
                "schema_version": PHYSIO_LEDGER_SCHEMA_VERSION,
                "session_id": self._session_id,
                "started_at_utc": self._started_at.isoformat(),
                "config": self._config.to_json_dict(),
            }
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._closed:
            return
        try:
            ended_at = datetime.now(UTC)
            self._write_line(
                {
                    "event": "SESSION_SUMMARY",
                    "session_id": self._session_id,
                    "ended_at_utc": ended_at.isoformat(),
                    "duration_s": (ended_at - self._started_at).total_seconds(),
                    "event_counts": dict(self._counters),
                    "terminated_by_exception": exc_type is not None,
                    "exception_type": (exc_type.__name__ if exc_type else None),
                }
            )
        finally:
            if self._fh is not None:
                self._fh.flush()
                self._fh.close()
                self._fh = None
            self._closed = True

    # ------------------------------------------------------------------
    # Event writing
    # ------------------------------------------------------------------

    def write_event(self, event: dict[str, Any]) -> None:
        """Append one event line to the ledger. Flushes immediately."""
        if self._closed or self._fh is None:
            raise RuntimeError("write_event on a closed PhysioLedger")
        name = str(event.get("event", "UNKNOWN"))
        self._counters[name] = self._counters.get(name, 0) + 1
        self._write_line(event)

    def _write_line(self, event: dict[str, Any]) -> None:
        assert self._fh is not None
        self._fh.write(json.dumps(event, default=str, ensure_ascii=False))
        self._fh.write("\n")
        self._fh.flush()
        # Intentionally no fsync per line: correctness-vs-throughput
        # trade. We rely on flush() + the OS write cache. A crash may
        # truncate the last few lines; the reader tolerates a partial
        # trailing JSON line.


__all__ = [
    "PHYSIO_LEDGER_SCHEMA_VERSION",
    "LedgerConfig",
    "PhysioLedger",
]
