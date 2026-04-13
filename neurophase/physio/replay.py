"""Replay input contract for timestamped RR intervals.

Reads a CSV whose schema is::

    timestamp_s,rr_ms
    0.000,820
    0.820,815
    1.635,830

and yields :class:`RRSample` instances after validating each row.
Malformed or physiologically impossible rows raise
:class:`ReplayIngestError` at read time. The reader does **not**
attempt to repair bad data — on any violation it raises and the
caller handles it (typical pipeline reaction: emit a fail-closed
frame and advance).

This module contains no live-device code. A live driver can be added
later by producing the same :class:`RRSample` stream; nothing in this
file needs to change for that to work.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Physiological plausibility envelope (open at the exact bounds).
# 30 bpm == 2000 ms, 200 bpm == 300 ms. Anything outside is either
# dropped sensor contact or an instrumentation artifact. These bounds
# are intentionally wide; clinical arrhythmia screening is NOT the
# purpose here.
RR_MIN_MS: float = 300.0
RR_MAX_MS: float = 2000.0


class ReplayIngestError(ValueError):
    """Raised when a replay row is malformed, impossible, or out of order."""


@dataclass(frozen=True)
class RRSample:
    """One validated RR-interval sample.

    Attributes
    ----------
    timestamp_s
        Monotonic sample time in seconds (must be strictly increasing
        across the stream).
    rr_ms
        R-R interval in milliseconds, inside ``[RR_MIN_MS, RR_MAX_MS]``.
    row_index
        Zero-based row index in the source file (CSV header row is not
        counted). Preserved for audit.
    """

    timestamp_s: float
    rr_ms: float
    row_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_s, float):
            raise ReplayIngestError(
                f"row {self.row_index}: timestamp_s must be float, "
                f"got {type(self.timestamp_s).__name__}"
            )
        if self.timestamp_s != self.timestamp_s:  # NaN
            raise ReplayIngestError(f"row {self.row_index}: timestamp_s is NaN")
        if self.timestamp_s < 0.0:
            raise ReplayIngestError(
                f"row {self.row_index}: timestamp_s={self.timestamp_s!r} is negative"
            )
        if not isinstance(self.rr_ms, float):
            raise ReplayIngestError(
                f"row {self.row_index}: rr_ms must be float, got {type(self.rr_ms).__name__}"
            )
        if self.rr_ms != self.rr_ms:  # NaN
            raise ReplayIngestError(f"row {self.row_index}: rr_ms is NaN")
        if not (RR_MIN_MS <= self.rr_ms <= RR_MAX_MS):
            raise ReplayIngestError(
                f"row {self.row_index}: rr_ms={self.rr_ms!r} outside "
                f"physiological envelope [{RR_MIN_MS}, {RR_MAX_MS}] ms"
            )


class RRReplayReader:
    """Iterable reader over a CSV of timestamped RR intervals.

    Parameters
    ----------
    path
        Path to a CSV file. The file must have header
        ``timestamp_s,rr_ms`` followed by numeric rows only.

    Notes
    -----
    * Validation is eager-per-row. A single bad row raises
      :class:`ReplayIngestError` at the point of iteration so the
      caller can decide how to react (default: emit a fail-closed
      frame and skip forward).
    * Monotonicity check: if row N has ``timestamp_s <= row N-1's
      timestamp_s``, that row raises. Duplicate timestamps are treated
      as non-monotonic and therefore rejected; a replay stream must
      be strictly ordered.
    """

    __slots__ = ("path",)

    _EXPECTED_HEADER: tuple[str, ...] = ("timestamp_s", "rr_ms")

    def __init__(self, path: str | Path) -> None:
        self.path: Path = Path(path)
        if not self.path.exists():
            raise ReplayIngestError(f"replay file not found: {self.path}")

    def __iter__(self) -> Iterator[RRSample]:
        with self.path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ReplayIngestError(f"{self.path}: file is empty") from exc
            normalised = tuple(col.strip() for col in header)
            if normalised != self._EXPECTED_HEADER:
                raise ReplayIngestError(
                    f"{self.path}: header {normalised} != expected {self._EXPECTED_HEADER}"
                )
            last_ts: float | None = None
            for row_index, row in enumerate(reader):
                if len(row) != 2:
                    raise ReplayIngestError(
                        f"row {row_index}: expected 2 columns, got {len(row)}: {row!r}"
                    )
                try:
                    ts = float(row[0])
                    rr = float(row[1])
                except ValueError as exc:
                    raise ReplayIngestError(
                        f"row {row_index}: non-numeric field in {row!r}"
                    ) from exc
                sample = RRSample(timestamp_s=ts, rr_ms=rr, row_index=row_index)
                if last_ts is not None and sample.timestamp_s <= last_ts:
                    raise ReplayIngestError(
                        f"row {row_index}: timestamp_s={sample.timestamp_s!r} "
                        f"is not strictly greater than previous {last_ts!r}"
                    )
                last_ts = sample.timestamp_s
                yield sample
