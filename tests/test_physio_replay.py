"""Tests for neurophase.physio.replay — replay ingest contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from neurophase.physio.replay import (
    RR_MAX_MS,
    RR_MIN_MS,
    ReplayIngestError,
    RRReplayReader,
    RRSample,
)


def _write_csv(tmp_path: Path, lines: list[str], name: str = "rr.csv") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestRRSample:
    def test_valid_sample_accepted(self) -> None:
        s = RRSample(timestamp_s=1.0, rr_ms=820.0, row_index=0)
        assert s.rr_ms == 820.0
        assert s.timestamp_s == 1.0

    def test_impossible_low_rr_rejected(self) -> None:
        with pytest.raises(ReplayIngestError, match="outside physiological envelope"):
            RRSample(timestamp_s=1.0, rr_ms=RR_MIN_MS - 1.0, row_index=0)

    def test_impossible_high_rr_rejected(self) -> None:
        with pytest.raises(ReplayIngestError, match="outside physiological envelope"):
            RRSample(timestamp_s=1.0, rr_ms=RR_MAX_MS + 1.0, row_index=0)

    def test_nan_timestamp_rejected(self) -> None:
        with pytest.raises(ReplayIngestError, match="NaN"):
            RRSample(timestamp_s=float("nan"), rr_ms=820.0, row_index=0)

    def test_nan_rr_rejected(self) -> None:
        with pytest.raises(ReplayIngestError, match="NaN"):
            RRSample(timestamp_s=1.0, rr_ms=float("nan"), row_index=0)

    def test_negative_timestamp_rejected(self) -> None:
        with pytest.raises(ReplayIngestError, match="negative"):
            RRSample(timestamp_s=-1.0, rr_ms=820.0, row_index=0)


class TestRRReplayReader:
    def test_valid_csv_ingest(self, tmp_path: Path) -> None:
        path = _write_csv(
            tmp_path, ["timestamp_s,rr_ms", "0.000,820.0", "0.820,815.5", "1.635,830.0"]
        )
        samples = list(RRReplayReader(path))
        assert len(samples) == 3
        assert samples[0].timestamp_s == 0.0
        assert samples[0].rr_ms == 820.0
        assert samples[-1].rr_ms == 830.0

    def test_missing_file_fails_cleanly(self, tmp_path: Path) -> None:
        with pytest.raises(ReplayIngestError, match="not found"):
            RRReplayReader(tmp_path / "nope.csv")

    def test_empty_file_fails_cleanly(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ReplayIngestError, match="empty"):
            list(RRReplayReader(path))

    def test_wrong_header_fails_cleanly(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, ["time,rr", "0,820"])
        with pytest.raises(ReplayIngestError, match="header"):
            list(RRReplayReader(path))

    def test_wrong_column_count_rejected(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, ["timestamp_s,rr_ms", "0.0,820.0,extra"])
        with pytest.raises(ReplayIngestError, match="expected 2 columns"):
            list(RRReplayReader(path))

    def test_non_numeric_field_rejected(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, ["timestamp_s,rr_ms", "0.0,820.0", "0.8,oops"])
        with pytest.raises(ReplayIngestError, match="non-numeric"):
            list(RRReplayReader(path))

    def test_impossible_rr_rejected_at_iteration(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, ["timestamp_s,rr_ms", "0.0,820.0", "0.8,100.0"])
        with pytest.raises(ReplayIngestError, match="outside physiological envelope"):
            list(RRReplayReader(path))

    def test_non_monotonic_timestamp_rejected(self, tmp_path: Path) -> None:
        path = _write_csv(
            tmp_path,
            ["timestamp_s,rr_ms", "0.0,820.0", "1.0,815.0", "0.5,830.0"],
        )
        with pytest.raises(ReplayIngestError, match="not strictly greater"):
            list(RRReplayReader(path))

    def test_duplicate_timestamp_rejected(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, ["timestamp_s,rr_ms", "1.0,820.0", "1.0,815.0"])
        with pytest.raises(ReplayIngestError, match="not strictly greater"):
            list(RRReplayReader(path))


class TestLeadingProvenanceComments:
    """# ... lines BEFORE the header are skipped; AFTER the header are data."""

    def test_single_leading_comment_skipped(self, tmp_path: Path) -> None:
        path = _write_csv(
            tmp_path,
            [
                "# SYNTHETIC REPLAY DATA - NOT REAL PHYSIOLOGICAL MEASUREMENTS",
                "timestamp_s,rr_ms",
                "0.0,820.0",
                "0.8,815.0",
            ],
        )
        samples = list(RRReplayReader(path))
        assert len(samples) == 2
        assert samples[0].rr_ms == 820.0

    def test_multiple_leading_comments_skipped(self, tmp_path: Path) -> None:
        path = _write_csv(
            tmp_path,
            [
                "# line 1",
                "# line 2: seed=42",
                "",  # blank line also skipped before header
                "timestamp_s,rr_ms",
                "0.0,820.0",
            ],
        )
        samples = list(RRReplayReader(path))
        assert len(samples) == 1

    def test_comment_after_header_rejected(self, tmp_path: Path) -> None:
        """Comments mid-stream must fail cleanly (fail-closed). An artefact
        must never be able to hide as a comment inside the data body."""
        path = _write_csv(
            tmp_path,
            [
                "timestamp_s,rr_ms",
                "0.0,820.0",
                "# sneaky mid-file comment",
                "1.0,815.0",
            ],
        )
        with pytest.raises(ReplayIngestError, match=r"expected 2 columns|non-numeric"):
            list(RRReplayReader(path))

    def test_comment_with_trailing_comma_after_header_still_rejected(self, tmp_path: Path) -> None:
        """A mid-file '#' row that happens to have two columns still fails:
        the first column is non-numeric, so the numeric parser rejects it."""
        path = _write_csv(
            tmp_path,
            [
                "timestamp_s,rr_ms",
                "0.0,820.0",
                "# sneaky,comment",
                "1.0,815.0",
            ],
        )
        with pytest.raises(ReplayIngestError, match="non-numeric"):
            list(RRReplayReader(path))

    def test_only_comments_file_fails_cleanly(self, tmp_path: Path) -> None:
        path = _write_csv(tmp_path, ["# comment 1", "# comment 2"])
        with pytest.raises(ReplayIngestError, match="empty"):
            list(RRReplayReader(path))

    def test_bundled_sample_csv_starts_with_provenance_marker(self) -> None:
        """The shipped replay sample must carry an explicit provenance note
        on its first line so a reviewer cannot mistake it for real data."""
        sample_path = (
            Path(__file__).resolve().parents[1] / "examples" / "data" / "physio_replay_sample.csv"
        )
        first_line = sample_path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("#")
        upper = first_line.upper()
        assert "SYNTHETIC" in upper
        assert "NOT REAL" in upper
