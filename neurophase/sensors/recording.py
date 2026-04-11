"""File-backed sensor source — deterministic JSONL replay.

Replays a committed JSONL file of timestamped phase snapshots and
emits :class:`NeuralFrame` objects one per :meth:`extract` call.
This is the canonical fixture for offline analysis and CI
regression, and the natural input shape for a future "record
from real hardware → replay in tests" workflow.

JSONL schema (one record per line)::

    {
      "phases": [0.1, -0.3, 1.2],   # radians, length = n_channels
      "channel_labels": ["ch0", "ch1", "ch2"],
      "sample_rate_hz": 256.0
    }

The source honours the :class:`NeuralPhaseExtractor` protocol
and is **deterministic**: two :class:`RecordingFileSource`
instances constructed from the same file produce byte-identical
frame sequences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neurophase.oscillators.neural_protocol import (
    NeuralFrame,
    SensorStatus,
)

__all__ = [
    "RecordingFileSource",
    "RecordingSample",
]


@dataclass(frozen=True)
class RecordingSample:
    """One decoded JSONL record."""

    phases: tuple[float, ...]
    channel_labels: tuple[str, ...]
    sample_rate_hz: float


class RecordingFileSource:
    """JSONL-backed replay source.

    Parameters
    ----------
    path
        Path to a JSONL file. Each line must be a valid JSON
        object matching the module-level schema.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If any line is not a well-formed JSON object or does not
        satisfy the expected schema.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"recording file not found: {self.path}")
        self._samples: list[RecordingSample] = list(self._load(self.path))
        self._cursor: int = 0

    def status(self) -> SensorStatus:
        if self._cursor >= len(self._samples):
            return SensorStatus.ABSENT
        return SensorStatus.LIVE

    def extract(self) -> NeuralFrame:
        if self._cursor >= len(self._samples):
            return NeuralFrame(
                status=SensorStatus.ABSENT,
                phases=np.array([], dtype=np.float64),
                channel_labels=(),
                sample_rate_hz=0.0,
            )
        sample = self._samples[self._cursor]
        self._cursor += 1
        return NeuralFrame(
            status=SensorStatus.LIVE,
            phases=np.array(sample.phases, dtype=np.float64),
            channel_labels=sample.channel_labels,
            sample_rate_hz=sample.sample_rate_hz,
        )

    def reset(self) -> None:
        """Rewind the cursor to the start of the recording."""
        self._cursor = 0

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    @property
    def position(self) -> int:
        return self._cursor

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> list[RecordingSample]:
        samples: list[RecordingSample] = []
        for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected object, got {type(payload).__name__}"
                )
            for required in ("phases", "channel_labels", "sample_rate_hz"):
                if required not in payload:
                    raise ValueError(f"{path}:{line_number}: missing required field {required!r}")
            phases = tuple(float(v) for v in payload["phases"])
            labels = tuple(str(v) for v in payload["channel_labels"])
            rate = float(payload["sample_rate_hz"])
            if len(phases) != len(labels):
                raise ValueError(
                    f"{path}:{line_number}: phases length {len(phases)} "
                    f"does not match channel_labels length {len(labels)}"
                )
            if rate <= 0:
                raise ValueError(f"{path}:{line_number}: sample_rate_hz must be > 0, got {rate}")
            samples.append(
                RecordingSample(
                    phases=phases,
                    channel_labels=labels,
                    sample_rate_hz=rate,
                )
            )
        return samples
