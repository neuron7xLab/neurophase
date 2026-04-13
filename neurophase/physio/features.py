"""HRV-style signal-quality features over a rolling RR window.

The features below are used by :mod:`neurophase.physio.gate` as
**signal-integrity indicators**, not as clinical or cognitive
biomarkers:

* ``rmssd_ms`` — root mean square of successive differences in RR.
  A numeric fingerprint of variability, not a health score.
* ``continuity_fraction`` — fraction of the window that is within the
  physiological plausibility envelope (see
  :func:`~neurophase.physio.replay`). Inserted samples failing the
  envelope are already rejected at ingest; ``continuity_fraction`` in
  this module reflects how full the rolling buffer is relative to its
  configured size.
* ``stability`` — 1 - (std / mean) of the RR series, clipped to
  ``[0, 1]``. Flatlined or wildly variable series both produce low
  stability.
* ``confidence`` — bounded ``[0, 1]`` score combining buffer fill,
  stability, and RMSSD plausibility. This is the scalar the gate uses
  to decide among EXECUTE_ALLOWED / EXECUTE_REDUCED / ABSTAIN / DEGRADED.

All thresholds are illustrative defaults. Tuning is expected in any
real deployment and must be documented alongside the deployment; this
module does not ship "production" thresholds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from neurophase.physio.replay import RRSample

# Buffer size defaults. 16 is the hard minimum below which RMSSD is
# unreliable on short HRV windows; 32 is the comfortable default.
MIN_WINDOW_SIZE: int = 16
DEFAULT_WINDOW_SIZE: int = 32

# RMSSD plausibility envelope (ms). Values outside this range are
# treated as artifact-dominated or flatlined, NOT as a health signal.
RMSSD_PLAUSIBLE_MIN_MS: float = 5.0
RMSSD_PLAUSIBLE_MAX_MS: float = 250.0


@dataclass(frozen=True)
class HRVFeatures:
    """Snapshot of the HRV-style signal-quality features at one tick."""

    rmssd_ms: float
    mean_rr_ms: float
    std_rr_ms: float
    stability: float  # [0, 1], 1 = perfectly stable series
    continuity_fraction: float  # [0, 1], 1 = buffer full to configured size
    confidence: float  # [0, 1], composite signal-quality score
    window_size: int  # number of samples contributing
    rmssd_plausible: bool  # RMSSD inside plausibility envelope?


class HRVWindow:
    """Rolling RR buffer with O(1) append and O(window) feature recomputation.

    Parameters
    ----------
    window_size
        Target buffer depth. Must be >= :data:`MIN_WINDOW_SIZE`.

    Notes
    -----
    The window is FIFO. Feature extraction requires at least
    :data:`MIN_WINDOW_SIZE` samples; calling
    :meth:`HRVWindow.features` on a shorter buffer returns features with
    ``confidence = 0.0`` and ``continuity_fraction < 1.0``, which the
    gate treats as fail-closed.
    """

    __slots__ = ("_buffer", "window_size")

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        if window_size < MIN_WINDOW_SIZE:
            raise ValueError(f"window_size must be >= {MIN_WINDOW_SIZE}, got {window_size}")
        self.window_size: int = window_size
        self._buffer: deque[float] = deque(maxlen=window_size)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, sample: RRSample) -> None:
        """Append the sample's RR value to the buffer."""
        self._buffer.append(sample.rr_ms)

    def reset(self) -> None:
        self._buffer.clear()

    def features(self) -> HRVFeatures:
        """Compute the current features snapshot.

        Returns a zeroed / low-confidence snapshot when the buffer has
        fewer than :data:`MIN_WINDOW_SIZE` samples; the gate treats
        low-confidence snapshots as fail-closed.
        """
        n = len(self._buffer)
        continuity_fraction = n / self.window_size if self.window_size > 0 else 0.0

        if n < MIN_WINDOW_SIZE:
            return HRVFeatures(
                rmssd_ms=0.0,
                mean_rr_ms=float(np.mean(self._buffer)) if n > 0 else 0.0,
                std_rr_ms=float(np.std(self._buffer, ddof=0)) if n > 0 else 0.0,
                stability=0.0,
                continuity_fraction=continuity_fraction,
                confidence=0.0,
                window_size=n,
                rmssd_plausible=False,
            )

        rr = np.asarray(self._buffer, dtype=np.float64)
        diffs = np.diff(rr)
        rmssd = float(np.sqrt(np.mean(diffs * diffs)))
        mean_rr = float(np.mean(rr))
        std_rr = float(np.std(rr, ddof=0))

        # Stability = 1 - (coefficient of variation), clipped.
        cov = std_rr / mean_rr if mean_rr > 0.0 else 1.0
        stability = float(np.clip(1.0 - cov, 0.0, 1.0))

        rmssd_plausible = RMSSD_PLAUSIBLE_MIN_MS <= rmssd <= RMSSD_PLAUSIBLE_MAX_MS

        # Composite confidence:
        #   * buffer fill (continuity_fraction)
        #   * stability (low-CoV)
        #   * RMSSD-plausibility gate (hard multiplicative factor — a
        #     series outside the RMSSD envelope is either a flatline
        #     or artifact-dominated and must NOT earn high confidence).
        confidence = continuity_fraction * stability
        if not rmssd_plausible:
            confidence *= 0.2
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return HRVFeatures(
            rmssd_ms=rmssd,
            mean_rr_ms=mean_rr,
            std_rr_ms=std_rr,
            stability=stability,
            continuity_fraction=continuity_fraction,
            confidence=confidence,
            window_size=n,
            rmssd_plausible=rmssd_plausible,
        )
