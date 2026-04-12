"""Clock synchronisation contract.

:class:`ClockSync` owns the invariant that the two ingress streams
(neural + market) arrive with timestamps consistent enough to be fused
into a single canonical tick. It refuses a pair whose inter-sample drift
exceeds a configured tolerance.

Fail-closed by construction: the only way to acknowledge a pair is to
call :meth:`ClockSync.fuse`, which raises :class:`ClockDesyncError` on
violation. There is no lenient mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from neurophase.bridges.errors import BridgeError
from neurophase.bridges.ingress import MarketTick, NeuralSample


class ClockDesyncError(BridgeError):
    """Raised when neural and market samples are too far apart in time."""


@dataclass(frozen=True)
class ClockSync:
    """Tolerance envelope for neural–market clock drift.

    Attributes
    ----------
    max_drift_seconds
        Maximum absolute difference between neural and market
        timestamps, in seconds. A pair outside this envelope is
        rejected.
    """

    max_drift_seconds: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_drift_seconds) or self.max_drift_seconds <= 0.0:
            raise BridgeError(
                f"ClockSync.max_drift_seconds must be a positive finite number, "
                f"got {self.max_drift_seconds!r}"
            )

    def fuse(self, neural: NeuralSample, market: MarketTick) -> float:
        """Return the fused timestamp (mean of the two) after a drift check.

        Raises
        ------
        ClockDesyncError
            When ``|neural.timestamp - market.timestamp| > max_drift_seconds``.
        """
        drift = abs(neural.timestamp - market.timestamp)
        if drift > self.max_drift_seconds:
            raise ClockDesyncError(
                f"clock desync: neural={neural.timestamp!r} "
                f"market={market.timestamp!r} drift={drift!r} "
                f"tolerance={self.max_drift_seconds!r}"
            )
        # Use market timestamp as the canonical clock — it is the
        # external signal the pipeline ultimately audits against.
        return market.timestamp
