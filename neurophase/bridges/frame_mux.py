"""Frame multiplexer — single ingress point for the canonical pipeline.

:class:`FrameMux` composes :class:`EegIngress`, :class:`MarketIngress`,
and :class:`ClockSync` into one callable that produces the ``(timestamp,
R, delta)`` triple consumed by
:meth:`~neurophase.runtime.orchestrator.RuntimeOrchestrator.tick`.

The mux is the **only** correct way to feed the runtime orchestrator
from paired neural + market streams. Direct construction of tick
arguments bypasses ingress validation and is forbidden on the live
path.
"""

from __future__ import annotations

from dataclasses import dataclass

from neurophase.bridges.clock_sync import ClockSync
from neurophase.bridges.ingress import EegIngress, MarketIngress


@dataclass(frozen=True)
class MuxedTick:
    """Canonical ``(timestamp, R, delta)`` triple plus provenance.

    Attributes
    ----------
    timestamp
        Fused timestamp from :class:`ClockSync`.
    R
        Market-side order parameter (pass-through from :class:`MarketTick`).
    delta
        Optional circular distance (pass-through).
    neural_source_id
        Ingress source id for the neural stream.
    market_source_id
        Ingress source id for the market stream.
    """

    timestamp: float
    R: float | None
    delta: float | None
    neural_source_id: str
    market_source_id: str


class FrameMux:
    """Compose neural + market ingress + clock sync into one ``poll()`` call."""

    __slots__ = ("_clock", "_eeg", "_market")

    def __init__(
        self,
        *,
        eeg: EegIngress,
        market: MarketIngress,
        clock: ClockSync,
    ) -> None:
        self._eeg = eeg
        self._market = market
        self._clock = clock

    def poll(self) -> MuxedTick:
        """Pull one paired sample from both ingresses, fused + validated.

        Raises
        ------
        BridgeError
            On any ingress failure, any validation failure, or any
            clock desync. The kernel is never fed partial data.
        """
        neural = self._eeg.poll()
        market = self._market.poll()
        ts = self._clock.fuse(neural, market)
        return MuxedTick(
            timestamp=ts,
            R=market.R,
            delta=market.delta,
            neural_source_id=neural.source_id,
            market_source_id=market.source_id,
        )
