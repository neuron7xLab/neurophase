"""bridges — ingress / egress contract layer.

Formalises the boundary between the kernel (``runtime``, ``gate``,
``contracts``) and the outside world. Each bridge is a typed, narrow,
fail-closed adapter.

Public surface::

    from neurophase.bridges import (
        BridgeError,
        ClockDesyncError,
        ClockSync,
        DownstreamAdapter,
        DownstreamAdapterError,
        DownstreamDispatchResult,
        EegIngress,
        FrameMux,
        MarketIngress,
        MarketTick,
        NeuralSample,
    )

See ``docs/BRIDGE_CONTRACTS.md`` for the full contract.
"""

from __future__ import annotations

from neurophase.bridges.clock_sync import ClockDesyncError, ClockSync
from neurophase.bridges.downstream_execution import (
    DownstreamAdapter,
    DownstreamAdapterError,
    DownstreamDispatchResult,
)
from neurophase.bridges.errors import BridgeError
from neurophase.bridges.frame_mux import FrameMux
from neurophase.bridges.ingress import (
    EegIngress,
    MarketIngress,
    MarketTick,
    NeuralSample,
)

__all__ = [
    "BridgeError",
    "ClockDesyncError",
    "ClockSync",
    "DownstreamAdapter",
    "DownstreamAdapterError",
    "DownstreamDispatchResult",
    "EegIngress",
    "FrameMux",
    "MarketIngress",
    "MarketTick",
    "NeuralSample",
]
