"""Ingress adapters — typed contracts for external observations.

Two ingress shapes are supported on the canonical path:

* :class:`NeuralSample` — one instantaneous neural / bio-sensor reading.
* :class:`MarketTick` — one market observation (price, R-proxy, optional
  delta).

Two adapters normalise raw driver payloads to the canonical shapes:

* :class:`EegIngress` — wraps any callable that produces a
  :class:`NeuralSample`.
* :class:`MarketIngress` — wraps any callable that produces a
  :class:`MarketTick`.

Adapters are fail-closed: a driver callable that raises, returns
``None``, or returns a non-canonical shape causes the adapter to raise
:class:`~neurophase.bridges.errors.BridgeError`. The kernel never sees
malformed data.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from neurophase.bridges.errors import BridgeError


@dataclass(frozen=True)
class NeuralSample:
    """One bio-sensor reading.

    Attributes
    ----------
    timestamp
        Monotonic sample time in seconds.
    phase
        Instantaneous neural phase in radians, in ``(-π, π]``.
    source_id
        Stable identifier for the sensor (e.g. ``"Fz"`` or ``"hilbert-0"``).
    quality
        Free-form quality annotation (``"good"``, ``"degraded"``, …) —
        opaque to the kernel but preserved for audit.
    """

    timestamp: float
    phase: float
    source_id: str
    quality: str = "good"

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise BridgeError(f"NeuralSample.timestamp is non-finite: {self.timestamp!r}")
        if not math.isfinite(self.phase):
            raise BridgeError(f"NeuralSample.phase is non-finite: {self.phase!r}")
        if not -math.pi - 1e-9 <= self.phase <= math.pi + 1e-9:
            raise BridgeError(
                f"NeuralSample.phase={self.phase!r} outside [-π, π]; "
                "ingress must normalise phases before handing them to the kernel."
            )
        if not self.source_id:
            raise BridgeError("NeuralSample.source_id must be non-empty")


@dataclass(frozen=True)
class MarketTick:
    """One market observation mapped to the canonical inputs.

    Attributes
    ----------
    timestamp
        Monotonic tick time in seconds.
    R
        Joint order parameter R(t) in ``[0, 1]``. ``None`` is legal —
        the gate routes that frame through the ``DEGRADED`` state.
    delta
        Circular distance. ``None`` is legal; the regime layer then
        skips.
    source_id
        Stable identifier for the data feed (e.g. ``"binance-spot"``).
    """

    timestamp: float
    R: float | None
    delta: float | None
    source_id: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp):
            raise BridgeError(f"MarketTick.timestamp is non-finite: {self.timestamp!r}")
        if self.R is not None:
            if not math.isfinite(self.R):
                raise BridgeError(f"MarketTick.R is non-finite: {self.R!r}")
            if not 0.0 <= self.R <= 1.0:
                raise BridgeError(f"MarketTick.R={self.R!r} outside [0, 1]")
        if self.delta is not None and not math.isfinite(self.delta):
            raise BridgeError(f"MarketTick.delta is non-finite: {self.delta!r}")
        if not self.source_id:
            raise BridgeError("MarketTick.source_id must be non-empty")


class EegIngress:
    """Adapter from a driver callable to :class:`NeuralSample`.

    The driver callable is injected at construction time. It is never
    called at import; it runs only inside :meth:`poll`.
    """

    __slots__ = ("_driver", "source_id")

    def __init__(
        self,
        driver: Callable[[], NeuralSample | None],
        *,
        source_id: str,
    ) -> None:
        if not source_id:
            raise BridgeError("EegIngress.source_id must be non-empty")
        self._driver = driver
        self.source_id: str = source_id

    def poll(self) -> NeuralSample:
        """Pull one sample. Raise :class:`BridgeError` on any failure."""
        try:
            sample = self._driver()
        except Exception as exc:
            raise BridgeError(f"EEG driver raised: {exc!r}") from exc
        if sample is None:
            raise BridgeError("EEG driver returned None")
        if not isinstance(sample, NeuralSample):
            raise BridgeError(f"EEG driver returned {type(sample).__name__}, expected NeuralSample")
        if sample.source_id != self.source_id:
            raise BridgeError(
                f"source_id mismatch: driver={sample.source_id!r} ingress={self.source_id!r}"
            )
        return sample


class MarketIngress:
    """Adapter from a driver callable to :class:`MarketTick`."""

    __slots__ = ("_driver", "source_id")

    def __init__(
        self,
        driver: Callable[[], MarketTick | None],
        *,
        source_id: str,
    ) -> None:
        if not source_id:
            raise BridgeError("MarketIngress.source_id must be non-empty")
        self._driver = driver
        self.source_id: str = source_id

    def poll(self) -> MarketTick:
        try:
            tick = self._driver()
        except Exception as exc:
            raise BridgeError(f"market driver raised: {exc!r}") from exc
        if tick is None:
            raise BridgeError("market driver returned None")
        if not isinstance(tick, MarketTick):
            raise BridgeError(f"market driver returned {type(tick).__name__}, expected MarketTick")
        if tick.source_id != self.source_id:
            raise BridgeError(
                f"source_id mismatch: driver={tick.source_id!r} ingress={self.source_id!r}"
            )
        return tick
