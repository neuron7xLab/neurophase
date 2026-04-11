"""Neural-side oscillator protocol.

This module defines the **contract** that a bio-sensor bridge must
satisfy for neurophase to fold its output into the Kuramoto network.
The protocol is deliberately abstract — neurophase itself knows nothing
about Tobii eye-trackers, OpenBCI, Muse, or Emotiv headsets. All of
those live downstream as concrete adapters implementing
``NeuralPhaseExtractor``.

Contract highlights:

    SensorStatus.{LIVE, ABSENT, DEGRADED}
        Honest null states — the system never fabricates phase data
        when the sensor is unreachable or returning garbage. Invariant
        I3 at the protocol boundary.

    NeuralFrame
        Immutable snapshot carrying channel-wise instantaneous phases
        plus a sensor-level liveness flag.

    NeuralPhaseExtractor
        Protocol with two methods — ``status()`` and ``extract()``. The
        extract method must return a ``NeuralFrame`` whose ``phases``
        field has length equal to the number of channels reported by
        the adapter.

    NullNeuralExtractor
        Always reports ``SensorStatus.ABSENT``. Used for unit tests and
        as the default when no real bridge is wired up. Zero side
        effects.

Reference theory: the neurophysiological backing for treating eye
movements, pupil dilation, heart rate variability, and EEG α/β bands
as coupled oscillators is spelled out in ``docs/theory/sensory_basis.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


class SensorStatus(Enum):
    """Liveness state of a bio-sensor bridge.

    LIVE      — hardware responding, phase data valid.
    ABSENT    — hardware not connected; enforce invariant I3.
    DEGRADED  — hardware connected but data quality insufficient
                (e.g. electrode impedance too high, blink saturation).
    """

    LIVE = auto()
    ABSENT = auto()
    DEGRADED = auto()


@dataclass(frozen=True)
class NeuralFrame:
    """A single snapshot from a bio-sensor bridge.

    Attributes
    ----------
    status : SensorStatus
        Liveness at the time of extraction.
    phases : FloatArray, shape (C,) or (C, T)
        Per-channel instantaneous phases in radians. Length 0 arrays are
        valid when status != LIVE (honest null).
    channel_labels : tuple[str, ...]
        Short human-readable labels for each channel (e.g. ``"alpha"``,
        ``"hrv"``, ``"pupil"``). Length must match ``phases.shape[0]``.
    sample_rate_hz : float
        Effective sampling rate of the underlying signals. Zero when
        status != LIVE.
    """

    status: SensorStatus
    phases: FloatArray
    channel_labels: tuple[str, ...]
    sample_rate_hz: float = 0.0

    def __post_init__(self) -> None:
        if self.status is SensorStatus.LIVE:
            if self.phases.size == 0:
                raise ValueError("LIVE frame must carry at least one phase channel")
            if len(self.channel_labels) != self.phases.shape[0]:
                raise ValueError(
                    f"channel_labels length ({len(self.channel_labels)}) "
                    f"must match phases.shape[0] ({self.phases.shape[0]})"
                )
            if self.sample_rate_hz <= 0:
                raise ValueError(
                    f"LIVE frame requires a positive sample_rate_hz, got {self.sample_rate_hz}"
                )
        # Non-LIVE frames are allowed to carry empty phases and zero rate.


@runtime_checkable
class NeuralPhaseExtractor(Protocol):
    """Contract for any bio-sensor bridge feeding neurophase.

    Implementations must be deterministic with respect to their inputs
    and must *never* fabricate phase data when the sensor is not live.
    """

    def status(self) -> SensorStatus:
        """Return the current bridge status without reading data."""
        ...

    def extract(self) -> NeuralFrame:
        """Return the latest neural-phase snapshot.

        Must honour the status contract: when status() is not LIVE the
        returned NeuralFrame must carry an empty ``phases`` array and
        sample_rate_hz = 0.
        """
        ...


class NullNeuralExtractor:
    """Always-absent bridge used as a default and for unit tests.

    This is the honest null: neurophase never synthesises brain activity.
    Plugging in a real adapter (Tobii, OpenBCI, ...) is a separate
    integration step outside this package.
    """

    def status(self) -> SensorStatus:
        return SensorStatus.ABSENT

    def extract(self) -> NeuralFrame:
        return NeuralFrame(
            status=SensorStatus.ABSENT,
            phases=np.array([], dtype=np.float64),
            channel_labels=(),
            sample_rate_hz=0.0,
        )
