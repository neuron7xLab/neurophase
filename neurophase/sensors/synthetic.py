"""Deterministic synthetic oscillator source.

Drives a bank of Kuramoto-like oscillators forward one tick at a
time and emits the resulting phases as a :class:`NeuralFrame`. The
source is **pure of its configuration and step count**: two
sources with the same config produce byte-identical frame
sequences. This is the fixture the calibration and resistance
suites use as their reference brain signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from neurophase.oscillators.neural_protocol import (
    NeuralFrame,
    SensorStatus,
)

__all__ = [
    "DEFAULT_SYNTHETIC_CONFIG",
    "SyntheticOscillatorConfig",
    "SyntheticOscillatorSource",
]


@dataclass(frozen=True)
class SyntheticOscillatorConfig:
    """Immutable configuration for :class:`SyntheticOscillatorSource`.

    Attributes
    ----------
    n_channels
        Number of oscillators (= channels in the emitted
        :class:`NeuralFrame`). Must be ≥ 1.
    base_frequency_hz
        Base rotation rate shared by every oscillator, in Hz.
    sample_rate_hz
        Output sample rate. Must be > 0.
    coupling_strength
        Mean-field coupling K in ``[0, 1]``. Zero → independent
        oscillators; one → strongly locked.
    channel_labels
        Human-readable labels, one per channel. If omitted the
        adapter synthesises ``("ch0", "ch1", ...)``.
    seed
        RNG seed for the initial phase spread. Default 0 gives
        byte-deterministic output.
    """

    n_channels: int = 4
    base_frequency_hz: float = 8.0  # α-band
    sample_rate_hz: float = 256.0
    coupling_strength: float = 0.4
    channel_labels: tuple[str, ...] | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_channels < 1:
            raise ValueError(f"n_channels must be >= 1, got {self.n_channels}")
        if self.sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be > 0, got {self.sample_rate_hz}")
        if not 0.0 <= self.coupling_strength <= 1.0:
            raise ValueError(f"coupling_strength must be in [0, 1], got {self.coupling_strength}")
        if self.channel_labels is not None and len(self.channel_labels) != self.n_channels:
            raise ValueError(
                f"channel_labels length ({len(self.channel_labels)}) must match "
                f"n_channels ({self.n_channels})"
            )


#: Default config — four α-band channels at 256 Hz with moderate coupling.
DEFAULT_SYNTHETIC_CONFIG: Final[SyntheticOscillatorConfig] = SyntheticOscillatorConfig()


class SyntheticOscillatorSource:
    """Deterministic Kuramoto-like source implementing :class:`NeuralPhaseExtractor`.

    The source owns an internal phase vector that advances by
    ``2π · base_frequency_hz / sample_rate_hz`` plus a coupling
    correction at every call to :meth:`extract`. No RNG is used
    per-step — the only randomness is the initial phase spread
    seeded at construction time — so two sources constructed with
    the same config produce bit-identical frame sequences.
    """

    def __init__(self, config: SyntheticOscillatorConfig | None = None) -> None:
        self.config: SyntheticOscillatorConfig = (
            config if config is not None else DEFAULT_SYNTHETIC_CONFIG
        )
        rng = np.random.default_rng(self.config.seed)
        # Initial phases on the principal branch.
        self._phases: NDArray[np.float64] = rng.uniform(-np.pi, np.pi, size=self.config.n_channels)
        self._step_count: int = 0

    def status(self) -> SensorStatus:
        return SensorStatus.LIVE

    def extract(self) -> NeuralFrame:
        cfg = self.config
        dt = 1.0 / cfg.sample_rate_hz
        omega = 2.0 * np.pi * cfg.base_frequency_hz

        # Mean-field Kuramoto step.
        r_vec = np.exp(1j * self._phases)
        mean_field = np.mean(r_vec)
        # Global complex order parameter magnitude + phase:
        psi = float(np.angle(mean_field))
        mag = float(np.abs(mean_field))

        # dφ_i/dt = ω + K·r·sin(ψ − φ_i)
        dphase = omega + cfg.coupling_strength * mag * np.sin(psi - self._phases)
        self._phases = np.mod(self._phases + dphase * dt + np.pi, 2.0 * np.pi) - np.pi
        self._step_count += 1

        labels = cfg.channel_labels or tuple(f"ch{i}" for i in range(cfg.n_channels))
        return NeuralFrame(
            status=SensorStatus.LIVE,
            phases=np.copy(self._phases),
            channel_labels=labels,
            sample_rate_hz=cfg.sample_rate_hz,
        )

    def reset(self) -> None:
        """Reset the internal state to the seeded initial spread."""
        rng = np.random.default_rng(self.config.seed)
        self._phases = rng.uniform(-np.pi, np.pi, size=self.config.n_channels)
        self._step_count = 0

    @property
    def n_steps(self) -> int:
        """Number of :meth:`extract` calls since construction / reset."""
        return self._step_count
