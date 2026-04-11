"""Oscillator bridges — market and neural phase extractors.

Market oscillators (price, volume, realized volatility) are fed into
``neurophase.core.phase`` to produce φ_market. Neural oscillators require
a bio-sensor bridge (EEG, HRV, pupil) — which lives behind the protocol
defined in ``neurophase.oscillators.neural_protocol``.

The protocol is deliberately *abstract*: neurophase defines the contract
and stores nothing hardware-specific. Concrete adapters (Tobii, OpenBCI,
Muse, Emotiv, ...) are downstream integrations.
"""

from __future__ import annotations

from neurophase.oscillators.market import MarketOscillators, extract_market_phase
from neurophase.oscillators.neural_protocol import (
    NeuralFrame,
    NeuralPhaseExtractor,
    NullNeuralExtractor,
    SensorStatus,
)

__all__ = [
    "MarketOscillators",
    "NeuralFrame",
    "NeuralPhaseExtractor",
    "NullNeuralExtractor",
    "SensorStatus",
    "extract_market_phase",
]
