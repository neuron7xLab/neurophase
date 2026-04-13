"""Oscillator bridges — market and neural phase extractors.

Market oscillators (price, volume, realized volatility) are fed into
``neurophase.core.phase`` to produce φ_market. Neural oscillators require
a bio-sensor bridge (EEG, HRV, pupil) — which lives behind the protocol
defined in ``neurophase.oscillators.neural_protocol``.

The protocol is deliberately *abstract*: neurophase defines the contract
and stores nothing hardware-specific. Concrete adapters (Tobii, OpenBCI,
Muse, Emotiv, ...) are downstream integrations.

Concrete extractor shipped in-repo:

* :class:`HRVPhaseExtractor` — RR intervals -> 4 Hz cubic-spline IBI ->
  :func:`~neurophase.core.phase.compute_phase` -> instantaneous HRV phase.
  Honest scope: HRV-phase, not EEG-phase; not a clinical metric.
"""

from __future__ import annotations

from neurophase.oscillators.hrv_phase_extractor import (
    DEFAULT_MIN_RR_SAMPLES,
    DEFAULT_MIN_WINDOW_S,
    DEFAULT_TARGET_SR_HZ,
    HRVPhaseExtractor,
    RRSource,
    ibi_to_phase_series,
)
from neurophase.oscillators.market import MarketOscillators, extract_market_phase
from neurophase.oscillators.neural_protocol import (
    NeuralFrame,
    NeuralPhaseExtractor,
    NullNeuralExtractor,
    SensorStatus,
)

__all__ = [
    "DEFAULT_MIN_RR_SAMPLES",
    "DEFAULT_MIN_WINDOW_S",
    "DEFAULT_TARGET_SR_HZ",
    "HRVPhaseExtractor",
    "MarketOscillators",
    "NeuralFrame",
    "NeuralPhaseExtractor",
    "NullNeuralExtractor",
    "RRSource",
    "SensorStatus",
    "extract_market_phase",
    "ibi_to_phase_series",
]
