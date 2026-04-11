"""Sensor adapter layer — concrete bridges implementing NeuralPhaseExtractor.

The :mod:`neurophase.oscillators.neural_protocol` module defines the
abstract ``NeuralPhaseExtractor`` protocol. This subpackage provides
the concrete adapters that satisfy it:

* :class:`SyntheticOscillatorSource` — deterministic Kuramoto-driven
  phase source, the canonical fixture for offline tests and
  calibration work.

* :class:`RecordingFileSource` — replays a JSONL file of timestamped
  phase snapshots, suitable for offline analysis and CI regression.

* :class:`AdapterRegistry` — named-factory registry with a stable
  ``build(name)`` surface. A reviewer can answer *"what sensors are
  installed?"* by inspecting one dict.

What this layer does NOT provide
--------------------------------

* Hardware-specific drivers (Tobii, OpenBCI, Polar, Muse, Emotiv).
  Real hardware integration lives in separate packages that depend
  on vendor SDKs; this layer provides the architectural seam they
  plug into, plus two in-memory implementations that close the
  contract loop without any external dependency.

* Clock synchronisation. Each adapter surfaces its own
  ``sample_rate_hz`` and timestamps; aligning multiple live sources
  is a downstream concern (see ``docs/theory/time_integrity.md``).

Every adapter in this subpackage is bound to HN34 in
``INVARIANTS.yaml`` via its protocol-compliance test.
"""

from __future__ import annotations

from neurophase.sensors.recording import RecordingFileSource, RecordingSample
from neurophase.sensors.registry import (
    DEFAULT_ADAPTER_REGISTRY,
    AdapterFactory,
    AdapterRegistry,
    SensorAdapterError,
)
from neurophase.sensors.synthetic import SyntheticOscillatorConfig, SyntheticOscillatorSource

__all__ = [
    "DEFAULT_ADAPTER_REGISTRY",
    "AdapterFactory",
    "AdapterRegistry",
    "RecordingFileSource",
    "RecordingSample",
    "SensorAdapterError",
    "SyntheticOscillatorConfig",
    "SyntheticOscillatorSource",
]
