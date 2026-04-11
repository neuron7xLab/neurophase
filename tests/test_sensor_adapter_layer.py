"""HN34 — sensor adapter layer contract tests.

Binds the :mod:`neurophase.sensors` subpackage to a minimal
contract surface:

1. Every built-in adapter implements :class:`NeuralPhaseExtractor`.
2. :class:`SyntheticOscillatorSource` is deterministic:
   two sources with the same config produce byte-identical frame
   sequences across 64 steps.
3. :class:`SyntheticOscillatorSource` emits LIVE frames with the
   expected channel count, labels, and sample rate.
4. :class:`RecordingFileSource` replays a committed JSONL file
   deterministically and transitions to ABSENT on exhaustion.
5. :class:`AdapterRegistry` rejects empty names, duplicate
   registrations, and unknown lookups. Build-time protocol check
   refuses non-conforming factories.
6. :data:`DEFAULT_ADAPTER_REGISTRY` ships with ``"synthetic"`` and
   ``"null"`` pre-registered and both build cleanly.
7. Round-trip: a synthetic source → extract N frames →
   write to JSONL → replay via RecordingFileSource → compare
   byte-identical phase arrays.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurophase.oscillators.neural_protocol import (
    NeuralFrame,
    NeuralPhaseExtractor,
    NullNeuralExtractor,
    SensorStatus,
)
from neurophase.sensors import (
    DEFAULT_ADAPTER_REGISTRY,
    AdapterRegistry,
    RecordingFileSource,
    SensorAdapterError,
    SyntheticOscillatorConfig,
    SyntheticOscillatorSource,
)

# ---------------------------------------------------------------------------
# 1. Protocol compliance.
# ---------------------------------------------------------------------------


def test_synthetic_source_implements_protocol() -> None:
    assert isinstance(SyntheticOscillatorSource(), NeuralPhaseExtractor)


def test_recording_source_implements_protocol(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text(
        json.dumps(
            {
                "phases": [0.1, -0.3],
                "channel_labels": ["a", "b"],
                "sample_rate_hz": 100.0,
            }
        )
        + "\n"
    )
    assert isinstance(RecordingFileSource(path), NeuralPhaseExtractor)


def test_null_extractor_implements_protocol() -> None:
    assert isinstance(NullNeuralExtractor(), NeuralPhaseExtractor)


# ---------------------------------------------------------------------------
# 2. Synthetic source determinism.
# ---------------------------------------------------------------------------


def test_synthetic_source_is_deterministic() -> None:
    cfg = SyntheticOscillatorConfig(n_channels=4, seed=7)
    a = SyntheticOscillatorSource(cfg)
    b = SyntheticOscillatorSource(cfg)
    for _ in range(64):
        fa = a.extract()
        fb = b.extract()
        assert np.array_equal(fa.phases, fb.phases)


def test_synthetic_source_reset_restores_initial_state() -> None:
    cfg = SyntheticOscillatorConfig(n_channels=3, seed=123)
    src = SyntheticOscillatorSource(cfg)
    first = src.extract()
    for _ in range(10):
        src.extract()
    assert src.n_steps == 11
    src.reset()
    assert src.n_steps == 0
    second = src.extract()
    assert np.array_equal(first.phases, second.phases)


# ---------------------------------------------------------------------------
# 3. Synthetic source frame shape.
# ---------------------------------------------------------------------------


def test_synthetic_frame_shape_matches_config() -> None:
    cfg = SyntheticOscillatorConfig(
        n_channels=5,
        sample_rate_hz=128.0,
        channel_labels=("alpha1", "alpha2", "beta1", "gamma1", "theta1"),
    )
    src = SyntheticOscillatorSource(cfg)
    frame: NeuralFrame = src.extract()
    assert frame.status is SensorStatus.LIVE
    assert frame.phases.shape == (5,)
    assert frame.channel_labels == ("alpha1", "alpha2", "beta1", "gamma1", "theta1")
    assert frame.sample_rate_hz == 128.0


def test_synthetic_source_invalid_config_rejected() -> None:
    with pytest.raises(ValueError, match="n_channels"):
        SyntheticOscillatorConfig(n_channels=0)
    with pytest.raises(ValueError, match="coupling_strength"):
        SyntheticOscillatorConfig(coupling_strength=1.5)
    with pytest.raises(ValueError, match="channel_labels"):
        SyntheticOscillatorConfig(n_channels=2, channel_labels=("only_one",))


# ---------------------------------------------------------------------------
# 4. RecordingFileSource behaviour.
# ---------------------------------------------------------------------------


def test_recording_source_replays_committed_sample() -> None:
    path = Path("data/sensors/sample_recording.jsonl")
    assert path.is_file(), f"committed sample recording missing at {path}"
    src = RecordingFileSource(path)
    assert src.n_samples == 10
    frames: list[NeuralFrame] = []
    for _ in range(10):
        assert src.status() is SensorStatus.LIVE
        frames.append(src.extract())
    # Exhausted — next call reports ABSENT.
    assert src.status() is SensorStatus.ABSENT
    exhausted = src.extract()
    assert exhausted.status is SensorStatus.ABSENT
    assert exhausted.phases.size == 0

    # All replayed frames match the expected shape.
    for frame in frames:
        assert frame.status is SensorStatus.LIVE
        assert frame.phases.shape == (4,)
        assert len(frame.channel_labels) == 4


def test_recording_source_reset_rewinds(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    lines = []
    for i in range(3):
        lines.append(
            json.dumps(
                {
                    "phases": [float(i), float(i + 1)],
                    "channel_labels": ["a", "b"],
                    "sample_rate_hz": 10.0,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")
    src = RecordingFileSource(path)
    first_pass = [src.extract().phases[0] for _ in range(3)]
    src.reset()
    second_pass = [src.extract().phases[0] for _ in range(3)]
    assert first_pass == second_pass


def test_recording_source_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        RecordingFileSource(tmp_path / "does_not_exist.jsonl")


def test_recording_source_rejects_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not valid json\n")
    with pytest.raises(ValueError, match="malformed JSON"):
        RecordingFileSource(bad)


def test_recording_source_rejects_schema_mismatch(tmp_path: Path) -> None:
    bad = tmp_path / "schema.jsonl"
    bad.write_text(
        json.dumps(
            {
                "phases": [0.1, 0.2],
                "channel_labels": ["only_one"],  # length mismatch
                "sample_rate_hz": 10.0,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="does not match channel_labels"):
        RecordingFileSource(bad)


# ---------------------------------------------------------------------------
# 5. AdapterRegistry contracts.
# ---------------------------------------------------------------------------


def test_registry_rejects_empty_name() -> None:
    reg = AdapterRegistry()
    with pytest.raises(SensorAdapterError, match="non-empty"):
        reg.register("", lambda: NullNeuralExtractor())


def test_registry_rejects_duplicate_registration() -> None:
    reg = AdapterRegistry()
    reg.register("x", lambda: NullNeuralExtractor())
    with pytest.raises(SensorAdapterError, match="already registered"):
        reg.register("x", lambda: NullNeuralExtractor())


def test_registry_unknown_build_raises() -> None:
    reg = AdapterRegistry()
    with pytest.raises(SensorAdapterError, match="unknown adapter"):
        reg.build("nope")


def test_registry_unknown_unregister_raises() -> None:
    reg = AdapterRegistry()
    with pytest.raises(KeyError):
        reg.unregister("nope")


def test_registry_unregister_happy_path() -> None:
    reg = AdapterRegistry()
    reg.register("x", lambda: NullNeuralExtractor())
    assert "x" in reg
    reg.unregister("x")
    assert "x" not in reg


def test_registry_build_checks_protocol_conformance() -> None:
    """A factory returning a non-conforming object is refused at build time."""
    reg = AdapterRegistry()

    class _FakeSensor:
        """Not a NeuralPhaseExtractor — missing extract()."""

        def status(self) -> SensorStatus:  # pragma: no cover — never reached
            return SensorStatus.LIVE

    # The registry accepts the registration but refuses at build time.
    reg.register("fake", lambda: _FakeSensor())  # type: ignore[arg-type,return-value]
    with pytest.raises(SensorAdapterError, match="NeuralPhaseExtractor"):
        reg.build("fake")


def test_registry_repr_and_len() -> None:
    reg = AdapterRegistry()
    assert len(reg) == 0
    reg.register("a", lambda: NullNeuralExtractor())
    reg.register("b", lambda: NullNeuralExtractor())
    assert len(reg) == 2
    r = repr(reg)
    assert "a" in r and "b" in r


# ---------------------------------------------------------------------------
# 6. Default registry surface.
# ---------------------------------------------------------------------------


def test_default_registry_ships_synthetic_and_null() -> None:
    assert "synthetic" in DEFAULT_ADAPTER_REGISTRY
    assert "null" in DEFAULT_ADAPTER_REGISTRY
    assert len(DEFAULT_ADAPTER_REGISTRY) >= 2


def test_default_synthetic_builds_clean() -> None:
    src = DEFAULT_ADAPTER_REGISTRY.build("synthetic")
    assert isinstance(src, NeuralPhaseExtractor)
    assert src.status() is SensorStatus.LIVE
    frame = src.extract()
    assert frame.phases.size >= 1


def test_default_null_builds_clean() -> None:
    src = DEFAULT_ADAPTER_REGISTRY.build("null")
    assert isinstance(src, NeuralPhaseExtractor)
    assert src.status() is SensorStatus.ABSENT
    frame = src.extract()
    assert frame.status is SensorStatus.ABSENT
    assert frame.phases.size == 0


# ---------------------------------------------------------------------------
# 7. Round-trip: synthetic → JSONL → recording replay → byte-identical.
# ---------------------------------------------------------------------------


def test_synthetic_to_recording_round_trip(tmp_path: Path) -> None:
    cfg = SyntheticOscillatorConfig(n_channels=3, seed=999)

    # 1. Drive the synthetic source and record to JSONL.
    src = SyntheticOscillatorSource(cfg)
    out = tmp_path / "round_trip.jsonl"
    recorded_phases: list[np.ndarray] = []
    with out.open("w", encoding="utf-8") as fh:
        for _ in range(20):
            frame = src.extract()
            recorded_phases.append(np.copy(frame.phases))
            fh.write(
                json.dumps(
                    {
                        "phases": frame.phases.tolist(),
                        "channel_labels": list(frame.channel_labels),
                        "sample_rate_hz": frame.sample_rate_hz,
                    }
                )
                + "\n"
            )

    # 2. Replay via RecordingFileSource.
    replay = RecordingFileSource(out)
    assert replay.n_samples == 20
    for i in range(20):
        replayed = replay.extract()
        assert np.array_equal(replayed.phases, recorded_phases[i])
    # Exhausted.
    assert replay.status() is SensorStatus.ABSENT


# ---------------------------------------------------------------------------
# 8. Sample recording file is well-formed and non-empty.
# ---------------------------------------------------------------------------


def test_committed_sample_recording_file_loads() -> None:
    path = Path("data/sensors/sample_recording.jsonl")
    src = RecordingFileSource(path)
    assert src.n_samples == 10
    # Every sample must have non-empty phases and a positive rate.
    for _ in range(10):
        frame = src.extract()
        assert frame.status is SensorStatus.LIVE
        assert frame.phases.size > 0
        assert frame.sample_rate_hz > 0.0
