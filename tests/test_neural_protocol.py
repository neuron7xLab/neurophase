"""Tests for neurophase.oscillators.neural_protocol."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.oscillators.neural_protocol import (
    NeuralFrame,
    NeuralPhaseExtractor,
    NullNeuralExtractor,
    SensorStatus,
)


def test_null_extractor_reports_absent() -> None:
    null = NullNeuralExtractor()
    assert null.status() is SensorStatus.ABSENT
    frame = null.extract()
    assert frame.status is SensorStatus.ABSENT
    assert frame.phases.size == 0
    assert frame.channel_labels == ()
    assert frame.sample_rate_hz == 0.0


def test_null_extractor_conforms_to_protocol() -> None:
    null = NullNeuralExtractor()
    # runtime_checkable Protocol duck-typing.
    assert isinstance(null, NeuralPhaseExtractor)


def test_live_frame_requires_matching_labels() -> None:
    with pytest.raises(ValueError, match="channel_labels length"):
        NeuralFrame(
            status=SensorStatus.LIVE,
            phases=np.zeros((3, 10), dtype=np.float64),
            channel_labels=("alpha", "hrv"),
            sample_rate_hz=250.0,
        )


def test_live_frame_requires_positive_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        NeuralFrame(
            status=SensorStatus.LIVE,
            phases=np.zeros((2, 10), dtype=np.float64),
            channel_labels=("alpha", "pupil"),
            sample_rate_hz=0.0,
        )


def test_live_frame_requires_non_empty_phases() -> None:
    with pytest.raises(ValueError, match="at least one phase channel"):
        NeuralFrame(
            status=SensorStatus.LIVE,
            phases=np.array([], dtype=np.float64),
            channel_labels=(),
            sample_rate_hz=250.0,
        )


def test_degraded_frame_may_be_empty() -> None:
    frame = NeuralFrame(
        status=SensorStatus.DEGRADED,
        phases=np.array([], dtype=np.float64),
        channel_labels=(),
        sample_rate_hz=0.0,
    )
    assert frame.status is SensorStatus.DEGRADED


def test_absent_frame_may_be_empty() -> None:
    frame = NeuralFrame(
        status=SensorStatus.ABSENT,
        phases=np.array([], dtype=np.float64),
        channel_labels=(),
        sample_rate_hz=0.0,
    )
    assert frame.status is SensorStatus.ABSENT
