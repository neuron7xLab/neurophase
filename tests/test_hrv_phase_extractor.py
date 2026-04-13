"""Tests for neurophase.oscillators.hrv_phase_extractor.

The HRV-phase extractor has two surfaces:

  * :func:`ibi_to_phase_series` — pure deterministic transform, tested
    with exact-value fixtures on a constructed RR sequence.
  * :class:`HRVPhaseExtractor` — stateless Protocol implementation whose
    status() / extract() contract must honour the repo-level invariant
    I₃ (never fabricate phase when the sensor cannot defend it).

These tests exercise both without any BLE / LSL transport.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from neurophase.oscillators.hrv_phase_extractor import (
    DEFAULT_MIN_RR_SAMPLES,
    DEFAULT_MIN_WINDOW_S,
    HRVPhaseExtractor,
    ibi_to_phase_series,
)
from neurophase.oscillators.neural_protocol import (
    NeuralFrame,
    NeuralPhaseExtractor,
    SensorStatus,
)
from neurophase.physio.replay import RRSample

# =======================================================================
#   Fixtures
# =======================================================================


def _rr_sequence(
    *,
    n: int,
    mean_ms: float = 820.0,
    jitter_ms: float = 10.0,
    t0: float = 100.0,
) -> list[RRSample]:
    """Build a monotonic RR sequence. Jitter alternates sign so the
    spline has a benign second derivative."""
    out: list[RRSample] = []
    t = t0
    for i in range(n):
        rr = mean_ms + (jitter_ms if i % 2 == 0 else -jitter_ms)
        t += rr / 1000.0
        out.append(RRSample(timestamp_s=t, rr_ms=rr, row_index=i))
    return out


def _hrv_modulated_sequence(
    *,
    n: int,
    mean_ms: float = 820.0,
    amp_ms: float = 25.0,
    hf_hz: float = 0.2,  # respiratory-coupled HRV
    t0: float = 100.0,
) -> list[RRSample]:
    """RR modulated by a slow sinusoid. Produces a phase series with
    enough structure for meaningful Hilbert output."""
    out: list[RRSample] = []
    t = t0
    elapsed = 0.0
    for i in range(n):
        rr = mean_ms + amp_ms * math.sin(2.0 * math.pi * hf_hz * elapsed)
        t += rr / 1000.0
        elapsed += rr / 1000.0
        out.append(RRSample(timestamp_s=t, rr_ms=rr, row_index=i))
    return out


def _static_source(samples: Sequence[RRSample]):
    """Build a zero-arg callable returning a fixed sample list."""

    def _source() -> Sequence[RRSample]:
        return samples

    return _source


# =======================================================================
#   ibi_to_phase_series: pure transform
# =======================================================================


class TestIbiToPhaseSeries:
    def test_smoke_runs_on_plausible_window(self) -> None:
        samples = _hrv_modulated_sequence(n=80)
        phase = ibi_to_phase_series(samples)
        # Phase must be non-empty and bounded in (-pi, pi].
        assert phase.size > 0
        assert np.all(phase > -math.pi - 1e-9)
        assert np.all(phase <= math.pi + 1e-9)

    def test_deterministic_for_same_input(self) -> None:
        samples = _hrv_modulated_sequence(n=80)
        a = ibi_to_phase_series(samples)
        b = ibi_to_phase_series(samples)
        np.testing.assert_array_equal(a, b)

    def test_rejects_fewer_than_four_samples(self) -> None:
        samples = _rr_sequence(n=3)
        with pytest.raises(ValueError, match="4 RR samples"):
            ibi_to_phase_series(samples)

    def test_rejects_non_monotonic_timestamps(self) -> None:
        samples = _rr_sequence(n=20)
        # Swap two mid-sequence timestamps to break monotonicity while
        # keeping the envelope validator of RRSample happy.
        samples[5] = RRSample(
            timestamp_s=samples[3].timestamp_s,
            rr_ms=820.0,
            row_index=5,
        )
        with pytest.raises(ValueError, match="monotonic"):
            ibi_to_phase_series(samples)

    def test_rejects_too_short_span(self) -> None:
        # Four samples clustered in a tiny time window: span * sr < 4.
        base_ts = 100.0
        samples = [
            RRSample(timestamp_s=base_ts + i * 0.001, rr_ms=820.0, row_index=i) for i in range(4)
        ]
        with pytest.raises(ValueError, match="too short"):
            ibi_to_phase_series(samples, target_sr_hz=4.0)

    def test_rejects_invalid_target_sr(self) -> None:
        samples = _rr_sequence(n=40)
        with pytest.raises(ValueError, match="target_sr_hz"):
            ibi_to_phase_series(samples, target_sr_hz=0.0)
        with pytest.raises(ValueError, match="target_sr_hz"):
            ibi_to_phase_series(samples, target_sr_hz=-1.0)

    def test_phase_length_matches_grid(self) -> None:
        samples = _hrv_modulated_sequence(n=80)
        ts = [s.timestamp_s for s in samples]
        sr = 4.0
        expected_n = int(np.arange(ts[0], ts[-1], 1.0 / sr).size)
        phase = ibi_to_phase_series(samples, target_sr_hz=sr)
        assert phase.shape == (expected_n,)


# =======================================================================
#   HRVPhaseExtractor: Protocol contract + fail-closed semantics
# =======================================================================


class TestHRVPhaseExtractorContract:
    def test_satisfies_neural_phase_extractor_protocol(self) -> None:
        ext = HRVPhaseExtractor(rr_source=_static_source([]))
        assert isinstance(ext, NeuralPhaseExtractor)

    def test_status_is_absent_before_first_extract(self) -> None:
        ext = HRVPhaseExtractor(rr_source=_static_source([]))
        assert ext.status() is SensorStatus.ABSENT

    def test_empty_source_yields_absent_frame(self) -> None:
        ext = HRVPhaseExtractor(rr_source=_static_source([]))
        frame = ext.extract()
        assert frame.status is SensorStatus.ABSENT
        assert frame.phases.size == 0
        assert frame.sample_rate_hz == 0.0
        assert ext.status() is SensorStatus.ABSENT

    def test_source_raising_yields_absent(self) -> None:
        def _boom() -> list[RRSample]:
            raise RuntimeError("source error")

        ext = HRVPhaseExtractor(rr_source=_boom)
        frame = ext.extract()
        assert frame.status is SensorStatus.ABSENT
        assert frame.phases.size == 0

    def test_too_few_samples_yields_degraded(self) -> None:
        # Need >= DEFAULT_MIN_RR_SAMPLES (20); give 10.
        ext = HRVPhaseExtractor(rr_source=_static_source(_rr_sequence(n=10)))
        frame = ext.extract()
        assert frame.status is SensorStatus.DEGRADED
        assert frame.phases.size == 0
        assert ext.status() is SensorStatus.DEGRADED

    def test_too_short_window_yields_degraded(self) -> None:
        # Enough samples but compressed timestamps -> span < min_window_s.
        samples = [
            RRSample(timestamp_s=100.0 + i * 0.05, rr_ms=820.0, row_index=i)
            for i in range(DEFAULT_MIN_RR_SAMPLES + 5)
        ]
        ext = HRVPhaseExtractor(rr_source=_static_source(samples))
        frame = ext.extract()
        assert frame.status is SensorStatus.DEGRADED

    def test_full_window_yields_live_frame(self) -> None:
        # Enough samples across > min_window_s.
        n = 60  # ~49 s at mean-RR 820 ms
        samples = _hrv_modulated_sequence(n=n)
        span = samples[-1].timestamp_s - samples[0].timestamp_s
        assert span >= DEFAULT_MIN_WINDOW_S, f"fixture produced span {span!r}"
        ext = HRVPhaseExtractor(rr_source=_static_source(samples))
        frame = ext.extract()
        assert frame.status is SensorStatus.LIVE
        assert frame.phases.ndim == 2
        assert frame.phases.shape[0] == 1  # one channel
        assert frame.phases.shape[1] > 0
        assert frame.channel_labels == ("hrv",)
        assert frame.sample_rate_hz == pytest.approx(4.0)
        # Live frames must validate NeuralFrame's own invariants.
        assert isinstance(frame, NeuralFrame)

    def test_extract_is_deterministic_for_static_source(self) -> None:
        samples = _hrv_modulated_sequence(n=60)
        ext_a = HRVPhaseExtractor(rr_source=_static_source(samples))
        ext_b = HRVPhaseExtractor(rr_source=_static_source(samples))
        frame_a = ext_a.extract()
        frame_b = ext_b.extract()
        assert frame_a.status is frame_b.status
        np.testing.assert_array_equal(frame_a.phases, frame_b.phases)


# =======================================================================
#   Construction guards
# =======================================================================


class TestHRVPhaseExtractorGuards:
    def test_min_window_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="min_window_s"):
            HRVPhaseExtractor(rr_source=_static_source([]), min_window_s=0.0)

    def test_min_rr_samples_floor(self) -> None:
        with pytest.raises(ValueError, match="min_rr_samples"):
            HRVPhaseExtractor(rr_source=_static_source([]), min_rr_samples=3)

    def test_target_sr_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="target_sr_hz"):
            HRVPhaseExtractor(rr_source=_static_source([]), target_sr_hz=0.0)

    def test_channel_label_non_empty(self) -> None:
        with pytest.raises(ValueError, match="channel_label"):
            HRVPhaseExtractor(rr_source=_static_source([]), channel_label="")
