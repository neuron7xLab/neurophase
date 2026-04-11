"""Tests for ``neurophase.analysis.prediction_error``."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from neurophase.analysis.prediction_error import (
    CognitiveState,
    PredictionErrorMonitor,
    PredictionErrorSample,
)

# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


class TestZeroError:
    def test_zero_error_when_phases_equal(self) -> None:
        m = PredictionErrorMonitor()
        out = m.update(psi_brain=0.3, psi_market=0.3)
        assert out["delta"] == pytest.approx(0.0, abs=1e-12)
        assert out["R_proxy"] == pytest.approx(1.0, abs=1e-12)
        assert out["cognitive_state"] is CognitiveState.SYNCHRONIZED

    def test_zero_error_modulo_2pi(self) -> None:
        m = PredictionErrorMonitor()
        out = m.update(psi_brain=0.3, psi_market=0.3 + 2 * math.pi)
        assert out["delta"] == pytest.approx(0.0, abs=1e-9)
        assert out["R_proxy"] == pytest.approx(1.0, abs=1e-9)


class TestMaxError:
    def test_maximum_error_at_pi_difference(self) -> None:
        m = PredictionErrorMonitor()
        out = m.update(psi_brain=0.0, psi_market=math.pi)
        assert out["delta"] == pytest.approx(math.pi, abs=1e-12)
        assert out["R_proxy"] == pytest.approx(0.0, abs=1e-12)
        assert out["cognitive_state"] is CognitiveState.SURRENDERED

    def test_half_cycle_is_half(self) -> None:
        m = PredictionErrorMonitor()
        out = m.update(psi_brain=0.0, psi_market=math.pi / 2)
        assert out["delta"] == pytest.approx(math.pi / 2, abs=1e-12)
        assert out["R_proxy"] == pytest.approx(0.5, abs=1e-12)


class TestSurrenderState:
    def test_surrendered_state_at_high_delta(self) -> None:
        m = PredictionErrorMonitor(sync_threshold=0.8, surrender_threshold=0.4)
        out = m.update(psi_brain=0.0, psi_market=2.8)  # far into anti-phase
        assert out["cognitive_state"] is CognitiveState.SURRENDERED
        assert out["R_proxy"] < 0.4

    def test_diverging_between_thresholds(self) -> None:
        m = PredictionErrorMonitor(sync_threshold=0.8, surrender_threshold=0.4)
        # Half a radian apart → R_proxy = (1 + cos 0.5)/2 ≈ 0.939 → SYNC
        # One radian → R_proxy = (1 + cos 1.0)/2 ≈ 0.77 → DIVERGING
        out = m.update(psi_brain=0.0, psi_market=1.0)
        assert out["cognitive_state"] is CognitiveState.DIVERGING


class TestHistorySchema:
    def test_history_schema(self) -> None:
        m = PredictionErrorMonitor(dt=0.01)
        for i in range(5):
            m.update(psi_brain=0.1 * i, psi_market=0.1 * i + 0.2)
        df = m.history()
        expected_columns = [
            "t",
            "psi_brain",
            "psi_market",
            "delta",
            "R_proxy",
            "cognitive_state",
        ]
        assert list(df.columns) == expected_columns
        assert len(df) == 5
        # cognitive_state is stored as the enum name for archive stability.
        assert df["cognitive_state"].isin([s.name for s in CognitiveState]).all()
        # Timestamps are monotonic under implicit dt.
        assert df["t"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------


class TestValidation:
    def test_bad_threshold_ordering_rejected(self) -> None:
        with pytest.raises(ValueError):
            PredictionErrorMonitor(sync_threshold=0.3, surrender_threshold=0.6)

    def test_bad_threshold_bounds_rejected(self) -> None:
        with pytest.raises(ValueError):
            PredictionErrorMonitor(sync_threshold=1.1)
        with pytest.raises(ValueError):
            PredictionErrorMonitor(surrender_threshold=0.0)

    def test_nonfinite_phases_rejected(self) -> None:
        m = PredictionErrorMonitor()
        with pytest.raises(ValueError):
            m.update(psi_brain=float("nan"), psi_market=0.0)
        with pytest.raises(ValueError):
            m.update(psi_brain=0.0, psi_market=float("inf"))

    def test_explicit_timestamp_required_when_dt_none(self) -> None:
        m = PredictionErrorMonitor(dt=None)
        with pytest.raises(ValueError):
            m.update(psi_brain=0.0, psi_market=0.0)
        # With explicit t it works.
        out = m.update(psi_brain=0.0, psi_market=0.0, t=1.23)
        assert out["t"] == 1.23


class TestReset:
    def test_reset_clears_archive(self) -> None:
        m = PredictionErrorMonitor()
        for i in range(3):
            m.update(psi_brain=0.0, psi_market=0.1 * i)
        assert len(m.history()) == 3
        m.reset()
        assert len(m.history()) == 0

    def test_empty_history_has_schema(self) -> None:
        m = PredictionErrorMonitor()
        df = m.history()
        assert list(df.columns) == [
            "t",
            "psi_brain",
            "psi_market",
            "delta",
            "R_proxy",
            "cognitive_state",
        ]
        assert len(df) == 0


class TestMonotonicity:
    def test_R_proxy_monotone_in_delta(self) -> None:
        """As Δ grows from 0 → π, R_proxy must monotonically decrease."""
        m = PredictionErrorMonitor()
        prev = 1.1
        for delta in np.linspace(0, math.pi, 25):
            out = m.update(psi_brain=0.0, psi_market=float(delta))
            R = float(out["R_proxy"])
            assert prev + 1e-12 >= R
            prev = R

    def test_sample_dataclass_frozen(self) -> None:
        s = PredictionErrorSample(
            t=0.0,
            psi_brain=0.0,
            psi_market=0.0,
            delta=0.0,
            R_proxy=1.0,
            cognitive_state=CognitiveState.SYNCHRONIZED,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.delta = 0.5  # type: ignore[misc]
