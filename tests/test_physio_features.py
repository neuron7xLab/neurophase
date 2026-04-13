"""Tests for neurophase.physio.features — HRV window and features."""

from __future__ import annotations

import pytest

from neurophase.physio.features import (
    DEFAULT_WINDOW_SIZE,
    MIN_WINDOW_SIZE,
    HRVWindow,
)
from neurophase.physio.replay import RRSample


def _push_n(win: HRVWindow, *, n: int, base_rr: float, jitter: float = 0.0) -> None:
    t = 0.0
    for i in range(n):
        rr = base_rr + (jitter if i % 2 == 0 else -jitter)
        t += rr / 1000.0
        win.push(RRSample(timestamp_s=t, rr_ms=rr, row_index=i))


class TestHRVWindowConstruction:
    def test_default_window_size(self) -> None:
        win = HRVWindow()
        assert win.window_size == DEFAULT_WINDOW_SIZE

    def test_min_window_enforced(self) -> None:
        with pytest.raises(ValueError, match=">= "):
            HRVWindow(window_size=MIN_WINDOW_SIZE - 1)


class TestFeaturesOnShortBuffer:
    def test_empty_buffer_returns_zero_confidence(self) -> None:
        win = HRVWindow()
        feats = win.features()
        assert feats.window_size == 0
        assert feats.confidence == 0.0
        assert feats.continuity_fraction == 0.0

    def test_below_min_window_returns_zero_confidence(self) -> None:
        win = HRVWindow()
        _push_n(win, n=MIN_WINDOW_SIZE - 1, base_rr=820.0, jitter=10.0)
        feats = win.features()
        assert feats.window_size == MIN_WINDOW_SIZE - 1
        assert feats.confidence == 0.0
        assert feats.rmssd_plausible is False


class TestFeaturesOnFullBuffer:
    def test_stable_full_buffer_yields_high_confidence(self) -> None:
        win = HRVWindow()
        _push_n(win, n=DEFAULT_WINDOW_SIZE, base_rr=820.0, jitter=10.0)
        feats = win.features()
        assert feats.window_size == DEFAULT_WINDOW_SIZE
        assert feats.continuity_fraction == pytest.approx(1.0)
        assert feats.rmssd_plausible is True
        assert feats.confidence > 0.9
        assert feats.stability > 0.9

    def test_flatlined_rr_fails_rmssd_plausibility(self) -> None:
        # Identical RRs -> RMSSD = 0 -> outside plausibility envelope.
        win = HRVWindow()
        _push_n(win, n=DEFAULT_WINDOW_SIZE, base_rr=820.0, jitter=0.0)
        feats = win.features()
        assert feats.rmssd_ms == pytest.approx(0.0)
        assert feats.rmssd_plausible is False
        # Confidence must be penalised by the RMSSD-plausibility factor.
        assert feats.confidence < 0.25

    def test_wild_artifact_window_lowers_confidence(self) -> None:
        win = HRVWindow()
        alternating = [400.0, 1800.0] * (DEFAULT_WINDOW_SIZE // 2)
        for i, rr in enumerate(alternating):
            win.push(RRSample(timestamp_s=float(i + 1), rr_ms=rr, row_index=i))
        feats = win.features()
        # Stability should collapse (CoV is huge).
        assert feats.stability < 0.5
        # RMSSD huge -> outside plausibility envelope.
        assert feats.rmssd_plausible is False
        assert feats.confidence < 0.25
