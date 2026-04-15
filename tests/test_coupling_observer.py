"""Tests for :mod:`neurophase.sync.coupling_observer`.

Covers:

* Warm-up — no snapshot until the buffer holds ``window`` samples.
* Hop semantics — emission cadence matches ``hop`` after warm-up;
  ``hop = 1`` emits every tick, ``hop = window`` emits over disjoint
  windows.
* Sliding contents — the buffer always reflects the latest ``window``
  samples, not the earliest.
* Determinism — under a shared seed, the observer emits byte-identical
  snapshots across independent runs over the same input stream.
* Reset semantics — clears buffer and re-warms.
* Validation contracts — invalid ``window`` / ``hop`` raise.
* End-to-end with the real :class:`CoupledBrainMarketSystem`.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.sync.coupled_brain_market import (
    CoupledBrainMarketSystem,
    CoupledStep,
)
from neurophase.sync.coupling_direction import CouplingDirection
from neurophase.sync.coupling_observer import CouplingObserver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(t: float, psi_brain: float, psi_market: float, R: float) -> CoupledStep:
    delta = float(np.arccos(np.clip(np.cos(psi_brain - psi_market), -1.0, 1.0)))
    return CoupledStep(
        t=t,
        R=R,
        psi_brain=psi_brain,
        psi_market=psi_market,
        delta=delta,
        execution_allowed=bool(R >= 0.5),
    )


def _synthetic_stream(n: int, *, seed: int = 0) -> list[CoupledStep]:
    """Brain → market causal stream wrapped as CoupledStep records."""
    rng = np.random.default_rng(seed)
    psi_brain = rng.uniform(-np.pi, np.pi, n)
    psi_market = np.empty(n)
    psi_market[0] = rng.uniform(-np.pi, np.pi)
    psi_market[1:] = 0.9 * psi_brain[:-1] + 0.3 * rng.standard_normal(n - 1)
    R = np.clip(np.abs(np.cos(psi_brain - psi_market)), 0.0, 1.0)
    return [
        _make_step(t=float(i) * 0.01, psi_brain=float(b), psi_market=float(m), R=float(r))
        for i, (b, m, r) in enumerate(zip(psi_brain, psi_market, R, strict=True))
    ]


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------


def test_no_snapshot_before_warm_up() -> None:
    obs = CouplingObserver(window=600, hop=10, n_surrogates=20, seed=0)
    stream = _synthetic_stream(599, seed=0)
    snapshots = [obs.observe(s) for s in stream]
    assert all(s is None for s in snapshots)
    assert obs.filled is False
    assert obs.emissions == 0
    assert obs.last_snapshot is None


def test_first_snapshot_lands_exactly_when_buffer_fills() -> None:
    obs = CouplingObserver(window=400, hop=50, n_surrogates=20, seed=0)
    stream = _synthetic_stream(600, seed=1)
    emit_indices = [i for i, s in enumerate(stream) if obs.observe(s) is not None]
    # First emission must be at index window - 1 (zero-based), i.e. the tick
    # that brings the buffer to exactly `window` samples.
    assert emit_indices[0] == obs.window - 1


# ---------------------------------------------------------------------------
# Hop semantics
# ---------------------------------------------------------------------------


def test_hop_one_emits_every_tick_after_warm_up() -> None:
    window, n = 200, 400
    obs = CouplingObserver(window=window, hop=1, n_surrogates=10, seed=0)
    stream = _synthetic_stream(n, seed=2)
    snapshots = [obs.observe(s) for s in stream]
    pre = sum(1 for s in snapshots[: window - 1] if s is not None)
    post = sum(1 for s in snapshots[window - 1 :] if s is not None)
    assert pre == 0
    assert post == n - window + 1
    assert obs.emissions == post


def test_hop_equal_window_emits_over_disjoint_windows() -> None:
    window, n = 200, 1000
    obs = CouplingObserver(window=window, hop=window, n_surrogates=10, seed=0)
    stream = _synthetic_stream(n, seed=3)
    emit_ticks = [i for i, s in enumerate(stream) if obs.observe(s) is not None]
    # First emission at warm-up (i = window - 1), then one every `window` ticks.
    expected = list(range(window - 1, n, window))
    assert emit_ticks == expected


def test_hop_arbitrary_emits_every_hop_after_warm_up() -> None:
    window, hop, n = 100, 40, 400
    obs = CouplingObserver(window=window, hop=hop, n_surrogates=10, seed=0)
    stream = _synthetic_stream(n, seed=4)
    emit_ticks = [i for i, s in enumerate(stream) if obs.observe(s) is not None]
    expected = [window - 1, *range(window - 1 + hop, n, hop)]
    assert emit_ticks == expected


# ---------------------------------------------------------------------------
# Sliding window contents
# ---------------------------------------------------------------------------


def test_buffer_holds_only_latest_window_samples() -> None:
    """An observer fed N > window steps must compute on the LAST `window`."""
    window = 200
    obs_full = CouplingObserver(window=window, hop=window, n_surrogates=20, seed=11)
    obs_tail = CouplingObserver(window=window, hop=window, n_surrogates=20, seed=11)
    stream = _synthetic_stream(window * 3, seed=5)
    # Feed the first observer the full stream; record its last snapshot.
    snap_full = None
    for s in stream:
        out = obs_full.observe(s)
        if out is not None:
            snap_full = out
    # Feed the second observer only the tail (the implicit window of obs_full).
    for s in stream[-window:]:
        snap_tail = obs_tail.observe(s)
    assert snap_full is not None
    assert snap_tail is not None
    assert snap_full == snap_tail


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_two_observers_agree_under_shared_seed() -> None:
    stream = _synthetic_stream(500, seed=7)
    a = CouplingObserver(window=300, hop=300, n_surrogates=30, seed=42)
    b = CouplingObserver(window=300, hop=300, n_surrogates=30, seed=42)
    out_a = [a.observe(s) for s in stream]
    out_b = [b.observe(s) for s in stream]
    assert out_a == out_b


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_buffer_and_emissions() -> None:
    obs = CouplingObserver(window=100, hop=100, n_surrogates=10, seed=0)
    for s in _synthetic_stream(250, seed=8):
        obs.observe(s)
    assert obs.emissions > 0
    assert obs.last_snapshot is not None
    obs.reset()
    assert obs.emissions == 0
    assert obs.filled is False
    assert obs.last_snapshot is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_rejects_too_small_window() -> None:
    with pytest.raises(ValueError, match="window"):
        CouplingObserver(window=3)


def test_rejects_non_positive_hop() -> None:
    with pytest.raises(ValueError, match="hop"):
        CouplingObserver(window=100, hop=0)


# ---------------------------------------------------------------------------
# End-to-end with the real coupled-Kuramoto system
# ---------------------------------------------------------------------------


def test_end_to_end_with_real_coupled_system_emits_well_formed_snapshots() -> None:
    sys = CoupledBrainMarketSystem(K=2.0, sigma=0.05, dt=0.01, seed=2026)
    obs = CouplingObserver(window=300, hop=300, n_surrogates=30, seed=2026)
    snapshots: list[CouplingDirection] = []
    for _ in range(900):
        R, psi_b, psi_m = sys.step()
        step = _make_step(t=0.0, psi_brain=psi_b, psi_market=psi_m, R=R)
        out = obs.observe(step)
        if out is not None:
            snapshots.append(out)
    # 900 steps, window=300, hop=300 ⇒ emissions at i = 299, 599, 899.
    assert len(snapshots) == 3
    for snap in snapshots:
        assert snap.te_brain_to_market >= 0.0
        assert snap.te_market_to_brain >= 0.0
        assert 0.0 < snap.p_brain_to_market <= 1.0
        assert 0.0 < snap.p_market_to_brain <= 1.0
        assert np.isfinite(snap.sigma_R)
        assert snap.n_samples == 300
