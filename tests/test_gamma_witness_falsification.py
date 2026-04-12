"""Falsification battery for the γ-verification witness.

Pre-registered hypotheses with fixed acceptance thresholds. Every test
either passes (hypothesis not falsified) or fails honestly — no post-hoc
parameter tuning.

H1 — Tick latency ≤ 10 ms. Hard contract from the integration protocol.
H2 — The two adapter facets produce distinct topo/cost dynamics — a
     structural prerequisite for non-trivial cross_coherence.
H3 — The witness does not alter KLR decisions under any trajectory.
H4 — Determinism: same seed + same state trajectory → identical reports.
H5 — The witness survives a long KLR run (500 ticks) without exceptions.

This module requires ``neosynaptex`` — it is skipped cleanly when the
optional dependency is not installed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from neurophase.reset import (
    GammaWitness,
    GammaWitnessReport,
    KLRPipeline,
    SystemMetrics,
    SystemState,
)
from neurophase.reset.gamma_witness import DEFAULT_WINDOW
from neurophase.reset.neosynaptex_adapter import KLRNeuronsAdapter, NeosynaptexResetAdapter

try:
    import neosynaptex  # noqa: F401
except Exception as _exc:
    pytest.skip(
        f"neosynaptex unavailable: {_exc}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _state(n: int = 8, seed: int = 0) -> SystemState:
    rng = np.random.default_rng(seed)
    base = np.full((n, n), 1.0 / n)
    noise = rng.normal(0.0, 0.001, size=(n, n))
    w = np.abs(base + noise)
    w = w / w.sum(axis=1, keepdims=True)
    return SystemState(
        weights=w,
        confidence=np.clip(np.full(n, 0.6) + rng.normal(0, 0.01, n), 0, 1),
        usage=np.clip(np.linspace(0.1, 0.9, n) + rng.normal(0, 0.02, n), 0, 1),
        utility=np.clip(np.linspace(0.2, 0.8, n) + rng.normal(0, 0.01, n), 0, 1),
        inhibition=np.clip(np.full(n, 0.5) + rng.normal(0, 0.01, n), 0, 1),
        topology=np.ones((n, n)),
    )


def _metrics() -> SystemMetrics:
    return SystemMetrics(0.9, 0.9, 0.2, 0.2, 0.1, 0.1)


# ===================================================================
# H1 — Tick latency under 10 ms
# Pre-registered threshold: median < 10 ms over 100 post-warmup ticks
# ===================================================================
class TestH1Latency:
    """H1: the witness must not add > 10 ms overhead to tick()."""

    THRESHOLD_MS: float = 10.0
    N_TICKS: int = 100

    def test_median_tick_latency_below_contract(self) -> None:
        pipe = KLRPipeline(_state())
        metrics = _metrics()

        # Warmup: let the witness and pipeline settle.
        for _ in range(DEFAULT_WINDOW + 5):
            pipe.tick(metrics)

        durations: list[float] = []
        for _ in range(self.N_TICKS):
            t0 = time.perf_counter()
            pipe.tick(metrics)
            durations.append(time.perf_counter() - t0)

        median_ms = float(np.median(durations)) * 1000.0
        assert median_ms < self.THRESHOLD_MS, (
            f"H1 FALSIFIED: median tick latency {median_ms:.2f} ms exceeds {self.THRESHOLD_MS} ms"
        )


# ===================================================================
# H2 — Distinct topo/cost signals across facets
# Pre-registered: at least 1 pair of (topo, cost) values must differ
# ===================================================================
class TestH2DistinctSignals:
    """H2: the two adapter facets emit distinguishable signals."""

    N_STATES: int = 20

    def test_signals_diverge_on_varied_trajectories(self) -> None:
        w = NeosynaptexResetAdapter()
        n = KLRNeuronsAdapter()
        topo_diffs: list[float] = []
        cost_diffs: list[float] = []
        for seed in range(self.N_STATES):
            state = _state(seed=seed)
            w.update(state)
            n.update(state)
            topo_diffs.append(abs(w.topo() - n.topo()))
            cost_diffs.append(abs(w.thermo_cost() - n.thermo_cost()))

        # At least half the states must produce non-zero topo difference.
        n_topo_distinct = sum(1 for d in topo_diffs if d > 1e-6)
        assert n_topo_distinct > self.N_STATES // 2, (
            f"H2 FALSIFIED: only {n_topo_distinct}/{self.N_STATES} states had distinct topo"
        )


# ===================================================================
# H3 — Witness never alters KLR decisions (extended trajectory)
# Pre-registered: decisions_with == decisions_without over 200 ticks
# ===================================================================
class TestH3Advisory:
    """H3: the witness is advisory-only under extended trajectories."""

    N_TICKS: int = 200

    def test_decision_equivalence_200_ticks(self) -> None:
        pipe_on = KLRPipeline(_state(seed=42))
        pipe_off = KLRPipeline(_state(seed=42), enable_witness=False)
        metrics = _metrics()

        d_on: list[str] = []
        d_off: list[str] = []
        for _ in range(self.N_TICKS):
            f_on = pipe_on.tick(metrics)
            f_off = pipe_off.tick(metrics)
            d_on.append(f_on.decision)
            d_off.append(f_off.decision)

        assert d_on == d_off, "H3 FALSIFIED: witness altered KLR decisions"
        assert np.allclose(pipe_on.twin_state.active.weights.sum(axis=1), 1.0)


# ===================================================================
# H4 — Determinism
# Pre-registered: same state trajectory → identical reports
# ===================================================================
class TestH4Determinism:
    """H4: identical inputs produce identical witness outputs."""

    N_TICKS: int = 50

    def test_reports_are_deterministic(self) -> None:
        reports_a: list[GammaWitnessReport] = []
        reports_b: list[GammaWitnessReport] = []

        for trial_reports in (reports_a, reports_b):
            witness = GammaWitness(window=DEFAULT_WINDOW)
            for i in range(self.N_TICKS):
                state = _state(seed=i)
                trial_reports.append(witness.observe(state))

        for i, (a, b) in enumerate(zip(reports_a, reports_b, strict=True)):
            assert a.phase == b.phase, f"H4 FALSIFIED: phase diverged at tick {i}"
            assert a.verdict == b.verdict, f"H4 FALSIFIED: verdict diverged at tick {i}"
            if np.isfinite(a.gamma_external) and np.isfinite(b.gamma_external):
                assert abs(a.gamma_external - b.gamma_external) < 1e-9, (
                    f"H4 FALSIFIED: gamma diverged at tick {i}"
                )


# ===================================================================
# H5 — Long-run stability (500 ticks via full KLR pipeline)
# Pre-registered: no exceptions, every frame has a valid witness_report
# ===================================================================
class TestH5LongRun:
    """H5: 500-tick KLR pipeline with witness runs without failure."""

    N_TICKS: int = 500

    def test_500_tick_pipeline_survives(self) -> None:
        pipe = KLRPipeline(_state(seed=7))
        metrics = _metrics()
        warmup_reports: list[GammaWitnessReport] = []
        post_warmup_reports: list[GammaWitnessReport] = []

        for i in range(self.N_TICKS):
            frame = pipe.tick(metrics)
            assert frame.witness_report is not None, f"witness_report was None at tick {i}"
            if i < DEFAULT_WINDOW:
                warmup_reports.append(frame.witness_report)
            else:
                post_warmup_reports.append(frame.witness_report)

        # All warmup reports must have verdict INSUFFICIENT_DATA.
        for r in warmup_reports:
            assert r.verdict == "INSUFFICIENT_DATA"

        # Post-warmup: every report must have a valid verdict and finite fields.
        for r in post_warmup_reports:
            assert r.verdict in {"COHERENT", "INCOHERENT", "INSUFFICIENT_DATA"}
            assert np.isfinite(r.gamma_external)
            assert np.isfinite(r.coherence)

        # Row-stochastic invariant survives.
        assert np.allclose(pipe.twin_state.active.weights.sum(axis=1), 1.0)
