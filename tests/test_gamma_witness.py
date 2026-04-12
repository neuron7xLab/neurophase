"""Tests for the external γ-verification witness.

Binds ``NEO-I1`` (read-only witness) and ``NEO-I2`` (advisory-only verdict)
from ``INVARIANTS.yaml``. Requires the optional ``neosynaptex`` extras
group — the tests are skipped (not xfailed) when it is not installed so
that the strict neurophase core suite remains unchanged.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from neurophase.reset import (
    GammaWitness,
    GammaWitnessReport,
    KLRFrame,
    KLRPipeline,
    SystemMetrics,
    SystemState,
)
from neurophase.reset.gamma_witness import COHERENCE_THRESHOLD, DEFAULT_WINDOW
from neurophase.reset.neosynaptex_adapter import KLRNeuronsAdapter, NeosynaptexResetAdapter

# ``pytest.importorskip`` only catches ImportError. Some releases of
# ``neosynaptex`` raise a domain-specific error during module initialisation
# when their bundled evidence ledger is missing — broaden the guard so the
# whole suite becomes a clean skip rather than a hard collection failure.
try:  # pragma: no cover - environment-dependent
    import neosynaptex  # noqa: F401
except Exception as _exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"neosynaptex witness unavailable in this environment: {_exc}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
_VALID_PHASES: frozenset[str] = frozenset(
    {
        "WARMUP",
        "INITIALIZING",
        "METASTABLE",
        "CONVERGING",
        "DIVERGING",
        "COLLAPSING",
        "DRIFTING",
        "DEGENERATE",
    }
)

_VALID_VERDICTS: frozenset[str] = frozenset({"COHERENT", "INCOHERENT", "INSUFFICIENT_DATA"})


def _state(n: int = 6) -> SystemState:
    return SystemState(
        weights=np.full((n, n), 1.0 / n),
        confidence=np.full(n, 0.6),
        usage=np.linspace(0.1, 0.9, n),
        utility=np.linspace(0.2, 0.8, n),
        inhibition=np.full(n, 0.5),
        topology=np.ones((n, n)),
    )


def _metrics() -> SystemMetrics:
    return SystemMetrics(0.9, 0.9, 0.2, 0.2, 0.1, 0.1)


def _fingerprint(state: SystemState) -> dict[str, np.ndarray]:
    return {
        "weights": state.weights.copy(),
        "confidence": state.confidence.copy(),
        "usage": state.usage.copy(),
        "utility": state.utility.copy(),
        "inhibition": state.inhibition.copy(),
        "topology": state.topology.copy(),
        "frozen": (state.frozen.copy() if state.frozen is not None else np.zeros(0, dtype=bool)),
    }


# ---------------------------------------------------------------------------
# Constants / schema sanity
# ---------------------------------------------------------------------------
def test_report_schema_is_stable() -> None:
    """``GammaWitnessReport`` is a 4-field frozen dataclass — its surface
    is load-bearing (wire format for the ``KLRFrame`` witness channel)."""

    report = GammaWitnessReport(
        gamma_external=0.0,
        phase="WARMUP",
        coherence=0.0,
        verdict="INSUFFICIENT_DATA",
    )
    assert report.phase in _VALID_PHASES
    assert report.verdict in _VALID_VERDICTS
    # Frozen dataclass — attribute assignment must raise FrozenInstanceError.
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.phase = "METASTABLE"  # type: ignore[misc]
    assert 0.0 <= COHERENCE_THRESHOLD <= 1.0


# ---------------------------------------------------------------------------
# NEO-I1 — adapter + witness are strictly read-only
# ---------------------------------------------------------------------------
def test_witness_readonly() -> None:
    """NEO-I1: witness.observe(state) never mutates any SystemState array."""

    witness = GammaWitness()
    state = _state()
    before = _fingerprint(state)

    for _ in range(DEFAULT_WINDOW * 2):
        report = witness.observe(state)
        assert isinstance(report, GammaWitnessReport)

    after = _fingerprint(state)
    for key, pre in before.items():
        assert np.array_equal(pre, after[key]), f"witness mutated state.{key}"


def test_adapter_state_projection() -> None:
    """The 3-key projection is finite, domain-stable and topology-positive."""

    adapter = NeosynaptexResetAdapter()
    state = _state()
    assert adapter.domain == "klr_weights"
    assert adapter.state_keys == ["ntk_rank", "frozen_ratio", "usage_entropy"]
    assert not adapter.has_snapshot()

    adapter.update(state)
    assert adapter.has_snapshot()

    projected = adapter.state()
    assert set(projected) == {"ntk_rank", "frozen_ratio", "usage_entropy"}
    for value in projected.values():
        assert np.isfinite(value)
        assert 0.0 <= value <= 1.0
    assert adapter.topo() > 0.0
    assert 0.0 < adapter.thermo_cost() <= 1.0


# ---------------------------------------------------------------------------
# Warmup behaviour
# ---------------------------------------------------------------------------
def test_warmup_period() -> None:
    """Before ``window`` observations the witness emits a deterministic
    WARMUP placeholder with ``verdict = INSUFFICIENT_DATA``."""

    witness = GammaWitness(window=DEFAULT_WINDOW)
    state = _state()
    reports = [witness.observe(state) for _ in range(DEFAULT_WINDOW - 1)]

    for r in reports:
        assert r.phase == "WARMUP"
        assert r.verdict == "INSUFFICIENT_DATA"
        assert r.gamma_external == 0.0
        assert r.coherence == 0.0


def test_report_after_warmup() -> None:
    """Once ``window`` ticks have been seen the witness surfaces a
    neosynaptex-derived phase and a tri-state verdict."""

    witness = GammaWitness(window=DEFAULT_WINDOW)
    state = _state()
    report = witness.observe(state)
    for _ in range(DEFAULT_WINDOW + 3):
        report = witness.observe(state)

    assert report.phase != "WARMUP"
    assert report.phase in _VALID_PHASES
    assert report.verdict in _VALID_VERDICTS
    assert np.isfinite(report.gamma_external)
    assert np.isfinite(report.coherence)


# ---------------------------------------------------------------------------
# NEO-I2 — witness never alters KLR decisions
# ---------------------------------------------------------------------------
def test_incoherent_does_not_block() -> None:
    """NEO-I2: whatever the witness says, KLR decisions must not change.

    We drive two pipelines through ``2 × window`` ticks — one with the
    witness enabled, one with it disabled. Every decision string must
    match, and the row-stochastic invariant must survive both runs.
    """

    pipe_with = KLRPipeline(_state())
    pipe_without = KLRPipeline(_state(), enable_witness=False)
    assert pipe_without.gamma_witness is None

    decisions_with: list[str] = []
    decisions_without: list[str] = []
    for _ in range(DEFAULT_WINDOW * 2):
        f_with = pipe_with.tick(_metrics())
        f_without = pipe_without.tick(_metrics())
        decisions_with.append(f_with.decision)
        decisions_without.append(f_without.decision)

    assert decisions_with == decisions_without, "witness output must never alter KLR decisions"
    assert np.allclose(pipe_with.twin_state.active.weights.sum(axis=1), 1.0)
    assert np.allclose(pipe_without.twin_state.active.weights.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Wiring into KLRFrame
# ---------------------------------------------------------------------------
def test_witness_in_frame() -> None:
    """After warmup every KLRFrame carries a non-null witness report."""

    pipe = KLRPipeline(_state())
    frame: KLRFrame | None = None
    for _ in range(DEFAULT_WINDOW + 2):
        frame = pipe.tick(_metrics())

    assert frame is not None
    assert frame.witness_report is not None
    assert isinstance(frame.witness_report, GammaWitnessReport)
    assert frame.witness_report.phase in _VALID_PHASES
    assert frame.witness_report.verdict in _VALID_VERDICTS


def test_witness_disabled_returns_none_frame() -> None:
    """With the witness switched off the frame channel remains ``None``."""

    pipe = KLRPipeline(_state(), enable_witness=False)
    frame = pipe.tick(_metrics())
    assert frame.witness_report is None


# ---------------------------------------------------------------------------
# Exception tolerance
# ---------------------------------------------------------------------------
def test_neosynaptex_exception_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raise inside ``Neosynaptex.observe`` must not propagate: the
    witness degrades silently and keeps :class:`KLRPipeline` healthy."""

    witness = GammaWitness(window=8)
    state = _state()
    # Prime the inner instance.
    for _ in range(10):
        witness.observe(state)

    class _Boom:
        def observe(self) -> object:
            raise RuntimeError("neosynaptex internal failure")

    witness._nx = _Boom()  # type: ignore[assignment]
    report = witness.observe(state)
    assert report.phase == "WARMUP"
    assert report.verdict == "INSUFFICIENT_DATA"
    assert report.gamma_external == 0.0
    assert report.coherence == 0.0
    assert witness.is_disabled
    # Subsequent calls stay in the degraded state.
    again = witness.observe(state)
    assert again.verdict == "INSUFFICIENT_DATA"


def test_adapter_update_never_mutates_state() -> None:
    """Spot-check the sole write path: :meth:`NeosynaptexResetAdapter.update`
    reads from ``state`` but leaves every array intact (NEO-I1)."""

    adapter = NeosynaptexResetAdapter()
    state = _state()
    before = _fingerprint(state)
    adapter.update(state)
    after = _fingerprint(state)
    for key, pre in before.items():
        assert np.array_equal(pre, after[key]), f"adapter mutated state.{key}"


# ---------------------------------------------------------------------------
# Neurons-facet adapter (dual-domain witness)
# ---------------------------------------------------------------------------
def test_neurons_adapter_projection() -> None:
    """KLRNeuronsAdapter produces a valid 3-key projection with orthogonal
    topo/cost signals relative to the weights-facet adapter."""

    adapter = KLRNeuronsAdapter()
    state = _state()
    assert adapter.domain == "klr_neurons"
    assert adapter.state_keys == ["frozen_ratio", "usage_entropy", "confidence_mean"]
    assert not adapter.has_snapshot()

    adapter.update(state)
    assert adapter.has_snapshot()

    projected = adapter.state()
    assert set(projected) == {"frozen_ratio", "usage_entropy", "confidence_mean"}
    for value in projected.values():
        assert np.isfinite(value)
    assert adapter.topo() > 0.0
    assert 0.0 < adapter.thermo_cost() <= 1.0


def test_neurons_adapter_readonly() -> None:
    """KLRNeuronsAdapter.update() never mutates SystemState (NEO-I1)."""

    adapter = KLRNeuronsAdapter()
    state = _state()
    before = _fingerprint(state)
    adapter.update(state)
    after = _fingerprint(state)
    for key, pre in before.items():
        assert np.array_equal(pre, after[key]), f"neurons adapter mutated state.{key}"


def test_dual_domain_distinct_signals() -> None:
    """The two adapters produce distinct topo/cost signals from the same state,
    which is the prerequisite for non-trivial cross_coherence."""

    weights_adapter = NeosynaptexResetAdapter()
    neurons_adapter = KLRNeuronsAdapter()
    state = _state()
    weights_adapter.update(state)
    neurons_adapter.update(state)

    assert weights_adapter.domain != neurons_adapter.domain
    # topo and cost should differ: weights uses non_frozen_count and 1 - ntk_rank;
    # neurons uses usage_mass and 1 - usage_entropy.
    w_topo = weights_adapter.topo()
    n_topo = neurons_adapter.topo()
    w_cost = weights_adapter.thermo_cost()
    n_cost = neurons_adapter.thermo_cost()

    # With a 6-node uniform state these should be meaningfully different.
    assert w_topo > 0.0 and n_topo > 0.0
    assert w_cost > 0.0 and n_cost > 0.0
    # Signals must differ for cross_coherence to be non-trivial.
    assert w_topo != n_topo or w_cost != n_cost


def test_witness_tick_latency_bounded() -> None:
    """Hot-path contract: median per-tick overhead of the witness must stay
    under 10 ms — the threshold stated in the integration protocol."""

    import time

    state = _state()
    metrics = _metrics()
    pipe = KLRPipeline(state)

    # Warmup phase — let neosynaptex initialise.
    for _ in range(DEFAULT_WINDOW + 5):
        pipe.tick(metrics)

    durations: list[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        pipe.tick(metrics)
        durations.append(time.perf_counter() - t0)

    median_ms = float(np.median(durations)) * 1000.0
    assert median_ms < 10.0, f"median tick latency {median_ms:.2f} ms exceeds 10 ms contract"
