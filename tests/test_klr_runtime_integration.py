"""RT-KLR-I1 — runtime integration contract tests.

Locks in the load-bearing contract: when a KLR pipeline is attached
to a StreamingPipeline, its output is **advisory-only**:

1. ``klr_decision``, ``klr_ntk_rank_delta``, and ``klr_warning``
   are attached to the emitted :class:`DecisionFrame`.
2. ``GateState`` and ``execution_allowed`` are never modified by
   the KLR advisory surface — the gate behaves EXACTLY as it would
   without KLR attached.
3. A KLR exception is caught, logged via ``klr_warning``, and
   never propagates out of :meth:`StreamingPipeline.tick`. The
   frame still carries a valid gate decision.
"""

from __future__ import annotations

import numpy as np

from neurophase.reset import KLRPipeline, SystemState
from neurophase.runtime.pipeline import PipelineConfig, StreamingPipeline


def _seed_state() -> SystemState:
    n = 6
    return SystemState(
        weights=np.full((n, n), 1.0 / n),
        confidence=np.full(n, 0.6),
        usage=np.linspace(0.1, 0.9, n),
        utility=np.linspace(0.2, 0.8, n),
        inhibition=np.full(n, 0.5),
        topology=np.ones((n, n)),
    )


def _config() -> PipelineConfig:
    return PipelineConfig(warmup_samples=2, stream_window=4, enable_stillness=False)


# ---------------------------------------------------------------------------
# 1. Advisory-only contract.
# ---------------------------------------------------------------------------


def test_klr_integration_advisory_only() -> None:
    """Gate state with KLR attached MUST equal gate state without KLR
    attached for the same input stream. KLR cannot widen or narrow
    the gate's permission surface."""
    cfg = _config()
    klr = KLRPipeline(_seed_state())
    p_with_klr = StreamingPipeline(cfg, klr_pipeline=klr)
    p_without = StreamingPipeline(cfg)

    seq = [(float(i) * 0.1, 0.9 - 0.05 * i, 0.05) for i in range(8)]
    with_states: list[str] = []
    no_states: list[str] = []
    with_allowed: list[bool] = []
    no_allowed: list[bool] = []
    for t, R, d in seq:
        fw = p_with_klr.tick(timestamp=t, R=R, delta=d)
        fn = p_without.tick(timestamp=t, R=R, delta=d)
        with_states.append(fw.gate.state.name)
        no_states.append(fn.gate.state.name)
        with_allowed.append(fw.execution_allowed)
        no_allowed.append(fn.execution_allowed)
        # KLR fields are attached on the with-KLR path.
        assert fw.klr_decision is not None
        # KLR fields are None on the without-KLR path.
        assert fn.klr_decision is None
        assert fn.klr_ntk_rank_delta is None
        assert fn.klr_warning is None

    # Load-bearing claim: KLR advisory is the zero-permission delta.
    assert with_states == no_states
    assert with_allowed == no_allowed


# ---------------------------------------------------------------------------
# 2. Backward compatibility — None klr_pipeline → unchanged behaviour.
# ---------------------------------------------------------------------------


def test_klr_none_backward_compat() -> None:
    """Omitting ``klr_pipeline`` keeps the pre-KLR behaviour exactly."""
    p = StreamingPipeline(_config())  # no klr_pipeline kwarg
    f = p.tick(timestamp=0.0, R=0.9, delta=0.05)
    assert f.klr_decision is None
    assert f.klr_ntk_rank_delta is None
    assert f.klr_warning is None
    # The to_json_dict projection still includes the three new keys
    # as None for downstream consumers that expect a stable schema.
    d = f.to_json_dict()
    assert d["klr_decision"] is None
    assert d["klr_ntk_rank_delta"] is None
    assert d["klr_warning"] is None


# ---------------------------------------------------------------------------
# 3. Exception containment — KLR failures never propagate.
# ---------------------------------------------------------------------------


class _ExplodingKLR:
    """Minimal fake KLR whose tick() always raises."""

    def tick(self, metrics: object) -> object:  # pragma: no cover - via pipeline
        raise RuntimeError("synthetic KLR failure")


def test_klr_exception_does_not_propagate() -> None:
    """A KLR tick() exception must NOT propagate. The frame must
    still carry a valid gate decision, and klr_warning must
    surface an ``"ERROR"`` marker."""
    fake: object = _ExplodingKLR()
    # mypy sees KLRPipeline type; cast via Any-ish attribute assignment
    # through the private slot to keep the test honest about the
    # structural contract.
    p = StreamingPipeline(_config())
    p._klr_pipeline = fake  # type: ignore[assignment]

    frame = p.tick(timestamp=0.0, R=0.9, delta=0.05)
    assert frame.klr_decision is None
    assert frame.klr_ntk_rank_delta is None
    assert frame.klr_warning is not None
    assert frame.klr_warning.startswith("ERROR:")
    # Gate is untouched.
    assert frame.gate.state.name in {"READY", "BLOCKED", "DEGRADED", "UNNECESSARY"}
