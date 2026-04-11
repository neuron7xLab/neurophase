"""G1 — contract tests for the deterministic regime taxonomy.

This test file is the HN23 binding for the four-state regime
classifier. It locks in:

1. **Exhaustive state coverage.** Every one of the four
   :class:`RegimeLabel` values is reachable on a synthetic
   :class:`DecisionFrame`.
2. **All 12 non-self transitions.** There are ``4 × 4 - 4 = 12``
   directed non-self transitions between regime labels, and every
   one is exercised by a concrete frame sequence.
3. **Determinism.** Two classifiers with identical thresholds fed
   the same frame sequence emit bit-identical :class:`RegimeState`
   sequences.
4. **Input validation.** ``frame.R is None`` and ``frame.delta is
   None`` raise ``ValueError`` — regime classification on missing
   inputs is meaningless and the classifier refuses to coerce.
5. **Confidence bounds.** Every emitted ``confidence_score`` lies
   in ``[0, 1]``, regardless of the input trajectory.
6. **Warmup semantics.** The first classified frame carries
   ``warm=False``; every subsequent frame carries ``warm=True``.
7. **State reset.** :meth:`RegimeClassifier.reset` clears the
   lag buffer so the next frame is a fresh warmup tick.
8. **Threshold validation.** :class:`RegimeThresholds`
   ``__post_init__`` rejects invalid parameter sets.

These bindings are registered as HN23 in ``INVARIANTS.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from neurophase.analysis.regime import (
    DEFAULT_REGIME_THRESHOLDS,
    RegimeClassifier,
    RegimeLabel,
    RegimeState,
    RegimeThresholds,
)
from neurophase.data.stream_detector import (
    StreamQualityDecision,
    StreamQualityStats,
    StreamRegime,
)
from neurophase.data.temporal_validator import (
    TemporalQualityDecision,
    TimeQuality,
)
from neurophase.gate.execution_gate import (
    GateDecision,
    GateState,
)
from neurophase.runtime.pipeline import DecisionFrame

# ---------------------------------------------------------------------------
# Synthetic DecisionFrame builder — keeps tests free of pipeline state.
# ---------------------------------------------------------------------------


def _frame(
    *,
    R: float | None,
    delta: float | None,
    tick: int = 0,
    ts: float = 0.0,
) -> DecisionFrame:
    """Build a minimal, valid DecisionFrame carrying the given ``R`` and ``delta``.

    The sub-decision fields are filled with benign values — the
    regime classifier only reads ``R``, ``delta``, ``tick_index``
    and ``timestamp``, so the rest is ballast.
    """
    temporal = TemporalQualityDecision(
        quality=TimeQuality.VALID,
        ts=ts,
        last_ts=None,
        gap_seconds=None,
        staleness_seconds=None,
        warmup_remaining=0,
        reason="valid: synthetic fixture",
    )
    stats = StreamQualityStats(
        total=1,
        valid=1,
        gapped=0,
        stale=0,
        reversed=0,
        duplicate=0,
        invalid=0,
        warmup=0,
        fault_rate=0.0,
    )
    stream = StreamQualityDecision(
        regime=StreamRegime.HEALTHY,
        stats=stats,
        last_quality=TimeQuality.VALID,
        held=False,
        reason="healthy: synthetic fixture",
    )
    # Pick a gate state that does not violate the __post_init__
    # invariant. If R is None or below threshold, route to DEGRADED/
    # BLOCKED; if R ≥ 0.5, route to READY.
    gate_state: GateState
    allowed: bool
    if R is None:
        gate_state, allowed = GateState.DEGRADED, False
    elif R < 0.5:
        gate_state, allowed = GateState.BLOCKED, False
    else:
        gate_state, allowed = GateState.READY, True
    gate = GateDecision(
        state=gate_state,
        execution_allowed=allowed,
        R=R,
        threshold=0.5,
        reason="synthetic",
    )
    return DecisionFrame(
        tick_index=tick,
        timestamp=ts,
        R=R,
        delta=delta,
        temporal=temporal,
        stream=stream,
        gate=gate,
        ledger_record=None,
    )


@dataclass(frozen=True)
class _TransitionCase:
    """A minimal two-frame recipe driving a specific regime transition."""

    name: str
    target: RegimeLabel
    R: float
    delta: float
    prev_R: float
    prev_delta: float


def _drive_from_prev(
    classifier: RegimeClassifier,
    *,
    prev_R: float,
    prev_delta: float,
    next_R: float,
    next_delta: float,
) -> RegimeState:
    """Seed the classifier with an anchor frame, then emit the target frame."""
    classifier.classify(_frame(R=prev_R, delta=prev_delta, tick=0, ts=0.0))
    return classifier.classify(_frame(R=next_R, delta=next_delta, tick=1, ts=0.1))


# ---------------------------------------------------------------------------
# 1. Basic construction and field invariants.
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_thresholds_are_usable(self) -> None:
        c = RegimeClassifier()
        assert c.thresholds is DEFAULT_REGIME_THRESHOLDS
        assert c.n_ticks == 0

    def test_custom_thresholds_are_stored(self) -> None:
        cfg = RegimeThresholds(low_R_threshold=0.30, high_R_threshold=0.80)
        c = RegimeClassifier(thresholds=cfg)
        assert c.thresholds is cfg

    def test_thresholds_reject_inverted_R_bounds(self) -> None:
        with pytest.raises(ValueError, match="low_R_threshold"):
            RegimeThresholds(low_R_threshold=0.80, high_R_threshold=0.40)

    def test_thresholds_reject_non_positive_motion(self) -> None:
        with pytest.raises(ValueError, match="rising_R_min"):
            RegimeThresholds(rising_R_min=0.0)

    def test_regime_state_is_frozen(self) -> None:
        c = RegimeClassifier()
        state = c.classify(_frame(R=0.9, delta=0.05))
        with pytest.raises((AttributeError, TypeError)):
            state.label = RegimeLabel.CHAOTIC  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Input validation.
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_R_none_raises(self) -> None:
        c = RegimeClassifier()
        with pytest.raises(ValueError, match=r"frame\.R"):
            c.classify(_frame(R=None, delta=0.05))

    def test_delta_none_raises(self) -> None:
        c = RegimeClassifier()
        with pytest.raises(ValueError, match=r"frame\.delta"):
            c.classify(_frame(R=0.8, delta=None))


# ---------------------------------------------------------------------------
# 3. First-tick reachability — all 4 states reachable on the first call.
# ---------------------------------------------------------------------------


class TestFirstTickReachability:
    def test_first_tick_trending_reachable(self) -> None:
        c = RegimeClassifier()
        state = c.classify(_frame(R=0.90, delta=0.05))
        assert state.label is RegimeLabel.TRENDING
        assert state.warm is False

    def test_first_tick_chaotic_reachable_via_low_R(self) -> None:
        c = RegimeClassifier()
        state = c.classify(_frame(R=0.20, delta=0.05))
        assert state.label is RegimeLabel.CHAOTIC

    def test_first_tick_chaotic_on_high_R_is_impossible(self) -> None:
        # With zero deltas on the first tick, high R always lands in
        # TRENDING — there is no way to produce CHAOTIC with high R
        # on a fresh classifier. This is a deliberate contract: a
        # single frame cannot carry Δδ information.
        c = RegimeClassifier()
        state = c.classify(_frame(R=0.95, delta=0.01))
        assert state.label is RegimeLabel.TRENDING


# ---------------------------------------------------------------------------
# 4. Second-tick reachability — COMPRESSING and REVERTING need motion.
# ---------------------------------------------------------------------------


class TestSecondTickReachability:
    def test_compressing_reachable(self) -> None:
        c = RegimeClassifier()
        state = _drive_from_prev(c, prev_R=0.70, prev_delta=0.30, next_R=0.82, next_delta=0.20)
        assert state.label is RegimeLabel.COMPRESSING
        assert state.warm is True
        assert state.dR > 0.01
        assert state.d_delta < -0.005

    def test_reverting_reachable(self) -> None:
        c = RegimeClassifier()
        state = _drive_from_prev(c, prev_R=0.90, prev_delta=0.05, next_R=0.75, next_delta=0.06)
        assert state.label is RegimeLabel.REVERTING
        assert state.dR < -0.01


# ---------------------------------------------------------------------------
# 5. All 12 non-self transitions reachable.
# ---------------------------------------------------------------------------

# Anchor frames for each label. Each anchor is a (R, δ) pair that,
# when fed as the *first* frame, lands in that label's bucket on
# the first classify() call. Then a second frame drives the
# specific target label.

_ANCHORS: dict[RegimeLabel, tuple[float, float]] = {
    RegimeLabel.TRENDING: (0.85, 0.05),  # high R, stable → TRENDING on tick 0
    RegimeLabel.CHAOTIC: (0.20, 0.05),  # low R → CHAOTIC on tick 0
    RegimeLabel.COMPRESSING: (0.85, 0.05),  # needs a prior tick → use trending anchor
    RegimeLabel.REVERTING: (0.85, 0.05),  # needs a prior tick → use trending anchor
}


def _target_frame(label: RegimeLabel, prev_R: float, prev_delta: float) -> tuple[float, float]:
    """Return ``(R, δ)`` that lands in ``label`` when preceded by ``(prev_R, prev_delta)``.

    Designed so every one of the 12 directed transitions between
    the four labels can be driven in two calls to
    :meth:`RegimeClassifier.classify` (one to seed the previous
    state, one to trigger the transition).
    """
    if label is RegimeLabel.TRENDING:
        # Small motion, high R. Keep dR and d_delta below the
        # rising/falling thresholds, and keep R above low_R.
        return (max(0.75, prev_R + 0.001), prev_delta + 0.001)
    if label is RegimeLabel.COMPRESSING:
        # Strong rising R AND strong narrowing δ.
        return (prev_R + 0.08, prev_delta - 0.08)
    if label is RegimeLabel.REVERTING:
        # Strong falling R; δ drift irrelevant as long as narrowing
        # is not paired with rising R.
        return (prev_R - 0.12, prev_delta + 0.01)
    if label is RegimeLabel.CHAOTIC:
        # The R-drop route to CHAOTIC is blocked whenever prev_R is
        # high: a huge negative ΔR fires REVERTING (rule 2) before
        # CHAOTIC (rule 3). Use the unstable-δ route instead: keep
        # R roughly constant, push |Δδ| above the
        # chaotic_delta_motion threshold (0.30 default). This
        # transitions to CHAOTIC from any high-R source in one step.
        return (prev_R + 0.001, prev_delta + 0.35)
    raise AssertionError("unreachable")


@pytest.mark.parametrize("source", list(RegimeLabel))
@pytest.mark.parametrize("target", list(RegimeLabel))
def test_all_transitions_reachable(source: RegimeLabel, target: RegimeLabel) -> None:
    """Every non-self transition (source → target) is reachable.

    Self-loops are skipped because a "transition" of a label onto
    itself is not a transition — the interesting contract is that
    every *change* between the four labels can be produced.
    """
    if source is target:
        pytest.skip("self-loops are not transitions")

    c = RegimeClassifier()
    anchor_R, anchor_delta = _ANCHORS[source]
    # For COMPRESSING and REVERTING anchors we first need a prior
    # frame so the classifier has a non-zero lag state. For
    # TRENDING and CHAOTIC, a single-frame anchor suffices — they
    # can be reached with zero deltas.
    if source is RegimeLabel.TRENDING or source is RegimeLabel.CHAOTIC:
        first_state = c.classify(_frame(R=anchor_R, delta=anchor_delta, tick=0, ts=0.0))
        assert first_state.label is source, (
            f"anchor for {source.name} did not land in {source.name}: {first_state}"
        )
        prev_R, prev_delta = anchor_R, anchor_delta
        next_tick = 1
    elif source is RegimeLabel.COMPRESSING:
        c.classify(_frame(R=0.60, delta=0.20, tick=0, ts=0.0))
        first_state = c.classify(_frame(R=0.72, delta=0.10, tick=1, ts=0.1))
        assert first_state.label is RegimeLabel.COMPRESSING
        prev_R, prev_delta = 0.72, 0.10
        next_tick = 2
    else:  # REVERTING
        c.classify(_frame(R=0.90, delta=0.05, tick=0, ts=0.0))
        first_state = c.classify(_frame(R=0.75, delta=0.06, tick=1, ts=0.1))
        assert first_state.label is RegimeLabel.REVERTING
        prev_R, prev_delta = 0.75, 0.06
        next_tick = 2

    next_R, next_delta = _target_frame(target, prev_R, prev_delta)
    out = c.classify(_frame(R=next_R, delta=next_delta, tick=next_tick, ts=0.1 * next_tick))
    assert out.label is target, f"transition {source.name} → {target.name} failed: got {out}"


# ---------------------------------------------------------------------------
# 6. Determinism.
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_sequence_same_labels(self) -> None:
        seq = [(0.80, 0.10), (0.85, 0.08), (0.72, 0.09), (0.25, 0.30), (0.90, 0.02)]
        c1 = RegimeClassifier()
        c2 = RegimeClassifier()
        out1 = [c1.classify(_frame(R=r, delta=d, tick=i)) for i, (r, d) in enumerate(seq)]
        out2 = [c2.classify(_frame(R=r, delta=d, tick=i)) for i, (r, d) in enumerate(seq)]
        assert [s.label for s in out1] == [s.label for s in out2]
        assert [s.confidence_score for s in out1] == [s.confidence_score for s in out2]

    def test_reset_produces_fresh_warmup(self) -> None:
        c = RegimeClassifier()
        c.classify(_frame(R=0.80, delta=0.10, tick=0))
        c.classify(_frame(R=0.70, delta=0.12, tick=1))
        assert c.n_ticks == 2
        c.reset()
        assert c.n_ticks == 0
        state = c.classify(_frame(R=0.60, delta=0.08, tick=0))
        assert state.warm is False
        assert state.dR == 0.0
        assert state.d_delta == 0.0


# ---------------------------------------------------------------------------
# 7. Confidence bounds and monotonicity.
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_confidence_is_in_unit_interval(self) -> None:
        c = RegimeClassifier()
        seq = [
            (0.90, 0.01),
            (0.92, 0.03),
            (0.70, 0.30),
            (0.10, 0.05),
            (0.50, 0.05),
            (0.99, 0.01),
        ]
        for i, (r, d) in enumerate(seq):
            state = c.classify(_frame(R=r, delta=d, tick=i, ts=0.1 * i))
            assert 0.0 <= state.confidence_score <= 1.0

    def test_deeper_chaos_has_higher_confidence(self) -> None:
        c1 = RegimeClassifier()
        c2 = RegimeClassifier()
        shallow = c1.classify(_frame(R=0.38, delta=0.05))
        deep = c2.classify(_frame(R=0.05, delta=0.05))
        assert shallow.label is RegimeLabel.CHAOTIC
        assert deep.label is RegimeLabel.CHAOTIC
        assert deep.confidence_score > shallow.confidence_score


# ---------------------------------------------------------------------------
# 8. Rich __repr__ — HN22 / HN23 canonical format.
# ---------------------------------------------------------------------------


class TestRepr:
    def test_regime_state_repr_format(self) -> None:
        c = RegimeClassifier()
        state = c.classify(_frame(R=0.85, delta=0.05))
        r = repr(state)
        assert r.startswith("RegimeState[")
        assert "TRENDING" in r
        assert "R=0.8500" in r
        assert "conf=" in r
        assert "cold" in r  # first tick → warm=False → cold

    def test_regime_state_repr_warm_after_second_tick(self) -> None:
        c = RegimeClassifier()
        c.classify(_frame(R=0.85, delta=0.05, tick=0))
        state = c.classify(_frame(R=0.86, delta=0.05, tick=1))
        r = repr(state)
        assert "warm" in r
        assert "cold" not in r


# ---------------------------------------------------------------------------
# 9. Immutable RegimeState validates confidence on construction.
# ---------------------------------------------------------------------------


class TestRegimeStateValidation:
    def test_out_of_range_confidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence_score"):
            RegimeState(
                label=RegimeLabel.TRENDING,
                R=0.9,
                dR=0.0,
                delta=0.05,
                d_delta=0.0,
                confidence_score=1.5,
                tick_index=0,
                timestamp=0.0,
                warm=False,
                reason="test",
            )


# ---------------------------------------------------------------------------
# 10. Integration — regime classifier runs on real StreamingPipeline frames.
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_regime_classifier_consumes_pipeline_frames(self) -> None:
        """Smoke test — a StreamingPipeline's emitted frames flow
        straight into the regime classifier without glue code."""
        from neurophase.runtime.pipeline import PipelineConfig, StreamingPipeline

        pipeline = StreamingPipeline(
            PipelineConfig(warmup_samples=2, stream_window=4, enable_stillness=False)
        )
        regime = RegimeClassifier()
        labels: list[RegimeLabel] = []
        for i in range(8):
            frame = pipeline.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.05)
            # Skip warmup frames where R might be passed through
            # unchanged but the pipeline hasn't yet emitted READY.
            if frame.R is None:
                continue
            labels.append(regime.classify(frame).label)
        assert len(labels) >= 4
        # A constant-R constant-δ feed should settle into TRENDING.
        assert labels[-1] is RegimeLabel.TRENDING


# ---------------------------------------------------------------------------
# 11. Defensive: classifier cannot be coerced into producing illegal labels.
# ---------------------------------------------------------------------------


class TestTotalClassification:
    def test_every_output_is_a_regime_label(self) -> None:
        """Fuzz 200 random (R, δ) pairs and check every emitted
        label is one of the four enum values.

        Uses a fixed seed so the test is deterministic despite the
        word "fuzz" — we are not random-sampling, we are running
        over a fixed lattice."""
        c = RegimeClassifier()
        grid_R = [0.05 + 0.05 * i for i in range(19)]
        grid_d = [0.02 + 0.04 * j for j in range(10)]
        count = 0
        for r in grid_R:
            for d in grid_d:
                state = c.classify(_frame(R=r, delta=d, tick=count, ts=0.1 * count))
                assert state.label in set(RegimeLabel)
                assert 0.0 <= state.confidence_score <= 1.0
                count += 1
        assert count == len(grid_R) * len(grid_d)


# ---------------------------------------------------------------------------
# 12. Type sanity — mypy hooks.
# ---------------------------------------------------------------------------


def _type_shape_hook() -> Any:
    """Compile-time shape check (not a runtime test).

    mypy --strict will reject this function if the public
    signatures drift."""
    cfg: RegimeThresholds = DEFAULT_REGIME_THRESHOLDS
    c: RegimeClassifier = RegimeClassifier(thresholds=cfg)
    frame: DecisionFrame = _frame(R=0.8, delta=0.05)
    state: RegimeState = c.classify(frame)
    label: RegimeLabel = state.label
    return (cfg, c, frame, state, label)
