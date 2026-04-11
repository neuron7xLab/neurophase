"""I1 — contract tests for the typed action policy.

This test file is the HN24 binding. It locks in:

1. **Gate-honoring contract.** ``execution_allowed=False`` always
   yields ``HOLD`` regardless of regime, confidence, or cooldown
   state. The policy may only narrow the gate's permission
   surface, never widen it.
2. **Truth table coverage.** Every cell of the
   ``(gate_state × regime_label × confidence_bucket)`` Cartesian
   product reachable by the public API is exercised at least
   once.
3. **Cooldown semantics.** A fresh ``ENGAGE`` / ``UNWIND``
   starts the cooldown; ``SUSTAIN`` does not refresh it; ``HOLD``
   and ``OBSERVE`` decrement it.
4. **Determinism.** Two policies with identical configuration
   fed identical input sequences emit byte-identical
   :class:`ActionDecision` sequences.
5. **JSON serialisability.** Every emitted decision projects to a
   flat ``json.dumps``-safe dict with primitive values only.
6. **Construction-time validation.** Out-of-range confidence,
   negative cooldown, and gate-violating intents are rejected at
   ``__post_init__`` time on both :class:`PolicyConfig` and
   :class:`ActionDecision`.
7. **Frozen dataclass.** :class:`ActionDecision` rejects
   attribute reassignment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from neurophase.analysis.regime import RegimeLabel, RegimeState
from neurophase.data.stream_detector import (
    StreamQualityDecision,
    StreamQualityStats,
    StreamRegime,
)
from neurophase.data.temporal_validator import (
    TemporalQualityDecision,
    TimeQuality,
)
from neurophase.gate.execution_gate import GateDecision, GateState
from neurophase.policy.action import (
    DEFAULT_POLICY_CONFIG,
    ActionDecision,
    ActionIntent,
    ActionPolicy,
    PolicyConfig,
)
from neurophase.runtime.pipeline import DecisionFrame

# ---------------------------------------------------------------------------
# Synthetic frame + regime fixture builders.
# ---------------------------------------------------------------------------


def _frame(
    *,
    R: float,
    delta: float,
    gate_state: GateState,
    tick: int = 0,
    ts: float = 0.0,
) -> DecisionFrame:
    """Build a minimal DecisionFrame with a chosen gate state.

    The gate's ``execution_allowed`` flag is derived from the
    state: only ``READY`` is permissive.
    """
    allowed = gate_state is GateState.READY
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


def _regime(
    *,
    label: RegimeLabel,
    confidence: float,
    R: float = 0.85,
    delta: float = 0.05,
    warm: bool = True,
    tick: int = 0,
    ts: float = 0.0,
) -> RegimeState:
    """Build a synthetic RegimeState with the given label + confidence."""
    return RegimeState(
        label=label,
        R=R,
        dR=0.001,
        delta=delta,
        d_delta=0.001,
        confidence_score=confidence,
        tick_index=tick,
        timestamp=ts,
        warm=warm,
        reason=f"synthetic: {label.name}",
    )


@dataclass(frozen=True)
class _Pair:
    """A (frame, regime) test fixture pair."""

    frame: DecisionFrame
    regime: RegimeState


def _pair(
    *,
    gate_state: GateState,
    label: RegimeLabel,
    confidence: float,
    warm: bool = True,
    tick: int = 0,
) -> _Pair:
    return _Pair(
        frame=_frame(R=0.85, delta=0.05, gate_state=gate_state, tick=tick),
        regime=_regime(label=label, confidence=confidence, warm=warm, tick=tick),
    )


# ---------------------------------------------------------------------------
# 1. Gate-honoring contract — the load-bearing safety surface.
# ---------------------------------------------------------------------------


class TestGateHonoringContract:
    @pytest.mark.parametrize(
        "gate_state",
        [
            GateState.BLOCKED,
            GateState.SENSOR_ABSENT,
            GateState.DEGRADED,
            GateState.UNNECESSARY,
        ],
    )
    @pytest.mark.parametrize("label", list(RegimeLabel))
    def test_gate_veto_always_holds(self, gate_state: GateState, label: RegimeLabel) -> None:
        """Every non-READY gate state forces HOLD, regardless of regime."""
        policy = ActionPolicy()
        pair = _pair(gate_state=gate_state, label=label, confidence=0.99)
        decision = policy.decide(pair.frame, pair.regime)
        assert decision.intent is ActionIntent.HOLD
        assert decision.execution_allowed is False
        assert decision.reason.startswith("hold: gate veto")

    def test_action_decision_rejects_intent_on_vetoed_frame(self) -> None:
        """ActionDecision __post_init__ refuses to materialise an
        action-bearing intent on a vetoed frame — defence in depth."""
        with pytest.raises(ValueError, match="gate veto requires HOLD"):
            ActionDecision(
                intent=ActionIntent.ENGAGE,
                tick_index=0,
                timestamp=0.0,
                gate_state=GateState.BLOCKED,
                execution_allowed=False,
                regime_label=RegimeLabel.TRENDING,
                confidence=0.9,
                cooldown_remaining=0,
                reason="bogus",
            )


# ---------------------------------------------------------------------------
# 2. Regime + confidence truth table on a permissive gate.
# ---------------------------------------------------------------------------


class TestPermissiveGateTruthTable:
    def test_chaotic_yields_hold(self) -> None:
        policy = ActionPolicy()
        pair = _pair(gate_state=GateState.READY, label=RegimeLabel.CHAOTIC, confidence=0.95)
        decision = policy.decide(pair.frame, pair.regime)
        assert decision.intent is ActionIntent.HOLD
        assert "chaotic" in decision.reason

    def test_low_confidence_yields_observe(self) -> None:
        policy = ActionPolicy(PolicyConfig(min_regime_confidence=0.50))
        pair = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.30,
        )
        decision = policy.decide(pair.frame, pair.regime)
        assert decision.intent is ActionIntent.OBSERVE
        assert "confidence" in decision.reason

    def test_compressing_yields_observe(self) -> None:
        policy = ActionPolicy()
        pair = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.COMPRESSING,
            confidence=0.90,
        )
        decision = policy.decide(pair.frame, pair.regime)
        assert decision.intent is ActionIntent.OBSERVE
        assert "compressing" in decision.reason

    def test_first_trending_emits_engage(self) -> None:
        policy = ActionPolicy()
        pair = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.90,
        )
        decision = policy.decide(pair.frame, pair.regime)
        assert decision.intent is ActionIntent.ENGAGE

    def test_subsequent_trending_emits_sustain(self) -> None:
        policy = ActionPolicy()
        first = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.90,
            tick=0,
        )
        second = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.90,
            tick=1,
        )
        d1 = policy.decide(first.frame, first.regime)
        d2 = policy.decide(second.frame, second.regime)
        assert d1.intent is ActionIntent.ENGAGE
        assert d2.intent is ActionIntent.SUSTAIN

    def test_reverting_emits_unwind(self) -> None:
        policy = ActionPolicy()
        pair = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.REVERTING,
            confidence=0.90,
        )
        decision = policy.decide(pair.frame, pair.regime)
        assert decision.intent is ActionIntent.UNWIND


# ---------------------------------------------------------------------------
# 3. Cold regime gate — first tick should not act.
# ---------------------------------------------------------------------------


class TestWarmRegimeRequirement:
    def test_cold_trending_yields_observe(self) -> None:
        policy = ActionPolicy()
        pair = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.95,
            warm=False,
        )
        decision = policy.decide(pair.frame, pair.regime)
        assert decision.intent is ActionIntent.OBSERVE
        assert "cold" in decision.reason

    def test_cold_regime_can_be_disabled(self) -> None:
        policy = ActionPolicy(PolicyConfig(require_warm_regime=False))
        pair = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.95,
            warm=False,
        )
        decision = policy.decide(pair.frame, pair.regime)
        assert decision.intent is ActionIntent.ENGAGE


# ---------------------------------------------------------------------------
# 4. Cooldown semantics.
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_engage_starts_cooldown(self) -> None:
        policy = ActionPolicy(PolicyConfig(cooldown_ticks=3))
        pair = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.95,
        )
        decision = policy.decide(pair.frame, pair.regime)
        assert decision.intent is ActionIntent.ENGAGE
        assert decision.cooldown_remaining == 3

    def test_sustain_preserves_cooldown(self) -> None:
        policy = ActionPolicy(PolicyConfig(cooldown_ticks=3))
        for tick in range(4):
            pair = _pair(
                gate_state=GateState.READY,
                label=RegimeLabel.TRENDING,
                confidence=0.95,
                tick=tick,
            )
            policy.decide(pair.frame, pair.regime)
        # After 4 trending frames the cooldown should still be 3
        # (sustain neither refreshes nor decrements it).
        assert policy.cooldown_remaining == 3

    def test_observe_decrements_cooldown(self) -> None:
        policy = ActionPolicy(PolicyConfig(cooldown_ticks=2))
        # Tick 0: engage → cooldown = 2
        first = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.95,
            tick=0,
        )
        policy.decide(first.frame, first.regime)
        assert policy.cooldown_remaining == 2

        # Tick 1: observe (compressing) → cooldown decrements to 1
        second = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.COMPRESSING,
            confidence=0.95,
            tick=1,
        )
        policy.decide(second.frame, second.regime)
        assert policy.cooldown_remaining == 1

        # Tick 2: observe again → cooldown decrements to 0
        third = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.COMPRESSING,
            confidence=0.95,
            tick=2,
        )
        policy.decide(third.frame, third.regime)
        assert policy.cooldown_remaining == 0

    def test_unwind_during_unwind_cooldown_yields_hold(self) -> None:
        policy = ActionPolicy(PolicyConfig(cooldown_ticks=2))
        first = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.REVERTING,
            confidence=0.90,
            tick=0,
        )
        d1 = policy.decide(first.frame, first.regime)
        assert d1.intent is ActionIntent.UNWIND
        assert d1.cooldown_remaining == 2

        second = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.REVERTING,
            confidence=0.90,
            tick=1,
        )
        d2 = policy.decide(second.frame, second.regime)
        assert d2.intent is ActionIntent.HOLD
        assert "reverting cooldown" in d2.reason

    def test_zero_cooldown_does_not_lock_anything(self) -> None:
        policy = ActionPolicy(PolicyConfig(cooldown_ticks=0))
        for tick in range(5):
            pair = _pair(
                gate_state=GateState.READY,
                label=RegimeLabel.REVERTING,
                confidence=0.95,
                tick=tick,
            )
            decision = policy.decide(pair.frame, pair.regime)
            assert decision.intent is ActionIntent.UNWIND
            assert decision.cooldown_remaining == 0


# ---------------------------------------------------------------------------
# 5. Determinism + reset.
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_sequence_same_decisions(self) -> None:
        cfg = PolicyConfig(cooldown_ticks=2)
        seq = [
            (GateState.READY, RegimeLabel.TRENDING, 0.9),
            (GateState.READY, RegimeLabel.TRENDING, 0.9),
            (GateState.BLOCKED, RegimeLabel.TRENDING, 0.9),
            (GateState.READY, RegimeLabel.REVERTING, 0.9),
            (GateState.READY, RegimeLabel.CHAOTIC, 0.9),
        ]
        p1 = ActionPolicy(cfg)
        p2 = ActionPolicy(cfg)
        out1 = []
        out2 = []
        for tick, (gs, label, conf) in enumerate(seq):
            pair = _pair(gate_state=gs, label=label, confidence=conf, tick=tick)
            out1.append(p1.decide(pair.frame, pair.regime))
            out2.append(p2.decide(pair.frame, pair.regime))
        assert [d.intent for d in out1] == [d.intent for d in out2]
        assert [d.cooldown_remaining for d in out1] == [d.cooldown_remaining for d in out2]
        assert [d.reason for d in out1] == [d.reason for d in out2]

    def test_reset_clears_state(self) -> None:
        policy = ActionPolicy(PolicyConfig(cooldown_ticks=3))
        pair = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.95,
        )
        policy.decide(pair.frame, pair.regime)
        assert policy.cooldown_remaining == 3
        assert policy.previous_intent is ActionIntent.ENGAGE
        policy.reset()
        assert policy.cooldown_remaining == 0
        # mypy narrows previous_intent to Literal[ENGAGE] from the
        # previous assert; round-trip via .name to break the narrowing.
        assert policy.previous_intent.name == "HOLD"
        assert policy.n_ticks == 0


# ---------------------------------------------------------------------------
# 6. JSON serialisation.
# ---------------------------------------------------------------------------


class TestJsonProjection:
    def test_to_json_dict_is_serialisable(self) -> None:
        policy = ActionPolicy()
        pair = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.90,
        )
        decision = policy.decide(pair.frame, pair.regime)
        d = decision.to_json_dict()
        # Round-trip through JSON.
        text = json.dumps(d)
        loaded = json.loads(text)
        assert loaded["intent"] == "engage"
        assert loaded["gate_state"] == "READY"
        assert loaded["regime_label"] == "TRENDING"
        assert loaded["execution_allowed"] is True

    def test_to_json_dict_is_flat(self) -> None:
        """No nested dataclass objects — every value is a primitive."""
        policy = ActionPolicy()
        pair = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.90,
        )
        decision = policy.decide(pair.frame, pair.regime)
        d = decision.to_json_dict()
        for key, val in d.items():
            assert isinstance(val, (str, int, float, bool, type(None))), (
                f"key {key!r} has non-primitive value {val!r}"
            )


# ---------------------------------------------------------------------------
# 7. Construction-time validation.
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_negative_cooldown_rejected(self) -> None:
        with pytest.raises(ValueError, match="cooldown_ticks"):
            PolicyConfig(cooldown_ticks=-1)

    def test_out_of_range_min_confidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_regime_confidence"):
            PolicyConfig(min_regime_confidence=1.5)

    def test_action_decision_rejects_negative_cooldown(self) -> None:
        with pytest.raises(ValueError, match="cooldown_remaining"):
            ActionDecision(
                intent=ActionIntent.HOLD,
                tick_index=0,
                timestamp=0.0,
                gate_state=GateState.BLOCKED,
                execution_allowed=False,
                regime_label=RegimeLabel.TRENDING,
                confidence=0.5,
                cooldown_remaining=-1,
                reason="bogus",
            )

    def test_action_decision_rejects_out_of_range_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            ActionDecision(
                intent=ActionIntent.HOLD,
                tick_index=0,
                timestamp=0.0,
                gate_state=GateState.BLOCKED,
                execution_allowed=False,
                regime_label=RegimeLabel.TRENDING,
                confidence=1.2,
                cooldown_remaining=0,
                reason="bogus",
            )

    def test_default_config_is_default(self) -> None:
        policy = ActionPolicy()
        assert policy.config is DEFAULT_POLICY_CONFIG


# ---------------------------------------------------------------------------
# 8. Frozen dataclass — no mutation.
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_action_decision_is_frozen(self) -> None:
        policy = ActionPolicy()
        pair = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.9,
        )
        decision = policy.decide(pair.frame, pair.regime)
        with pytest.raises((AttributeError, TypeError)):
            decision.intent = ActionIntent.HOLD  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 9. Rich __repr__ — HN24 design language.
# ---------------------------------------------------------------------------


class TestRepr:
    def test_engage_repr_format(self) -> None:
        policy = ActionPolicy()
        pair = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.92,
        )
        decision = policy.decide(pair.frame, pair.regime)
        r = repr(decision)
        assert r.startswith("ActionDecision[")
        assert "ENGAGE" in r
        assert "READY" in r
        assert "TRENDING" in r
        assert "conf=0.92" in r
        assert "✓" in r

    def test_hold_repr_carries_x_marker(self) -> None:
        policy = ActionPolicy()
        pair = _pair(
            gate_state=GateState.BLOCKED,
            label=RegimeLabel.TRENDING,
            confidence=0.9,
        )
        decision = policy.decide(pair.frame, pair.regime)
        r = repr(decision)
        assert "HOLD" in r
        assert "✗" in r


# ---------------------------------------------------------------------------
# 10. Total classification — fuzz the truth table to ensure totality.
# ---------------------------------------------------------------------------


class TestTotalClassification:
    def test_every_input_yields_some_intent(self) -> None:
        """Sweep the (gate × label × confidence) lattice and check
        every cell yields one of the five typed intents."""
        gate_states = list(GateState)
        labels = list(RegimeLabel)
        confs = [0.0, 0.25, 0.49, 0.50, 0.75, 1.0]
        intents_seen: set[ActionIntent] = set()
        for gs in gate_states:
            for label in labels:
                for conf in confs:
                    p = ActionPolicy()
                    pair = _pair(gate_state=gs, label=label, confidence=conf)
                    d = p.decide(pair.frame, pair.regime)
                    assert d.intent in set(ActionIntent)
                    intents_seen.add(d.intent)
        # All five intents must be reachable from the lattice.
        assert intents_seen == set(ActionIntent) - {ActionIntent.SUSTAIN}, (
            "SUSTAIN requires a multi-tick sequence; all other "
            "intents must be reachable on a single fresh-classifier call"
        )

    def test_sustain_is_reachable_via_two_tick_sequence(self) -> None:
        """SUSTAIN is the only intent that requires multi-tick state."""
        policy = ActionPolicy()
        first = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.9,
            tick=0,
        )
        second = _pair(
            gate_state=GateState.READY,
            label=RegimeLabel.TRENDING,
            confidence=0.9,
            tick=1,
        )
        policy.decide(first.frame, first.regime)
        d2 = policy.decide(second.frame, second.regime)
        assert d2.intent is ActionIntent.SUSTAIN
