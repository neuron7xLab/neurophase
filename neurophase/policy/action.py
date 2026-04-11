"""I1 — typed, deterministic action policy above the execution gate.

The execution gate (B1 + I₁–I₄) decides **whether** the system is
*permitted* to act on the current frame. The regime classifier
(G1) decides **what kind** of regime the frame lives in. Neither
of those, on its own, answers the question a downstream consumer
actually asks: *"given this gate decision and this regime label,
what should I do?"*

This module is the bridge. :class:`ActionPolicy` consumes a
``(DecisionFrame, RegimeState)`` pair and emits a typed
:class:`ActionIntent` token plus a frozen
:class:`ActionDecision` envelope carrying full provenance for the
audit log. The policy is:

* **Pure** with respect to its inputs — no clocks, no randomness,
  no learned weights. Same inputs + same cooldown state → same
  intent, byte-identically.
* **Total** — every valid input lands in exactly one
  :class:`ActionIntent`.
* **Gate-honoring** — the gate's veto is absolute. If the gate
  rejects the frame (``execution_allowed=False``), the policy
  emits :attr:`ActionIntent.HOLD` regardless of regime, regardless
  of cooldown, regardless of confidence. The gate's contract is
  the load-bearing safety surface; the policy may only *narrow*
  that surface, never widen it.
* **Cooldown-aware** — to avoid flapping, the policy enforces a
  configurable minimum dwell time between intent transitions.
  Cooldown only applies *between* non-HOLD intents — a HOLD
  caused by gate veto resets the dwell counter so the next
  permissive frame is not artificially delayed.
* **Auditable** — every :class:`ActionDecision` is frozen and has
  a flat :meth:`ActionDecision.to_json_dict` projection suitable
  for direct ``json.dumps``.

What the policy does **not** do
-------------------------------

* It does not execute trades. The policy is a planner; the
  executor lives downstream and is out of scope.
* It does not size positions. Sizing is a separate concern (the
  ``risk`` package); the policy emits an *intent token* that the
  sizer maps to a notional.
* It does not override the gate. Ever. If you find yourself
  wanting the policy to admit a frame the gate has rejected, the
  bug is in the gate, not in the policy.
* It does not introduce randomness or hidden state beyond the
  cooldown counter. Two policies with the same configuration fed
  the same input sequence emit byte-identical
  :class:`ActionDecision` sequences.

Truth table (high level)
------------------------

==================  ===================  ====================  ============================
gate_state          regime_label          confidence            intent
==================  ===================  ====================  ============================
✗ (any)             —                     —                    ``HOLD``      (gate veto)
``READY``           ``CHAOTIC``           any                  ``HOLD``      (no regime)
``READY``           any                   < min_confidence      ``OBSERVE``   (low conviction)
``READY``           ``COMPRESSING``       ≥ min_confidence      ``OBSERVE``   (build-up)
``READY``           ``TRENDING``          ≥ min_confidence      ``ENGAGE`` / ``SUSTAIN``
``READY``           ``REVERTING``         ≥ min_confidence      ``UNWIND``
==================  ===================  ====================  ============================

The TRENDING branch emits ``ENGAGE`` on the *first* permissive
frame after a non-engaged tick, and ``SUSTAIN`` on every
subsequent permissive trending frame. This gives downstream
consumers a clean "open" / "hold" distinction without coupling
the policy to broker semantics.

Cooldown
--------

Whenever the emitted intent is one of the action-bearing tokens
(``ENGAGE``, ``UNWIND``, ``SUSTAIN``), the policy starts (or
refreshes) a cooldown counter of length :attr:`PolicyConfig.cooldown_ticks`.
While the counter is positive, the policy will downgrade
intents that would otherwise be ``ENGAGE`` to ``SUSTAIN`` (so the
*direction* of an active engagement is preserved) and will keep
emitting ``HOLD`` on a fresh ``UNWIND`` until the cooldown clears.
``HOLD`` (gate-induced) does not consume the cooldown; ``OBSERVE``
also does not consume it. The cooldown is purely a "no flapping
between action-bearing intents" mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from neurophase.analysis.regime import RegimeState
    from neurophase.runtime.pipeline import DecisionFrame

from neurophase.analysis.regime import RegimeLabel
from neurophase.gate.execution_gate import GateState

__all__ = [
    "DEFAULT_POLICY_CONFIG",
    "ActionDecision",
    "ActionIntent",
    "ActionPolicy",
    "PolicyConfig",
]


class ActionIntent(Enum):
    """The five typed intents the policy can emit.

    Each value is a stable string token suitable for downstream
    log processors. Adding a new value is a breaking change to the
    HN24 contract.
    """

    HOLD = "hold"
    """Do nothing — gate veto, chaotic regime, or cooldown lockout."""

    OBSERVE = "observe"
    """Gate is permissive but conviction is too low to act on; the
    caller should wait for confirmation."""

    ENGAGE = "engage"
    """High-conviction *new* action — first permissive trending
    frame after a non-engaged tick. The downstream executor opens a
    new position."""

    SUSTAIN = "sustain"
    """High-conviction *continuation* — subsequent permissive
    trending frames. The downstream executor maintains the existing
    position."""

    UNWIND = "unwind"
    """High-conviction reversal — the regime has flipped to
    REVERTING and the executor should unwind any open position."""


@dataclass(frozen=True)
class PolicyConfig:
    """Immutable configuration for an :class:`ActionPolicy`.

    Attributes
    ----------
    min_regime_confidence
        Lower bound on :attr:`RegimeState.confidence_score` for an
        action-bearing intent (``ENGAGE`` / ``SUSTAIN`` /
        ``UNWIND``) to be considered. Below this, the policy emits
        ``OBSERVE``. Must lie in ``[0, 1]``.
    cooldown_ticks
        Number of ticks during which a fresh action-bearing intent
        is suppressed after a transition. Zero disables the
        cooldown entirely. Must be a non-negative integer.
    require_warm_regime
        When ``True`` (the default), the policy refuses to emit an
        action-bearing intent on a regime state with
        :attr:`RegimeState.warm` ``= False`` (i.e. the very first
        classified frame). The first frame has zero ΔR / Δδ
        information by construction, so action-bearing intents on
        it would be conviction-free.
    """

    min_regime_confidence: float = 0.50
    cooldown_ticks: int = 0
    require_warm_regime: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_regime_confidence <= 1.0:
            raise ValueError(
                f"min_regime_confidence must be in [0, 1], got {self.min_regime_confidence}"
            )
        if self.cooldown_ticks < 0:
            raise ValueError(f"cooldown_ticks must be ≥ 0, got {self.cooldown_ticks}")


#: Default policy configuration. Frozen — do not mutate.
DEFAULT_POLICY_CONFIG: Final[PolicyConfig] = PolicyConfig()


@dataclass(frozen=True, repr=False)
class ActionDecision:
    """Frozen, JSON-serializable outcome of one
    :meth:`ActionPolicy.decide` call.

    Attributes
    ----------
    intent
        The emitted :class:`ActionIntent`.
    tick_index
        Copied from the input frame for log correlation.
    timestamp
        Copied from the input frame.
    gate_state
        The gate state observed at this tick (mirrors
        ``frame.gate.state``).
    execution_allowed
        Mirrors ``frame.gate.execution_allowed`` — surfaced here
        so a downstream consumer never has to crack open the
        nested gate decision.
    regime_label
        The classified regime label (mirrors
        ``regime.label``).
    confidence
        The classified regime confidence
        (``regime.confidence_score``).
    cooldown_remaining
        Number of ticks remaining on the cooldown counter
        *after* this decision was emitted. Zero means the policy
        is free to transition on the next tick.
    reason
        One-line human-readable explainer. Stable first token
        (``hold:`` / ``observe:`` / ``engage:`` / ``sustain:`` /
        ``unwind:``); the structured fields above are the
        machine-parseable surface.
    """

    intent: ActionIntent
    tick_index: int
    timestamp: float
    gate_state: GateState
    execution_allowed: bool
    regime_label: RegimeLabel
    confidence: float
    cooldown_remaining: int
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.cooldown_remaining < 0:
            raise ValueError(f"cooldown_remaining must be ≥ 0, got {self.cooldown_remaining}")
        # Gate-honoring contract: an action-bearing intent on a
        # gated frame is structurally impossible.
        if self.execution_allowed is False and self.intent is not ActionIntent.HOLD:
            raise ValueError(
                f"gate veto requires HOLD, got {self.intent.name} on execution_allowed=False"
            )

    def __repr__(self) -> str:  # aesthetic rich repr (HN24)
        flag = "✓" if self.execution_allowed else "✗"
        return (
            f"ActionDecision[{self.intent.name} · "
            f"tick={self.tick_index} · "
            f"{self.gate_state.name} · "
            f"{self.regime_label.name} · "
            f"conf={self.confidence:.2f} · {flag}]"
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe projection — no nested dataclass objects."""
        return {
            "intent": self.intent.value,
            "tick_index": self.tick_index,
            "timestamp": self.timestamp,
            "gate_state": self.gate_state.name,
            "execution_allowed": self.execution_allowed,
            "regime_label": self.regime_label.name,
            "confidence": self.confidence,
            "cooldown_remaining": self.cooldown_remaining,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# ActionPolicy — stateful, deterministic, gate-honoring.
# ---------------------------------------------------------------------------


class ActionPolicy:
    """Stateful, deterministic policy mapping ``(frame, regime) → intent``.

    Parameters
    ----------
    config
        Optional :class:`PolicyConfig`. Defaults to
        :data:`DEFAULT_POLICY_CONFIG`.

    Notes
    -----
    The policy carries minimal state: the cooldown counter and a
    one-tick memory of the previous emitted intent (used to
    distinguish ``ENGAGE`` from ``SUSTAIN``). Both are reset by
    :meth:`reset`.

    Two policies with the same :class:`PolicyConfig` fed the same
    ``(frame, regime)`` sequence emit bit-identical
    :class:`ActionDecision` sequences. There are no clocks, no RNG,
    no learned weights.
    """

    __slots__ = ("_cooldown", "_n_ticks", "_prev_intent", "config")

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config: PolicyConfig = config if config is not None else DEFAULT_POLICY_CONFIG
        self._cooldown: int = 0
        self._prev_intent: ActionIntent = ActionIntent.HOLD
        self._n_ticks: int = 0

    def reset(self) -> None:
        """Clear cooldown + previous-intent memory. Use at session boundaries."""
        self._cooldown = 0
        self._prev_intent = ActionIntent.HOLD
        self._n_ticks = 0

    @property
    def n_ticks(self) -> int:
        """Number of decisions emitted since construction / reset."""
        return self._n_ticks

    @property
    def cooldown_remaining(self) -> int:
        return self._cooldown

    @property
    def previous_intent(self) -> ActionIntent:
        return self._prev_intent

    def decide(self, frame: DecisionFrame, regime: RegimeState) -> ActionDecision:
        """Emit a typed :class:`ActionDecision` for one frame + regime pair.

        Parameters
        ----------
        frame
            The :class:`DecisionFrame` produced by a
            :class:`~neurophase.runtime.pipeline.StreamingPipeline`.
        regime
            The :class:`~neurophase.analysis.regime.RegimeState`
            classified for the same tick. The policy does not
            verify temporal alignment between the two — that is the
            caller's contract — but it will refuse to emit an
            action-bearing intent on a regime where
            :attr:`RegimeState.warm` is ``False`` when
            :attr:`PolicyConfig.require_warm_regime` is enabled.

        Returns
        -------
        ActionDecision
            Frozen, JSON-safe envelope.
        """
        intent, reason = self._classify(frame=frame, regime=regime)
        # Cooldown bookkeeping happens *after* classification but
        # *before* the decision is materialised, so the
        # cooldown_remaining field reflects the post-tick state.
        self._update_cooldown(intent=intent)
        decision = ActionDecision(
            intent=intent,
            tick_index=frame.tick_index,
            timestamp=frame.timestamp,
            gate_state=frame.gate.state,
            execution_allowed=frame.gate.execution_allowed,
            regime_label=regime.label,
            confidence=regime.confidence_score,
            cooldown_remaining=self._cooldown,
            reason=reason,
        )
        self._prev_intent = intent
        self._n_ticks += 1
        return decision

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _classify(self, *, frame: DecisionFrame, regime: RegimeState) -> tuple[ActionIntent, str]:
        """Run the priority tree and return ``(intent, reason)``.

        The order is **fixed** and corresponds to the truth table
        in the module docstring. Any future change to this order
        is a breaking HN24 modification.
        """
        # 1. Gate veto — absolute.
        if not frame.gate.execution_allowed:
            return (
                ActionIntent.HOLD,
                f"hold: gate veto ({frame.gate.state.name})",
            )

        # 2. Chaotic regime — no actionable structure.
        if regime.label is RegimeLabel.CHAOTIC:
            return (
                ActionIntent.HOLD,
                f"hold: chaotic regime (R={regime.R:.4f})",
            )

        # 3. Cold regime (warmup tick) — refuse action-bearing intents.
        if self.config.require_warm_regime and not regime.warm:
            return (
                ActionIntent.OBSERVE,
                "observe: regime is cold (no ΔR/Δδ available)",
            )

        # 4. Low conviction → OBSERVE.
        if regime.confidence_score < self.config.min_regime_confidence:
            return (
                ActionIntent.OBSERVE,
                (
                    f"observe: confidence={regime.confidence_score:.2f} < "
                    f"{self.config.min_regime_confidence:.2f}"
                ),
            )

        # 5. Compressing regime — wait for trend confirmation.
        if regime.label is RegimeLabel.COMPRESSING:
            return (
                ActionIntent.OBSERVE,
                f"observe: compressing build-up (conf={regime.confidence_score:.2f})",
            )

        # 6. Cooldown gate — only applies to fresh ENGAGE / UNWIND.
        # Inside cooldown, an active TRENDING frame downgrades to
        # SUSTAIN (preserves direction) and an active REVERTING
        # frame downgrades to HOLD (we already unwound).
        cooldown_active = self._cooldown > 0

        if regime.label is RegimeLabel.TRENDING:
            if cooldown_active or self._prev_intent in {
                ActionIntent.ENGAGE,
                ActionIntent.SUSTAIN,
            }:
                return (
                    ActionIntent.SUSTAIN,
                    (
                        f"sustain: trending (conf={regime.confidence_score:.2f}, "
                        f"cooldown={self._cooldown})"
                    ),
                )
            return (
                ActionIntent.ENGAGE,
                f"engage: trending (conf={regime.confidence_score:.2f})",
            )

        if regime.label is RegimeLabel.REVERTING:
            if cooldown_active and self._prev_intent is ActionIntent.UNWIND:
                return (
                    ActionIntent.HOLD,
                    f"hold: reverting cooldown ({self._cooldown})",
                )
            return (
                ActionIntent.UNWIND,
                f"unwind: reverting (conf={regime.confidence_score:.2f})",
            )

        # The classifier is total over four labels and we have
        # branched on all of them. Reaching this point indicates a
        # silent enum extension — fail loud rather than coerce.
        raise AssertionError(  # pragma: no cover
            f"unhandled RegimeLabel: {regime.label}"
        )

    def _update_cooldown(self, *, intent: ActionIntent) -> None:
        """Apply the cooldown rules for the just-emitted intent.

        * ``ENGAGE`` and ``UNWIND`` *start* a fresh cooldown.
        * ``SUSTAIN`` does **not** restart the cooldown — a long
          run of trending frames must not lock the policy out
          forever — but it also does not advance it, because
          ``SUSTAIN`` represents continuity of an existing
          engagement.
        * Every other intent (``HOLD``, ``OBSERVE``) decrements
          the counter toward zero.
        """
        if intent in {ActionIntent.ENGAGE, ActionIntent.UNWIND}:
            self._cooldown = self.config.cooldown_ticks
            return
        if intent is ActionIntent.SUSTAIN:
            return
        if self._cooldown > 0:
            self._cooldown -= 1
