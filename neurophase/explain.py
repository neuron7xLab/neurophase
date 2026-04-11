"""Structured interpretability layer for gate decisions.

Every :class:`~neurophase.runtime.pipeline.DecisionFrame` carries a
``reason: str`` on each of its nested sub-decisions (temporal, stream,
gate). That is **readable** but it is not **interpretable** — a
downstream tool that wants to answer *"why did this frame land in
UNNECESSARY?"* has to string-parse the reason field, and string
parsing is exactly the kind of fragile coupling that doctrine item
*"honest naming"* forbids.

This module replaces that pattern with :class:`DecisionExplanation`
— a frozen, deterministic, JSON-serializable causal chain that is
generated from a :class:`DecisionFrame` by walking the same strict
priority order the gate itself uses:

    B₁ temporal → I₂ sensor → I₃ R validity → I₁ threshold → I₄ stillness

Each layer emits exactly one :class:`ExplanationStep` with a typed
verdict (``pass`` / ``fail`` / ``skipped``) and a typed contract id
(``B1`` / ``I1`` / ``I2`` / ``I3`` / ``I4`` / ``READY``). The first
``fail`` step is the **causal root** of the final state; every
subsequent layer is marked ``skipped`` because the gate short-
circuited there. A ``READY`` decision produces a chain of all
``pass`` steps followed by a final ``READY`` step.

Contracts enforced by this module
---------------------------------

* **Determinism.** ``explain_decision(frame) == explain_decision(frame)``
  byte-identically for any frame. No randomness, no clocks, no
  wall-time side effects.
* **Loss-free.** The sum of information in the
  :class:`DecisionExplanation` is ≥ the information in the original
  ``reason`` strings — the explanation carries every typed field the
  gate used, not just the narrative line.
* **Reproducible projection.** ``explanation.as_dict()`` is a flat
  JSON-safe projection suitable for log pipelines. No nested
  dataclasses.
* **Stable first-token tags.** Each step's ``verdict`` field is one
  of exactly three strings (``"pass"`` / ``"fail"`` / ``"skipped"``)
  and each ``contract`` field is one of exactly six strings
  (``"B1"`` / ``"I1"`` / ``"I2"`` / ``"I3"`` / ``"I4"`` / ``"READY"``).
  These are frozen enums — adding a new value is a breaking change.

What this module does NOT do
----------------------------

* It does not mutate any state. Explanations are pure projections.
* It does not reach into the stillness / temporal detectors. It
  reconstructs the chain from the *materialised* frame only — this
  is what makes it safe to call on frames loaded from a replayed
  ledger (F2) where the original detectors no longer exist.
* It does not attempt to predict what would have happened under
  different inputs. That is the A3 matrix's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from neurophase.data.stream_detector import StreamRegime
from neurophase.data.temporal_validator import TimeQuality
from neurophase.gate.execution_gate import GateDecision, GateState
from neurophase.runtime.pipeline import DecisionFrame

#: Stable, closed set of verdict values. First token of every
#: :class:`ExplanationStep` reason line.
_VERDICT_PASS: Final[str] = "pass"
_VERDICT_FAIL: Final[str] = "fail"
_VERDICT_SKIPPED: Final[str] = "skipped"


class Verdict(Enum):
    """Typed verdict for one step in a decision's causal chain."""

    PASS = _VERDICT_PASS
    FAIL = _VERDICT_FAIL
    SKIPPED = _VERDICT_SKIPPED


class Contract(Enum):
    """The contract each explanation step evaluates.

    ``READY`` is the terminal sentinel emitted on an all-``pass``
    chain. It is not an invariant — it marks the absence of any
    failing invariant.
    """

    B1 = "B1"  # temporal precondition
    I2 = "I2"  # sensor presence
    I3 = "I3"  # R validity
    I1 = "I1"  # R threshold
    I4 = "I4"  # stillness
    READY = "READY"  # no failing contract → permissive state


@dataclass(frozen=True)
class ExplanationStep:
    """One causal step in a :class:`DecisionExplanation` chain.

    Attributes
    ----------
    contract
        Which contract this step evaluates (``B1``..``I4`` /
        ``READY``).
    verdict
        ``PASS`` — this layer passed, control moved on.
        ``FAIL`` — this layer rejected the frame; the final state is
        owned by this step.
        ``SKIPPED`` — a strictly-earlier step already failed; this
        layer was never evaluated.
    observed
        The typed observation at this layer. For ``B1`` this is a
        :class:`TimeQuality`; for ``I1`` it is the numeric ``R``;
        for ``I4`` it is the :class:`StillnessState` name (or
        ``"skipped"`` when ``δ`` was missing). Always JSON-safe.
    detail
        One-line human-readable detail. **Not** parsed by any
        downstream tool — the typed fields above are the structured
        surface.
    """

    contract: Contract
    verdict: Verdict
    observed: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "contract": self.contract.value,
            "verdict": self.verdict.value,
            "observed": self.observed,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DecisionExplanation:
    """The structured causal chain of a single :class:`DecisionFrame`.

    Attributes
    ----------
    tick_index
        Copied from the frame for log correlation.
    timestamp
        Copied from the frame.
    final_state
        The resolved :class:`GateState` — exactly what the gate
        emitted. Not recomputed.
    execution_allowed
        Shortcut for ``final_state is GateState.READY``.
    causal_contract
        The contract that owns the final state: the first ``FAIL``
        step in the chain, or ``Contract.READY`` if every step
        passed. This is the load-bearing interpretability field —
        it answers "why is this frame in this state?" in one
        structured lookup, no string parsing.
    chain
        Ordered tuple of :class:`ExplanationStep` objects, one per
        layer, in the priority order the gate uses.
    summary
        One-line narrative for human viewers. Derived from the
        typed fields above; never written by hand.

    Determinism
    -----------
    ``explain_decision(frame) == explain_decision(frame)`` is
    guaranteed. This enables byte-identical explanation traces
    under F2 replay.
    """

    tick_index: int
    timestamp: float
    final_state: GateState
    execution_allowed: bool
    causal_contract: Contract
    chain: tuple[ExplanationStep, ...] = field(default_factory=tuple)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe flat projection.

        Every value is a primitive or a list of dicts; no nested
        dataclass objects. Suitable for direct ``json.dumps``.
        """
        return {
            "tick_index": self.tick_index,
            "timestamp": self.timestamp,
            "final_state": self.final_state.name,
            "execution_allowed": self.execution_allowed,
            "causal_contract": self.causal_contract.value,
            "chain": [step.to_dict() for step in self.chain],
            "summary": self.summary,
        }

    def as_text(self) -> str:
        """Deterministic tree-style rendering for human reviewers.

        Example output for a frame that lands in ``BLOCKED``::

            tick 42 @ t=4.2000s
              ├─ [pass]    B1   VALID
              ├─ [pass]    I2   sensor_present=True
              ├─ [pass]    I3   R=0.4200
              ├─ [fail]    I1   R=0.42 < threshold=0.65  ← causal root
              ├─ [skipped] I4   threshold failed upstream
              └─ final: BLOCKED  (execution_allowed=False)
        """
        lines: list[str] = [f"tick {self.tick_index} @ t={self.timestamp:.4f}s"]
        last_index = len(self.chain) - 1
        for i, step in enumerate(self.chain):
            corner = "└─" if i == last_index else "├─"
            marker = (
                "  ← causal root"
                if step.contract is self.causal_contract and step.verdict is Verdict.FAIL
                else ""
            )
            lines.append(
                f"  {corner} [{step.verdict.value:<7}] "
                f"{step.contract.value:<5} {step.detail}{marker}"
            )
        lines.append(
            f"  final: {self.final_state.name}  (execution_allowed={self.execution_allowed})"
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The derivation function — pure, deterministic, total.
# ---------------------------------------------------------------------------


def explain_decision(
    frame: DecisionFrame, *, threshold: float | None = None
) -> DecisionExplanation:
    """Reconstruct the causal chain from a materialised decision frame.

    Parameters
    ----------
    frame
        A :class:`DecisionFrame` produced by
        :class:`~neurophase.runtime.pipeline.StreamingPipeline.tick`
        or loaded from a ledger replay.
    threshold
        Optional override of the gate threshold used for the ``I1``
        detail line. Defaults to ``frame.gate.threshold``.

    Returns
    -------
    DecisionExplanation
        Frozen, deterministic, JSON-safe.
    """
    if threshold is None:
        threshold = frame.gate.threshold

    chain: list[ExplanationStep] = []

    final_state = frame.gate.state
    final_allowed = frame.gate.execution_allowed
    causal: Contract | None = None

    # ---- B1 temporal precondition -----------------------------------
    tq = frame.temporal.quality
    if tq is TimeQuality.VALID:
        chain.append(
            ExplanationStep(
                contract=Contract.B1,
                verdict=Verdict.PASS,
                observed=tq.name,
                detail="temporal quality is VALID",
            )
        )
        b1_passed = True
    else:
        chain.append(
            ExplanationStep(
                contract=Contract.B1,
                verdict=Verdict.FAIL,
                observed=tq.name,
                detail=f"temporal quality = {tq.name} (expected VALID)",
            )
        )
        b1_passed = False
        causal = Contract.B1

    # ---- Stream regime (rolled into B1 surface at the runtime layer)-
    # Not part of the strict priority order, but the pipeline forces
    # DEGRADED on non-HEALTHY stream regimes. Surface it as a B1
    # sub-step so the explanation is lossless.
    stream_regime = frame.stream.regime
    if stream_regime is not StreamRegime.HEALTHY and b1_passed:
        # The stream detector flagged the frame even though the
        # per-sample B1 was VALID. Pipeline.tick() then synthesized a
        # GAPPED time_quality placeholder, which is *already* accounted
        # for by the first B1 step above if we walked the pipeline
        # path. We add a separate, explicit "stream" step so the
        # reviewer sees both layers.
        chain.append(
            ExplanationStep(
                contract=Contract.B1,
                verdict=Verdict.FAIL,
                observed=stream_regime.name,
                detail=(f"stream regime = {stream_regime.name} (expected HEALTHY)"),
            )
        )
        b1_passed = False
        if causal is None:
            causal = Contract.B1

    # ---- I2 sensor presence -----------------------------------------
    # The pipeline does not expose a sensor_present hook, so we infer
    # it from the final gate state: SENSOR_ABSENT iff I2 failed. All
    # other paths pass this layer.
    if not b1_passed:
        chain.append(_skipped_step(Contract.I2, "B1 failed upstream"))
        sensor_passed = None  # undefined — layer did not run
    elif final_state is GateState.SENSOR_ABSENT:
        chain.append(
            ExplanationStep(
                contract=Contract.I2,
                verdict=Verdict.FAIL,
                observed="sensor_absent",
                detail="sensor not present",
            )
        )
        if causal is None:
            causal = Contract.I2
        sensor_passed = False
    else:
        chain.append(
            ExplanationStep(
                contract=Contract.I2,
                verdict=Verdict.PASS,
                observed="sensor_present",
                detail="sensor present",
            )
        )
        sensor_passed = True

    # ---- I3 R validity ----------------------------------------------
    upstream_failed = (not b1_passed) or sensor_passed is False or sensor_passed is None
    if upstream_failed:
        chain.append(
            _skipped_step(
                Contract.I3,
                ("B1 failed upstream" if not b1_passed else "I2 failed upstream"),
            )
        )
        i3_passed = None
    else:
        R_val = frame.R
        if R_val is None or not math.isfinite(R_val) or not 0.0 <= R_val <= 1.0:
            # I3 failed at the gate, but the gate may or may not have
            # emitted DEGRADED depending on B1 ordering. We label this
            # failure only when the final state is DEGRADED — otherwise
            # the run went through I3 cleanly via a stream-degraded
            # path.
            if final_state is GateState.DEGRADED and causal is None:
                chain.append(
                    ExplanationStep(
                        contract=Contract.I3,
                        verdict=Verdict.FAIL,
                        observed=repr(R_val),
                        detail=f"R = {R_val!r} is not in [0, 1]",
                    )
                )
                causal = Contract.I3
                i3_passed = False
            else:
                chain.append(_skipped_step(Contract.I3, "R invalid but upstream dominated"))
                i3_passed = None
        else:
            chain.append(
                ExplanationStep(
                    contract=Contract.I3,
                    verdict=Verdict.PASS,
                    observed=f"{R_val:.4f}",
                    detail=f"R = {R_val:.4f} ∈ [0, 1]",
                )
            )
            i3_passed = True

    # ---- I1 R threshold ---------------------------------------------
    if i3_passed is not True:
        chain.append(_skipped_step(Contract.I1, "I3 failed or skipped upstream"))
        i1_passed = None
    else:
        R_val = frame.R
        assert R_val is not None  # narrowed by i3_passed is True
        if R_val < threshold:
            chain.append(
                ExplanationStep(
                    contract=Contract.I1,
                    verdict=Verdict.FAIL,
                    observed=f"{R_val:.4f}",
                    detail=(f"R = {R_val:.4f} < threshold = {threshold:.4f}"),
                )
            )
            if causal is None:
                causal = Contract.I1
            i1_passed = False
        else:
            chain.append(
                ExplanationStep(
                    contract=Contract.I1,
                    verdict=Verdict.PASS,
                    observed=f"{R_val:.4f}",
                    detail=(f"R = {R_val:.4f} ≥ threshold = {threshold:.4f}"),
                )
            )
            i1_passed = True

    # ---- I4 stillness ------------------------------------------------
    if i1_passed is not True:
        chain.append(_skipped_step(Contract.I4, "I1 failed or skipped upstream"))
    elif frame.gate.stillness_state is None:
        chain.append(
            ExplanationStep(
                contract=Contract.I4,
                verdict=Verdict.SKIPPED,
                observed="no_detector_or_missing_delta",
                detail="stillness layer was not evaluated",
            )
        )
    elif final_state is GateState.UNNECESSARY:
        chain.append(
            ExplanationStep(
                contract=Contract.I4,
                verdict=Verdict.FAIL,
                observed=frame.gate.stillness_state.name,
                detail=(f"stillness = {frame.gate.stillness_state.name} → no new information"),
            )
        )
        if causal is None:
            causal = Contract.I4
    else:
        chain.append(
            ExplanationStep(
                contract=Contract.I4,
                verdict=Verdict.PASS,
                observed=frame.gate.stillness_state.name,
                detail=(f"stillness = {frame.gate.stillness_state.name} → active"),
            )
        )

    # ---- Final resolution -------------------------------------------
    if causal is None:
        causal = Contract.READY

    summary = _summary_line(
        final_state=final_state,
        execution_allowed=final_allowed,
        causal=causal,
    )

    return DecisionExplanation(
        tick_index=frame.tick_index,
        timestamp=frame.timestamp,
        final_state=final_state,
        execution_allowed=final_allowed,
        causal_contract=causal,
        chain=tuple(chain),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _skipped_step(contract: Contract, reason: str) -> ExplanationStep:
    return ExplanationStep(
        contract=contract,
        verdict=Verdict.SKIPPED,
        observed="upstream_short_circuit",
        detail=reason,
    )


def _summary_line(*, final_state: GateState, execution_allowed: bool, causal: Contract) -> str:
    if execution_allowed and final_state is GateState.READY:
        return "READY: every contract passed — execution permitted"
    return f"{final_state.name}: execution blocked by {causal.value} ({final_state.name.lower()})"


def explain_gate(
    decision: GateDecision, *, tick_index: int = 0, timestamp: float = 0.0
) -> DecisionExplanation:
    """Convenience wrapper for a bare :class:`GateDecision`.

    Most callers should use :func:`explain_decision` on a full
    :class:`DecisionFrame`. This shortcut is intended for unit tests
    and for direct ``ExecutionGate.evaluate`` consumers that never
    went through a pipeline.
    """
    # Build a synthetic frame-shaped envelope. We only need the fields
    # ``explain_decision`` actually reads.
    from dataclasses import dataclass as _dc

    @_dc(frozen=True)
    class _MinimalTemporal:
        quality: TimeQuality = TimeQuality.VALID

    @_dc(frozen=True)
    class _MinimalStreamStats:
        fault_rate: float = 0.0

    @_dc(frozen=True)
    class _MinimalStream:
        regime: StreamRegime = StreamRegime.HEALTHY
        reason: str = "healthy: synthetic fixture"
        stats: _MinimalStreamStats = field(default_factory=_MinimalStreamStats)

    @_dc(frozen=True)
    class _MinimalFrame:
        tick_index: int
        timestamp: float
        R: float | None
        delta: float | None
        temporal: _MinimalTemporal
        stream: _MinimalStream
        gate: GateDecision

    synthetic: Any = _MinimalFrame(
        tick_index=tick_index,
        timestamp=timestamp,
        R=decision.R,
        delta=None,
        temporal=_MinimalTemporal(),
        stream=_MinimalStream(stats=_MinimalStreamStats()),
        gate=decision,
    )
    return explain_decision(synthetic, threshold=decision.threshold)
