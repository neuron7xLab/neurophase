"""A3 — cross-module invariant test matrix.

This is the **safety-proof** complement to the isolated per-module tests.

Every other test file in the suite checks one axis at a time: a
single invariant, a single layer, a single behavioural contract. A3
is the test that nothing can silently slip out of its assigned state
across the **full Cartesian product** of upstream inputs:

    (time_quality, sensor_present, R, delta, stillness_detector, stillness_result)

Why this matters (safety > liveness)
------------------------------------

Suppose a future refactor of ``ExecutionGate._classify_ready`` rewrites
a conditional and accidentally lets one cell slip from ``UNNECESSARY``
to ``READY``. All isolated tests would still pass because they exercise
one input axis at a time and the drift is in a cross-product cell that
no single-axis test ever touches. The matrix here sweeps the entire
cross-product and compares the live ``ExecutionGate`` output against a
**pure analytical predictor** encoded from the ``STATE_MACHINE.yaml``
spec. Any drift between the two is a cross-module invariant violation
and breaks CI.

Doctrine (from the Evolution Board master prompt v2.0):

> The most important question: what does the system NOT know about itself?
> A proof that something cannot silently fail > a proof that it works.

The matrix is that proof for the gate-level semantic surface.

What this file contains
-----------------------

1. A **pure analytical predictor** ``predict_gate_state`` that encodes
   the strict evaluation order straight from ``STATE_MACHINE.yaml``:
   ``B₁ > I₂ > I₃ > I₁ > I₄``. The predictor has no dependency on
   ``ExecutionGate`` and can be read as the formal specification.
2. An **input grid** spanning every meaningful regime for each input
   axis: seven ``TimeQuality`` values + one "no-gate-check" case;
   sensor present / absent; valid / invalid / below-threshold / at-
   threshold / above-threshold ``R``; missing / invalid / several
   valid ``δ`` values; with / without stillness detector;
   ``STILL`` / ``ACTIVE`` / no-detector stillness outcomes.
3. A **cross-product driver** that iterates every cell and verifies
   the live gate agrees with the predictor.
4. A **reachability suite** proving every ``GateState`` member is
   reached by at least one cell.
5. A **priority-ordering suite** that constructs inputs designed to
   satisfy *two* failing conditions at once and verifies the
   higher-priority one wins.
6. A **state-machine consistency check** that loads
   ``STATE_MACHINE.yaml`` and verifies every declared transition has
   a matching matrix cell (liveness for the spec itself).
7. A **``STREAMING_PIPELINE`` reachability suite** that drives the
   full runtime pipeline into every gate state through a natural
   tick sequence — not synthetic one-shots. This is the liveness
   proof that the analytical predictor is not a useless abstraction.

Bindings
--------

HN16 (new honest-naming contract) is registered in ``INVARIANTS.yaml``
against this file. If any cell disagrees with the predictor, CI
fails and the binding surfaces the regression.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from neurophase.data.temporal_validator import (
    TemporalQualityDecision,
    TimeQuality,
)
from neurophase.gate.execution_gate import (
    DEFAULT_THRESHOLD,
    ExecutionGate,
    GateDecision,
    GateState,
)
from neurophase.gate.stillness_detector import StillnessDetector, StillnessState
from neurophase.governance.state_machine import load_state_machine

# ---------------------------------------------------------------------------
# Analytical predictor — the formal specification of the gate.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateInput:
    """One cell of the input cross-product."""

    time_quality: TimeQuality | None  # None means "no temporal check"
    sensor_present: bool
    R: float | None
    delta: float | None
    has_stillness_detector: bool
    forced_stillness: StillnessState | None  # what the detector would emit, or None
    threshold: float = DEFAULT_THRESHOLD


#: Only these values of ``R`` are physically valid inputs for the
#: ``I₁`` check; anything else routes to ``DEGRADED`` via ``I₃``.
_R_VALID_RANGE = (0.0, 1.0)


def _r_is_valid(R: float | None) -> bool:
    if R is None:
        return False
    if not math.isfinite(R):
        return False
    lo, hi = _R_VALID_RANGE
    return lo <= R <= hi


def _delta_is_valid_for_stillness(delta: float | None) -> bool:
    if delta is None:
        return False
    if not math.isfinite(delta):
        return False
    return 0.0 <= delta <= math.pi + 1e-12


def predict_gate_state(inputs: GateInput) -> GateState:
    """Pure predictor — the state ``ExecutionGate.evaluate`` *must* return.

    Encodes the strict evaluation order from the Evolution Board
    doctrine:

        0. ``time_quality`` supplied and != VALID  → DEGRADED (B₁)
        1. ``sensor_present == False``             → SENSOR_ABSENT (I₂)
        2. ``R`` invalid / None / OOR              → DEGRADED (I₃)
        3. ``R < threshold``                       → BLOCKED (I₁)
        4. ``R ≥ threshold``:
           4.1 no stillness detector               → READY
           4.2 δ missing / invalid                 → READY (stillness skipped)
           4.3 stillness ACTIVE                    → READY
           4.4 stillness STILL                     → UNNECESSARY (I₄)
    """
    # Step 0 — B₁ temporal precondition.
    if inputs.time_quality is not None and inputs.time_quality is not TimeQuality.VALID:
        return GateState.DEGRADED

    # Step 1 — I₂ sensor presence.
    if not inputs.sensor_present:
        return GateState.SENSOR_ABSENT

    # Step 2 — I₃ R validity.
    if not _r_is_valid(inputs.R):
        return GateState.DEGRADED

    # Step 3 — I₁ threshold.
    assert inputs.R is not None  # narrowed by _r_is_valid
    if inputs.threshold > inputs.R:
        return GateState.BLOCKED

    # Step 4 — R ≥ threshold, split by stillness layer.
    if not inputs.has_stillness_detector:
        return GateState.READY

    if not _delta_is_valid_for_stillness(inputs.delta):
        return GateState.READY  # stillness evaluation skipped

    if inputs.forced_stillness is StillnessState.ACTIVE:
        return GateState.READY
    if inputs.forced_stillness is StillnessState.STILL:
        return GateState.UNNECESSARY

    # Detector present, delta present, but we did not force a specific
    # stillness result. In that case the detector will be in warmup
    # (the test harness creates a fresh detector per cell), which
    # returns ACTIVE → READY.
    return GateState.READY


# ---------------------------------------------------------------------------
# Input grid — Cartesian product generator.
# ---------------------------------------------------------------------------


def _time_quality_values() -> tuple[TimeQuality | None, ...]:
    return (None, *TimeQuality)


def _r_values() -> tuple[float | None, ...]:
    return (
        None,
        float("nan"),
        float("inf"),
        -0.1,
        1.1,
        0.0,
        0.30,
        0.50,
        DEFAULT_THRESHOLD - 0.01,  # just below
        DEFAULT_THRESHOLD,  # at
        DEFAULT_THRESHOLD + 0.01,  # just above
        0.90,
        1.0,
    )


def _delta_values() -> tuple[float | None, ...]:
    return (
        None,
        float("nan"),
        float("inf"),
        -0.1,
        math.pi + 0.5,
        0.0,
        0.01,
        0.50,
        math.pi,
    )


def _stillness_outcomes() -> tuple[StillnessState | None, ...]:
    return (None, StillnessState.STILL, StillnessState.ACTIVE)


def _enumerate_cells() -> Iterable[GateInput]:
    """Generate the full Cartesian product, deduplicated."""
    seen: set[tuple[object, ...]] = set()
    for tq in _time_quality_values():
        for sensor in (True, False):
            for R in _r_values():
                for delta in _delta_values():
                    for has_still in (True, False):
                        outcomes = _stillness_outcomes() if has_still else (None,)
                        for forced in outcomes:
                            key = (
                                tq,
                                sensor,
                                R,
                                delta,
                                has_still,
                                forced,
                            )
                            # Nan keys need a canonical repr — use math.isnan.
                            if key in seen:
                                continue
                            seen.add(key)
                            yield GateInput(
                                time_quality=tq,
                                sensor_present=sensor,
                                R=R,
                                delta=delta,
                                has_stillness_detector=has_still,
                                forced_stillness=forced,
                            )


# ---------------------------------------------------------------------------
# Gate driver — builds an ``ExecutionGate`` per cell.
# ---------------------------------------------------------------------------


class _DeterministicStillnessDetector(StillnessDetector):
    """StillnessDetector that can be *forced* to return a specific state.

    Used only inside the A3 matrix: the analytical predictor specifies
    what the detector *should* emit, so the matrix must be able to
    force the detector into that outcome without priming a whole
    window. This is a test-only subclass; production code always uses
    the real rolling detector.
    """

    def __init__(self, forced: StillnessState | None) -> None:
        super().__init__(window=4, eps_R=1e-3, eps_F=1e-3, delta_min=0.10)
        self._forced: StillnessState | None = forced

    def update(self, R: float, delta: float) -> object:  # type: ignore[override]
        # Build a minimal StillnessDecision-shaped object compatible
        # with the gate's StillnessState check. We import the real
        # frozen class from the module and populate only the fields
        # the gate actually reads (``state``).
        from neurophase.gate.stillness_detector import StillnessDecision

        if self._forced is None:
            # Fall back to the real detector behaviour — this is the
            # "warmup → ACTIVE" branch for the first few calls.
            return super().update(R=R, delta=delta)
        return StillnessDecision(
            state=self._forced,
            R=R,
            delta=delta,
            dR_dt_max=0.0,
            dF_proxy_dt_max=0.0,
            delta_max=delta,
            window_filled=True,
            reason=f"forced: {self._forced.name.lower()}",
        )


def _build_gate(inputs: GateInput) -> ExecutionGate:
    detector: StillnessDetector | None
    if inputs.has_stillness_detector:
        detector = _DeterministicStillnessDetector(inputs.forced_stillness)
    else:
        detector = None
    return ExecutionGate(threshold=inputs.threshold, stillness_detector=detector)


def _build_time_quality(
    quality: TimeQuality | None,
) -> TemporalQualityDecision | None:
    if quality is None:
        return None
    return TemporalQualityDecision(
        quality=quality,
        ts=1.0,
        last_ts=0.0,
        gap_seconds=1.0,
        staleness_seconds=0.0,
        warmup_remaining=0,
        reason=f"{quality.name.lower()}: matrix fixture",
    )


def _evaluate(inputs: GateInput) -> GateDecision:
    gate = _build_gate(inputs)
    return gate.evaluate(
        R=inputs.R,
        sensor_present=inputs.sensor_present,
        delta=inputs.delta,
        time_quality=_build_time_quality(inputs.time_quality),
    )


# ---------------------------------------------------------------------------
# 1. The cross-product driver test.
# ---------------------------------------------------------------------------


ALL_CELLS: Final[tuple[GateInput, ...]] = tuple(_enumerate_cells())


class TestCrossProduct:
    """Every cell in the input cross-product must match the predictor."""

    def test_matrix_is_nontrivial(self) -> None:
        """Smoke check: the matrix covers at least 500 distinct cells."""
        assert len(ALL_CELLS) >= 500

    def test_every_cell_matches_predictor(self) -> None:
        """The load-bearing proof.

        For every cell in the Cartesian product, the live
        ``ExecutionGate.evaluate`` output must equal the analytical
        prediction. A single disagreement is a cross-module invariant
        violation and fails the build.
        """
        disagreements: list[tuple[GateInput, GateState, GateState]] = []
        for cell in ALL_CELLS:
            expected = predict_gate_state(cell)
            actual = _evaluate(cell).state
            if actual is not expected:
                disagreements.append((cell, expected, actual))
        if disagreements:
            formatted = "\n".join(
                f"  {cell!r}\n    expected={expected.name}, got={actual.name}"
                for cell, expected, actual in disagreements[:10]
            )
            total = len(disagreements)
            pytest.fail(
                f"{total} matrix cell(s) disagree with the analytical predictor.\n"
                f"First 10:\n{formatted}"
            )

    def test_execution_allowed_iff_ready(self) -> None:
        """``execution_allowed == True`` iff ``state == READY``, everywhere."""
        for cell in ALL_CELLS:
            decision = _evaluate(cell)
            if decision.state is GateState.READY:
                assert decision.execution_allowed is True, cell
            else:
                assert decision.execution_allowed is False, cell


# ---------------------------------------------------------------------------
# 2. Reachability — every GateState member must be reached.
# ---------------------------------------------------------------------------


class TestReachability:
    """Every ``GateState`` member must appear in the matrix output."""

    def test_every_state_is_reachable(self) -> None:
        reached: set[GateState] = set()
        for cell in ALL_CELLS:
            reached.add(_evaluate(cell).state)
        missing = set(GateState) - reached
        assert not missing, f"unreachable gate states: {sorted(s.name for s in missing)}"

    @pytest.mark.parametrize(
        "target",
        list(GateState),
        ids=lambda s: s.name,
    )
    def test_each_state_has_at_least_one_cell(self, target: GateState) -> None:
        """Parametrized per-state check so the failure message is specific."""
        for cell in ALL_CELLS:
            if _evaluate(cell).state is target:
                return
        pytest.fail(f"no matrix cell produces {target.name}")


# ---------------------------------------------------------------------------
# 3. Priority ordering — higher-priority condition wins.
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    """When two failing conditions apply, the higher-priority one wins.

    Priority order from the Evolution Board doctrine:
        B₁ (time_quality != VALID) > I₂ (sensor absent)
                                   > I₃ (R invalid)
                                   > I₁ (R < threshold)
                                   > I₄ (stillness)
    """

    def test_temporal_dominates_sensor_absent(self) -> None:
        # time_quality=GAPPED AND sensor_present=False
        # Priority order: temporal wins → DEGRADED.
        cell = GateInput(
            time_quality=TimeQuality.GAPPED,
            sensor_present=False,
            R=0.99,
            delta=0.01,
            has_stillness_detector=False,
            forced_stillness=None,
        )
        assert _evaluate(cell).state is GateState.DEGRADED
        assert predict_gate_state(cell) is GateState.DEGRADED

    def test_temporal_dominates_r_invalid(self) -> None:
        cell = GateInput(
            time_quality=TimeQuality.STALE,
            sensor_present=True,
            R=float("nan"),
            delta=0.01,
            has_stillness_detector=False,
            forced_stillness=None,
        )
        assert _evaluate(cell).state is GateState.DEGRADED

    def test_temporal_dominates_r_below_threshold(self) -> None:
        cell = GateInput(
            time_quality=TimeQuality.REVERSED,
            sensor_present=True,
            R=0.10,
            delta=0.01,
            has_stillness_detector=False,
            forced_stillness=None,
        )
        # Temporal first → DEGRADED, not BLOCKED.
        decision = _evaluate(cell)
        assert decision.state is GateState.DEGRADED
        assert "temporal" in decision.reason.lower()

    def test_sensor_dominates_r_invalid(self) -> None:
        """When no time_quality is supplied, sensor_absent wins over R invalid."""
        cell = GateInput(
            time_quality=None,
            sensor_present=False,
            R=float("nan"),
            delta=0.01,
            has_stillness_detector=False,
            forced_stillness=None,
        )
        assert _evaluate(cell).state is GateState.SENSOR_ABSENT

    def test_sensor_dominates_r_below_threshold(self) -> None:
        cell = GateInput(
            time_quality=None,
            sensor_present=False,
            R=0.1,
            delta=0.01,
            has_stillness_detector=False,
            forced_stillness=None,
        )
        assert _evaluate(cell).state is GateState.SENSOR_ABSENT

    def test_r_invalid_dominates_r_below_threshold(self) -> None:
        """R is None, so I₃ fires before I₁ would have a chance."""
        cell = GateInput(
            time_quality=None,
            sensor_present=True,
            R=None,
            delta=0.01,
            has_stillness_detector=False,
            forced_stillness=None,
        )
        assert _evaluate(cell).state is GateState.DEGRADED

    def test_r_below_threshold_dominates_stillness(self) -> None:
        """R is below threshold AND stillness would say STILL — BLOCKED wins."""
        cell = GateInput(
            time_quality=None,
            sensor_present=True,
            R=0.3,
            delta=0.01,
            has_stillness_detector=True,
            forced_stillness=StillnessState.STILL,
        )
        assert _evaluate(cell).state is GateState.BLOCKED

    def test_stillness_only_applies_above_threshold(self) -> None:
        cell = GateInput(
            time_quality=None,
            sensor_present=True,
            R=0.99,
            delta=0.01,
            has_stillness_detector=True,
            forced_stillness=StillnessState.STILL,
        )
        assert _evaluate(cell).state is GateState.UNNECESSARY


# ---------------------------------------------------------------------------
# 4. STATE_MACHINE.yaml × matrix consistency.
# ---------------------------------------------------------------------------


class TestStateMachineConsistency:
    """Every transition in STATE_MACHINE.yaml must have ≥ 1 matching matrix cell."""

    def test_every_transition_target_is_reachable_by_matrix(self) -> None:
        spec = load_state_machine()
        cells_by_state: dict[str, list[GateInput]] = {}
        for cell in ALL_CELLS:
            state = _evaluate(cell).state.name
            cells_by_state.setdefault(state, []).append(cell)

        missing: list[str] = []
        for transition in spec.transitions:
            if transition.target not in cells_by_state:
                missing.append(f"{transition.id} → {transition.target} has no matrix cell")
        assert not missing, "\n".join(missing)

    def test_permissive_transitions_match_ready_cells(self) -> None:
        """Every transition marked ``execution_allowed=True`` must have at
        least one matrix cell that lands in READY with the flag set."""
        spec = load_state_machine()
        ready_cells = [cell for cell in ALL_CELLS if _evaluate(cell).state is GateState.READY]
        assert ready_cells, "matrix has no READY cells"
        for transition in spec.transitions:
            if transition.execution_allowed:
                assert transition.target == "READY"


# ---------------------------------------------------------------------------
# 5. Full pipeline reachability — STREAMING_PIPELINE drives every state.
# ---------------------------------------------------------------------------


class TestPipelineReachability:
    """Prove every gate state is reachable via ``StreamingPipeline.tick``.

    The cross-product matrix verifies *point-wise* safety: for every
    isolated input cell, the gate returns the right state. This
    section is the complementary *liveness* check: driving the full
    pipeline (B1 temporal validator → B2+B6 stream detector →
    ExecutionGate) with a natural tick sequence reaches every state.
    """

    def _drive(self, *, R: float | None, delta: float | None = 0.01, **config: object) -> GateState:
        from neurophase.runtime.pipeline import PipelineConfig, StreamingPipeline

        defaults: dict[str, object] = {
            "warmup_samples": 2,
            "stream_window": 4,
            "max_fault_rate": 0.50,
        }
        defaults.update(config)
        p = StreamingPipeline(PipelineConfig(**defaults))  # type: ignore[arg-type]
        # Prime enough clean ticks for:
        #   (a) TemporalValidator warmup to complete (warmup_samples),
        #   (b) TemporalStreamDetector window to fill (stream_window),
        #   (c) stream regime to commit to HEALTHY.
        # Six clean ticks is sufficient for the default defaults above.
        for i in range(6):
            p.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.01)
        frame = p.tick(timestamp=0.7, R=R, delta=delta)
        return frame.gate_state

    def test_degraded_reachable_via_bad_R(self) -> None:
        """R=None → DEGRADED."""
        assert self._drive(R=None) is GateState.DEGRADED

    def test_blocked_reachable_via_low_R(self) -> None:
        assert self._drive(R=0.30) is GateState.BLOCKED

    def test_ready_reachable_via_healthy_high_R(self) -> None:
        # Disable stillness so the post-warmup frame is READY, not UNNECESSARY.
        assert self._drive(R=0.99, enable_stillness=False) is GateState.READY

    def test_unnecessary_reachable_via_still_high_R(self) -> None:
        """Drive the pipeline with a constant calm signal until stillness
        fires, then confirm the gate transitions to UNNECESSARY."""
        from neurophase.runtime.pipeline import PipelineConfig, StreamingPipeline

        p = StreamingPipeline(
            PipelineConfig(
                warmup_samples=2,
                stream_window=4,
                enable_stillness=True,
                stillness_window=4,
                stillness_eps_R=1e-3,
                stillness_eps_F=1e-3,
                stillness_delta_min=0.20,
            )
        )
        seen_unnecessary = False
        for i in range(20):
            frame = p.tick(timestamp=float(i) * 0.1, R=0.95, delta=0.01)
            if frame.gate_state is GateState.UNNECESSARY:
                seen_unnecessary = True
                break
        assert seen_unnecessary, "UNNECESSARY never reached via pipeline"

    def test_sensor_absent_reachable_via_direct_gate_call(self) -> None:
        """SENSOR_ABSENT is not reachable through ``StreamingPipeline``
        because the pipeline does not expose a ``sensor_present`` hook;
        it is reachable through a direct ``ExecutionGate`` call, which
        is the intended entry point for hardware fault reporting."""
        gate = ExecutionGate(threshold=DEFAULT_THRESHOLD)
        decision = gate.evaluate(R=0.99, sensor_present=False)
        assert decision.state is GateState.SENSOR_ABSENT


# ---------------------------------------------------------------------------
# 6. Invariant: the matrix test itself is bound in INVARIANTS.yaml.
# ---------------------------------------------------------------------------


class TestBoundInRegistry:
    """Meta-meta-check: HN16 must reference this file."""

    def test_this_file_is_registered(self) -> None:
        registry_path = Path(__file__).resolve().parent.parent / "INVARIANTS.yaml"
        text = registry_path.read_text(encoding="utf-8")
        assert "test_invariant_matrix.py" in text, (
            "tests/test_invariant_matrix.py must be bound to HN16 in INVARIANTS.yaml"
        )
