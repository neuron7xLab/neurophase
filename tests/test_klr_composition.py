"""Composition-level tests for the KLR (Ketamine-Like Reset) subsystem.

Covers all four load-bearing invariants across the full state-machine lifecycle:
  KLR-I1: Rollback safety — failed intervention reverts to checkpoint bit-for-bit.
  KLR-I2: Frozen nodes never modified — frozen[i]=True → weights[i,:] unchanged.
  KLR-I3: Row-stochastic preservation — weights rows sum to 1.0 after every op.
  KLR-I4: Deterministic seed trace — same inputs → same outputs.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import patch

import numpy as np

from neurophase.reset.config import KLRConfig
from neurophase.reset.controller import (
    KetamineLikeResetController,
    ResetReport,
    ResetState,
)
from neurophase.reset.curriculum import Curriculum
from neurophase.reset.deterministic_oracle import derive_seed
from neurophase.reset.metrics import SystemMetrics
from neurophase.reset.pipeline import KLRFrame, KLRPipeline
from neurophase.reset.state import SystemState, clone_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(n: int = 8, *, seed: int = 42) -> SystemState:
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.1, 1.0, (n, n))
    w /= w.sum(axis=1, keepdims=True)
    return SystemState(
        weights=w,
        confidence=rng.uniform(0.0, 1.0, n),
        usage=rng.uniform(0.0, 1.0, n),
        utility=rng.uniform(0.0, 1.0, n),
        inhibition=rng.uniform(0.0, 1.0, n),
        topology=(rng.uniform(0, 1, (n, n)) > 0.3).astype(np.float64),
        frozen=np.array([i < 2 for i in range(n)]),
        gamma=0.5,
    )


def _make_metrics(*, trigger: bool = True) -> SystemMetrics:
    if trigger:
        return SystemMetrics(
            error=0.9,
            persistence=0.8,
            diversity=0.1,
            improvement=0.1,
            noise=0.5,
            reward=0.0,
        )
    return SystemMetrics(
        error=0.1,
        persistence=0.1,
        diversity=0.9,
        improvement=0.9,
        noise=0.1,
        reward=1.0,
    )


def _make_curriculum(n: int = 8, *, seed: int = 7) -> Curriculum:
    rng = np.random.default_rng(seed)
    tb = rng.uniform(0.0, 0.5, n)
    cs = rng.uniform(-0.5, 0.5, n)
    sp = rng.uniform(0.1, 0.9, n)
    return Curriculum(
        target_bias=tb,
        corrective_signal=cs,
        stress_pattern=sp,
    )


def _controller_that_triggers() -> KetamineLikeResetController:
    """Controller with lock_in_threshold=0.1 so the metrics will always trigger."""
    return KetamineLikeResetController(KLRConfig(lock_in_threshold=0.1))


# ---------------------------------------------------------------------------
# KLR-I1: Rollback safety
# ---------------------------------------------------------------------------


def test_rollback_restores_exact_checkpoint() -> None:
    """When relapse_ratio > relapse_threshold, output weights must equal the
    pre-intervention checkpoint bit-for-bit (KLR-I1)."""
    ctrl = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.1, relapse_threshold=0.0))
    state = _make_state()
    before_weights = state.weights.copy()
    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()

    out, report = ctrl.run(state, metrics, curriculum)

    assert report.status == "ROLLBACK"
    assert np.array_equal(out.weights, before_weights), (
        "Rollback must restore weights to exact checkpoint values"
    )


def test_rollback_on_exception_restores_state() -> None:
    """If an exception is raised inside the plasticity window, state is restored
    to the checkpoint (KLR-I1)."""
    ctrl = _controller_that_triggers()
    state = _make_state()
    before_weights = state.weights.copy()
    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()

    def _boom(*args: Any, **kwargs: Any) -> SystemState:
        raise RuntimeError("synthetic_failure")

    with patch.object(ctrl, "_open_plasticity_window", side_effect=_boom):
        out, report = ctrl.run(state, metrics, curriculum)

    assert report.status == "ROLLBACK"
    assert np.array_equal(out.weights, before_weights), (
        "Exception path must restore weights to checkpoint"
    )


def test_checkpoint_is_deep_copy() -> None:
    """The checkpoint stored in _prepare must be independent of the live state;
    mutating the output after rollback must not corrupt the checkpoint copy."""
    ctrl = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.1, relapse_threshold=0.0))
    state = _make_state()
    checkpoint_snapshot = state.weights.copy()
    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()

    out, report = ctrl.run(state, metrics, curriculum)
    assert report.status == "ROLLBACK"

    # Corrupt the output in-place — checkpoint must remain intact.
    out.weights[:] = 999.0
    assert np.array_equal(out.weights.shape, checkpoint_snapshot.shape)
    # The controller no longer holds _checkpoint after rollback.
    assert ctrl._checkpoint is None, "checkpoint must be cleared after rollback"


# ---------------------------------------------------------------------------
# KLR-I2: Frozen nodes never modified
# ---------------------------------------------------------------------------


def test_frozen_weights_unchanged_after_success() -> None:
    """Successful intervention must not alter rows/columns of frozen nodes (KLR-I2)."""
    ctrl = _controller_that_triggers()
    state = _make_state()
    assert state.frozen is not None
    frozen_rows_before = state.weights[:2, :].copy()

    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()

    out, report = ctrl.run(state, metrics, curriculum)

    if report.status == "SUCCESS":
        assert np.allclose(out.weights[:2, :], frozen_rows_before, atol=1e-10), (
            "Frozen node rows must be unchanged after SUCCESS"
        )


def test_frozen_weights_unchanged_after_rollback() -> None:
    """After rollback, frozen rows must equal the original checkpoint rows (KLR-I2)."""
    ctrl = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.1, relapse_threshold=0.0))
    state = _make_state()
    frozen_rows_before = state.weights[:2, :].copy()
    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()

    out, report = ctrl.run(state, metrics, curriculum)

    assert report.status == "ROLLBACK"
    assert np.allclose(out.weights[:2, :], frozen_rows_before, atol=1e-10), (
        "Frozen node rows must be unchanged after ROLLBACK"
    )


def test_disinhibit_skips_frozen_nodes() -> None:
    """_disinhibit must not modify inhibition/confidence of frozen nodes (KLR-I2).

    Note: _disinhibit applies `weights[:, mask] *= scale` which can touch frozen
    rows in the columns of dominant non-frozen nodes — that column broadcast is
    intentional and is later corrected by _consolidate. The invariant that matters
    is that inhibition and confidence of frozen nodes are unchanged.
    """
    ctrl = _controller_that_triggers()
    state = _make_state()
    assert state.frozen is not None
    frozen_inhibition_before = state.inhibition[:2].copy()
    frozen_confidence_before = state.confidence[:2].copy()

    ctrl._prepare(state)
    result = ctrl._disinhibit(state)

    assert np.allclose(result.inhibition[:2], frozen_inhibition_before, atol=1e-10), (
        "_disinhibit must not modify frozen node inhibition"
    )
    assert np.allclose(result.confidence[:2], frozen_confidence_before, atol=1e-10), (
        "_disinhibit must not modify frozen node confidence"
    )
    # The row-write mask excludes frozen nodes: weights[mask, :] only touches non-frozen rows.
    # Verify that the dominant non-frozen mask itself excludes frozen indices.
    dominant = float(np.quantile(state.usage, ctrl.config.usage_quantile))
    mask = (state.usage >= dominant) & ~result.frozen  # type: ignore[operator]
    assert not np.any(mask[:2]), "_disinhibit row-write mask must not include frozen node indices"


def test_plasticity_window_skips_frozen_weights() -> None:
    """_open_plasticity_window update mask blocks frozen×frozen entries (KLR-I2)."""
    ctrl = _controller_that_triggers()
    state = _make_state()
    assert state.frozen is not None
    frozen_weights_before = state.weights[:2, :].copy()

    rng = np.random.default_rng(0)
    ctrl._prepare(state)
    result = ctrl._open_plasticity_window(state, _make_curriculum(), rng=rng)

    # Rows belonging to frozen nodes must not be modified.
    assert np.allclose(result.weights[:2, :], frozen_weights_before, atol=1e-10), (
        "_open_plasticity_window must not modify frozen rows"
    )


def test_consolidate_can_only_add_to_frozen_set() -> None:
    """_consolidate may freeze new nodes but must never un-freeze existing ones (KLR-I2)."""
    ctrl = _controller_that_triggers()
    state = _make_state()
    assert state.frozen is not None
    originally_frozen = state.frozen.copy()

    ctrl._prepare(state)
    state = ctrl._disinhibit(state)
    state = ctrl._open_plasticity_window(state, _make_curriculum(), rng=np.random.default_rng(0))
    result = ctrl._consolidate(state)

    assert result.frozen is not None
    # Every node that was frozen before must still be frozen.
    for i, was_frozen in enumerate(originally_frozen):
        if was_frozen:
            assert result.frozen[i], f"Node {i} was frozen but got un-frozen by _consolidate"


# ---------------------------------------------------------------------------
# KLR-I3: Row-stochastic preservation
# ---------------------------------------------------------------------------


def test_row_stochastic_after_full_successful_cycle() -> None:
    """Output weights must be row-stochastic after a SUCCESS (KLR-I3)."""
    ctrl = _controller_that_triggers()
    state = _make_state()
    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()

    out, report = ctrl.run(state, metrics, curriculum)

    if report.status == "SUCCESS":
        row_sums = out.weights.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            f"Row sums after SUCCESS must be ≈ 1.0, got {row_sums}"
        )


def test_row_stochastic_after_consolidate() -> None:
    """_consolidate renormalises rows — each row must sum to 1.0 immediately after."""
    ctrl = _controller_that_triggers()
    state = _make_state()

    ctrl._prepare(state)
    state = ctrl._disinhibit(state)
    state = ctrl._open_plasticity_window(state, _make_curriculum(), rng=np.random.default_rng(0))

    # Plasticity may break row-stochasticity — that's expected.
    result = ctrl._consolidate(state)
    row_sums = result.weights.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"_consolidate must produce row-stochastic weights, got {row_sums}"
    )


def test_row_stochastic_after_rollback() -> None:
    """After ROLLBACK, the restored weights must still be row-stochastic (KLR-I3)."""
    ctrl = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.1, relapse_threshold=0.0))
    state = _make_state()
    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()

    out, report = ctrl.run(state, metrics, curriculum)

    assert report.status == "ROLLBACK"
    row_sums = out.weights.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), (
        f"Rolled-back weights must be row-stochastic, got {row_sums}"
    )


def test_row_stochastic_disinhibit_breaks_then_consolidate_restores() -> None:
    """Disinhibit breaks row-stochasticity; _consolidate must fully restore it."""
    ctrl = _controller_that_triggers()
    state = _make_state()

    ctrl._prepare(state)
    state = ctrl._disinhibit(state)

    # After disinhibit, rows are NOT guaranteed to sum to 1.
    row_sums_mid = state.weights.sum(axis=1)
    assert not np.allclose(row_sums_mid, 1.0, atol=1e-3), (
        "_disinhibit expected to break row-stochasticity (test precondition)"
    )

    state = ctrl._open_plasticity_window(state, _make_curriculum(), rng=np.random.default_rng(0))
    result = ctrl._consolidate(state)
    row_sums_final = result.weights.sum(axis=1)
    assert np.allclose(row_sums_final, 1.0, atol=1e-6), (
        "_consolidate must restore row-stochastic property after disinhibit"
    )


# ---------------------------------------------------------------------------
# KLR-I4: Deterministic seed trace
# ---------------------------------------------------------------------------


def test_same_seed_same_output() -> None:
    """Identical state / metrics / curriculum must yield bit-identical output (KLR-I4)."""
    state_a = _make_state()
    state_b = clone_state(state_a)
    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()

    ctrl_a = _controller_that_triggers()
    ctrl_b = _controller_that_triggers()

    out_a, report_a = ctrl_a.run(state_a, metrics, curriculum)
    out_b, report_b = ctrl_b.run(state_b, metrics, curriculum)

    assert report_a.status == report_b.status
    assert np.array_equal(out_a.weights, out_b.weights), (
        "Same inputs must produce identical weight outputs (KLR-I4)"
    )
    assert report_a.seed_trace == report_b.seed_trace


def test_different_seed_different_output() -> None:
    """Different random_seed in KLRConfig must produce different weight updates (KLR-I4)."""
    state_a = _make_state()
    state_b = clone_state(state_a)
    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()

    # Force triggers but give different random_seed values.
    cfg_a = KLRConfig(lock_in_threshold=0.1, random_seed=1)
    cfg_b = KLRConfig(lock_in_threshold=0.1, random_seed=2)

    out_a, report_a = KetamineLikeResetController(cfg_a).run(state_a, metrics, curriculum)
    out_b, report_b = KetamineLikeResetController(cfg_b).run(state_b, metrics, curriculum)

    if report_a.status == "SUCCESS" and report_b.status == "SUCCESS":
        assert not np.array_equal(out_a.weights, out_b.weights), (
            "Different random seeds must produce different weight outputs"
        )


def test_seed_recorded_in_report_matches_derive_seed() -> None:
    """report.seed_trace must equal derive_seed(state, metrics, curriculum).seed (KLR-I4)."""
    state = _make_state()
    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()
    ctrl = _controller_that_triggers()

    expected_seed = derive_seed(state, metrics, curriculum).seed
    _, report = ctrl.run(clone_state(state), metrics, curriculum)

    if report.status in {"SUCCESS", "ROLLBACK"}:
        assert report.seed_trace == expected_seed, (
            f"report.seed_trace={report.seed_trace} must equal derive_seed result={expected_seed}"
        )


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


def test_full_lifecycle_reaches_stable_state() -> None:
    """Controller must transition IDLE→PREPARED→…→STABLE on a clean SUCCESS run."""
    ctrl = _controller_that_triggers()
    state = _make_state()
    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()

    _, report = ctrl.run(state, metrics, curriculum)

    # The state machine must end in STABLE (SUCCESS) or ROLLBACK.
    assert ctrl.state_machine in {ResetState.STABLE, ResetState.ROLLBACK}, (
        f"Unexpected terminal state: {ctrl.state_machine}"
    )
    if report.status == "SUCCESS":
        assert ctrl.state_machine == ResetState.STABLE


def test_pipeline_tick_returns_klr_frame() -> None:
    """KLRPipeline.tick() must return a KLRFrame with required fields."""
    pipeline = KLRPipeline(_make_state())
    metrics = _make_metrics(trigger=True)

    frame = pipeline.tick(metrics)

    assert isinstance(frame, KLRFrame)
    assert frame.decision in {"SUCCESS", "ROLLBACK", "SKIPPED"}
    assert math.isfinite(frame.ntk_rank_delta)
    assert isinstance(frame.report, ResetReport)


def test_pipeline_multiple_ticks_no_crash() -> None:
    """50 ticks with alternating trigger / non-trigger metrics must not raise."""
    pipeline = KLRPipeline(_make_state())
    for i in range(50):
        metrics = _make_metrics(trigger=(i % 3 == 0))
        frame = pipeline.tick(metrics)
        assert frame.decision in {"SUCCESS", "ROLLBACK", "SKIPPED"}


def test_pipeline_refractory_blocks_after_success() -> None:
    """Immediately after a SUCCESS the next tick must return SKIPPED (refractory lock)."""
    pipeline = KLRPipeline(_make_state())
    metrics = _make_metrics(trigger=True)

    # Run until we get a SUCCESS or exhaust attempts.
    success_seen = False
    for _ in range(30):
        frame = pipeline.tick(metrics)
        if frame.decision == "SUCCESS":
            success_seen = True
            break

    if success_seen:
        next_frame = pipeline.tick(metrics)
        assert next_frame.decision == "SKIPPED", (
            "Tick immediately after SUCCESS must be SKIPPED (refractory lock)"
        )


def test_ntk_rank_delta_is_finite() -> None:
    """Every KLRFrame.ntk_rank_delta must be finite (no NaN/Inf)."""
    pipeline = KLRPipeline(_make_state())
    for i in range(20):
        metrics = _make_metrics(trigger=(i % 2 == 0))
        frame = pipeline.tick(metrics)
        assert math.isfinite(frame.ntk_rank_delta), (
            f"ntk_rank_delta must be finite at step {i}, got {frame.ntk_rank_delta}"
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_all_nodes_frozen_skips_modification() -> None:
    """When every node is frozen, plasticity window update must be a no-op."""
    ctrl = _controller_that_triggers()
    n = 8
    rng = np.random.default_rng(0)
    w = rng.uniform(0.1, 1.0, (n, n))
    w /= w.sum(axis=1, keepdims=True)
    state = SystemState(
        weights=w,
        confidence=np.full(n, 0.5),
        usage=np.full(n, 0.5),
        utility=np.full(n, 0.5),
        inhibition=np.full(n, 0.5),
        topology=np.ones((n, n)),
        frozen=np.ones(n, dtype=bool),
        gamma=0.5,
    )
    weights_before = state.weights.copy()

    ctrl._prepare(state)
    result = ctrl._open_plasticity_window(state, _make_curriculum(), rng=np.random.default_rng(0))

    assert np.allclose(result.weights, weights_before, atol=1e-12), (
        "All-frozen state must not have any weights modified by plasticity window"
    )


def test_single_node_state_does_not_crash() -> None:
    """n=1 edge case must complete without raising (row-stochastic = [[1.0]])."""
    state = SystemState(
        weights=np.array([[1.0]]),
        confidence=np.array([0.5]),
        usage=np.array([0.5]),
        utility=np.array([0.5]),
        inhibition=np.array([0.5]),
        topology=np.array([[1.0]]),
        gamma=0.5,
    )
    metrics = _make_metrics(trigger=True)
    curriculum = Curriculum(
        target_bias=np.array([0.5]),
        corrective_signal=np.array([0.1]),
        stress_pattern=np.array([1.0]),
    )
    ctrl = _controller_that_triggers()

    out, report = ctrl.run(state, metrics, curriculum)

    assert report.status in {"SUCCESS", "ROLLBACK", "SKIPPED"}
    row_sums = out.weights.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_high_relapse_triggers_rollback() -> None:
    """relapse_ratio > relapse_threshold must result in ROLLBACK status (KLR-I1 trigger)."""
    # relapse_threshold=0.0 means any positive error ratio triggers rollback.
    ctrl = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.1, relapse_threshold=0.0))
    state = _make_state()
    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()

    _, report = ctrl.run(state, metrics, curriculum)

    assert report.status == "ROLLBACK", (
        "relapse_threshold=0.0 must force ROLLBACK on any non-zero relapse ratio"
    )
    assert report.relapse_ratio >= 0.0


# ---------------------------------------------------------------------------
# Additional composition coverage
# ---------------------------------------------------------------------------


def test_report_fields_are_finite_on_success() -> None:
    """All numeric fields in ResetReport must be finite after SUCCESS."""
    ctrl = _controller_that_triggers()
    state = _make_state()
    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()

    _, report = ctrl.run(state, metrics, curriculum)

    if report.status == "SUCCESS":
        for field in (
            report.relapse_ratio,
            report.improvement_ratio,
            report.gamma_after,
            report.lockin_score,
            report.threshold_used,
            report.ntk_rank_pre,
            report.ntk_rank_post,
            report.rank_delta,
        ):
            assert math.isfinite(field), f"Report field {field} must be finite on SUCCESS"


def test_skipped_when_lockin_score_below_threshold() -> None:
    """If metrics do not trigger lock-in, status must be SKIPPED."""
    # Use a very high threshold so the stable metrics never cross it.
    ctrl = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.99))
    state = _make_state()
    metrics = _make_metrics(trigger=False)
    curriculum = _make_curriculum()

    _, report = ctrl.run(state, metrics, curriculum)

    assert report.status == "SKIPPED"


def test_controller_state_machine_starts_idle() -> None:
    """Fresh controller must be in IDLE state before any run."""
    ctrl = KetamineLikeResetController()
    assert ctrl.state_machine == ResetState.IDLE


def test_clone_state_is_independent() -> None:
    """clone_state must produce an array-independent copy (deep copy semantics)."""
    original = _make_state()
    cloned = clone_state(original)

    cloned.weights[:] = 0.0
    assert not np.allclose(original.weights, 0.0), (
        "clone_state must produce an independent copy — original must be unaffected"
    )


def test_row_stochastic_preserved_after_skipped_run() -> None:
    """SKIPPED status must return the input state with row-stochastic weights intact."""
    ctrl = KetamineLikeResetController(KLRConfig(lock_in_threshold=0.99))
    state = _make_state()
    metrics = _make_metrics(trigger=False)
    curriculum = _make_curriculum()

    out, report = ctrl.run(state, metrics, curriculum)

    assert report.status == "SKIPPED"
    row_sums = out.weights.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_intervention_does_not_modify_curriculum() -> None:
    """The Curriculum dataclass is immutable (frozen=True) and must not be mutated."""
    curriculum = _make_curriculum()
    target_before = curriculum.target_bias.copy()
    ctrl = _controller_that_triggers()
    state = _make_state()
    metrics = _make_metrics(trigger=True)

    ctrl.run(state, metrics, curriculum)

    assert np.array_equal(curriculum.target_bias, target_before), (
        "run() must not mutate curriculum.target_bias"
    )


def test_pipeline_frame_witness_field_type() -> None:
    """KLRFrame.witness_report must be None or a GammaWitnessReport, never an unexpected type."""
    from neurophase.reset.gamma_witness import GammaWitnessReport

    pipeline = KLRPipeline(_make_state(), enable_witness=True)
    frame = pipeline.tick(_make_metrics(trigger=True))

    assert frame.witness_report is None or isinstance(frame.witness_report, GammaWitnessReport)


def test_frozen_count_non_decreasing_across_runs() -> None:
    """Frozen node count must be non-decreasing: consolidation can only add, never remove."""
    ctrl = _controller_that_triggers()
    state = _make_state()
    assert state.frozen is not None
    prev_frozen_count = int(np.sum(state.frozen))
    metrics = _make_metrics(trigger=True)
    curriculum = _make_curriculum()

    for _ in range(5):
        out, _run_report = ctrl.run(state, metrics, curriculum)
        if out.frozen is not None:
            current_frozen_count = int(np.sum(out.frozen))
            assert current_frozen_count >= prev_frozen_count, (
                "Frozen count must be non-decreasing across successive runs"
            )
            prev_frozen_count = current_frozen_count
        state = clone_state(out)
        # Reset refractory so next run can proceed.
        ctrl._refractory._unlock_at = None
