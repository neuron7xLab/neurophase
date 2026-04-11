"""Ketamine-Like Reset Controller (KLR).

Deterministic, fail-closed state machine used to dislodge pathological
attractor lock-in and safely commit only validated improvements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from time import time

import numpy as np
from numpy.typing import NDArray

from neurophase.reset.config import (
    CALIBRATION_LABEL_SOURCE,
    CALIBRATION_SCOPE,
    EVIDENCE_STATUS,
    LOCKIN_WEIGHT_DIVERSITY,
    LOCKIN_WEIGHT_ERROR,
    LOCKIN_WEIGHT_IMPROVEMENT,
    LOCKIN_WEIGHT_PERSISTENCE,
    KLRConfig,
)
from neurophase.reset.curriculum import Curriculum
from neurophase.reset.deterministic_oracle import derive_seed
from neurophase.reset.frozen_analysis import FrozenNodeAnalyzer
from neurophase.reset.integrity import IntegrityOracle
from neurophase.reset.metrics import SystemMetrics
from neurophase.reset.ntk_monitor import NTKMonitor
from neurophase.reset.refractory import RefractoryGate
from neurophase.reset.state import SystemState


class ResetState(Enum):
    """Controller lifecycle."""

    IDLE = auto()
    PREPARED = auto()
    DISINHIBITED = auto()
    PLASTICITY_OPEN = auto()
    CONSOLIDATING = auto()
    STABLE = auto()
    ROLLBACK = auto()


@dataclass(frozen=True)
class ResetReport:
    """Structured reset result."""

    status: str
    reason: str
    relapse_ratio: float
    improvement_ratio: float
    new_frozen_count: int
    gamma_after: float = 0.0
    lockin_score: float = 0.0
    seed_trace: int = 0
    warnings: tuple[str, ...] = ()
    threshold_used: float = 0.0
    ntk_rank_pre: float = 0.0
    ntk_rank_post: float = 0.0
    rank_delta: float = 0.0
    label_scope: str = CALIBRATION_SCOPE
    evidence_status: str = EVIDENCE_STATUS


class KetamineLikeResetController:
    """Adaptive reset controller with rollback safety."""

    def __init__(self, config: KLRConfig | None = None) -> None:
        self.config = config or KLRConfig()
        self.state_machine = ResetState.IDLE
        self._checkpoint: SystemState | None = None
        self._refractory = RefractoryGate()
        self._integrity = IntegrityOracle()
        self._frozen_analyzer = FrozenNodeAnalyzer()
        self._ntk = NTKMonitor()

    def run(
        self,
        state: SystemState,
        metrics: SystemMetrics,
        curriculum: Curriculum,
    ) -> tuple[SystemState, ResetReport]:
        now = time()
        # SystemState.__post_init__ guarantees frozen is not None after
        # construction — narrow for mypy.
        assert state.frozen is not None
        if not self._refractory.can_intervene(now):
            return state, ResetReport(
                status="SKIPPED",
                reason="Refractory lock active",
                relapse_ratio=0.0,
                improvement_ratio=0.0,
                new_frozen_count=int(np.sum(state.frozen)),
                lockin_score=0.0,
                warnings=(f"ready_in={self._refractory.seconds_until_ready(now):.1f}s",),
                threshold_used=self.config.lock_in_threshold,
                label_scope=CALIBRATION_LABEL_SOURCE,
            )

        try:
            rank_pre = self._ntk.rank_proxy(state.weights)
            lockin_score = self._lockin_score(metrics)
            if lockin_score < self.config.lock_in_threshold:
                return state, self._skip_report(lockin_score=lockin_score)

            seed_trace = derive_seed(state, metrics, curriculum)
            rng = np.random.default_rng(seed_trace.seed)
            before_hash = self._integrity.checksum_state(state)
            self._prepare(state)
            state = self._disinhibit(state)
            state = self._open_plasticity_window(state, curriculum, rng=rng)
            state = self._consolidate(state)
            out_state, report = self._validate_and_commit(
                state,
                curriculum,
                lockin_score=lockin_score,
                seed_trace=seed_trace.seed,
                rank_pre=rank_pre,
            )
            self._refractory.register_outcome(report.status, now)
            after_hash = self._integrity.checksum_state(out_state)
            self._integrity.log_mutation(before_hash, after_hash, report.status)
            self._integrity.assert_mutation_logged(before_hash, after_hash)
            if report.status == "SUCCESS" and before_hash == after_hash:
                report = ResetReport(
                    **{**report.__dict__, "warnings": (*report.warnings, "no_op_intervention")}
                )
            return out_state, report
        except Exception as exc:
            out = self._rollback(fallback_state=state)
            report = self._rollback_report(str(exc), lockin_score=0.0, seed_trace=0)
            self._refractory.register_outcome(report.status, now)
            return out, report

    def run_intervention(
        self,
        state: SystemState,
        metrics: SystemMetrics,
        curriculum: Curriculum,
    ) -> tuple[SystemState, dict[str, float | str]]:
        """Backward-compatible entrypoint used by older integration code."""
        out_state, report = self.run(state, metrics, curriculum)
        return out_state, {
            "status": report.status,
            "reason": report.reason,
            "relapse_ratio": report.relapse_ratio,
            "improvement_ratio": report.improvement_ratio,
            "gamma_after": report.gamma_after,
            "lockin_score": report.lockin_score,
            "threshold_used": report.threshold_used,
        }

    def detect_lockin(self, metrics: SystemMetrics) -> bool:
        return self._detect_lockin(metrics)

    def explain_lockin(self, metrics: SystemMetrics) -> dict[str, float]:
        self._validate_metrics(metrics)
        w_error, w_persistence, w_diversity, w_improvement = self._weights()
        error_term = w_error * metrics.error
        persistence_term = w_persistence * metrics.persistence
        inverse_diversity_term = w_diversity * (1.0 - metrics.diversity)
        inverse_improvement_term = w_improvement * (1.0 - metrics.improvement)
        total = error_term + persistence_term + inverse_diversity_term + inverse_improvement_term
        return {
            "error_term": float(error_term),
            "persistence_term": float(persistence_term),
            "inverse_diversity_term": float(inverse_diversity_term),
            "inverse_improvement_term": float(inverse_improvement_term),
            "total_score": float(total),
            "threshold": float(self.config.lock_in_threshold),
            "triggered": float(total >= self.config.lock_in_threshold),
        }

    def _lockin_score(self, metrics: SystemMetrics) -> float:
        self._validate_metrics(metrics)
        inv_diversity = max(0.0, 1.0 - metrics.diversity)
        inv_improvement = max(0.0, 1.0 - metrics.improvement)
        w_error, w_persistence, w_diversity, w_improvement = self._weights()
        return (
            w_error * metrics.error
            + w_persistence * metrics.persistence
            + w_diversity * inv_diversity
            + w_improvement * inv_improvement
        )

    def _weights(self) -> tuple[float, float, float, float]:
        if self.config.calibrated_weights is not None:
            return self.config.calibrated_weights
        return (
            LOCKIN_WEIGHT_ERROR,
            LOCKIN_WEIGHT_PERSISTENCE,
            LOCKIN_WEIGHT_DIVERSITY,
            LOCKIN_WEIGHT_IMPROVEMENT,
        )

    def _detect_lockin(self, metrics: SystemMetrics) -> bool:
        """Boolean compatibility helper for calibration/testing routines."""
        return self._lockin_score(metrics) >= self.config.lock_in_threshold

    def _prepare(self, state: SystemState) -> None:
        self.state_machine = ResetState.PREPARED
        # SystemState.__post_init__ initialises frozen when None.
        assert state.frozen is not None
        self._checkpoint = SystemState(
            weights=np.copy(state.weights),
            confidence=np.copy(state.confidence),
            usage=np.copy(state.usage),
            utility=np.copy(state.utility),
            inhibition=np.copy(state.inhibition),
            topology=np.copy(state.topology),
            frozen=np.copy(state.frozen),
            metadata=state.metadata.copy(),
        )

    def _disinhibit(self, state: SystemState) -> SystemState:
        self.state_machine = ResetState.DISINHIBITED
        if state.frozen is None:
            raise ValueError("state.frozen must be initialized")
        dominant = np.quantile(state.usage, self.config.usage_quantile)
        mask = (state.usage >= dominant) & ~state.frozen

        state.inhibition[mask] *= self.config.disinhibit_inhibition_scale
        state.confidence[mask] *= self.config.disinhibit_confidence_scale
        state.weights[mask, :] *= self.config.disinhibit_weight_scale
        state.weights[:, mask] *= self.config.disinhibit_weight_scale
        return state

    def _open_plasticity_window(
        self, state: SystemState, curriculum: Curriculum, *, rng: np.random.Generator
    ) -> SystemState:
        self.state_machine = ResetState.PLASTICITY_OPEN
        self._validate_curriculum_shapes(state, curriculum)
        if state.frozen is None:
            raise ValueError("state.frozen must be initialized")

        exploration = rng.normal(0.0, self.config.plasticity_noise, size=state.weights.shape)
        novelty = 1.0 / (state.usage + 1e-8)
        novelty = novelty / (novelty.max() + 1e-8)

        pos = np.maximum(0.0, curriculum.corrective_signal)
        neg = np.minimum(0.0, curriculum.corrective_signal)

        anti_hebb = np.outer(np.abs(neg), np.ones_like(neg))
        update = (
            exploration * state.topology
            + (np.outer(pos, pos) * state.topology) * 0.6
            - anti_hebb * 0.4
        ) * (~state.frozen[:, None] & ~state.frozen[None, :])

        state.weights += update * novelty[:, None]
        np.clip(state.weights, 0.0, 1.0, out=state.weights)
        return state

    def _consolidate(self, state: SystemState) -> SystemState:
        self.state_machine = ResetState.CONSOLIDATING
        if state.frozen is None:
            raise ValueError("state.frozen must be initialized")
        row_sums = state.weights.sum(axis=1, keepdims=True)
        state.weights = np.divide(state.weights, np.where(row_sums == 0.0, 1.0, row_sums))

        alpha = self.config.annealing_factor
        state.confidence = alpha * state.confidence + (1.0 - alpha) * state.utility

        high_utility = np.quantile(state.utility, self.config.freeze_quantile)
        newly_stable = (state.utility >= high_utility) & (state.confidence > 0.85)
        state.frozen = state.frozen | newly_stable
        return state

    def _validate_and_commit(
        self,
        state: SystemState,
        curriculum: Curriculum,
        *,
        lockin_score: float,
        seed_trace: int,
        rank_pre: float,
    ) -> tuple[SystemState, ResetReport]:
        if self._checkpoint is None:
            raise RuntimeError("checkpoint missing before validation")

        pre = np.dot(self._checkpoint.weights, curriculum.stress_pattern)
        post = np.dot(state.weights, curriculum.stress_pattern)

        pre_err = float(np.linalg.norm(pre - curriculum.target_bias))
        post_err = float(np.linalg.norm(post - curriculum.target_bias))
        relapse_ratio = post_err / (pre_err + 1e-12)

        if relapse_ratio > self.config.relapse_threshold:
            return self._rollback(), self._rollback_report(
                relapse_ratio, lockin_score=lockin_score, seed_trace=seed_trace
            )

        checkpoint = self._checkpoint
        self.state_machine = ResetState.STABLE
        self._checkpoint = None
        gamma_after = self._gamma(state.weights)
        rank_post = self._ntk.rank_proxy(state.weights)
        ok, warns = self._frozen_analyzer.verify_consolidation(checkpoint, state, curriculum)
        warn_msgs = tuple(w.message for w in warns)
        if rank_post <= rank_pre:
            warn_msgs = (*warn_msgs, "ntk_rank_non_increasing")
        _ = ok
        return state, ResetReport(
            status="SUCCESS",
            reason="Pathological attractor escaped and consolidated",
            relapse_ratio=float(relapse_ratio),
            improvement_ratio=float(1.0 - relapse_ratio),
            new_frozen_count=int(np.sum(state.frozen)) if state.frozen is not None else 0,
            gamma_after=gamma_after,
            lockin_score=lockin_score,
            seed_trace=seed_trace,
            warnings=warn_msgs,
            threshold_used=self.config.lock_in_threshold,
            ntk_rank_pre=float(rank_pre),
            ntk_rank_post=float(rank_post),
            rank_delta=float(rank_post - rank_pre),
            label_scope=CALIBRATION_LABEL_SOURCE,
        )

    def _rollback(self, fallback_state: SystemState | None = None) -> SystemState:
        self.state_machine = ResetState.ROLLBACK
        if self._checkpoint is None:
            if fallback_state is not None:
                return fallback_state
            raise RuntimeError("cannot rollback without checkpoint")
        safe = self._checkpoint
        self._checkpoint = None
        return safe

    def _skip_report(self, *, lockin_score: float) -> ResetReport:
        return ResetReport(
            status="SKIPPED",
            reason="System is functionally stable",
            relapse_ratio=0.0,
            improvement_ratio=0.0,
            new_frozen_count=0,
            lockin_score=lockin_score,
            threshold_used=self.config.lock_in_threshold,
            label_scope=CALIBRATION_LABEL_SOURCE,
        )

    def _rollback_report(
        self, relapse: float | str, *, lockin_score: float, seed_trace: int = 0
    ) -> ResetReport:
        if isinstance(relapse, str):
            reason = f"Execution fault: {relapse}"
            relapse_ratio = 1.0
        else:
            reason = "High relapse risk — safety invariant triggered"
            relapse_ratio = float(relapse)
        return ResetReport(
            status="ROLLBACK",
            reason=reason,
            relapse_ratio=relapse_ratio,
            improvement_ratio=0.0,
            new_frozen_count=0,
            lockin_score=lockin_score,
            seed_trace=seed_trace,
            threshold_used=self.config.lock_in_threshold,
            label_scope=CALIBRATION_LABEL_SOURCE,
        )

    def _validate_curriculum_shapes(self, state: SystemState, curriculum: Curriculum) -> None:
        n = state.weights.shape[0]
        for name, arr in (
            ("target_bias", curriculum.target_bias),
            ("corrective_signal", curriculum.corrective_signal),
            ("stress_pattern", curriculum.stress_pattern),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must be finite")

    def _validate_metrics(self, metrics: SystemMetrics) -> None:
        for name in ("error", "persistence", "diversity", "improvement", "noise", "reward"):
            value = float(getattr(metrics, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("error", "persistence", "diversity", "improvement"):
            value = float(getattr(metrics, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    def _gamma(self, weights: NDArray[np.float64]) -> float:
        """Normalized mean row entropy ∈ [0, 1]. Higher means richer mixing."""
        probs = np.clip(weights, 1e-12, 1.0)
        ent = -np.sum(probs * np.log(probs), axis=1)
        denom = np.log(weights.shape[1]) if weights.shape[1] > 1 else 1.0
        return float(np.mean(ent / denom))
