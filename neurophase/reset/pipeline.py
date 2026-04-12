"""Unified KLR pipeline orchestration API."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurophase.reset.adaptive_threshold import AdaptiveThreshold
from neurophase.reset.config import KLRConfig
from neurophase.reset.controller import KetamineLikeResetController, ResetReport
from neurophase.reset.curriculum import Curriculum
from neurophase.reset.gamma_witness import GammaWitness, GammaWitnessReport
from neurophase.reset.integrity import IntegrityOracle
from neurophase.reset.ledger import RollbackLedger
from neurophase.reset.metrics import SystemMetrics
from neurophase.reset.ntk_monitor import NTKMonitor
from neurophase.reset.plasticity_injector import PlasticityInjector
from neurophase.reset.plasticity_monitor import PlasticityMonitor, PlasticityReport
from neurophase.reset.refractory import RefractoryGate
from neurophase.reset.state import SystemState, clone_state
from neurophase.reset.twin_state import TwinStateManager


@dataclass(frozen=True)
class KLRFrame:
    decision: str
    ntk_rank_delta: float
    report: ResetReport
    plasticity_report: PlasticityReport
    witness_report: GammaWitnessReport | None = None


class KLRPipeline:
    def __init__(
        self,
        initial_state: SystemState,
        config: KLRConfig | None = None,
        *,
        enable_witness: bool = True,
    ) -> None:
        self.config = config or KLRConfig()
        self.controller = KetamineLikeResetController(self.config)
        self.adaptive_threshold = AdaptiveThreshold(
            fallback_threshold=self.config.lock_in_threshold
        )
        self.ledger = RollbackLedger()
        self.refractory = RefractoryGate()
        self.integrity = IntegrityOracle()
        self.ntk_monitor = NTKMonitor()
        self.plasticity_injector = PlasticityInjector()
        self.plasticity_monitor = PlasticityMonitor()
        self.twin_state = TwinStateManager(active=initial_state, passive=clone_state(initial_state))
        self.gamma_witness: GammaWitness | None = GammaWitness() if enable_witness else None
        self.step_counter = 0

    def auto_curriculum(self) -> Curriculum:
        state = self.twin_state.active
        target_bias = state.utility / (float(np.sum(state.utility)) + 1e-8)
        corrective_signal = state.utility - float(np.mean(state.utility))
        stress_pattern = state.usage / (float(np.sum(state.usage)) + 1e-8)
        return Curriculum(
            target_bias=target_bias,
            corrective_signal=corrective_signal,
            stress_pattern=stress_pattern,
        )

    def tick(self, metrics: SystemMetrics) -> KLRFrame:
        self.step_counter += 1
        active = self.twin_state.active
        _ = self.integrity.checksum_state(active)
        refractory_active = not self.refractory.can_intervene(float(self.step_counter))
        rank_pre = self.ntk_monitor.rank_proxy(active.weights)

        # Advisory γ-verification witness (NEO-I1 / NEO-I2). Observe before
        # any mutating component so the snapshot is the pre-intervention state.
        witness_report: GammaWitnessReport | None = None
        if self.gamma_witness is not None:
            try:
                witness_report = self.gamma_witness.observe(active)
            except Exception:
                witness_report = None

        injection_triggered = False
        if not refractory_active:
            injection_triggered = self.plasticity_injector.maybe_inject(
                active,
                ntk_rank_normalized=rank_pre,
                config=self.config,
            )

        decision = "SKIPPED"
        report: ResetReport
        try:
            if self.refractory.can_intervene(float(self.step_counter)):
                self.adaptive_threshold.freeze()
                threshold = self.adaptive_threshold.current()
                run_cfg = self.config
                run_cfg = KLRConfig(**{**run_cfg.__dict__, "lock_in_threshold": threshold})
                controller = KetamineLikeResetController(run_cfg)
                out_state, report = controller.run(active, metrics, self.auto_curriculum())
                self.twin_state.active = out_state
                self.ledger.record_attempt(out_state, metrics, report)
                self.adaptive_threshold.unfreeze()
                self.adaptive_threshold.update(self.ledger._entries)
                self.refractory.register_outcome(report.status, float(self.step_counter))
                decision = report.status
            else:
                report = ResetReport(
                    status="SKIPPED",
                    reason="Refractory lock active",
                    relapse_ratio=0.0,
                    improvement_ratio=0.0,
                    new_frozen_count=int(np.sum(active.frozen)) if active.frozen is not None else 0,
                )
        except Exception as exc:
            report = ResetReport(
                status="ROLLBACK",
                reason=f"pipeline_exception: {exc}",
                relapse_ratio=1.0,
                improvement_ratio=0.0,
                new_frozen_count=0,
            )
            decision = report.status

        rank_post = self.ntk_monitor.rank_proxy(self.twin_state.active.weights)
        _ = self.twin_state.tick(self.ledger._entries, refractory_active)
        plast_report = self.plasticity_monitor.compute(
            self.twin_state.active.usage,
            self.twin_state.active.frozen
            if self.twin_state.active.frozen is not None
            else np.zeros(0, dtype=bool),
            rank_post,
            plasticity_floor=self.config.plasticity_floor,
            injection_triggered=injection_triggered,
        )

        return KLRFrame(
            decision=decision,
            ntk_rank_delta=float(rank_post - rank_pre),
            report=report,
            plasticity_report=plast_report,
            witness_report=witness_report,
        )
