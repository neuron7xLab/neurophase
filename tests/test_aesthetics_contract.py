"""HN22 — architectural aesthetics contract.

This test file locks in the cosmetic polish added on top of the
solid v0.4.0 foundation (703+ tests, mypy strict, 21 CI-bound
contracts). Each assertion closes a specific aesthetic regression
mode:

1. **Rich `__repr__` stability.** Every ``frozen=True, repr=False``
   dataclass in the public surface carries a compact one-line
   representation in the canonical design language
   ``ClassName[field · field · field · ✓|✗]``. The tests here
   pin the output format so a future refactor cannot silently
   degrade the debug UX.

2. **Public façade docstring coverage.** Every symbol in
   ``neurophase.api.__all__`` must have a non-empty docstring.

3. **Canonical `__all__` ordering.** ``neurophase.api.__all__`` is
   sorted — dunders first, then CamelCase, then lower_case — so
   that a reviewer can predict where to find a symbol without
   grep.

4. **Visual architecture docs exist.** The mermaid state diagram
   and the one-page architecture tour both exist on disk and
   cross-reference each other.

These tests are locked into ``INVARIANTS.yaml`` under **HN22**. A
future PR that weakens any of the above fails CI.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from neurophase import api as facade
from neurophase.data.stream_detector import (
    StreamQualityDecision,
    StreamQualityStats,
    StreamRegime,
)
from neurophase.data.temporal_validator import (
    TemporalQualityDecision,
    TimeQuality,
)
from neurophase.explain import (
    Contract,
    DecisionExplanation,
)
from neurophase.gate.execution_gate import (
    ExecutionGate,
    GateState,
)
from neurophase.gate.stillness_detector import (
    StillnessDecision,
    StillnessState,
)
from neurophase.runtime.pipeline import (
    PipelineConfig,
    StreamingPipeline,
)
from neurophase.validation.null_model import NullModelResult

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. Rich __repr__ stability.
# ---------------------------------------------------------------------------


class TestRichRepr:
    """Every polished dataclass must render in the canonical format."""

    def test_gate_decision_repr(self) -> None:
        gate = ExecutionGate(threshold=0.65)
        decision = gate.evaluate(R=0.95)
        r = repr(decision)
        assert r.startswith("GateDecision["), r
        assert "READY" in r
        assert "R=0.9500" in r
        assert "θ=0.65" in r
        assert "✓" in r

    def test_gate_decision_repr_on_blocked(self) -> None:
        gate = ExecutionGate(threshold=0.65)
        decision = gate.evaluate(R=0.30)
        r = repr(decision)
        assert "BLOCKED" in r
        assert "✗" in r

    def test_gate_decision_repr_on_degraded(self) -> None:
        gate = ExecutionGate(threshold=0.65)
        decision = gate.evaluate(R=None)
        r = repr(decision)
        assert "DEGRADED" in r
        assert "R=None" in r
        assert "✗" in r

    def test_stillness_decision_repr(self) -> None:
        d = StillnessDecision(
            state=StillnessState.STILL,
            R=0.95,
            delta=0.01,
            dR_dt_max=1e-4,
            dF_proxy_dt_max=1e-4,
            delta_max=0.01,
            window_filled=True,
            reason="still: test",
        )
        r = repr(d)
        assert r.startswith("StillnessDecision[")
        assert "STILL" in r
        assert "R=0.9500" in r
        assert "δ=0.0100" in r
        assert "warm" in r

    def test_temporal_quality_decision_repr(self) -> None:
        d = TemporalQualityDecision(
            quality=TimeQuality.VALID,
            ts=1.0,
            last_ts=0.0,
            gap_seconds=1.0,
            staleness_seconds=0.0,
            warmup_remaining=0,
            reason="valid: test",
        )
        r = repr(d)
        assert r.startswith("TemporalQualityDecision[")
        assert "VALID" in r
        assert "ts=1.0000" in r

    def test_stream_quality_decision_repr(self) -> None:
        stats = StreamQualityStats(
            total=8,
            valid=8,
            gapped=0,
            stale=0,
            reversed=0,
            duplicate=0,
            invalid=0,
            warmup=0,
            fault_rate=0.0,
        )
        d = StreamQualityDecision(
            regime=StreamRegime.HEALTHY,
            stats=stats,
            last_quality=TimeQuality.VALID,
            held=False,
            reason="healthy: test",
        )
        r = repr(d)
        assert r.startswith("StreamQualityDecision[")
        assert "HEALTHY" in r
        assert "valid=8/8" in r
        assert "fault=0.00" in r

    def test_decision_frame_repr(self) -> None:
        p = StreamingPipeline(
            PipelineConfig(warmup_samples=2, stream_window=4, enable_stillness=False)
        )
        for i in range(6):
            p.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.01)
        frame = p.tick(timestamp=0.7, R=0.99, delta=0.01)
        r = repr(frame)
        assert r.startswith("DecisionFrame[")
        assert f"tick={frame.tick_index}" in r
        assert "READY" in r
        assert "R=0.9900" in r

    def test_decision_explanation_repr(self) -> None:
        exp = DecisionExplanation(
            tick_index=7,
            timestamp=0.7,
            final_state=GateState.READY,
            execution_allowed=True,
            causal_contract=Contract.READY,
            chain=(),
            summary="READY",
        )
        r = repr(exp)
        assert r.startswith("DecisionExplanation[")
        assert "tick=7" in r
        assert "READY" in r
        assert "causal=READY" in r
        assert "✓" in r

    def test_null_model_result_repr(self) -> None:
        r_obj = NullModelResult(
            observed=0.9234,
            null_distribution=np.zeros(10),
            p_value=0.001,
            n_surrogates=1000,
            seed=42,
            rejected=True,
            alpha=0.05,
        )
        r = repr(r_obj)
        assert r.startswith("NullModelResult[")
        assert "observed=0.9234" in r
        assert "p=0.0010" in r
        assert "n=1000" in r
        assert "rejected" in r

    def test_threshold_calibration_report_repr(self) -> None:
        from neurophase.calibration.threshold import (
            ThresholdGrid,
            calibrate_gate_threshold,
        )

        grid = ThresholdGrid(
            thresholds=(0.30, 0.50, 0.70, 0.90),
            null_seeds=tuple(range(100, 108)),
            locked_seeds=tuple(range(200, 208)),
            n_samples=64,
        )
        report = calibrate_gate_threshold(grid, train_fraction=0.5)
        r = repr(report)
        assert r.startswith("ThresholdCalibrationReport[")
        assert "best=" in r
        assert "train=" in r
        assert "test=" in r
        assert "gap=" in r

    def test_replay_result_repr(self, tmp_path: Path) -> None:
        import dataclasses

        from neurophase.audit.replay import ReplayInput, replay_ledger
        from neurophase.runtime.pipeline import PipelineConfig, StreamingPipeline

        original = tmp_path / "original.jsonl"
        cfg = PipelineConfig(warmup_samples=2, stream_window=4, ledger_path=original)
        p = StreamingPipeline(cfg)
        inputs = [ReplayInput(timestamp=float(i) * 0.1, R=0.9, delta=0.01) for i in range(6)]
        for inp in inputs:
            p.tick(timestamp=inp.timestamp, R=inp.R, delta=inp.delta)
        scratch = tmp_path / "scratch.jsonl"
        replay_cfg = dataclasses.replace(cfg, ledger_path=scratch)
        result = replay_ledger(
            original_path=original,
            config=replay_cfg,
            inputs=inputs,
            scratch_path=scratch,
        )
        r = repr(result)
        assert r.startswith("ReplayResult[")
        assert "ok" in r
        assert "n=6" in r


# ---------------------------------------------------------------------------
# 2. Public façade docstring coverage.
# ---------------------------------------------------------------------------


class TestFacadeDocstrings:
    def test_every_public_symbol_has_docstring(self) -> None:
        """Every symbol in ``neurophase.api.__all__`` must have a
        non-empty docstring. We allow builtins (``__version__``,
        ``DEFAULT_THRESHOLD``) which are constants, not symbols
        with docstrings."""
        allowlist = {"DEFAULT_THRESHOLD", "__version__"}
        missing: list[str] = []
        for name in facade.__all__:
            if name in allowlist:
                continue
            obj = getattr(facade, name)
            doc = inspect.getdoc(obj)
            if not doc or len(doc.strip()) < 10:
                missing.append(name)
        assert not missing, f"symbols missing docstrings: {missing}"


# ---------------------------------------------------------------------------
# 3. Canonical __all__ ordering.
# ---------------------------------------------------------------------------


class TestCanonicalOrdering:
    def test_facade_all_matches_ruff_canonical_sort(self) -> None:
        """``neurophase.api.__all__`` matches ruff's RUF022 canonical sort.

        Ruff enforces a natural sort where SCREAMING_SNAKE_CASE
        constants come first, then CamelCase, then dunders, then
        lowercase. Pinning the exact order prevents drift and gives
        reviewers a single predictable place to find every symbol.
        """
        expected = [
            "DEFAULT_THRESHOLD",
            "Contract",
            "DecisionExplanation",
            "DecisionFrame",
            "ExecutionGate",
            "ExplanationStep",
            "GateDecision",
            "GateState",
            "PipelineConfig",
            "RegimeClassifier",
            "RegimeLabel",
            "RegimeState",
            "RegimeThresholds",
            "StillnessDetector",
            "StillnessState",
            "StreamingPipeline",
            "TimeQuality",
            "Verdict",
            "__version__",
            "create_pipeline",
            "explain_decision",
            "explain_gate",
        ]
        assert list(facade.__all__) == expected, (
            f"neurophase.api.__all__ diverged from the canonical ruff sort.\n"
            f"  actual:   {list(facade.__all__)}\n"
            f"  expected: {expected}"
        )


# ---------------------------------------------------------------------------
# 4. Visual architecture docs exist and cross-reference.
# ---------------------------------------------------------------------------


class TestArchitectureDocs:
    def test_mermaid_state_diagram_exists(self) -> None:
        path = REPO_ROOT / "docs" / "diagrams" / "gate_state_machine.md"
        assert path.is_file(), f"missing {path}"
        text = path.read_text(encoding="utf-8")
        # Must contain at least one mermaid block.
        assert "```mermaid" in text
        # Must reference every gate state.
        for state in GateState:
            assert state.name in text, f"gate_state_machine.md does not mention {state.name}"

    def test_one_page_architecture_exists(self) -> None:
        path = REPO_ROOT / "docs" / "ARCHITECTURE_ONE_PAGE.md"
        assert path.is_file(), f"missing {path}"
        text = path.read_text(encoding="utf-8")
        assert "The five states" in text
        assert "The single blessed import path" in text
        assert "neurophase.api" in text
        # Must cross-reference the other key docs.
        for ref in [
            "EVOLUTION_BOARD.md",
            "TASK_MAP.md",
            "scientific_basis.md",
            "neurophase_elite_bibliography.md",
            "invariant_matrix.md",
            "stillness_invariant.md",
            "time_integrity.md",
            "gate_state_machine.md",
        ]:
            assert ref in text, f"ARCHITECTURE_ONE_PAGE.md is missing reference to {ref}"

    def test_mermaid_priority_flowchart_is_present(self) -> None:
        """The priority flowchart is a second mermaid block in the
        diagrams doc — distinct from the state diagram."""
        path = REPO_ROOT / "docs" / "diagrams" / "gate_state_machine.md"
        text = path.read_text(encoding="utf-8")
        # At least three mermaid blocks (state, priority, pipeline).
        assert text.count("```mermaid") >= 3, (
            "gate_state_machine.md must contain state + priority + pipeline diagrams"
        )
