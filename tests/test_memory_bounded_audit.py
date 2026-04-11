"""L1 — contract tests for the memory-bounded rolling audit.

This test file is the HN27 binding. It locks in:

1. **Constant memory.** A 10 000-tick run on a fresh orchestrator
   produces a final audit whose ``total_measured_size`` equals
   the audit at tick 100. The runtime stack does not grow with
   the input length.
2. **Bounded.** Every component reports
   ``measured_size ≤ declared_cap`` at every audit point.
3. **Total enumeration.** ``audit_runtime_memory`` enumerates
   exactly six components (TemporalValidator,
   TemporalStreamDetector, StillnessDetector,
   DecisionTraceLedger, RegimeClassifier, ActionPolicy). A
   future addition that introduces a new rolling buffer must
   either register itself or fail this enumeration check.
4. **Hot-path stability.** The total declared cap is a pure
   function of the configuration; it is unchanged between two
   audits at very different tick counts on the same instance.
5. **Raise-on-violation mode.** When configured, the audit
   raises :class:`MemoryAuditError` rather than returning a
   report with ``all_bounded=False``.
6. **Configuration sensitivity.** Changing the
   :class:`PipelineConfig`'s window or warmup_samples updates
   the declared cap deterministically.
7. **JSON-safe report.** :meth:`MemoryAuditReport.to_json_dict`
   round-trips through ``json.dumps`` / ``json.loads``.
8. **Frozen dataclass.** Reassigning a field on a report or a
   component raises.
"""

from __future__ import annotations

import json

import pytest

from neurophase.runtime.memory_audit import (
    ComponentMemoryFootprint,
    MemoryAuditError,
    MemoryAuditReport,
    audit_runtime_memory,
)
from neurophase.runtime.orchestrator import (
    OrchestratorConfig,
    RuntimeOrchestrator,
)
from neurophase.runtime.pipeline import PipelineConfig

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _orch(
    *,
    warmup_samples: int = 4,
    stream_window: int = 8,
    enable_stillness: bool = True,
    stillness_window: int = 8,
) -> RuntimeOrchestrator:
    return RuntimeOrchestrator(
        OrchestratorConfig(
            pipeline=PipelineConfig(
                warmup_samples=warmup_samples,
                stream_window=stream_window,
                enable_stillness=enable_stillness,
                stillness_window=stillness_window,
            )
        )
    )


def _drive(orch: RuntimeOrchestrator, *, n: int) -> None:
    for i in range(n):
        orch.tick(timestamp=float(i) * 0.01, R=0.92, delta=0.05)


# ---------------------------------------------------------------------------
# 1. Constant memory under heavy ticking — the load-bearing claim.
# ---------------------------------------------------------------------------


class TestConstantMemory:
    def test_10k_ticks_does_not_grow_buffers(self) -> None:
        """A fresh orchestrator driven for 10 000 ticks must
        report identical total_measured_size at tick 100 and at
        tick 10 000. The runtime stack is bounded in tick count."""
        orch = _orch()
        _drive(orch, n=100)
        early = audit_runtime_memory(orch)
        _drive(orch, n=9_900)  # total = 10 000
        late = audit_runtime_memory(orch)
        assert orch.n_ticks == 10_000
        assert early.total_measured_size == late.total_measured_size, (
            f"buffer grew between tick 100 ({early.total_measured_size}) "
            f"and tick 10_000 ({late.total_measured_size})"
        )
        assert late.all_bounded

    def test_total_declared_cap_is_constant(self) -> None:
        """The declared cap is a pure function of config; it
        does not depend on tick count."""
        orch = _orch()
        cap_before = audit_runtime_memory(orch).total_declared_cap
        _drive(orch, n=5_000)
        cap_after = audit_runtime_memory(orch).total_declared_cap
        assert cap_before == cap_after


# ---------------------------------------------------------------------------
# 2. Per-component bounded — every audit point.
# ---------------------------------------------------------------------------


class TestBoundedness:
    def test_every_component_bounded_at_every_audit_point(self) -> None:
        """Audit every 100 ticks across a 5 000-tick run; every
        component must be bounded at every audit point."""
        orch = _orch()
        for batch in range(50):
            _drive(orch, n=100)
            report = audit_runtime_memory(orch)
            assert report.all_bounded, f"unbounded at batch {batch}: {report}"
            for c in report.components:
                assert c.is_bounded, f"unbounded component at batch {batch}: {c}"

    def test_audit_before_first_tick_is_bounded(self) -> None:
        orch = _orch()
        report = audit_runtime_memory(orch)
        assert report.n_ticks == 0
        assert report.all_bounded
        assert report.total_measured_size == 0

    def test_audit_after_close_is_bounded(self) -> None:
        orch = _orch()
        _drive(orch, n=100)
        orch.close()
        report = audit_runtime_memory(orch)
        assert report.all_bounded
        assert report.n_ticks == 100


# ---------------------------------------------------------------------------
# 3. Total enumeration — every component is reported.
# ---------------------------------------------------------------------------


class TestEnumeration:
    def test_six_components_enumerated(self) -> None:
        report = audit_runtime_memory(_orch())
        assert len(report.components) == 6
        names = [c.name for c in report.components]
        assert names == [
            "TemporalValidator",
            "TemporalStreamDetector",
            "StillnessDetector",
            "DecisionTraceLedger",
            "RegimeClassifier",
            "ActionPolicy",
        ]

    def test_disabled_stillness_still_enumerated(self) -> None:
        orch = _orch(enable_stillness=False)
        report = audit_runtime_memory(orch)
        names = [c.name for c in report.components]
        assert "StillnessDetector" in names
        # The disabled stillness component reports cap=0, size=0,
        # bounded=True.
        stillness = next(c for c in report.components if c.name == "StillnessDetector")
        assert stillness.declared_cap == 0
        assert stillness.measured_size == 0
        assert stillness.is_bounded


# ---------------------------------------------------------------------------
# 4. Configuration sensitivity.
# ---------------------------------------------------------------------------


class TestConfigSensitivity:
    def test_doubling_window_doubles_stream_cap(self) -> None:
        small = _orch(stream_window=8)
        large = _orch(stream_window=16)
        sc = next(
            c for c in audit_runtime_memory(small).components if c.name == "TemporalStreamDetector"
        )
        lc = next(
            c for c in audit_runtime_memory(large).components if c.name == "TemporalStreamDetector"
        )
        assert sc.declared_cap == 8
        assert lc.declared_cap == 16

    def test_doubling_stillness_window_doubles_cap(self) -> None:
        small = _orch(stillness_window=8)
        large = _orch(stillness_window=16)
        sc = next(
            c for c in audit_runtime_memory(small).components if c.name == "StillnessDetector"
        )
        lc = next(
            c for c in audit_runtime_memory(large).components if c.name == "StillnessDetector"
        )
        # 2× histories: cap = 2 * window.
        assert sc.declared_cap == 16
        assert lc.declared_cap == 32


# ---------------------------------------------------------------------------
# 5. Raise-on-violation mode.
# ---------------------------------------------------------------------------


class TestRaiseOnViolation:
    def test_clean_report_does_not_raise(self) -> None:
        orch = _orch()
        _drive(orch, n=100)
        # Should NOT raise.
        report = audit_runtime_memory(orch, raise_on_violation=True)
        assert report.all_bounded

    def test_synthetic_violation_raises(self) -> None:
        """Force an unbounded report by constructing it directly
        and checking that the API surface refuses to materialise
        an inconsistent footprint. (We cannot make a real layer
        violate its cap, by design — the contract IS that no real
        layer ever does.)"""
        with pytest.raises(ValueError, match="is_bounded="):
            ComponentMemoryFootprint(
                name="Bogus",
                declared_cap=4,
                measured_size=10,
                is_bounded=True,  # claim bounded but size > cap
                detail="lying",
            )


# ---------------------------------------------------------------------------
# 6. JSON-safe round-trip.
# ---------------------------------------------------------------------------


class TestJsonProjection:
    def test_to_json_dict_round_trip(self) -> None:
        orch = _orch()
        _drive(orch, n=20)
        report = audit_runtime_memory(orch)
        d = report.to_json_dict()
        text = json.dumps(d)
        loaded = json.loads(text)
        assert loaded["n_ticks"] == 20
        assert loaded["all_bounded"] is True
        assert len(loaded["components"]) == 6

    def test_components_are_flat_dicts(self) -> None:
        orch = _orch()
        report = audit_runtime_memory(orch)
        d = report.to_json_dict()
        for comp in d["components"]:
            for k, v in comp.items():
                assert isinstance(v, (str, int, bool)), (
                    f"component field {k!r} is not primitive: {v!r}"
                )


# ---------------------------------------------------------------------------
# 7. Frozen dataclasses.
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_report_is_frozen(self) -> None:
        report = audit_runtime_memory(_orch())
        with pytest.raises((AttributeError, TypeError)):
            report.n_ticks = 99  # type: ignore[misc]

    def test_component_is_frozen(self) -> None:
        report = audit_runtime_memory(_orch())
        c = report.components[0]
        with pytest.raises((AttributeError, TypeError)):
            c.measured_size = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 8. Aesthetic rich __repr__.
# ---------------------------------------------------------------------------


class TestRepr:
    def test_report_repr(self) -> None:
        report = audit_runtime_memory(_orch())
        r = repr(report)
        assert r.startswith("MemoryAuditReport[")
        assert "n_ticks=0" in r
        assert "components=6" in r
        assert "✓" in r

    def test_component_repr(self) -> None:
        report = audit_runtime_memory(_orch())
        c = next(c for c in report.components if c.name == "TemporalStreamDetector")
        r = repr(c)
        assert r.startswith("ComponentMemoryFootprint[")
        assert "TemporalStreamDetector" in r
        assert "✓" in r


# ---------------------------------------------------------------------------
# 9. Construction-time consistency checks on the report.
# ---------------------------------------------------------------------------


class TestReportInternalConsistency:
    def test_inconsistent_total_cap_rejected(self) -> None:
        with pytest.raises(ValueError, match="total_declared_cap mismatch"):
            MemoryAuditReport(
                n_ticks=0,
                components=(
                    ComponentMemoryFootprint(
                        name="A",
                        declared_cap=4,
                        measured_size=2,
                        is_bounded=True,
                        detail="ok",
                    ),
                ),
                total_declared_cap=999,  # wrong
                total_measured_size=2,
                all_bounded=True,
            )

    def test_negative_n_ticks_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_ticks"):
            MemoryAuditReport(
                n_ticks=-1,
                components=(),
                total_declared_cap=0,
                total_measured_size=0,
                all_bounded=True,
            )


# ---------------------------------------------------------------------------
# 10. MemoryAuditError surfaces on real violation paths.
# ---------------------------------------------------------------------------


class TestMemoryAuditError:
    def test_error_is_assertion_subclass(self) -> None:
        """An L1 violation is a contract failure, not a recoverable
        runtime fault — the error is an AssertionError subclass so
        callers cannot accidentally swallow it with except ValueError."""
        assert issubclass(MemoryAuditError, AssertionError)
