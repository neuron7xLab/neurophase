"""HN32 — Seventh axis (Resistance) contract tests.

The first six formal review axes describe how the system *reads*.
The seventh axis describes how it *holds under attack*. This test
file binds every :class:`~neurophase.governance.resistance.ResistanceScenario`
to an individual pytest node id so a single failing adversarial
scenario is a merge block.

Locks in:

1. **Gate never widens under adversarial KLR.** A fake KLR
   lying about every frame as SUCCESS with positive rank delta
   cannot alter the gate sequence relative to a clean pipeline.
2. **Memory bounded under 10 000-tick endurance load.** L1 law
   holds after a decade of ticks.
3. **Ledger tamper is detected.** A single-byte edit of any
   committed ledger record fails :func:`verify_ledger`.
4. **Claim forgery rejected.** A ``CLAIMS.yaml`` declaring
   status=FACT with < 3 supporting citations is refused at
   load time.
5. **KLR curriculum shape attack collapses to ROLLBACK.** A
   mis-shaped curriculum vector cannot yield SUCCESS.
6. **Monograph drift is caught by the regeneration guard.** A
   hand-edit that inserts non-generated content cannot
   impersonate the generator.

These are **composition-level** tests — each scenario only
breaks if multiple layers cooperate to fail, so a green
HN32 row is a strong claim about the whole stack.
"""

from __future__ import annotations

from neurophase.governance.resistance import (
    SEVENTH_AXIS_SCENARIOS,
    ResistanceReport,
    ResistanceScenario,
    ResistanceSuite,
)


def test_suite_enumerates_six_scenarios() -> None:
    """The registry is stable at exactly six scenarios. Adding or
    removing a scenario without updating HN32 bindings is a contract
    violation."""
    assert len(SEVENTH_AXIS_SCENARIOS) == 6
    ids = {s.id for s in SEVENTH_AXIS_SCENARIOS}
    assert ids == {
        "GATE_NEVER_WIDENS",
        "MEMORY_BOUNDED_10K",
        "LEDGER_TAMPER_DETECTED",
        "CLAIM_FORGERY_REJECTED",
        "CURRICULUM_SHAPE_CORRUPTION",
        "MONOGRAPH_DRIFT",
    }


def test_resistance_scenario_is_frozen() -> None:
    """ResistanceScenario + ResistanceReport are frozen dataclasses."""
    s = SEVENTH_AXIS_SCENARIOS[0]
    assert isinstance(s, ResistanceScenario)
    # id, statement, run must not be mutated after construction.
    try:
        s.id = "bogus"  # type: ignore[misc]
    except (AttributeError, TypeError):
        pass
    else:
        raise AssertionError("ResistanceScenario must be frozen")


def test_gate_never_widens_under_adversarial_klr() -> None:
    """Scenario 1: RT-KLR-I1 survives a KLR that lies about every frame."""
    report = ResistanceSuite().run_one("GATE_NEVER_WIDENS")
    assert isinstance(report, ResistanceReport)
    assert report.passed, report.detail


def test_memory_bounded_under_10k_ticks() -> None:
    """Scenario 2: L1 survives 10 000 ticks of endurance load."""
    report = ResistanceSuite().run_one("MEMORY_BOUNDED_10K")
    assert report.passed, report.detail


def test_ledger_tamper_detected() -> None:
    """Scenario 3: F1 detects a single-byte tamper at the exact index."""
    report = ResistanceSuite().run_one("LEDGER_TAMPER_DETECTED")
    assert report.passed, report.detail
    # The detail line must carry the broken index.
    assert "index" in report.detail


def test_claim_forgery_rejected() -> None:
    """Scenario 4: HN30 mechanical promotion rule rejects a forged FACT."""
    report = ResistanceSuite().run_one("CLAIM_FORGERY_REJECTED")
    assert report.passed, report.detail


def test_curriculum_shape_corruption_rollback() -> None:
    """Scenario 5: KLR shape-mismatched curriculum cannot reach SUCCESS."""
    report = ResistanceSuite().run_one("CURRICULUM_SHAPE_CORRUPTION")
    assert report.passed, report.detail


def test_monograph_drift_detected() -> None:
    """Scenario 6: HN29 regeneration guard catches a forged monograph."""
    report = ResistanceSuite().run_one("MONOGRAPH_DRIFT")
    assert report.passed, report.detail


def test_full_suite_passes() -> None:
    """Run the whole suite end-to-end in declaration order and
    assert every scenario passes. This is the load-bearing
    HN32 row."""
    reports = ResistanceSuite().run_all()
    assert len(reports) == 6
    failed = [r for r in reports if not r.passed]
    assert not failed, "seventh-axis resistance suite failed on " + ", ".join(
        f"{r.scenario_id}: {r.detail}" for r in failed
    )


def test_unknown_scenario_raises_key_error() -> None:
    """Defensive: asking for a scenario id that does not exist is a
    KeyError, never a silent fallback."""
    import pytest

    with pytest.raises(KeyError):
        ResistanceSuite().run_one("DEFINITELY_NOT_A_SCENARIO")


def test_resistance_report_repr_format() -> None:
    """Aesthetic: ResistanceReport carries a rich __repr__ with the
    pass/fail flag and the scenario id."""
    report = ResistanceSuite().run_one("GATE_NEVER_WIDENS")
    r = repr(report)
    assert r.startswith("ResistanceReport[")
    assert "GATE_NEVER_WIDENS" in r
    assert "✓" in r
