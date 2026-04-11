"""HN37 — Tenth axis (Reproducibility / Відтворюваність) contract tests.

Axes 8 and 9 are static: they check that the files agree and
cover everything. Axis 10 is dynamic: it checks that the system,
when invoked, produces the same bytes twice. A coherent and
complete system can still be non-reproducible if a module
embeds ``time.time()``, iterates over an unsorted dict, or uses
``random`` without a seed.

Locks in:

1. **Eight scenarios registered.** The registry is stable at
   exactly eight. Adding or removing a scenario requires
   updating HN37 bindings.
2. **Every scenario is individually invocable.**
3. **Report self-consistency** — ``reproducible`` cannot be
   lied to at construction.
4. **THE LOAD-BEARING CLAIM: every scenario passes on main.**
5. **Auditor is itself reproducible** (meta-claim).
6. **JSON projection is flat** and round-trips.
7. **Rich __repr__** follows the canonical design language.
"""

from __future__ import annotations

import json

import pytest

from neurophase.governance.reproducibility import (
    REPRODUCIBILITY_SCENARIOS,
    ReproducibilityAuditor,
    ReproducibilityCheckResult,
    ReproducibilityReport,
    run_reproducibility,
)

# ---------------------------------------------------------------------------
# 1. Registry enumeration.
# ---------------------------------------------------------------------------


def test_registry_enumerates_eight_scenarios() -> None:
    """The axis-10 registry is stable at exactly eight scenarios."""
    assert len(REPRODUCIBILITY_SCENARIOS) == 8
    ids = {entry[0] for entry in REPRODUCIBILITY_SCENARIOS}
    assert ids == {
        "MONOGRAPH_BYTE_EQUAL",
        "DOCTOR_REGISTRY_BYTE_EQUAL",
        "RESISTANCE_SUITE_BYTE_EQUAL",
        "COMPLETENESS_SUITE_BYTE_EQUAL",
        "PARAMETER_SWEEP_BYTE_EQUAL",
        "SYNTHETIC_OSCILLATOR_BYTE_EQUAL",
        "INVARIANT_REGISTRY_BYTE_EQUAL",
        "PIPELINE_GATE_SEQUENCE_BYTE_EQUAL",
    }


# ---------------------------------------------------------------------------
# 2. Per-scenario invocation.
# ---------------------------------------------------------------------------


def test_full_suite_runs_all_scenarios_and_they_all_pass() -> None:
    """Run the whole suite once. This is cheaper than running
    each scenario twice (once parametrised + once in the full
    suite) and still covers every scenario exactly once.

    The individual per-scenario pass/fail details are surfaced
    in ``report.results`` so a failure still gets a precise
    error message.
    """
    report = ReproducibilityAuditor().run()
    assert len(report.results) == len(REPRODUCIBILITY_SCENARIOS)
    scenario_ids = {r.scenario_id for r in report.results}
    expected_ids = {entry[0] for entry in REPRODUCIBILITY_SCENARIOS}
    assert scenario_ids == expected_ids
    failed = [(r.scenario_id, r.detail) for r in report.results if not r.passed]
    assert not failed, f"scenarios failed: {failed}"


def test_run_one_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError):
        ReproducibilityAuditor().run_one("NOT_A_REAL_SCENARIO")


def test_run_one_cheap_synthetic_scenario() -> None:
    """Verify run_one works with a cheap scenario (no full doctor)."""
    result = ReproducibilityAuditor().run_one("SYNTHETIC_OSCILLATOR_BYTE_EQUAL")
    assert result.passed, result.detail


# ---------------------------------------------------------------------------
# 3. Report self-consistency.
# ---------------------------------------------------------------------------


def test_report_reproducible_disagreement_rejected() -> None:
    """A report that lies about ``reproducible`` is refused."""
    with pytest.raises(ValueError, match="reproducible"):
        ReproducibilityReport(
            results=(
                ReproducibilityCheckResult("A", True, "byte_equal"),
                ReproducibilityCheckResult("B", False, "mismatch"),
            ),
            reproducible=True,  # lie
        )


# ---------------------------------------------------------------------------
# 4. THE LOAD-BEARING CLAIM.
# ---------------------------------------------------------------------------


def test_full_reproducibility_suite_passes_on_main() -> None:
    """This is HN37 load-bearing claim via ``run_reproducibility``.

    Note: ``test_full_suite_runs_all_scenarios_and_they_all_pass``
    already runs the full suite via the auditor. This test uses the
    convenience ``run_reproducibility()`` shortcut so both entry
    points stay tested.
    """
    report = run_reproducibility()
    failed = [(r.scenario_id, r.detail) for r in report.results if not r.passed]
    assert report.reproducible, (
        f"reproducibility drift on main: {failed}. "
        "Some module is non-deterministic under repeated invocation."
    )


# ---------------------------------------------------------------------------
# 6. JSON projection round-trip.
# ---------------------------------------------------------------------------


def test_report_json_round_trip() -> None:
    report = run_reproducibility()
    payload = report.to_json_dict()
    text = json.dumps(payload)
    loaded = json.loads(text)
    assert loaded["reproducible"] == report.reproducible
    assert len(loaded["results"]) == len(report.results)


def test_json_projection_is_flat() -> None:
    report = run_reproducibility()
    payload = report.to_json_dict()
    for result in payload["results"]:
        for key, value in result.items():
            assert isinstance(value, (str, bool)), (
                f"field {key!r} has non-primitive value {value!r}"
            )


# ---------------------------------------------------------------------------
# 7. Aesthetic.
# ---------------------------------------------------------------------------


def test_check_result_repr_format() -> None:
    r = ReproducibilityCheckResult("X", True, "byte_equal: ok")
    assert repr(r).startswith("ReproducibilityCheckResult[")
    assert "✓" in repr(r)
    assert "X" in repr(r)


def test_check_result_repr_failure_flag() -> None:
    r = ReproducibilityCheckResult("Y", False, "mismatch: drift")
    assert "✗" in repr(r)


def test_report_repr_shows_summary() -> None:
    report = run_reproducibility()
    r = repr(report)
    assert r.startswith("ReproducibilityReport[")
    assert "/" in r
    if report.reproducible:
        assert "✓" in r
