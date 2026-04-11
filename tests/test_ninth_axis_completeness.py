"""HN35 — Ninth axis (Completeness / Повнота) contract tests.

Axis 8 asks *"do the self-descriptions AGREE?"*. Axis 9 asks
*"do they COVER everything?"*. A system that passes axis 8 can
still be incomplete — a public symbol exported without a test,
an orphan module, a dead pytest binding, a registered adapter
nobody tests. Axis 9 closes those gaps mechanically.

Locks in:

1. **Five completeness checks.** The registry is stable at
   exactly five. Adding or removing a check requires updating
   HN35 bindings.
2. **Every check is individually invocable.**
3. **Report self-consistency** — ``complete`` and
   ``total_gaps`` cannot be lied to at construction.
4. **Per-check gap validation** — a check with ``passed=True``
   cannot also carry gaps.
5. **THE LOAD-BEARING CLAIM: the whole suite runs green on main.**
6. **Axis 9 is surfaced inside the doctor** as the 9th check.
7. **JSON-safe projection** round-trips.
8. **Rich __repr__** follows the canonical design language.
"""

from __future__ import annotations

import json

import pytest

from neurophase.governance.completeness import (
    COMPLETENESS_CHECKS,
    CompletenessAuditor,
    CompletenessCheckResult,
    CompletenessReport,
    run_completeness,
)

# ---------------------------------------------------------------------------
# 1. Registry enumeration.
# ---------------------------------------------------------------------------


def test_registry_enumerates_five_checks() -> None:
    """The completeness registry is stable at exactly five checks."""
    assert len(COMPLETENESS_CHECKS) == 5
    ids = {entry[0] for entry in COMPLETENESS_CHECKS}
    assert ids == {
        "API_SYMBOL_TEST_COVERAGE",
        "PUBLIC_MODULE_REACHABLE",
        "INVARIANT_TEST_NODE_EXISTS",
        "SENSOR_ADAPTER_HAS_TEST",
        "MONOGRAPH_MENTIONS_EVERY_HN",
    }


# ---------------------------------------------------------------------------
# 2. Per-check invocation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("check_id", [entry[0] for entry in COMPLETENESS_CHECKS])
def test_individual_check_runs_complete(check_id: str) -> None:
    """Every registered check passes on the committed repo state."""
    result = CompletenessAuditor().run_one(check_id)
    assert isinstance(result, CompletenessCheckResult)
    assert result.passed, f"{check_id} failed: {result.detail} / gaps={result.gaps}"


def test_run_one_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError):
        CompletenessAuditor().run_one("NOT_A_REAL_CHECK")


# ---------------------------------------------------------------------------
# 3. Report self-consistency.
# ---------------------------------------------------------------------------


def test_report_complete_disagreement_rejected() -> None:
    """A report that lies about ``complete`` is refused."""
    with pytest.raises(ValueError, match="complete"):
        CompletenessReport(
            results=(
                CompletenessCheckResult("A", True, "ok"),
                CompletenessCheckResult("B", False, "gap", gaps=("x",)),
            ),
            complete=True,  # lie
            total_gaps=1,
        )


def test_report_total_gaps_disagreement_rejected() -> None:
    """A report that lies about ``total_gaps`` is refused."""
    with pytest.raises(ValueError, match="total_gaps"):
        CompletenessReport(
            results=(CompletenessCheckResult("A", False, "gap", gaps=("a", "b", "c")),),
            complete=False,
            total_gaps=99,  # lie
        )


# ---------------------------------------------------------------------------
# 4. Per-check validation.
# ---------------------------------------------------------------------------


def test_check_result_rejects_gaps_with_passed_true() -> None:
    """A check that reports gaps cannot also claim to have passed."""
    with pytest.raises(ValueError, match="gaps but passed=True"):
        CompletenessCheckResult("X", True, "ok", gaps=("a",))


def test_check_result_accepts_clean_pass() -> None:
    """A clean pass has no gaps and passed=True."""
    r = CompletenessCheckResult("X", True, "ok")
    assert r.passed
    assert r.gaps == ()


# ---------------------------------------------------------------------------
# 5. THE LOAD-BEARING CLAIM.
# ---------------------------------------------------------------------------


def test_full_completeness_suite_runs_green_on_main() -> None:
    """This is HN35 in one line. If this fails, the repository
    has a completeness gap — some public surface exists that is
    not tested, not reachable, or not documented."""
    report = run_completeness()
    failed = [(r.check_id, r.gaps) for r in report.results if not r.passed]
    assert report.complete, (
        f"completeness drift on main: {failed}. "
        "Run `python -c 'from neurophase.governance.completeness import "
        "CompletenessAuditor; print(CompletenessAuditor().run())'` for details."
    )


# ---------------------------------------------------------------------------
# 6. Doctor integration.
# ---------------------------------------------------------------------------


def test_doctor_exposes_completeness_check() -> None:
    """Axis 9 is surfaced inside the doctor as a registered check."""
    from neurophase.governance.doctor import DOCTOR_CHECKS, Doctor

    ids = [entry[0] for entry in DOCTOR_CHECKS]
    assert "COMPLETENESS_SUITE_GREEN" in ids

    result = Doctor().run_one("COMPLETENESS_SUITE_GREEN")
    assert result.passed, result.detail


# ---------------------------------------------------------------------------
# 7. JSON-safe projection.
# ---------------------------------------------------------------------------


def test_report_json_round_trip() -> None:
    report = run_completeness()
    payload = report.to_json_dict()
    text = json.dumps(payload)
    loaded = json.loads(text)
    assert loaded["complete"] == report.complete
    assert loaded["total_gaps"] == report.total_gaps
    assert len(loaded["results"]) == len(report.results)


def test_json_projection_is_flat() -> None:
    report = run_completeness()
    payload = report.to_json_dict()
    for result in payload["results"]:
        for key, value in result.items():
            assert isinstance(value, (str, bool, list)), (
                f"field {key!r} has non-primitive value {value!r}"
            )
            if isinstance(value, list):
                for item in value:
                    assert isinstance(item, str)


# ---------------------------------------------------------------------------
# 8. Aesthetic.
# ---------------------------------------------------------------------------


def test_check_result_repr_format() -> None:
    r = CompletenessCheckResult("X", True, "ok")
    assert repr(r).startswith("CompletenessCheckResult[")
    assert "✓" in repr(r)
    assert "X" in repr(r)


def test_check_result_repr_shows_gap_count() -> None:
    r = CompletenessCheckResult("X", False, "drift", gaps=("a", "b"))
    assert "✗" in repr(r)
    assert "2 gap" in repr(r)


def test_report_repr_shows_summary() -> None:
    report = run_completeness()
    r = repr(report)
    assert r.startswith("CompletenessReport[")
    assert "/" in r
    if report.complete:
        assert "✓" in r
