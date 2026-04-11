"""HN33 — Eighth axis (Coherence / Цілісність) contract tests.

The eighth axis says: every source of truth about the system
tells the same story. INVARIANTS.yaml, STATE_MACHINE.yaml,
CLAIMS.yaml, the monograph, the bibliography, the exported API,
and the runtime stack must all agree mechanically — and the
doctor CLI is the single command that makes it so.

Locks in:

1. **Registry enumerates eight checks.** Adding or removing a
   check without updating HN33 bindings is a contract violation.
2. **Every check is individually invocable.**
3. **DoctorReport is self-consistent** — the ``all_healthy``
   and ``total_warnings`` fields cannot be lied to at construction.
4. **The doctor runs green on main.** This is the load-bearing
   HN33 row: if any coherence check fails on the committed repo
   state, the PR cannot merge.
5. **JSON projection is flat** and round-trips.
6. **Markdown rendering is deterministic** — two invocations
   produce byte-identical output.
7. **CLI subcommand** ``python -m neurophase doctor`` exits
   0 on healthy state and 1 on any drift.
8. **Every DOI in CLAIMS.yaml resolves in the bibliography** —
   the doctor catches drift that no per-module test can see.
"""

from __future__ import annotations

import json

import pytest

from neurophase.governance.doctor import (
    DOCTOR_CHECKS,
    Doctor,
    DoctorCheckResult,
    DoctorReport,
    run_doctor,
)

# ---------------------------------------------------------------------------
# 1. Registry enumeration.
# ---------------------------------------------------------------------------


def test_registry_enumerates_ten_checks() -> None:
    """The doctor registry is stable at exactly ten checks.

    Check #10 is the self-enforcing loop: the doctor verifies
    that CI runs the doctor. If the CI step is removed, the
    doctor flags it on the next local run.
    """
    assert len(DOCTOR_CHECKS) == 10
    ids = {entry[0] for entry in DOCTOR_CHECKS}
    assert ids == {
        "INVARIANT_REGISTRY_SCHEMA",
        "STATE_MACHINE_SCHEMA",
        "CLAIM_REGISTRY_SCHEMA",
        "MONOGRAPH_FRESH",
        "BIBLIOGRAPHY_DOI_COHERENCE",
        "API_FACADE_SURFACE",
        "RESISTANCE_SUITE_GREEN",
        "RUNTIME_MEMORY_BOUNDED",
        "COMPLETENESS_SUITE_GREEN",
        "CI_WORKFLOW_DOCTOR_STEP_PRESENT",
    }


# ---------------------------------------------------------------------------
# 2. Per-check invocation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("check_id", [entry[0] for entry in DOCTOR_CHECKS])
def test_individual_check_runs_green(check_id: str) -> None:
    """Every registered check passes when run against the
    committed repository state."""
    result = Doctor().run_one(check_id)
    assert isinstance(result, DoctorCheckResult)
    assert result.passed, f"{check_id} failed: {result.detail}"


def test_run_one_unknown_raises_key_error() -> None:
    """Defensive: unknown check id is a loud error, never a silent fallback."""
    with pytest.raises(KeyError):
        Doctor().run_one("DEFINITELY_NOT_A_CHECK")


# ---------------------------------------------------------------------------
# 3. Report self-consistency.
# ---------------------------------------------------------------------------


def test_report_all_healthy_disagreement_rejected() -> None:
    """A report that lies about ``all_healthy`` is refused at construction."""
    with pytest.raises(ValueError, match="all_healthy"):
        DoctorReport(
            results=(
                DoctorCheckResult("A", True, "ok"),
                DoctorCheckResult("B", False, "drift"),
            ),
            all_healthy=True,  # lie — B failed
            total_warnings=0,
        )


def test_report_total_warnings_disagreement_rejected() -> None:
    """A report that lies about ``total_warnings`` is refused."""
    with pytest.raises(ValueError, match="total_warnings"):
        DoctorReport(
            results=(DoctorCheckResult("A", True, "ok", warnings=("w1", "w2")),),
            all_healthy=True,
            total_warnings=99,  # lie
        )


# ---------------------------------------------------------------------------
# 4. THE LOAD-BEARING CLAIM: doctor runs green on main.
# ---------------------------------------------------------------------------


def test_full_doctor_runs_green_on_main() -> None:
    """This is HN33 in one line. If this test fails, the repository
    has a coherence drift that must be fixed before merge."""
    report = run_doctor()
    failed = [r.check_id for r in report.results if not r.passed]
    assert report.all_healthy, (
        f"coherence drift on main: {len(failed)} check(s) failed: {failed}. "
        f"Run `python -m neurophase doctor` for details."
    )


# ---------------------------------------------------------------------------
# 5. JSON projection + round-trip.
# ---------------------------------------------------------------------------


def test_report_json_round_trip() -> None:
    report = run_doctor()
    payload = report.to_json_dict()
    text = json.dumps(payload)
    loaded = json.loads(text)
    assert loaded["all_healthy"] == report.all_healthy
    assert loaded["total_warnings"] == report.total_warnings
    assert len(loaded["results"]) == len(report.results)
    for i, r in enumerate(report.results):
        assert loaded["results"][i]["check_id"] == r.check_id
        assert loaded["results"][i]["passed"] == r.passed


def test_json_projection_is_flat() -> None:
    report = run_doctor()
    payload = report.to_json_dict()
    for result in payload["results"]:
        for key, value in result.items():
            assert isinstance(value, (str, bool, list)), (
                f"result field {key!r} has non-primitive value {value!r}"
            )
            if isinstance(value, list):
                for item in value:
                    assert isinstance(item, str)


# ---------------------------------------------------------------------------
# 6. Markdown rendering is deterministic.
# ---------------------------------------------------------------------------


def test_markdown_rendering_is_deterministic() -> None:
    """Two calls to :meth:`DoctorReport.as_markdown` on the same
    report produce byte-identical output."""
    report = run_doctor()
    a = report.as_markdown()
    b = report.as_markdown()
    assert a == b
    assert "neurophase doctor" in a
    assert "10 / 10" in a  # pinned to the current ten checks


def test_markdown_shows_drift_banner_on_failure() -> None:
    drifty = DoctorReport(
        results=(
            DoctorCheckResult("PASS", True, "ok"),
            DoctorCheckResult("FAIL", False, "boom"),
        ),
        all_healthy=False,
        total_warnings=0,
    )
    md = drifty.as_markdown()
    assert "DRIFT DETECTED" in md
    assert "✗" in md


# ---------------------------------------------------------------------------
# 7. CLI subcommand.
# ---------------------------------------------------------------------------


def test_cli_doctor_exit_code_healthy(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI exits 0 on a healthy state."""
    from neurophase.__main__ import main

    exit_code = main(["doctor"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "neurophase doctor" in captured.out
    assert "HEALTHY" in captured.out


def test_cli_doctor_json_mode(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI supports a JSON-output mode for tooling."""
    from neurophase.__main__ import main

    exit_code = main(["doctor", "--json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["all_healthy"] is True
    assert len(payload["results"]) == 10


# ---------------------------------------------------------------------------
# 8. Aesthetic.
# ---------------------------------------------------------------------------


def test_doctor_check_result_repr_format() -> None:
    """Rich __repr__ follows the canonical HN22 design language."""
    r = DoctorCheckResult("X", True, "ok", warnings=("a",))
    assert repr(r).startswith("DoctorCheckResult[")
    assert "X" in repr(r)
    assert "✓" in repr(r)
    assert "1 warn" in repr(r)


def test_doctor_report_repr_shows_summary() -> None:
    report = run_doctor()
    r = repr(report)
    assert r.startswith("DoctorReport[")
    assert "/" in r
    if report.all_healthy:
        assert "✓" in r


# ---------------------------------------------------------------------------
# 9. Self-enforcing loop: CI step present + doctor check of the CI step.
# ---------------------------------------------------------------------------


def test_ci_workflow_contains_doctor_step() -> None:
    """The committed CI workflow must contain the doctor step.

    This is the load-bearing enforcement claim: every PR MUST
    run ``python -m neurophase doctor`` in CI, not just pass
    ``pytest``. Removing the step makes the doctor a theoretical
    tool — axis 7/8/9 become opt-in instead of enforced.
    """
    from pathlib import Path

    workflow_path = (
        Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    )
    assert workflow_path.is_file(), f"CI workflow missing at {workflow_path}"
    text = workflow_path.read_text(encoding="utf-8")
    assert "python -m neurophase doctor" in text, (
        "CI workflow does not run the doctor — the self-enforcing loop is broken"
    )


def test_doctor_self_enforcement_check_passes() -> None:
    """The 10th doctor check (CI_WORKFLOW_DOCTOR_STEP_PRESENT)
    verifies the CI step exists. Must be green on main."""
    from neurophase.governance.doctor import Doctor

    result = Doctor().run_one("CI_WORKFLOW_DOCTOR_STEP_PRESENT")
    assert result.passed, result.detail


def test_doctor_self_enforcement_check_reads_real_workflow() -> None:
    """The self-enforcement check reads the committed CI
    workflow and succeeds iff the doctor step is present.

    This is the falsification proof for the self-enforcing
    loop: the check is substring-sensitive to the marker
    ``python -m neurophase doctor``, and the marker is
    currently present in ``.github/workflows/ci.yml``.
    Removing it would flip this test to red.
    """
    from pathlib import Path

    from neurophase.governance.doctor import _check_ci_workflow_has_doctor_step

    workflow = (
        Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    )
    text = workflow.read_text(encoding="utf-8")
    marker = "python -m neurophase doctor"
    assert marker in text

    result = _check_ci_workflow_has_doctor_step()
    assert result.passed
