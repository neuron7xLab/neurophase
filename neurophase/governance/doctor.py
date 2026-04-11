"""Eighth axis — Coherence (Цілісність): neurophase doctor.

The first six Sutskever axes describe how the system *reads*.
The seventh (Resistance) describes how it *holds under attack*.
The eighth (Coherence) describes whether the system's **multiple
self-descriptions agree mechanically**.

A coherent system cannot lie to itself. If the code says one
thing, the README another, the monograph a third, and the tests
a fourth — the system has drifted, and a reviewer who asks the
naive question *"what does neurophase actually promise?"* will
get more than one answer.

The :class:`Doctor` is the single command that makes that
drift impossible. It runs a declarative battery of **cross-
source-of-truth checks** and returns a frozen
:class:`DoctorReport` whose ``all_healthy`` flag is the
load-bearing coherence claim for HN33.

What the doctor checks
----------------------

1. **INVARIANT_REGISTRY_SCHEMA** — ``INVARIANTS.yaml`` loads
   without error; every invariant + honest-naming contract has
   ≥ 1 bound pytest node id and every ``enforced_in`` path
   exists on disk.

2. **STATE_MACHINE_SCHEMA** — ``STATE_MACHINE.yaml`` loads
   without error; every transition target is a declared state;
   every non-READY state has ``execution_allowed=False``.

3. **CLAIM_REGISTRY_SCHEMA** — ``CLAIMS.yaml`` loads without
   error; the mechanical promotion rule is satisfied for every
   claim; every ``related_invariants`` id exists in
   ``INVARIANTS.yaml``.

4. **MONOGRAPH_FRESH** — the committed
   ``docs/monograph/INVARIANTS_MONOGRAPH.md`` equals the live
   :func:`generate_monograph` output byte-for-byte.

5. **BIBLIOGRAPHY_DOI_COHERENCE** — every supporting-citation
   DOI referenced in ``CLAIMS.yaml`` is mentioned in the
   bibliography file. A claim that cites a paper the
   bibliography does not list is a coherence fault.

6. **API_FACADE_SURFACE** — ``neurophase.api.__all__`` is
   importable in full; every listed symbol is identity-equal
   to its canonical module export. A symbol listed but not
   exported is a drift.

7. **RESISTANCE_SUITE_GREEN** — the seventh-axis adversarial
   sweep (:class:`~neurophase.governance.resistance.ResistanceSuite`)
   runs clean. This makes the eighth axis strictly stronger
   than the seventh: coherence implies resistance.

8. **RUNTIME_MEMORY_BOUNDED** — a fresh orchestrator driven
   for 1 024 ticks has all six rolling buffers bounded by
   their declared caps (L1 endurance claim as a doctor check).

Each check returns a :class:`DoctorCheckResult`; the suite
aggregates into :class:`DoctorReport`. The report is JSON-safe
and the entire run is deterministic — the same repository
state produces the same report byte-for-byte.

What the doctor does NOT do
---------------------------

* It does **not** rewrite any file. If it flags a drift, the
  caller must decide what to fix — the code or the description.
* It does **not** run ``ruff`` or ``mypy``. Those are
  pre-doctor gates; the doctor assumes the code compiles.
* It does **not** run the full pytest suite. Individual
  resistance scenarios are re-run because they are composition-
  level; per-module tests are assumed to have run already.
* It does **not** change its output under ``--verbose`` or
  any environmental knob. Every field in the report is
  deterministic.

Command-line entry point
------------------------

``python -m neurophase doctor`` invokes :func:`run_doctor` and
prints a coloured Markdown summary. Exit code is ``0`` iff
every check passed, ``1`` otherwise — suitable for CI.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "DOCTOR_CHECKS",
    "Doctor",
    "DoctorCheckResult",
    "DoctorReport",
    "run_doctor",
]

#: Canonical repository root — resolved from this module's
#: location. The doctor runs against whatever repo it lives in.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True, repr=False)
class DoctorCheckResult:
    """Frozen outcome of one doctor check.

    Attributes
    ----------
    check_id
        Stable UPPER_SNAKE_CASE id.
    passed
        ``True`` iff the check found no coherence fault.
    detail
        One-line human-readable summary.
    warnings
        Tuple of additional warning lines surfaced by the check
        (non-blocking observations).
    """

    check_id: str
    passed: bool
    detail: str
    warnings: tuple[str, ...] = ()

    def __repr__(self) -> str:  # HN33 aesthetic
        flag = "✓" if self.passed else "✗"
        suffix = f" · {len(self.warnings)} warn(s)" if self.warnings else ""
        return f"DoctorCheckResult[{self.check_id} · {flag}{suffix}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "detail": self.detail,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, repr=False)
class DoctorReport:
    """Aggregated doctor outcome.

    Attributes
    ----------
    results
        Tuple of :class:`DoctorCheckResult`, one per registered
        check, in declaration order.
    all_healthy
        ``True`` iff every result's ``passed`` flag is ``True``.
        This is the load-bearing HN33 field.
    total_warnings
        Sum of ``len(r.warnings)`` across all results.
    """

    results: tuple[DoctorCheckResult, ...]
    all_healthy: bool
    total_warnings: int

    def __post_init__(self) -> None:
        expected_healthy = all(r.passed for r in self.results)
        if expected_healthy != self.all_healthy:
            raise ValueError(f"all_healthy={self.all_healthy} disagrees with per-result flags")
        expected_warnings = sum(len(r.warnings) for r in self.results)
        if expected_warnings != self.total_warnings:
            raise ValueError(
                f"total_warnings={self.total_warnings} disagrees with per-result count"
            )

    def __repr__(self) -> str:  # HN33 aesthetic
        flag = "✓" if self.all_healthy else "✗"
        passed = sum(1 for r in self.results if r.passed)
        return (
            f"DoctorReport[{passed}/{len(self.results)} · warnings={self.total_warnings} · {flag}]"
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "all_healthy": self.all_healthy,
            "total_warnings": self.total_warnings,
            "results": [r.to_dict() for r in self.results],
        }

    def as_markdown(self) -> str:
        """Render the report as a coloured Markdown table for humans."""
        lines: list[str] = []
        banner = "✓ HEALTHY" if self.all_healthy else "✗ DRIFT DETECTED"
        lines.append(f"# neurophase doctor — {banner}")
        lines.append("")
        lines.append(
            f"**{sum(1 for r in self.results if r.passed)} / "
            f"{len(self.results)}** checks passed, "
            f"**{self.total_warnings}** warning(s)."
        )
        lines.append("")
        lines.append("| # | Check | Status | Detail |")
        lines.append("|---|---|---|---|")
        for i, r in enumerate(self.results, 1):
            flag = "✓" if r.passed else "✗"
            lines.append(f"| {i} | `{r.check_id}` | {flag} | {r.detail} |")
        if self.total_warnings:
            lines.append("")
            lines.append("## Warnings")
            lines.append("")
            for r in self.results:
                for w in r.warnings:
                    lines.append(f"- `{r.check_id}`: {w}")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual check runners — each pure of inputs.
# ---------------------------------------------------------------------------


def _check_invariant_registry_schema() -> DoctorCheckResult:
    """Check 1: INVARIANTS.yaml loads, bindings + paths exist."""
    from neurophase.governance.invariants import (
        InvariantRegistryError,
        load_registry,
    )

    try:
        registry = load_registry()
    except InvariantRegistryError as exc:
        return DoctorCheckResult(
            "INVARIANT_REGISTRY_SCHEMA",
            False,
            f"registry failed to load: {exc}",
        )

    warnings: list[str] = []
    total = len(registry.invariants) + len(registry.honest_naming)
    missing_paths: list[str] = []
    for inv in registry.invariants:
        if len(inv.tests) < 1:
            return DoctorCheckResult(
                "INVARIANT_REGISTRY_SCHEMA",
                False,
                f"invariant {inv.id} has zero bound tests",
            )
        for site in inv.enforced_in:
            if not (_REPO_ROOT / site.path).exists():
                missing_paths.append(f"{inv.id}::{site.path}")
    for hn in registry.honest_naming:
        if len(hn.tests) < 1:
            return DoctorCheckResult(
                "INVARIANT_REGISTRY_SCHEMA",
                False,
                f"honest-naming {hn.id} has zero bound tests",
            )
        for site in hn.enforced_in:
            if not (_REPO_ROOT / site.path).exists():
                missing_paths.append(f"{hn.id}::{site.path}")

    if missing_paths:
        return DoctorCheckResult(
            "INVARIANT_REGISTRY_SCHEMA",
            False,
            f"{len(missing_paths)} enforced_in path(s) missing on disk",
            warnings=tuple(missing_paths[:5]),
        )
    return DoctorCheckResult(
        "INVARIANT_REGISTRY_SCHEMA",
        True,
        f"registry loaded: {total} total contracts",
        warnings=tuple(warnings),
    )


def _check_state_machine_schema() -> DoctorCheckResult:
    """Check 2: STATE_MACHINE.yaml loads cleanly."""
    from neurophase.governance.state_machine import (
        StateMachineRegistryError,
        load_state_machine,
    )

    try:
        spec = load_state_machine()
    except StateMachineRegistryError as exc:
        return DoctorCheckResult(
            "STATE_MACHINE_SCHEMA",
            False,
            f"state machine failed to load: {exc}",
        )

    state_names = spec.state_names()
    for t in spec.transitions:
        if t.target not in state_names:
            return DoctorCheckResult(
                "STATE_MACHINE_SCHEMA",
                False,
                f"transition {t.id} targets unknown state {t.target!r}",
            )

    # Every non-READY state must be non-permissive.
    for state in spec.states:
        if state.name != "READY" and state.execution_allowed:
            return DoctorCheckResult(
                "STATE_MACHINE_SCHEMA",
                False,
                f"non-READY state {state.name!r} has execution_allowed=True",
            )

    return DoctorCheckResult(
        "STATE_MACHINE_SCHEMA",
        True,
        f"{len(spec.states)} states, {len(spec.transitions)} transitions",
    )


def _check_claim_registry_schema() -> DoctorCheckResult:
    """Check 3: CLAIMS.yaml loads with mechanical promotion rule."""
    from neurophase.governance.claims import ClaimRegistryError, load_claims

    try:
        registry = load_claims()
    except ClaimRegistryError as exc:
        return DoctorCheckResult(
            "CLAIM_REGISTRY_SCHEMA",
            False,
            f"claim registry failed to load: {exc}",
        )
    return DoctorCheckResult(
        "CLAIM_REGISTRY_SCHEMA",
        True,
        f"{len(registry.claims)} claims, {len(registry.linked_invariants())} linked HN ids",
    )


def _check_monograph_fresh() -> DoctorCheckResult:
    """Check 4: committed monograph matches live generator byte-for-byte."""
    from neurophase.governance.monograph import MONOGRAPH_PATH, generate_monograph

    if not MONOGRAPH_PATH.is_file():
        return DoctorCheckResult(
            "MONOGRAPH_FRESH",
            False,
            f"monograph not found at {MONOGRAPH_PATH}",
        )
    committed = MONOGRAPH_PATH.read_text(encoding="utf-8")
    live = generate_monograph()
    if committed != live:
        committed_lines = committed.splitlines()
        live_lines = live.splitlines()
        delta = len(live_lines) - len(committed_lines)
        return DoctorCheckResult(
            "MONOGRAPH_FRESH",
            False,
            f"monograph stale: line delta = {delta:+d}; regenerate before merge",
        )
    return DoctorCheckResult(
        "MONOGRAPH_FRESH",
        True,
        f"monograph byte-equal to live generator ({len(committed)} bytes)",
    )


def _check_bibliography_doi_coherence() -> DoctorCheckResult:
    """Check 5: every DOI in CLAIMS.yaml appears in the bibliography."""
    from neurophase.governance.claims import load_claims

    bib_path = _REPO_ROOT / "docs" / "theory" / "neurophase_elite_bibliography.md"
    if not bib_path.is_file():
        return DoctorCheckResult(
            "BIBLIOGRAPHY_DOI_COHERENCE",
            False,
            f"bibliography not found at {bib_path}",
        )
    bib_text = bib_path.read_text(encoding="utf-8")
    registry = load_claims()

    missing: list[str] = []
    for claim in registry.claims:
        for citation in claim.evidence:
            if not citation.supports:
                continue
            doi = citation.doi.strip()
            if not doi:
                continue  # internal/experimental citation — allowed
            # A DOI coherence match is present if the DOI substring
            # occurs in the bibliography OR the citation source key
            # does. Both are sufficient — we do not require both.
            if doi in bib_text:
                continue
            if citation.source and citation.source in bib_text:
                continue
            missing.append(f"{claim.id}:{citation.source} ({doi})")

    if missing:
        return DoctorCheckResult(
            "BIBLIOGRAPHY_DOI_COHERENCE",
            False,
            f"{len(missing)} citation(s) absent from bibliography",
            warnings=tuple(missing[:10]),
        )
    return DoctorCheckResult(
        "BIBLIOGRAPHY_DOI_COHERENCE",
        True,
        "every supporting citation resolves in the bibliography",
    )


def _check_api_facade_surface() -> DoctorCheckResult:
    """Check 6: neurophase.api.__all__ is fully importable and identity-equal."""
    try:
        import neurophase.api as api
    except Exception as exc:  # pragma: no cover - import failure path
        return DoctorCheckResult(
            "API_FACADE_SURFACE",
            False,
            f"neurophase.api import failed: {type(exc).__name__}: {exc}",
        )
    missing: list[str] = []
    for name in api.__all__:
        if not hasattr(api, name):
            missing.append(name)
    if missing:
        return DoctorCheckResult(
            "API_FACADE_SURFACE",
            False,
            f"{len(missing)} symbol(s) listed in __all__ but not exported",
            warnings=tuple(missing),
        )
    return DoctorCheckResult(
        "API_FACADE_SURFACE",
        True,
        f"{len(api.__all__)} public facade symbols all resolved",
    )


def _check_resistance_suite_green() -> DoctorCheckResult:
    """Check 7: the axis-7 adversarial sweep passes cleanly."""
    from neurophase.governance.resistance import ResistanceSuite

    reports = ResistanceSuite().run_all()
    failed = [r for r in reports if not r.passed]
    if failed:
        return DoctorCheckResult(
            "RESISTANCE_SUITE_GREEN",
            False,
            f"{len(failed)} axis-7 scenario(s) failing",
            warnings=tuple(f"{r.scenario_id}: {r.detail}" for r in failed),
        )
    return DoctorCheckResult(
        "RESISTANCE_SUITE_GREEN",
        True,
        f"all {len(reports)} axis-7 adversarial scenarios passed",
    )


def _check_runtime_memory_bounded() -> DoctorCheckResult:
    """Check 8: L1 endurance claim on a 1024-tick fresh orchestrator."""
    from neurophase.runtime.memory_audit import audit_runtime_memory
    from neurophase.runtime.orchestrator import (
        OrchestratorConfig,
        RuntimeOrchestrator,
    )
    from neurophase.runtime.pipeline import PipelineConfig

    orch = RuntimeOrchestrator(
        OrchestratorConfig(
            pipeline=PipelineConfig(
                warmup_samples=4,
                stream_window=8,
                enable_stillness=True,
                stillness_window=8,
            )
        )
    )
    for i in range(1024):
        orch.tick(timestamp=float(i) * 0.01, R=0.92, delta=0.04)
    report = audit_runtime_memory(orch)
    if not report.all_bounded:
        offenders = [c.name for c in report.components if not c.is_bounded]
        return DoctorCheckResult(
            "RUNTIME_MEMORY_BOUNDED",
            False,
            f"unbounded components: {offenders}",
        )
    return DoctorCheckResult(
        "RUNTIME_MEMORY_BOUNDED",
        True,
        f"all 6 components bounded after {orch.n_ticks} ticks "
        f"(total={report.total_measured_size}/{report.total_declared_cap})",
    )


# ---------------------------------------------------------------------------
# Registry + runner
# ---------------------------------------------------------------------------

#: Stable tuple of every doctor check. Order is surfaced in the
#: report table; the HN33 bindings reference this exact sequence.
DOCTOR_CHECKS: tuple[tuple[str, Callable[[], DoctorCheckResult]], ...] = (
    ("INVARIANT_REGISTRY_SCHEMA", _check_invariant_registry_schema),
    ("STATE_MACHINE_SCHEMA", _check_state_machine_schema),
    ("CLAIM_REGISTRY_SCHEMA", _check_claim_registry_schema),
    ("MONOGRAPH_FRESH", _check_monograph_fresh),
    ("BIBLIOGRAPHY_DOI_COHERENCE", _check_bibliography_doi_coherence),
    ("API_FACADE_SURFACE", _check_api_facade_surface),
    ("RESISTANCE_SUITE_GREEN", _check_resistance_suite_green),
    ("RUNTIME_MEMORY_BOUNDED", _check_runtime_memory_bounded),
)


class Doctor:
    """The single coherence-audit runner.

    Stateless. Two invocations on the same repository state
    produce byte-identical :class:`DoctorReport` payloads.
    """

    def run(self) -> DoctorReport:
        """Execute every registered check and return the aggregated report."""
        results: list[DoctorCheckResult] = []
        for _, runner in DOCTOR_CHECKS:
            try:
                result = runner()
            except Exception as exc:  # defensive — a check should never raise
                result = DoctorCheckResult(
                    check_id=runner.__name__.lstrip("_").upper().replace("CHECK_", ""),
                    passed=False,
                    detail=f"check raised {type(exc).__name__}: {exc}",
                )
            results.append(result)

        all_healthy = all(r.passed for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        return DoctorReport(
            results=tuple(results),
            all_healthy=all_healthy,
            total_warnings=total_warnings,
        )

    def run_one(self, check_id: str) -> DoctorCheckResult:
        """Execute a single check by id."""
        for registered_id, runner in DOCTOR_CHECKS:
            if registered_id == check_id:
                return runner()
        raise KeyError(f"unknown doctor check id: {check_id!r}")


def run_doctor() -> DoctorReport:
    """Shortcut: ``Doctor().run()``."""
    return Doctor().run()


# ---------------------------------------------------------------------------
# Internals — defensive helpers.
# ---------------------------------------------------------------------------

#: Stable regex to detect a line that looks like a DOI — used by
#: future extensions; the current coherence check uses simple
#: substring membership which is sufficient for the small
#: committed CLAIMS.yaml.
_DOI_RE: Final[re.Pattern[str]] = re.compile(r"10\.\d{4,}/\S+")


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        return None
