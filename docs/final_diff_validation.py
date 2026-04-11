#!/usr/bin/env python3
"""Ω-level final diff validator (20+ checks).

Offline-only validator that enforces documentation + CI + typing integration
artifacts for elite neurophase governance.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    label: str


def exists(rel: str) -> CheckResult:
    return CheckResult((ROOT / rel).exists(), f"exists: {rel}")


def contains(rel: str, pattern: str, label: str) -> CheckResult:
    p = ROOT / rel
    if not p.exists():
        return CheckResult(False, f"missing file: {rel}")
    text = p.read_text(encoding="utf-8")
    return CheckResult(re.search(pattern, text, re.MULTILINE) is not None, label)


def count_at_least(rel: str, pattern: str, n: int, label: str) -> CheckResult:
    p = ROOT / rel
    if not p.exists():
        return CheckResult(False, f"missing file: {rel}")
    text = p.read_text(encoding="utf-8")
    count = len(re.findall(pattern, text, re.MULTILINE))
    return CheckResult(count >= n, f"{label}: {count} >= {n}")


def build_checks() -> list[CheckResult]:
    checks: list[CheckResult] = []

    # 1-12 file existence
    for rel in [
        "docs/theory/neurophase_elite_bibliography.md",
        "docs/theory/hierarchical_status_bibliography.md",
        "docs/theory/neurophase_scientific_basis.md",
        "docs/science_basis.md",
        "docs/validation/integration_readiness_protocol.md",
        "docs/validation/evidence_labeling_style_guide.md",
        "docs/validation/MERGE_SIGN_OFF_TEMPLATE.txt",
        "docs/validation/FINAL_PRODUCTION_SIGNOFF_2026-04-11.md",
        "docs/maintenance/2026_q2_calibration.md",
        "docs/audit_evidence_labels.py",
        "docs/validate_bibliography_dois.py",
        "docs/evidence_enforcer.py",
        "docs/oracle/evidence_oracle.py",
        "docs/oracle/predictive_docs_dynamics.py",
        "docs/singularity_manifest.md",
        "omega_governance_kernel.py",
        "neurophase/kernel/omega_invariants.py",
        "neurophase/agents/pi_core.py",
        "neurophase/calibration/singularity_attractor.py",
    ]:
        checks.append(exists(rel))

    # 13-14 CI hooks present
    checks.append(contains(".github/workflows/ci.yml", r"omega_governance_kernel.py", "ci: omega governance kernel"))
    checks.append(contains("omega_governance_kernel.py", r"docs/evidence_enforcer.py", "kernel: includes evidence enforcer"))

    # 15-17 bibliography integrity
    checks.append(count_at_least("docs/theory/neurophase_elite_bibliography.md", r"DOI:\s*10\.", 12, "elite DOI anchors"))
    checks.append(contains("docs/theory/neurophase_elite_bibliography.md", r"TRACEABILITY MATRIX", "elite traceability matrix"))
    checks.append(contains("docs/theory/neurophase_elite_bibliography.md", r"INTEGRATION READINESS CHECKLIST", "elite integration checklist"))

    # 20-22 evidence taxonomy consistency
    checks.append(contains("docs/validation/evidence_labeling_style_guide.md", r"Established", "style: Established"))
    checks.append(contains("docs/validation/evidence_labeling_style_guide.md", r"Strongly Plausible", "style: Strongly Plausible"))
    checks.append(contains("docs/validation/evidence_labeling_style_guide.md", r"Tentative", "style: Tentative"))

    # 23-24 typing hardening checks
    checks.append(
        contains(
            "neurophase/metrics/asymmetry.py",
            r"def _is_effectively_constant\(arr: npt\.NDArray\[np\.float64\], std: float\)",
            "typing: asymmetry ndarray signature",
        )
    )
    checks.append(
        contains(
            "neurophase/metrics/ricci.py",
            r"tuple\[npt\.NDArray\[np\.float64\], npt\.NDArray\[np\.float64\]\]",
            "typing: ricci local distribution signature",
        )
    )

    return checks


def main() -> int:
    checks = build_checks()
    passed = sum(1 for c in checks if c.ok)
    total = len(checks)

    print(f"FINAL DIFF VALIDATION: {'PASS' if passed == total else 'FAIL'} ({passed}/{total})")
    for c in checks:
        print(f" - {'✓' if c.ok else '✗'} {c.label}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
