#!/usr/bin/env python3
"""Hyper-strict evidence enforcer for docs merge gating.

Rules enforced:
1) Core theory docs must include evidence taxonomy labels.
2) Elite bibliography must include DOI anchors and traceability matrix.
3) Integration protocol must include calibration + failure mode + CI automation sections.
4) Scientific basis docs must point to elite bibliography + style guide.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class GateError(Exception):
    """Raised when compliance gate fails."""


def read(rel: str) -> str:
    path = ROOT / rel
    if not path.exists():
        raise GateError(f"missing required file: {rel}")
    return path.read_text(encoding="utf-8")


def must(pattern: str, text: str, message: str) -> None:
    if re.search(pattern, text, re.MULTILINE) is None:
        raise GateError(message)


def must_count(pattern: str, text: str, n: int, message: str) -> None:
    count = len(re.findall(pattern, text, re.MULTILINE))
    if count < n:
        raise GateError(f"{message} (found {count}, required {n})")


def main() -> int:
    try:
        elite = read("theory/neurophase_elite_bibliography.md")
        style = read("validation/evidence_labeling_style_guide.md")
        protocol = read("validation/integration_readiness_protocol.md")
        science = read("science_basis.md")
        theory = read("theory/neurophase_scientific_basis.md")

        must_count(r"DOI:\s*10\.", elite, 12, "elite bibliography lacks sufficient DOI anchors")
        must(r"TRACEABILITY MATRIX", elite, "elite bibliography missing traceability matrix")
        must(r"INTEGRATION READINESS CHECKLIST", elite, "elite bibliography missing integration checklist")

        for label in ["Established", "Strongly Plausible", "Tentative", "Unsupported"]:
            must(label, style, f"style guide missing label: {label}")

        must(r"PHASE 2: METRIC CALIBRATION LOOP", protocol, "protocol missing calibration loop")
        must(r"PHASE 4: FAILURE MODES", protocol, "protocol missing failure modes")
        must(r"PHASE 7: CI/CD", protocol, "protocol missing CI/CD automation section")

        must(r"neurophase_elite_bibliography\.md", science, "science basis missing elite bibliography anchor")
        must(r"evidence_labeling_style_guide\.md", science, "science basis missing evidence style anchor")
        must(r"neurophase_elite_bibliography\.md", theory, "theory basis missing elite bibliography anchor")

    except GateError as exc:
        print(f"EVIDENCE ENFORCER: FAIL - {exc}")
        return 1

    print("EVIDENCE ENFORCER: PASS - all Ω-level documentation gates satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
