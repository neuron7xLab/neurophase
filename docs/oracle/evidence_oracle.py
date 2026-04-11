#!/usr/bin/env python3
"""Self-referential evidence oracle.

Scans repository docs, verifies evidence labels, and generates a compact
traceability report. Non-network, deterministic.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
LABEL_RE = re.compile(r"\b(Established|Strongly Plausible|Tentative|Unsupported(?:/Weak|-Weak)?)\b")


@dataclass(frozen=True)
class OracleReport:
    files_scanned: int
    labels_found: int
    unlabeled_claim_lines: int
    pass_gate: bool


def run_oracle() -> OracleReport:
    files = list(DOCS.rglob("*.md"))
    labels = 0
    unlabeled = 0

    for path in files:
        text = path.read_text(encoding="utf-8")
        labels += len(LABEL_RE.findall(text))
        for line in text.splitlines():
            low = line.lower().strip()
            if low.startswith("claim:") and not LABEL_RE.search(line):
                unlabeled += 1
            if "todo" in low and "evidence" in low:
                unlabeled += 1

    return OracleReport(
        files_scanned=len(files),
        labels_found=labels,
        unlabeled_claim_lines=unlabeled,
        pass_gate=labels > 0 and unlabeled == 0,
    )


def main() -> int:
    report = run_oracle()
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.pass_gate else 1


if __name__ == "__main__":
    sys.exit(main())
