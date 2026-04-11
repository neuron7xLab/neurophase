#!/usr/bin/env python3
"""Validate DOI presence/shape in elite bibliography.

Offline check only: verifies DOI-like tokens and minimal cardinality.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIB = ROOT / "theory" / "neurophase_elite_bibliography.md"
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9<>]+")


def main() -> int:
    if not BIB.exists():
        print(f"DOI VALIDATION: FAIL - missing file {BIB.relative_to(ROOT)}")
        return 1

    text = BIB.read_text(encoding="utf-8")
    dois = DOI_RE.findall(text)
    unique_dois = sorted(set(dois))

    if len(unique_dois) < 12:
        print(
            "DOI VALIDATION: FAIL - expected at least 12 DOI anchors "
            f"in elite bibliography, found {len(unique_dois)}"
        )
        return 1

    print(
        "DOI VALIDATION: PASS - "
        f"{len(unique_dois)} unique DOI anchors detected in {BIB.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
