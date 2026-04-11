#!/usr/bin/env python3
"""Lightweight evidence-label audit for neurophase docs.

Gate intent: ensure core theory/validation docs keep explicit evidence taxonomy.
This script is intentionally conservative to avoid false failures on narrative text.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET_FILES = [
    ROOT / "theory" / "neurophase_elite_bibliography.md",
    ROOT / "theory" / "neurophase_scientific_basis.md",
    ROOT / "validation" / "evidence_labeling_style_guide.md",
]
LABEL_RE = re.compile(r"\b(Established|Strongly Plausible|Tentative|Unsupported(?:/Weak|-Weak)?)\b")


def main() -> int:
    missing: list[str] = []
    stats: dict[str, int] = {}

    for path in TARGET_FILES:
        if not path.exists():
            missing.append(f"missing file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        count = len(LABEL_RE.findall(text))
        stats[str(path.relative_to(ROOT))] = count
        if count == 0:
            missing.append(f"no evidence labels found: {path.relative_to(ROOT)}")

    if missing:
        print("EVIDENCE AUDIT: FAIL")
        for m in missing:
            print(f" - {m}")
        return 1

    print("EVIDENCE AUDIT: PASS")
    for file, count in stats.items():
        print(f" - {file}: {count} labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
