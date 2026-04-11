#!/usr/bin/env python3
"""π-agent self-healing hooks for documentation consistency.

This script performs safe, idempotent repairs for missing anchors only.
It does NOT alter scientific claims; it only restores required links.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ensure_anchor(path: Path, marker: str, snippet: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    path.write_text(text.rstrip() + "\n\n" + snippet + "\n", encoding="utf-8")
    return True


def main() -> int:
    changed = 0

    science = ROOT / "science_basis.md"
    changed += int(
        ensure_anchor(
            science,
            "neurophase_elite_bibliography.md",
            "Reference anchor: `docs/theory/neurophase_elite_bibliography.md`",
        )
    )

    protocol = ROOT / "validation" / "integration_readiness_protocol.md"
    changed += int(
        ensure_anchor(
            protocol,
            "docs/evidence_enforcer.py",
            "Governance enforcement hook: `python docs/evidence_enforcer.py`",
        )
    )

    print(f"PI SELF-HEAL: {'CHANGED' if changed else 'NOOP'} ({changed} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
