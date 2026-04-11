#!/usr/bin/env python3
"""Zero-defect governance singularity kernel.

Runs all governance checks in one pass.
"""

from __future__ import annotations

import subprocess
import sys

COMMANDS = [
    ["python", "docs/pi_self_heal_hooks.py"],
    ["pytest", "-q"],
    ["ruff", "check", "."],
    ["mypy", "neurophase"],
    ["python", "docs/audit_evidence_labels.py"],
    ["python", "docs/validate_bibliography_dois.py"],
    ["python", "docs/final_diff_validation.py"],
    ["python", "docs/evidence_enforcer.py"],
    ["python", "docs/oracle/evidence_oracle.py"],
]


def main() -> int:
    for cmd in COMMANDS:
        print(f"\n[Ω] running: {' '.join(cmd)}")
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            print(f"[Ω] FAIL at: {' '.join(cmd)}")
            return completed.returncode
    print("\n[Ω] PASS - zero-defect governance kernel completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
