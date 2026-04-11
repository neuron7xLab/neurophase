"""Predictive documentation dynamics using Kuramoto-inspired synchronization."""

from __future__ import annotations

import math
from pathlib import Path


def _safe_ratio(num: float, den: float) -> float:
    return 0.0 if den <= 0 else num / den


def compute_r_docs(repo_root: Path) -> float:
    """Compute R_docs in [0,1] from docs-code link coherence.

    Heuristic: compares referenced module paths in docs with actual file existence.
    """
    docs = list((repo_root / "docs").rglob("*.md"))
    total_refs = 0.0
    valid_refs = 0.0

    for d in docs:
        text = d.read_text(encoding="utf-8")
        for token in ["neurophase/", "tests/test_"]:
            idx = 0
            while True:
                i = text.find(token, idx)
                if i < 0:
                    break
                j = i
                while j < len(text) and text[j] not in " \n`|)":
                    j += 1
                rel = text[i:j].strip()
                total_refs += 1.0
                if (repo_root / rel).exists():
                    valid_refs += 1.0
                idx = j

    coherence = _safe_ratio(valid_refs, total_refs)
    # Kuramoto-like order parameter with unit phase approximation.
    return max(0.0, min(1.0, abs(math.cos(math.pi * (1.0 - coherence)))))
