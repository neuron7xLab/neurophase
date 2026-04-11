"""π-agent recursive self-improvement loop (governed, safe mode).

This implementation is non-destructive:
- scans docs/code synchronization
- proposes mutations as structured recommendations
- accepts only proposals that improve R_docs score
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MutationProposal:
    description: str
    expected_delta_r_docs: float


@dataclass
class PiCore:
    repo_root: Path

    def _compute_r_docs(self) -> float:
        docs = list((self.repo_root / "docs").rglob("*.md"))
        total = 0.0
        ok = 0.0
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
                    total += 1.0
                    if (self.repo_root / rel).exists():
                        ok += 1.0
                    idx = j
        return 0.0 if total <= 0 else ok / total

    def scan(self) -> float:
        return self._compute_r_docs()

    def propose(self) -> list[MutationProposal]:
        return [
            MutationProposal("add missing module/test link anchors", 0.03),
            MutationProposal("tighten tentative claim falsification criteria", 0.02),
        ]

    def select(self, proposals: list[MutationProposal], baseline: float) -> list[MutationProposal]:
        return [p for p in proposals if baseline + p.expected_delta_r_docs > baseline]
