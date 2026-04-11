"""Bibliography & honest-citation contract (HN15).

Enforces, at CI time:

1. **No fabricated future citations.** Any markdown file in ``docs/``
   that references a specific researcher + a future year (e.g.
   ``Friston/Clark, 2026``, ``Ming/Wharton (2026)``, ``NIH Resilience
   Consortium 2026``) fails the build. The 2026 year label in
   headers / copyright / "in preparation" self-references is
   allowed; only *researcher + future-year* pairings are rejected.

2. **Elite bibliography presence.** The canonical
   ``docs/theory/neurophase_elite_bibliography.md`` exists, contains
   ≥ 15 real DOI anchors, carries the traceability matrix, and lists
   every evidence-status label.

3. **Evidence-labeling style guide presence.** The
   ``docs/validation/evidence_labeling_style_guide.md`` exists and
   defines all four evidence labels.

4. **Cross-referencing.** The scientific basis docs
   (``docs/theory/scientific_basis.md``,
   ``docs/theory/neurophase_scientific_basis.md``,
   ``docs/science_basis.md``, ``README.md``) reference the elite
   bibliography by relative path.

Earlier drafts of the repo contained the following fabrications:

* ``Clark, 2026`` — no such book
* ``Ming / Wharton, 2026`` — no such paper
* ``NIH Resilience Consortium, 2026`` — no such consortium
* ``Capital-Weighted Kuramoto Working Group (2026)`` — no such group

This test exists specifically to prevent their re-introduction.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"


def _all_docs() -> list[Path]:
    return sorted(DOCS.rglob("*.md"))


# ---------------------------------------------------------------------------
# No fabricated-researcher-2026 citations
# ---------------------------------------------------------------------------


#: Patterns that were fabricated in earlier drafts.
#: If any of these show up verbatim in any docs/ markdown, CI fails.
_FABRICATED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Clark[,\s]*\(?2026\)?", re.IGNORECASE),
    re.compile(r"Friston[,/\s]*Clark[,\s]*\(?2026\)?", re.IGNORECASE),
    re.compile(r"Friston[,\s]*2026", re.IGNORECASE),
    re.compile(r"Ming\s*[/,]\s*Wharton[,\s]*\(?2026\)?", re.IGNORECASE),
    re.compile(r"Wharton[,\s]*\(?2026\)?", re.IGNORECASE),
    re.compile(r"Ming[,\s]*\(?2026\)?", re.IGNORECASE),
    re.compile(r"NIH\s+Resilience\s+Consortium[,\s]*\(?2026\)?", re.IGNORECASE),
    re.compile(r"NIH[,\s]*\(?2026\)?", re.IGNORECASE),
    re.compile(
        r"Capital[- ]Weighted\s+Kuramoto\s+(?:Working\s+Group|WG)[,\s]*\(?2026\)?",
        re.IGNORECASE,
    ),
)

#: Docs that deliberately document the fabricated patterns as a warning
#: (and therefore legitimately mention them). These are allow-listed.
_DOCUMENTATION_ALLOWLIST: frozenset[str] = frozenset(
    {
        "docs/theory/neurophase_elite_bibliography.md",
        "docs/theory/scientific_basis.md",
    }
)


def _strip_documentation_lines(text: str) -> str:
    """Drop lines that explicitly call out the fabrications.

    A line is skipped when it contains words that mark it as a
    warning/explanation rather than a live citation: ``fabricat``,
    ``fabrication``, ``fake``, ``removed``, ``slipped past review``.
    """
    out: list[str] = []
    warning_markers = (
        "fabricat",
        "fake",
        "removed",
        "slipped past review",
    )
    for line in text.splitlines():
        lower = line.lower()
        if any(marker in lower for marker in warning_markers):
            continue
        out.append(line)
    return "\n".join(out)


class TestNoFabricatedCitations:
    """HN15: docs/ must not contain fabricated future-dated citations."""

    @pytest.mark.parametrize("path", _all_docs(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
    def test_no_fabricated_pattern(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(REPO_ROOT))
        # Allow-listed docs mention the patterns as warnings; strip
        # warning lines before scanning so we still catch any *live*
        # citation in the rest of the document.
        if rel in _DOCUMENTATION_ALLOWLIST:
            text = _strip_documentation_lines(text)
        hits: list[str] = []
        for pattern in _FABRICATED_PATTERNS:
            for match in pattern.finditer(text):
                hits.append(match.group(0))
        assert not hits, (
            f"{rel} contains fabricated citations: {hits}. "
            f"Real bibliography lives in docs/theory/neurophase_elite_bibliography.md."
        )

    def test_test_file_is_its_own_exception(self) -> None:
        """The test file itself mentions the patterns by design —
        it must not regress into a prose doc."""
        self_path = REPO_ROOT / "tests" / "test_bibliography_contract.py"
        assert self_path.is_file()


# ---------------------------------------------------------------------------
# Elite bibliography presence and contents
# ---------------------------------------------------------------------------


_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9<>]+")


class TestEliteBibliography:
    @pytest.fixture(scope="class")
    def elite_text(self) -> str:
        path = DOCS / "theory" / "neurophase_elite_bibliography.md"
        assert path.is_file(), f"missing elite bibliography: {path}"
        return path.read_text(encoding="utf-8")

    def test_has_enough_real_dois(self, elite_text: str) -> None:
        dois = sorted(set(_DOI_RE.findall(elite_text)))
        assert len(dois) >= 15, (
            f"elite bibliography has only {len(dois)} unique DOI anchors; "
            f"expected ≥ 15 real peer-reviewed sources"
        )

    def test_has_traceability_matrix(self, elite_text: str) -> None:
        assert re.search(r"traceability\s+matrix", elite_text, re.IGNORECASE) is not None, (
            "elite bibliography must carry a traceability matrix"
        )

    def test_has_integration_readiness_checklist(self, elite_text: str) -> None:
        assert (
            re.search(r"integration\s+readiness\s+checklist", elite_text, re.IGNORECASE) is not None
        ), "elite bibliography must carry an integration readiness checklist"

    def test_has_all_four_evidence_labels(self, elite_text: str) -> None:
        for label in ("Established", "Strongly Plausible", "Tentative", "Unsupported"):
            assert label in elite_text, f"elite bibliography missing evidence label {label!r}"


# ---------------------------------------------------------------------------
# Evidence-labeling style guide
# ---------------------------------------------------------------------------


class TestEvidenceStyleGuide:
    @pytest.fixture(scope="class")
    def style_text(self) -> str:
        path = DOCS / "validation" / "evidence_labeling_style_guide.md"
        assert path.is_file(), f"missing style guide: {path}"
        return path.read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "label",
        ["Established", "Strongly Plausible", "Tentative", "Unsupported"],
    )
    def test_defines_label(self, style_text: str, label: str) -> None:
        assert label in style_text, (
            f"evidence_labeling_style_guide.md must define the {label!r} label"
        )


# ---------------------------------------------------------------------------
# Cross-referencing contract
# ---------------------------------------------------------------------------


class TestCrossReferences:
    """Every scientific basis doc + README must point at the elite bibliography."""

    @pytest.mark.parametrize(
        "rel",
        [
            "docs/theory/scientific_basis.md",
            "docs/theory/neurophase_scientific_basis.md",
            "docs/science_basis.md",
            "README.md",
        ],
    )
    def test_anchors_elite_bibliography(self, rel: str) -> None:
        path = REPO_ROOT / rel
        assert path.is_file(), f"missing {rel}"
        text = path.read_text(encoding="utf-8")
        assert "neurophase_elite_bibliography.md" in text, (
            f"{rel} must reference docs/theory/neurophase_elite_bibliography.md"
        )
