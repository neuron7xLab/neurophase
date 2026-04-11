"""C8 — claim registry: hypothesis → theory → fact status tracker.

A complement to ``INVARIANTS.yaml``. Where the invariant registry
tracks **mechanical contracts** (gate behaviour, runtime safety,
honest naming), the claim registry tracks **scientific claims**
the system makes about the world, each one promoted through three
status levels:

* ``HYPOTHESIS`` — proposed but not yet tested.
* ``THEORY`` — supported by at least one piece of peer-reviewed
  or experimentally-validated evidence.
* ``FACT`` — supported by at least three independent supporting
  citations AND no contradicting citation.

The promotion rule is **mechanical**, not editorial: the loader
verifies that the declared status matches the evidence count and
sign. A claim that is marked ``FACT`` but only carries one
supporting citation fails validation at load time. This makes
the rung on doctrine item *"hypothesis → theory → fact"* a CI-
enforced invariant.

Why this exists
---------------

The doctrine carries a strong commitment: every load-bearing
scientific claim must be traceable to a real, dated, peer-
reviewed source via the bibliography traceability matrix
(HN15). C8 turns that commitment into a **registry** so that:

1. A reviewer can answer *"what does the system claim is true?"*
   by reading one file.
2. A claim cannot drift in status without a corresponding
   change to its evidence list — promotion is mechanically
   gated.
3. A claim that loses its evidence (e.g. a paper retracted)
   demotes back to ``HYPOTHESIS`` automatically as soon as the
   citation is removed.
4. Cross-links from claims to honest-naming contracts make the
   bibliography ↔ contract ↔ test triangle queryable end-to-end.

Contract (HN30)
---------------

* The loader enforces:
  - Unique claim ids.
  - Schema version matches :data:`SCHEMA_VERSION`.
  - Every claim has at least one supporting evidence citation
    (a HYPOTHESIS with zero evidence is allowed; a THEORY/FACT
    with insufficient evidence is rejected).
  - Status matches the evidence count:
    - ``HYPOTHESIS``: 0+ supporting, 0+ contradicting.
    - ``THEORY``: ≥ 1 supporting, 0 contradicting.
    - ``FACT``: ≥ 3 supporting, 0 contradicting.
  - Every cross-referenced ``related_invariants`` HN id exists
    in ``INVARIANTS.yaml`` (no dangling links).
* Determinism: same YAML → bit-identical
  :class:`ClaimRegistry`.

What C8 is NOT
--------------

* It is **not** a literature manager. The registry stores DOIs
  and one-line summaries; the actual papers live in the
  bibliography file.
* It is **not** a peer review system. A claim's status reflects
  the count of supporting citations, not the editorial judgement
  of any reviewer.
* It does **not** promote claims automatically based on test
  results. Tests prove that the *implementation* matches the
  *contract*; promoting a scientific claim from theory to fact
  requires a new citation, not a green test run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

import yaml

from neurophase.governance.invariants import (
    InvariantRegistry,
    load_registry,
)

#: Canonical path of ``CLAIMS.yaml`` relative to the package root.
DEFAULT_CLAIMS_PATH: Final[Path] = Path(__file__).resolve().parent.parent.parent / "CLAIMS.yaml"

#: Supported registry schema version.
SCHEMA_VERSION: Final[int] = 1

#: Minimum number of supporting citations required for FACT.
FACT_MIN_SUPPORTING: Final[int] = 3

#: Minimum number of supporting citations required for THEORY.
THEORY_MIN_SUPPORTING: Final[int] = 1

__all__ = [
    "DEFAULT_CLAIMS_PATH",
    "FACT_MIN_SUPPORTING",
    "SCHEMA_VERSION",
    "THEORY_MIN_SUPPORTING",
    "Claim",
    "ClaimRegistry",
    "ClaimRegistryError",
    "ClaimStatus",
    "EvidenceCitation",
    "load_claims",
]


class ClaimRegistryError(ValueError):
    """Raised when the claim registry fails schema validation."""


class ClaimStatus(Enum):
    """Three-rung status ladder for scientific claims."""

    HYPOTHESIS = "hypothesis"
    """Proposed but not yet supported by published evidence."""

    THEORY = "theory"
    """Supported by ≥ 1 supporting citation, no contradicting evidence."""

    FACT = "fact"
    """Supported by ≥ 3 supporting citations, no contradicting evidence."""


@dataclass(frozen=True)
class EvidenceCitation:
    """One piece of evidence attached to a :class:`Claim`.

    Attributes
    ----------
    source
        Short author/year-style citation key (e.g.
        ``"Friston 2010"``). Free-form but stable.
    doi
        DOI string. Empty when the source is not a DOI-bearing
        publication (e.g. an internal experimental result).
    supports
        ``True`` iff the citation supports the claim. ``False``
        means it contradicts the claim — a contradicting citation
        forces the claim to ``HYPOTHESIS`` regardless of how
        many supporting citations exist.
    summary
        One-line description of how the citation bears on the
        claim.
    """

    source: str
    doi: str
    supports: bool
    summary: str

    def __post_init__(self) -> None:
        if not self.source:
            raise ClaimRegistryError("EvidenceCitation.source must be non-empty")
        if not self.summary:
            raise ClaimRegistryError("EvidenceCitation.summary must be non-empty")


@dataclass(frozen=True, repr=False)
class Claim:
    """One scientific claim in the registry.

    Attributes
    ----------
    id
        Stable short identifier (``"C1"``, ``"C2"``, …).
    statement
        One-line claim. The narrative form lives in the linked
        documentation; the registry holds only the headline.
    status
        Current :class:`ClaimStatus`. The loader verifies the
        status against the evidence count + sign.
    evidence
        Tuple of :class:`EvidenceCitation`. Order is preserved.
    related_invariants
        Tuple of HN ids in ``INVARIANTS.yaml`` that are
        downstream of this claim. The loader verifies every id
        exists.
    introduced_in
        PR or version where the claim was first registered.
    notes
        Optional free-form notes.
    """

    id: str
    statement: str
    status: ClaimStatus
    evidence: tuple[EvidenceCitation, ...]
    related_invariants: tuple[str, ...]
    introduced_in: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ClaimRegistryError("Claim.id must be non-empty")
        if not self.statement:
            raise ClaimRegistryError(f"Claim {self.id!r}.statement must be non-empty")

    def __repr__(self) -> str:  # aesthetic rich repr (HN30)
        icon = {
            ClaimStatus.HYPOTHESIS: "○",
            ClaimStatus.THEORY: "◐",
            ClaimStatus.FACT: "●",
        }[self.status]
        return (
            f"Claim[{icon} {self.id} · {self.status.value} · "
            f"evidence={len(self.evidence)} · "
            f"links={len(self.related_invariants)}]"
        )

    @property
    def n_supporting(self) -> int:
        return sum(1 for e in self.evidence if e.supports)

    @property
    def n_contradicting(self) -> int:
        return sum(1 for e in self.evidence if not e.supports)

    @property
    def headline_dois(self) -> tuple[str, ...]:
        """Return the DOIs of every supporting citation."""
        return tuple(e.doi for e in self.evidence if e.supports and e.doi)


@dataclass(frozen=True, repr=False)
class ClaimRegistry:
    """Full loaded claim registry.

    Attributes
    ----------
    version
        Schema version read from ``CLAIMS.yaml``.
    claims
        Tuple of :class:`Claim` records, in file order.
    path
        Absolute path the registry was loaded from.
    """

    version: int
    claims: tuple[Claim, ...]
    path: Path

    def __repr__(self) -> str:  # aesthetic rich repr (HN30)
        n_hyp = sum(1 for c in self.claims if c.status is ClaimStatus.HYPOTHESIS)
        n_thy = sum(1 for c in self.claims if c.status is ClaimStatus.THEORY)
        n_fact = sum(1 for c in self.claims if c.status is ClaimStatus.FACT)
        return (
            f"ClaimRegistry[v{self.version} · "
            f"○{n_hyp} ◐{n_thy} ●{n_fact} · "
            f"total={len(self.claims)}]"
        )

    def by_id(self, claim_id: str) -> Claim:
        """Return the claim with the given ``id``.

        Raises
        ------
        KeyError
            If no claim with that ``id`` exists.
        """
        for claim in self.claims:
            if claim.id == claim_id:
                return claim
        raise KeyError(f"unknown claim id: {claim_id!r}")

    def by_status(self, status: ClaimStatus) -> tuple[Claim, ...]:
        """Return every claim with the given status, in file order."""
        return tuple(c for c in self.claims if c.status is status)

    def linked_invariants(self) -> tuple[str, ...]:
        """Return every HN id referenced by any claim, deduped, sorted."""
        seen: set[str] = set()
        for claim in self.claims:
            seen.update(claim.related_invariants)
        return tuple(sorted(seen))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_claims(
    path: Path | str | None = None,
    *,
    invariant_registry: InvariantRegistry | None = None,
) -> ClaimRegistry:
    """Load and validate ``CLAIMS.yaml``.

    Parameters
    ----------
    path
        Optional override for the registry path. Defaults to
        :data:`DEFAULT_CLAIMS_PATH`.
    invariant_registry
        Optional pre-loaded :class:`InvariantRegistry` for the
        cross-link check. When ``None``, the loader loads
        ``INVARIANTS.yaml`` from the canonical path.

    Returns
    -------
    ClaimRegistry
        The parsed, schema-validated registry.

    Raises
    ------
    ClaimRegistryError
        If the file is missing, unparseable, schema-invalid, or
        contains a status that does not match the evidence count.
    """
    resolved = Path(path) if path is not None else DEFAULT_CLAIMS_PATH
    if not resolved.is_file():
        raise ClaimRegistryError(f"claim registry not found at {resolved}")

    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ClaimRegistryError(f"claim registry at {resolved} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ClaimRegistryError(f"claim registry at {resolved} must be a mapping at the top level")

    version = raw.get("version")
    if version != SCHEMA_VERSION:
        raise ClaimRegistryError(
            f"claim registry version mismatch: expected {SCHEMA_VERSION}, got {version!r}"
        )

    claims_raw = raw.get("claims", [])
    if not isinstance(claims_raw, list):
        raise ClaimRegistryError("'claims' must be a list")

    inv_reg = invariant_registry if invariant_registry is not None else load_registry()
    valid_hn_ids = {hn.id for hn in inv_reg.honest_naming} | {i.id for i in inv_reg.invariants}

    claims: list[Claim] = []
    seen: set[str] = set()
    for i, entry in enumerate(claims_raw):
        claim = _parse_claim(entry, index=i, valid_hn_ids=valid_hn_ids)
        if claim.id in seen:
            raise ClaimRegistryError(f"duplicate claim id: {claim.id!r} at index {i}")
        seen.add(claim.id)
        claims.append(claim)

    return ClaimRegistry(
        version=version,
        claims=tuple(claims),
        path=resolved,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _parse_claim(entry: Any, *, index: int, valid_hn_ids: set[str]) -> Claim:
    if not isinstance(entry, dict):
        raise ClaimRegistryError(f"claims[{index}] must be a mapping")

    required = ("id", "statement", "status", "evidence", "introduced_in")
    for key in required:
        if key not in entry:
            raise ClaimRegistryError(f"claims[{index}] missing required field {key!r}")

    status_raw = entry["status"]
    try:
        status = ClaimStatus(status_raw)
    except ValueError as exc:
        raise ClaimRegistryError(
            f"claims[{index}] has invalid status {status_raw!r}; "
            f"expected one of {[s.value for s in ClaimStatus]}"
        ) from exc

    evidence_raw = entry["evidence"]
    if not isinstance(evidence_raw, list):
        raise ClaimRegistryError(f"claims[{index}].evidence must be a list")

    evidence: list[EvidenceCitation] = []
    for j, ev in enumerate(evidence_raw):
        if not isinstance(ev, dict):
            raise ClaimRegistryError(f"claims[{index}].evidence[{j}] must be a mapping")
        for key in ("source", "supports", "summary"):
            if key not in ev:
                raise ClaimRegistryError(
                    f"claims[{index}].evidence[{j}] missing required field {key!r}"
                )
        if not isinstance(ev["supports"], bool):
            raise ClaimRegistryError(f"claims[{index}].evidence[{j}].supports must be a boolean")
        evidence.append(
            EvidenceCitation(
                source=str(ev["source"]),
                doi=str(ev.get("doi", "")),
                supports=bool(ev["supports"]),
                summary=str(ev["summary"]),
            )
        )

    related_raw = entry.get("related_invariants", [])
    if not isinstance(related_raw, list):
        raise ClaimRegistryError(f"claims[{index}].related_invariants must be a list if provided")
    related = tuple(str(r) for r in related_raw)
    for hn_id in related:
        if hn_id not in valid_hn_ids:
            raise ClaimRegistryError(
                f"claims[{index}].related_invariants references unknown id "
                f"{hn_id!r}; check INVARIANTS.yaml"
            )

    claim = Claim(
        id=str(entry["id"]),
        statement=str(entry["statement"]),
        status=status,
        evidence=tuple(evidence),
        related_invariants=related,
        introduced_in=str(entry["introduced_in"]),
        notes=str(entry.get("notes", "")),
    )

    # Status-vs-evidence consistency check (the load-bearing
    # mechanical promotion rule).
    n_supp = claim.n_supporting
    n_contra = claim.n_contradicting

    if status is ClaimStatus.FACT:
        if n_contra > 0:
            raise ClaimRegistryError(
                f"claims[{index}] {claim.id!r}: status is FACT but has "
                f"{n_contra} contradicting citation(s)"
            )
        if n_supp < FACT_MIN_SUPPORTING:
            raise ClaimRegistryError(
                f"claims[{index}] {claim.id!r}: status is FACT but only "
                f"{n_supp} supporting citation(s) (need ≥ {FACT_MIN_SUPPORTING})"
            )
    elif status is ClaimStatus.THEORY:
        if n_contra > 0:
            raise ClaimRegistryError(
                f"claims[{index}] {claim.id!r}: status is THEORY but has "
                f"{n_contra} contradicting citation(s)"
            )
        if n_supp < THEORY_MIN_SUPPORTING:
            raise ClaimRegistryError(
                f"claims[{index}] {claim.id!r}: status is THEORY but only "
                f"{n_supp} supporting citation(s) (need ≥ {THEORY_MIN_SUPPORTING})"
            )
    # HYPOTHESIS has no constraints — that is the default rung.

    return claim
