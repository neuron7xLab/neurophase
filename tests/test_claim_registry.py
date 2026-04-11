"""C8 — contract tests for the claim registry.

This test file is the HN30 binding. It locks in:

1. **Schema correctness.** ``CLAIMS.yaml`` parses cleanly and
   conforms to schema version 1.
2. **Mechanical promotion rule.** A claim marked ``FACT`` MUST
   carry ≥ 3 supporting citations and zero contradicting; a
   ``THEORY`` MUST carry ≥ 1 supporting and zero contradicting;
   a ``HYPOTHESIS`` is unconstrained. The loader rejects status
   inconsistencies at load time.
3. **Cross-link integrity.** Every ``related_invariants`` HN id
   declared in a claim must exist in ``INVARIANTS.yaml``. A
   dangling reference fails the loader.
4. **Determinism.** Two loads of the same file produce equal
   :class:`ClaimRegistry` objects.
5. **Frozen dataclasses.** :class:`Claim` and :class:`ClaimRegistry`
   reject attribute reassignment.
6. **Invariant cross-link sanity.** Every committed claim
   currently links to ≥ 1 honest-naming or invariant id (no
   orphan claims) and every linked id is real.
7. **Aesthetic.** Rich __repr__ on both :class:`Claim` and
   :class:`ClaimRegistry`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neurophase.governance.claims import (
    DEFAULT_CLAIMS_PATH,
    FACT_MIN_SUPPORTING,
    SCHEMA_VERSION,
    THEORY_MIN_SUPPORTING,
    Claim,
    ClaimRegistry,
    ClaimRegistryError,
    ClaimStatus,
    EvidenceCitation,
    load_claims,
)
from neurophase.governance.invariants import load_registry

INVARIANT_REGISTRY = load_registry()


# ---------------------------------------------------------------------------
# Fixture helpers.
# ---------------------------------------------------------------------------


def _write_claims(tmp_path: Path, claims_payload: list[dict[str, object]]) -> Path:
    path = tmp_path / "CLAIMS.yaml"
    payload = {"version": SCHEMA_VERSION, "claims": claims_payload}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _supporting(source: str = "Smith 2020") -> dict[str, object]:
    return {
        "source": source,
        "doi": "10.0000/test",
        "supports": True,
        "summary": "supporting evidence",
    }


def _contradicting(source: str = "Jones 2021") -> dict[str, object]:
    return {
        "source": source,
        "doi": "10.0000/contra",
        "supports": False,
        "summary": "contradicting evidence",
    }


# ---------------------------------------------------------------------------
# 1. Schema correctness on the committed CLAIMS.yaml.
# ---------------------------------------------------------------------------


class TestCommittedRegistry:
    def test_committed_file_loads(self) -> None:
        registry = load_claims(invariant_registry=INVARIANT_REGISTRY)
        assert registry.version == SCHEMA_VERSION
        assert len(registry.claims) >= 1

    def test_default_path_is_repo_root(self) -> None:
        assert DEFAULT_CLAIMS_PATH.name == "CLAIMS.yaml"
        assert DEFAULT_CLAIMS_PATH.is_file()

    def test_every_committed_claim_has_unique_id(self) -> None:
        registry = load_claims(invariant_registry=INVARIANT_REGISTRY)
        ids = [c.id for c in registry.claims]
        assert len(ids) == len(set(ids))

    def test_every_committed_claim_links_to_at_least_one_invariant(
        self,
    ) -> None:
        """Every claim in the committed registry must declare at
        least one related_invariants link. Orphan claims (no
        cross-references to the invariant registry) are not
        currently allowed."""
        registry = load_claims(invariant_registry=INVARIANT_REGISTRY)
        for claim in registry.claims:
            assert len(claim.related_invariants) >= 1, f"claim {claim.id} has no related_invariants"

    def test_every_linked_invariant_is_real(self) -> None:
        registry = load_claims(invariant_registry=INVARIANT_REGISTRY)
        valid = {hn.id for hn in INVARIANT_REGISTRY.honest_naming} | {
            i.id for i in INVARIANT_REGISTRY.invariants
        }
        for claim in registry.claims:
            for linked in claim.related_invariants:
                assert linked in valid, f"claim {claim.id} links to unknown id {linked}"


# ---------------------------------------------------------------------------
# 2. Mechanical promotion rule.
# ---------------------------------------------------------------------------


class TestPromotionRule:
    def test_fact_with_three_supporting_loads(self, tmp_path: Path) -> None:
        path = _write_claims(
            tmp_path,
            [
                {
                    "id": "T1",
                    "statement": "test claim",
                    "status": "fact",
                    "evidence": [
                        _supporting("A"),
                        _supporting("B"),
                        _supporting("C"),
                    ],
                    "introduced_in": "test",
                    "related_invariants": [],
                }
            ],
        )
        registry = load_claims(path, invariant_registry=INVARIANT_REGISTRY)
        assert registry.claims[0].status is ClaimStatus.FACT

    def test_fact_with_two_supporting_rejected(self, tmp_path: Path) -> None:
        path = _write_claims(
            tmp_path,
            [
                {
                    "id": "T2",
                    "statement": "test claim",
                    "status": "fact",
                    "evidence": [_supporting("A"), _supporting("B")],
                    "introduced_in": "test",
                    "related_invariants": [],
                }
            ],
        )
        with pytest.raises(ClaimRegistryError, match="FACT"):
            load_claims(path, invariant_registry=INVARIANT_REGISTRY)

    def test_fact_with_contradicting_rejected(self, tmp_path: Path) -> None:
        path = _write_claims(
            tmp_path,
            [
                {
                    "id": "T3",
                    "statement": "test claim",
                    "status": "fact",
                    "evidence": [
                        _supporting("A"),
                        _supporting("B"),
                        _supporting("C"),
                        _contradicting("D"),
                    ],
                    "introduced_in": "test",
                    "related_invariants": [],
                }
            ],
        )
        with pytest.raises(ClaimRegistryError, match="contradicting"):
            load_claims(path, invariant_registry=INVARIANT_REGISTRY)

    def test_theory_with_one_supporting_loads(self, tmp_path: Path) -> None:
        path = _write_claims(
            tmp_path,
            [
                {
                    "id": "T4",
                    "statement": "test claim",
                    "status": "theory",
                    "evidence": [_supporting("A")],
                    "introduced_in": "test",
                    "related_invariants": [],
                }
            ],
        )
        registry = load_claims(path, invariant_registry=INVARIANT_REGISTRY)
        assert registry.claims[0].status is ClaimStatus.THEORY

    def test_theory_with_zero_supporting_rejected(self, tmp_path: Path) -> None:
        path = _write_claims(
            tmp_path,
            [
                {
                    "id": "T5",
                    "statement": "test claim",
                    "status": "theory",
                    "evidence": [],
                    "introduced_in": "test",
                    "related_invariants": [],
                }
            ],
        )
        with pytest.raises(ClaimRegistryError, match="THEORY"):
            load_claims(path, invariant_registry=INVARIANT_REGISTRY)

    def test_theory_with_contradicting_rejected(self, tmp_path: Path) -> None:
        path = _write_claims(
            tmp_path,
            [
                {
                    "id": "T6",
                    "statement": "test claim",
                    "status": "theory",
                    "evidence": [_supporting("A"), _contradicting("B")],
                    "introduced_in": "test",
                    "related_invariants": [],
                }
            ],
        )
        with pytest.raises(ClaimRegistryError, match="contradicting"):
            load_claims(path, invariant_registry=INVARIANT_REGISTRY)

    def test_hypothesis_with_zero_evidence_loads(self, tmp_path: Path) -> None:
        path = _write_claims(
            tmp_path,
            [
                {
                    "id": "T7",
                    "statement": "test claim",
                    "status": "hypothesis",
                    "evidence": [],
                    "introduced_in": "test",
                    "related_invariants": [],
                }
            ],
        )
        registry = load_claims(path, invariant_registry=INVARIANT_REGISTRY)
        assert registry.claims[0].status is ClaimStatus.HYPOTHESIS

    def test_hypothesis_with_contradicting_loads(self, tmp_path: Path) -> None:
        """A claim with mixed evidence stays at HYPOTHESIS — it
        is the unconstrained rung."""
        path = _write_claims(
            tmp_path,
            [
                {
                    "id": "T8",
                    "statement": "test claim",
                    "status": "hypothesis",
                    "evidence": [_supporting("A"), _contradicting("B")],
                    "introduced_in": "test",
                    "related_invariants": [],
                }
            ],
        )
        registry = load_claims(path, invariant_registry=INVARIANT_REGISTRY)
        assert registry.claims[0].status is ClaimStatus.HYPOTHESIS


# ---------------------------------------------------------------------------
# 3. Cross-link integrity.
# ---------------------------------------------------------------------------


class TestCrossLinks:
    def test_dangling_invariant_link_rejected(self, tmp_path: Path) -> None:
        path = _write_claims(
            tmp_path,
            [
                {
                    "id": "T9",
                    "statement": "test claim",
                    "status": "hypothesis",
                    "evidence": [],
                    "introduced_in": "test",
                    "related_invariants": ["HN9999"],  # nonexistent
                }
            ],
        )
        with pytest.raises(ClaimRegistryError, match="HN9999"):
            load_claims(path, invariant_registry=INVARIANT_REGISTRY)

    def test_real_invariant_link_accepted(self, tmp_path: Path) -> None:
        path = _write_claims(
            tmp_path,
            [
                {
                    "id": "T10",
                    "statement": "test claim",
                    "status": "hypothesis",
                    "evidence": [],
                    "introduced_in": "test",
                    "related_invariants": ["HN1"],
                }
            ],
        )
        registry = load_claims(path, invariant_registry=INVARIANT_REGISTRY)
        assert registry.claims[0].related_invariants == ("HN1",)


# ---------------------------------------------------------------------------
# 4. Schema-level validation.
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ClaimRegistryError, match="not found"):
            load_claims(tmp_path / "missing.yaml")

    def test_wrong_version_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump({"version": 999, "claims": []}), encoding="utf-8")
        with pytest.raises(ClaimRegistryError, match="version"):
            load_claims(path, invariant_registry=INVARIANT_REGISTRY)

    def test_duplicate_id_rejected(self, tmp_path: Path) -> None:
        path = _write_claims(
            tmp_path,
            [
                {
                    "id": "DUP",
                    "statement": "first",
                    "status": "hypothesis",
                    "evidence": [],
                    "introduced_in": "test",
                    "related_invariants": [],
                },
                {
                    "id": "DUP",
                    "statement": "second",
                    "status": "hypothesis",
                    "evidence": [],
                    "introduced_in": "test",
                    "related_invariants": [],
                },
            ],
        )
        with pytest.raises(ClaimRegistryError, match="duplicate"):
            load_claims(path, invariant_registry=INVARIANT_REGISTRY)

    def test_invalid_status_rejected(self, tmp_path: Path) -> None:
        path = _write_claims(
            tmp_path,
            [
                {
                    "id": "T11",
                    "statement": "test",
                    "status": "speculation",  # not in enum
                    "evidence": [],
                    "introduced_in": "test",
                    "related_invariants": [],
                }
            ],
        )
        with pytest.raises(ClaimRegistryError, match="invalid status"):
            load_claims(path, invariant_registry=INVARIANT_REGISTRY)

    def test_supports_must_be_bool(self, tmp_path: Path) -> None:
        path = _write_claims(
            tmp_path,
            [
                {
                    "id": "T12",
                    "statement": "test",
                    "status": "hypothesis",
                    "evidence": [
                        {
                            "source": "X",
                            "doi": "",
                            "supports": "yes",  # not a bool
                            "summary": "x",
                        }
                    ],
                    "introduced_in": "test",
                    "related_invariants": [],
                }
            ],
        )
        with pytest.raises(ClaimRegistryError, match="supports"):
            load_claims(path, invariant_registry=INVARIANT_REGISTRY)


# ---------------------------------------------------------------------------
# 5. Determinism.
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_loads_equal(self) -> None:
        a = load_claims(invariant_registry=INVARIANT_REGISTRY)
        b = load_claims(invariant_registry=INVARIANT_REGISTRY)
        assert a == b


# ---------------------------------------------------------------------------
# 6. Frozen dataclasses.
# ---------------------------------------------------------------------------


class TestFrozen:
    def test_claim_is_frozen(self) -> None:
        registry = load_claims(invariant_registry=INVARIANT_REGISTRY)
        claim = registry.claims[0]
        with pytest.raises((AttributeError, TypeError)):
            claim.status = ClaimStatus.HYPOTHESIS  # type: ignore[misc]

    def test_evidence_citation_is_frozen(self) -> None:
        ec = EvidenceCitation(source="X", doi="10.0/y", supports=True, summary="x")
        with pytest.raises((AttributeError, TypeError)):
            ec.doi = "10.0/changed"  # type: ignore[misc]

    def test_registry_is_frozen(self) -> None:
        registry = load_claims(invariant_registry=INVARIANT_REGISTRY)
        with pytest.raises((AttributeError, TypeError)):
            registry.version = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 7. Aesthetic rich __repr__.
# ---------------------------------------------------------------------------


class TestRepr:
    def test_claim_repr_contains_status_icon(self) -> None:
        registry = load_claims(invariant_registry=INVARIANT_REGISTRY)
        for claim in registry.claims:
            r = repr(claim)
            assert r.startswith("Claim[")
            assert claim.id in r
            # Status icon must be one of the three.
            assert any(icon in r for icon in ("○", "◐", "●"))

    def test_registry_repr_shows_status_breakdown(self) -> None:
        registry = load_claims(invariant_registry=INVARIANT_REGISTRY)
        r = repr(registry)
        assert r.startswith("ClaimRegistry[")
        assert f"v{SCHEMA_VERSION}" in r
        assert "○" in r
        assert "◐" in r
        assert "●" in r


# ---------------------------------------------------------------------------
# 8. Helper API.
# ---------------------------------------------------------------------------


class TestHelperAPI:
    def test_by_id_returns_claim(self) -> None:
        registry = load_claims(invariant_registry=INVARIANT_REGISTRY)
        first = registry.claims[0]
        assert registry.by_id(first.id) is first

    def test_by_id_unknown_raises(self) -> None:
        registry = load_claims(invariant_registry=INVARIANT_REGISTRY)
        with pytest.raises(KeyError):
            registry.by_id("DEFINITELY_NOT_A_CLAIM")

    def test_by_status_filters(self) -> None:
        registry = load_claims(invariant_registry=INVARIANT_REGISTRY)
        for status in ClaimStatus:
            for claim in registry.by_status(status):
                assert claim.status is status

    def test_linked_invariants_dedupes_and_sorts(self) -> None:
        registry = load_claims(invariant_registry=INVARIANT_REGISTRY)
        linked = registry.linked_invariants()
        assert linked == tuple(sorted(set(linked)))

    def test_n_supporting_n_contradicting(self) -> None:
        registry = load_claims(invariant_registry=INVARIANT_REGISTRY)
        for claim in registry.claims:
            assert claim.n_supporting == sum(1 for e in claim.evidence if e.supports)
            assert claim.n_contradicting == sum(1 for e in claim.evidence if not e.supports)

    def test_constants_match_promotion_rule(self) -> None:
        assert FACT_MIN_SUPPORTING == 3
        assert THEORY_MIN_SUPPORTING == 1


# ---------------------------------------------------------------------------
# 9. Mypy type sanity hook.
# ---------------------------------------------------------------------------


def _type_shape_hook() -> tuple[ClaimRegistry, Claim, ClaimStatus, EvidenceCitation]:
    """Pin the public type shape; mypy --strict rejects drift."""
    registry: ClaimRegistry = load_claims(invariant_registry=INVARIANT_REGISTRY)
    claim: Claim = registry.claims[0]
    status: ClaimStatus = claim.status
    ev: EvidenceCitation = (
        claim.evidence[0]
        if claim.evidence
        else EvidenceCitation(source="x", doi="", supports=True, summary="x")
    )
    return (registry, claim, status, ev)
