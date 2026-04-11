"""Meta-test for the ``INVARIANTS.yaml`` registry (A1).

This test is the **enforcement mechanism** behind the machine-readable
invariant registry. It does four things:

1. Loads the registry via ``neurophase.governance.load_registry``.
2. Verifies that every registered invariant binds to at least one
   pytest node id.
3. Verifies that every referenced test file exists on disk (a stronger
   check is impossible without collecting the pytest session).
4. Verifies that every enforcement-site path and every doc path exists.

A CI failure here means one of two things:

* A PR weakened an invariant by removing the enforcing test without
  updating ``INVARIANTS.yaml``.
* A PR removed the invariant's enforcement site or documentation.

Either case is a governance regression and must be resolved before
merge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurophase.governance.invariants import (
    DEFAULT_REGISTRY_PATH,
    InvariantRegistry,
    InvariantRegistryError,
    InvariantSeverity,
    load_registry,
)

REPO_ROOT: Path = DEFAULT_REGISTRY_PATH.parent


@pytest.fixture(scope="module")
def registry() -> InvariantRegistry:
    return load_registry()


# ---------------------------------------------------------------------------
# Presence and schema
# ---------------------------------------------------------------------------


class TestRegistryPresence:
    def test_registry_file_exists(self) -> None:
        assert DEFAULT_REGISTRY_PATH.is_file(), (
            f"INVARIANTS.yaml must exist at {DEFAULT_REGISTRY_PATH}"
        )

    def test_registry_loads_without_errors(self, registry: InvariantRegistry) -> None:
        assert registry.version == 1
        assert len(registry.invariants) >= 4

    def test_registry_contains_core_invariants(self, registry: InvariantRegistry) -> None:
        ids = {inv.id for inv in registry.invariants}
        assert {"I1", "I2", "I3", "I4"} <= ids
        # B1 must be in the registry once PR #11 is merged.
        assert "B1" in ids


# ---------------------------------------------------------------------------
# Every invariant must bind to ≥ 1 test
# ---------------------------------------------------------------------------


class TestInvariantTestBinding:
    def test_every_invariant_has_at_least_one_test(self, registry: InvariantRegistry) -> None:
        for inv in registry.invariants:
            assert len(inv.tests) >= 1, (
                f"invariant {inv.id} has zero bound tests — governance regression"
            )

    def test_every_honest_naming_contract_has_at_least_one_test(
        self, registry: InvariantRegistry
    ) -> None:
        for hn in registry.honest_naming:
            assert len(hn.tests) >= 1, f"honest-naming contract {hn.id} has zero bound tests"

    def test_every_test_file_exists(self, registry: InvariantRegistry) -> None:
        """The file portion of every bound test node id must exist."""
        for node_id in registry.all_test_ids():
            file_part = node_id.split("::", 1)[0]
            path = REPO_ROOT / file_part
            assert path.is_file(), (
                f"registry references non-existent test file {file_part} (node id: {node_id})"
            )


# ---------------------------------------------------------------------------
# Enforcement sites and docs exist
# ---------------------------------------------------------------------------


class TestEnforcementSites:
    def test_every_enforcement_path_exists(self, registry: InvariantRegistry) -> None:
        for inv in registry.invariants:
            for site in inv.enforced_in:
                path = REPO_ROOT / site.path
                assert path.is_file(), (
                    f"invariant {inv.id} enforcement site {site.path} does not exist"
                )

    def test_every_doc_path_exists(self, registry: InvariantRegistry) -> None:
        for inv in registry.invariants:
            for doc in inv.docs:
                path = REPO_ROOT / doc
                assert path.is_file(), f"invariant {inv.id} doc path {doc} does not exist"

    def test_honest_naming_paths_exist(self, registry: InvariantRegistry) -> None:
        for hn in registry.honest_naming:
            for site in hn.enforced_in:
                path = REPO_ROOT / site.path
                assert path.is_file(), f"honest-naming contract {hn.id} site {site.path} missing"
            for doc in hn.docs:
                path = REPO_ROOT / doc
                assert path.is_file(), f"honest-naming contract {hn.id} doc {doc} missing"


# ---------------------------------------------------------------------------
# Schema validator negative tests
# ---------------------------------------------------------------------------


class TestSchemaValidator:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InvariantRegistryError):
            load_registry(tmp_path / "nonexistent.yaml")

    def test_bad_yaml_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("::: not valid yaml :::", encoding="utf-8")
        with pytest.raises(InvariantRegistryError):
            load_registry(p)

    def test_wrong_version_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "wrong_version.yaml"
        p.write_text("version: 999\ninvariants:\n  - id: I1\n", encoding="utf-8")
        with pytest.raises(InvariantRegistryError, match="version"):
            load_registry(p)

    def test_empty_invariants_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("version: 1\ninvariants: []\n", encoding="utf-8")
        with pytest.raises(InvariantRegistryError, match="non-empty"):
            load_registry(p)

    def test_missing_required_field_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "incomplete.yaml"
        p.write_text(
            "version: 1\ninvariants:\n  - id: X1\n    symbol: X\n",
            encoding="utf-8",
        )
        with pytest.raises(InvariantRegistryError, match="missing required"):
            load_registry(p)

    def test_zero_tests_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "no_tests.yaml"
        p.write_text(
            "version: 1\n"
            "invariants:\n"
            "  - id: X1\n"
            "    symbol: X\n"
            "    statement: test\n"
            "    severity: hard\n"
            "    enforced_in:\n"
            "      - path: neurophase/gate/execution_gate.py\n"
            "        symbol: X\n"
            "    tests: []\n"
            "    docs: []\n"
            "    introduced_in: test\n",
            encoding="utf-8",
        )
        with pytest.raises(InvariantRegistryError, match="bind ≥ 1 test"):
            load_registry(p)

    def test_duplicate_id_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "dup.yaml"
        p.write_text(
            "version: 1\n"
            "invariants:\n"
            "  - id: X1\n"
            "    symbol: X\n"
            "    statement: test\n"
            "    severity: hard\n"
            "    enforced_in:\n"
            "      - path: neurophase/gate/execution_gate.py\n"
            "        symbol: X\n"
            "    tests:\n"
            "      - tests/test_execution_gate.py::test_blocks_below_threshold\n"
            "    docs: []\n"
            "    introduced_in: test\n"
            "  - id: X1\n"
            "    symbol: Y\n"
            "    statement: test\n"
            "    severity: hard\n"
            "    enforced_in:\n"
            "      - path: neurophase/gate/execution_gate.py\n"
            "        symbol: Y\n"
            "    tests:\n"
            "      - tests/test_execution_gate.py::test_permits_above_threshold\n"
            "    docs: []\n"
            "    introduced_in: test\n",
            encoding="utf-8",
        )
        with pytest.raises(InvariantRegistryError, match="duplicate"):
            load_registry(p)

    def test_bad_severity_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad_severity.yaml"
        p.write_text(
            "version: 1\n"
            "invariants:\n"
            "  - id: X1\n"
            "    symbol: X\n"
            "    statement: test\n"
            "    severity: catastrophic\n"
            "    enforced_in:\n"
            "      - path: neurophase/gate/execution_gate.py\n"
            "        symbol: X\n"
            "    tests:\n"
            "      - tests/test_execution_gate.py::test_blocks_below_threshold\n"
            "    docs: []\n"
            "    introduced_in: test\n",
            encoding="utf-8",
        )
        with pytest.raises(InvariantRegistryError, match="severity"):
            load_registry(p)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


class TestLookup:
    def test_by_id(self, registry: InvariantRegistry) -> None:
        inv = registry.by_id("I1")
        assert inv.severity is InvariantSeverity.HARD
        assert "R(t) < threshold" in inv.statement

    def test_by_id_unknown_raises(self, registry: InvariantRegistry) -> None:
        with pytest.raises(KeyError):
            registry.by_id("I999")
