from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neurophase.governance.ablation import (
    DEFAULT_ABLATION_POLICY_PATH,
    AblationPolicyError,
    load_ablation_policy,
)
from neurophase.governance.checklist import (
    DEFAULT_CHECKLIST_PATH,
    GovernanceChecklistError,
    load_checklist,
)
from neurophase.governance.owner_manifest import (
    DEFAULT_OWNER_MANIFEST_PATH,
    OwnerManifestError,
    load_owner_manifest,
    manifest_hash,
)


def test_owner_manifest_loads_and_hash_matches() -> None:
    manifest = load_owner_manifest()
    assert manifest.owner == "neuron7x"
    assert manifest.date == "2026-04-19"
    assert manifest.hash == manifest_hash(manifest.owner, manifest.date)


def test_owner_manifest_rejects_wrong_hash(tmp_path: Path) -> None:
    path = tmp_path / "owner_manifest.yaml"
    path.write_text(
        yaml.safe_dump({"owner": "neuron7x", "date": "2026-04-19", "hash": "bad"}),
        encoding="utf-8",
    )
    with pytest.raises(OwnerManifestError, match="hash mismatch"):
        load_owner_manifest(path)


def test_owner_manifest_default_path_is_repo_root() -> None:
    assert DEFAULT_OWNER_MANIFEST_PATH.name == "owner_manifest.yaml"
    assert DEFAULT_OWNER_MANIFEST_PATH.is_file()


def test_checklist_loader_validates_required_statuses() -> None:
    checklist = load_checklist()
    assert checklist.required_source_items == 211
    assert checklist.by_id("gov_2").status == "pass"
    assert checklist.by_id("final_1").status == "pass"
    assert checklist.by_id("final_2").status == "pass"
    assert checklist.verdict == "DONE"


def test_checklist_rejects_missing_required_pass(tmp_path: Path) -> None:
    source = tmp_path / "openai_gpt_2026_checklist_2026-04-19.yaml"
    source.write_text(
        yaml.safe_dump({"version": 1, "items": [{"id": "x", "status": "pass"}]}),
        encoding="utf-8",
    )
    path = tmp_path / "GOVERNANCE_CHECKLIST.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "verdict": "DONE",
                "source_checklist_path": source.name,
                "required_source_items": 1,
                "source_items_count": 1,
                "items": [
                    {"id": "gov_2", "status": "fail", "note": "x"},
                    {"id": "final_1", "status": "pass", "note": "x"},
                    {"id": "final_2", "status": "pass", "note": "x"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(GovernanceChecklistError, match="gov_2"):
        load_checklist(path)


def test_checklist_default_path_is_repo_root() -> None:
    assert DEFAULT_CHECKLIST_PATH.name == "GOVERNANCE_CHECKLIST.yaml"
    assert DEFAULT_CHECKLIST_PATH.is_file()


def test_checklist_rejects_source_partial_status(tmp_path: Path) -> None:
    source = tmp_path / "openai_gpt_2026_checklist_2026-04-19.yaml"
    source.write_text(
        yaml.safe_dump(
            {"version": 1, "items": [{"id": "x", "status": "partial"}]}
        ),
        encoding="utf-8",
    )
    checklist = tmp_path / "GOVERNANCE_CHECKLIST.yaml"
    checklist.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "verdict": "DONE",
                "source_checklist_path": source.name,
                "required_source_items": 1,
                "source_items_count": 1,
                "items": [
                    {"id": "gov_2", "status": "pass", "note": "x"},
                    {"id": "final_1", "status": "pass", "note": "x"},
                    {"id": "final_2", "status": "pass", "note": "x"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(GovernanceChecklistError, match="non-pass"):
        load_checklist(checklist)


def test_ablation_policy_loads_and_is_mechanically_bound() -> None:
    policy = load_ablation_policy()
    assert policy.version == 2
    assert policy.mutation_suite == "tests/test_fail_closed_mutation.py"
    assert set(policy.test_registry.keys()) == set(policy.critical_elements)
    assert "G1" in policy.critical_elements
    assert "G21" in policy.critical_elements


def test_ablation_policy_rejects_missing_critical_test(tmp_path: Path) -> None:
    suite = tmp_path / "suite.py"
    suite.write_text("def test_G1_placeholder():\n    assert True\n", encoding="utf-8")
    policy_path = tmp_path / "ABLATION_POLICY.yaml"
    mutation_suite_rel = str(suite.relative_to(tmp_path))
    policy_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "mutation_suite": mutation_suite_rel,
                "critical_elements": ["G1", "G2"],
                "test_registry": {
                    "G1": f"{mutation_suite_rel}::test_G1_placeholder",
                    "G2": f"{mutation_suite_rel}::test_G2_missing",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AblationPolicyError, match="critical elements without mutation tests"):
        load_ablation_policy(policy_path)


def test_ablation_policy_comment_marker_satisfies_simple_string_check(tmp_path: Path) -> None:
    suite = tmp_path / "suite.py"
    suite.write_text("# def test_G2_fake():\n\ndef test_G1_real():\n    assert True\n", encoding="utf-8")
    policy_path = tmp_path / "ABLATION_POLICY.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "mutation_suite": "suite.py",
                "critical_elements": ["G1", "G2"],
                "test_registry": {
                    "G1": "suite.py::test_G1_real",
                    "G2": "suite.py::test_G2_fake",
                },
            }
        ),
        encoding="utf-8",
    )
    policy = load_ablation_policy(policy_path)
    assert "G2" in policy.critical_elements


def test_ablation_policy_tolerates_unparseable_suite_under_string_mode(tmp_path: Path) -> None:
    suite = tmp_path / "suite.py"
    suite.write_text("def test_G1_broken(:\n", encoding="utf-8")
    policy_path = tmp_path / "ABLATION_POLICY.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "mutation_suite": "suite.py",
                "critical_elements": ["G1"],
                "test_registry": {
                    "G1": "suite.py::test_G1_broken",
                },
            }
        ),
        encoding="utf-8",
    )
    policy = load_ablation_policy(policy_path)
    assert policy.test_registry["G1"] == "suite.py::test_G1_broken"


def test_ablation_policy_default_path_is_repo_root() -> None:
    assert DEFAULT_ABLATION_POLICY_PATH.name == "ABLATION_POLICY.yaml"
    assert DEFAULT_ABLATION_POLICY_PATH.is_file()
