from __future__ import annotations

from pathlib import Path

import pytest

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
st = hypothesis.strategies

from neurophase.gate.execution_gate import ExecutionGate
from neurophase.governance.ablation import load_ablation_policy
from neurophase.governance.checklist import load_checklist


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.mark.strict
def test_governance_checklist_schema_strict() -> None:
    checklist = load_checklist()
    assert checklist.verdict == "DONE"
    assert checklist.required_source_items >= 211
    assert checklist.source_items_count == checklist.required_source_items


@given(st.sampled_from([p.name for p in _repo_root().glob("*.yaml")]))
def test_hn39_no_gpt_labeled_yaml_artifacts(artifact_name: str) -> None:
    assert "gpt" not in artifact_name.lower()


def test_t8_transition_blocks_gate_on_failed_governance(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> object:
        raise ValueError("broken checklist")

    monkeypatch.setattr("neurophase.gate.execution_gate.load_checklist", _boom)
    with pytest.raises(ValueError, match="T8 governance guard failed"):
        ExecutionGate()


def test_mutation_binding_coverage_is_total() -> None:
    policy = load_ablation_policy()
    required = set(policy.critical_elements)
    bound = set(policy.test_registry.keys())
    kill_rate = len(required & bound) / len(required)
    assert kill_rate >= 1.0, "FAIL-CLOSED: ablation binding coverage < 100%"
