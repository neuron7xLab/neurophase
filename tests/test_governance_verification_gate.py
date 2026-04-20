from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neurophase.gate.execution_gate import ExecutionGate, GateState
from neurophase.governance.ablation import load_ablation_policy
from neurophase.governance.checklist import governance_closure_valid, load_checklist

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
st = hypothesis.strategies


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


def test_hn39_no_gpt_reference_in_source_checklist_payload() -> None:
    payload = yaml.safe_load(
        (_repo_root() / "mechanical_governance_source_checklist_2026-04-20.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(payload, dict)
    source = payload.get("source")
    assert isinstance(source, str)
    assert "gpt" not in source.lower()
    assert "openai" not in source.lower()


def test_t8_transition_blocks_gate_on_failed_governance(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> object:
        raise ValueError("broken checklist")

    monkeypatch.setattr("neurophase.governance.checklist.load_checklist", _boom)
    with pytest.raises(ValueError, match="T8 governance guard failed"):
        ExecutionGate()


@pytest.mark.parametrize(
    "failure_mode,patch_target,exc",
    [
        ("missing_checklist", "neurophase.governance.checklist.load_checklist", FileNotFoundError),
        ("verdict_not_DONE", "neurophase.governance.checklist.load_checklist", RuntimeError),
        (
            "invalid_owner_hash",
            "neurophase.governance.owner_manifest.load_owner_manifest",
            ValueError,
        ),
        ("unbound_ablation", "neurophase.governance.ablation.load_ablation_policy", ValueError),
    ],
)
def test_hn39_blocks_on_any_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
    patch_target: str,
    exc: type[Exception],
) -> None:
    def _boom() -> object:
        raise exc(failure_mode)

    monkeypatch.setattr(patch_target, _boom)
    assert governance_closure_valid() is False


def test_no_path_to_ready_bypasses_governance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("neurophase.governance.checklist.governance_closure_valid", lambda: False)
    gate = ExecutionGate(enforce_governance=False)
    decision = gate.evaluate(R=0.99)
    assert decision.state is GateState.BLOCKED
    assert decision.execution_allowed is False


def test_mutation_binding_coverage_is_total() -> None:
    policy = load_ablation_policy()
    required = set(policy.critical_elements)
    bound = set(policy.test_registry.keys())
    kill_rate = len(required & bound) / len(required)
    assert kill_rate >= 1.0, "FAIL-CLOSED: ablation binding coverage < 100%"
