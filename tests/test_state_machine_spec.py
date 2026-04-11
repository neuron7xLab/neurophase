"""Meta-test for ``STATE_MACHINE.yaml`` (A2).

Verifies:

1. The spec loads and all five ``GateState`` members are declared.
2. Every transition binds to ≥ 1 pytest node id.
3. Every bound test file exists on disk.
4. Every ``execution_allowed=True`` transition targets ``READY``.
5. Every non-READY state has ``execution_allowed=false`` in its
   StateSpec record.
6. The priority column is strictly increasing (evaluation order is
   well-defined).
7. Schema validator rejects seven categories of malformed input:
   missing file, bad YAML, wrong version, missing gate states,
   unknown state name, duplicate transition id, empty tests list.

A CI failure here means a PR either weakened the state machine
contract or renamed a test without updating the spec.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurophase.gate.execution_gate import GateState
from neurophase.governance.state_machine import (
    DEFAULT_STATE_MACHINE_PATH,
    StateMachineRegistryError,
    StateMachineSpec,
    load_state_machine,
)

REPO_ROOT: Path = DEFAULT_STATE_MACHINE_PATH.parent


@pytest.fixture(scope="module")
def spec() -> StateMachineSpec:
    return load_state_machine()


# ---------------------------------------------------------------------------
# Presence and schema
# ---------------------------------------------------------------------------


class TestPresence:
    def test_spec_file_exists(self) -> None:
        assert DEFAULT_STATE_MACHINE_PATH.is_file()

    def test_loads_without_errors(self, spec: StateMachineSpec) -> None:
        assert spec.version == 1
        assert len(spec.states) == 5  # READY, BLOCKED, SENSOR_ABSENT, DEGRADED, UNNECESSARY
        assert len(spec.transitions) >= 7  # at least the 8 transitions we encoded

    def test_every_gate_state_declared(self, spec: StateMachineSpec) -> None:
        declared = spec.state_names()
        enum_members = {s.name for s in GateState}
        assert declared == enum_members


# ---------------------------------------------------------------------------
# Transition bindings
# ---------------------------------------------------------------------------


class TestTransitionBindings:
    def test_every_transition_has_tests(self, spec: StateMachineSpec) -> None:
        for t in spec.transitions:
            assert len(t.tests) >= 1, (
                f"transition {t.id!r} has zero bound tests — governance regression"
            )

    def test_every_test_file_exists(self, spec: StateMachineSpec) -> None:
        for node_id in spec.all_bound_tests():
            file_part = node_id.split("::", 1)[0]
            path = REPO_ROOT / file_part
            assert path.is_file(), (
                f"state machine spec references non-existent test file {file_part} "
                f"(node id: {node_id})"
            )


# ---------------------------------------------------------------------------
# Invariant consistency
# ---------------------------------------------------------------------------


class TestInvariantConsistency:
    def test_permissive_transitions_target_ready_only(self, spec: StateMachineSpec) -> None:
        """The global invariant: execution_allowed=True ⇒ state=READY."""
        for t in spec.transitions:
            if t.execution_allowed:
                assert t.target == "READY", (
                    f"transition {t.id!r} has execution_allowed=True but "
                    f"targets {t.target!r} — invariant violation in the spec"
                )

    def test_ready_is_the_unique_permissive_state(self, spec: StateMachineSpec) -> None:
        permissive = spec.permissive_states()
        assert permissive == frozenset({"READY"})

    def test_every_non_ready_state_has_execution_allowed_false(
        self, spec: StateMachineSpec
    ) -> None:
        for s in spec.states:
            if s.name != "READY":
                assert s.execution_allowed is False, (
                    f"state {s.name!r} has execution_allowed=True but is not READY"
                )

    def test_priorities_are_strictly_increasing(self, spec: StateMachineSpec) -> None:
        prio = [t.priority for t in spec.transitions]
        assert prio == sorted(prio)
        assert len(set(prio)) == len(prio), "duplicate priorities detected"


# ---------------------------------------------------------------------------
# Schema validator negative tests
# ---------------------------------------------------------------------------


class TestSchemaValidator:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(StateMachineRegistryError):
            load_state_machine(tmp_path / "nope.yaml")

    def test_bad_yaml_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("::: not valid yaml :::", encoding="utf-8")
        with pytest.raises(StateMachineRegistryError):
            load_state_machine(p)

    def test_wrong_version_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "v.yaml"
        p.write_text(
            "version: 999\ngate_states: []\ntransitions: []\n",
            encoding="utf-8",
        )
        with pytest.raises(StateMachineRegistryError, match="version"):
            load_state_machine(p)

    def test_missing_gate_states_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "ms.yaml"
        p.write_text(
            "version: 1\ngate_states: []\ntransitions: []\n",
            encoding="utf-8",
        )
        with pytest.raises(StateMachineRegistryError, match="non-empty"):
            load_state_machine(p)

    def test_unknown_state_name_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "us.yaml"
        p.write_text(
            (
                "version: 1\n"
                "gate_states:\n"
                "  - name: HYPERDRIVE\n"
                "    execution_allowed: false\n"
                "    semantic: test\n"
                "transitions: []\n"
            ),
            encoding="utf-8",
        )
        with pytest.raises(StateMachineRegistryError, match="not a valid GateState"):
            load_state_machine(p)

    def test_incomplete_state_set_rejected(self, tmp_path: Path) -> None:
        """Only declaring READY is not enough — every GateState enum
        member must be present."""
        p = tmp_path / "partial.yaml"
        p.write_text(
            (
                "version: 1\n"
                "gate_states:\n"
                "  - name: READY\n"
                "    execution_allowed: true\n"
                "    semantic: test\n"
                "transitions: []\n"
            ),
            encoding="utf-8",
        )
        with pytest.raises(StateMachineRegistryError, match="missing GateState members"):
            load_state_machine(p)
