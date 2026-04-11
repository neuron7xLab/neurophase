"""Loader and typed records for ``STATE_MACHINE.yaml`` — task A2.

The state machine spec is the canonical declarative specification of
the 5-state ``ExecutionGate``. It is loaded and validated here, and
its bindings are verified by ``tests/test_state_machine_spec.py``.

This module mirrors the shape of ``neurophase.governance.invariants``
— it provides typed records, a loader, and a schema validator that
raises :class:`StateMachineRegistryError` on any malformed input. No
runtime enforcement — the spec is consumed by CI only.

Load-bearing contract: every :class:`Transition` must have at least
one bound pytest node id, and every ``target`` / ``from_state`` must
be a valid :class:`~neurophase.gate.execution_gate.GateState` member.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from neurophase.gate.execution_gate import GateState

#: Canonical path of ``STATE_MACHINE.yaml`` relative to the package root.
DEFAULT_STATE_MACHINE_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent.parent / "STATE_MACHINE.yaml"
)

#: Supported schema version.
_SCHEMA_VERSION: Final[int] = 1


class StateMachineRegistryError(ValueError):
    """Raised when ``STATE_MACHINE.yaml`` fails schema validation."""


@dataclass(frozen=True)
class StateSpec:
    """Typed record of a single state in the spec."""

    name: str
    execution_allowed: bool
    semantic: str


@dataclass(frozen=True)
class Transition:
    """Typed record of a single transition rule."""

    id: str
    priority: int
    from_state: str
    trigger: str
    guard: str
    target: str
    execution_allowed: bool
    invariant: str | None
    tests: tuple[str, ...]


@dataclass(frozen=True)
class GlobalInvariant:
    """Typed record of the state-machine-wide invariant."""

    statement: str
    enforced_in: tuple[tuple[str, str], ...]
    tests: tuple[str, ...]


@dataclass(frozen=True)
class StateMachineSpec:
    """Full loaded spec."""

    version: int
    states: tuple[StateSpec, ...]
    transitions: tuple[Transition, ...]
    global_invariant: GlobalInvariant
    path: Path

    def state_names(self) -> frozenset[str]:
        return frozenset(s.name for s in self.states)

    def permissive_states(self) -> frozenset[str]:
        return frozenset(s.name for s in self.states if s.execution_allowed)

    def all_bound_tests(self) -> tuple[str, ...]:
        out: list[str] = []
        for t in self.transitions:
            out.extend(t.tests)
        out.extend(self.global_invariant.tests)
        return tuple(out)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_state_machine(path: Path | str | None = None) -> StateMachineSpec:
    """Load and validate ``STATE_MACHINE.yaml``.

    Raises :class:`StateMachineRegistryError` on any malformed input
    (missing file, bad YAML, wrong version, unknown state name,
    invalid execution_allowed flag, empty test bindings).
    """
    resolved = Path(path) if path is not None else DEFAULT_STATE_MACHINE_PATH
    if not resolved.is_file():
        raise StateMachineRegistryError(f"state machine spec not found at {resolved}")

    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise StateMachineRegistryError(
            f"state machine spec at {resolved} is not valid YAML: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise StateMachineRegistryError(
            f"state machine spec at {resolved} must be a top-level mapping"
        )

    version = raw.get("version")
    if version != _SCHEMA_VERSION:
        raise StateMachineRegistryError(
            f"state machine spec version mismatch: expected {_SCHEMA_VERSION}, got {version!r}"
        )

    states = _parse_states(raw.get("gate_states", []))
    transitions = _parse_transitions(
        raw.get("transitions", []), valid_states=frozenset(s.name for s in states)
    )
    global_invariant = _parse_global_invariant(raw.get("global_invariant"))

    return StateMachineSpec(
        version=version,
        states=tuple(states),
        transitions=tuple(transitions),
        global_invariant=global_invariant,
        path=resolved,
    )


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_states(raw: Any) -> list[StateSpec]:
    if not isinstance(raw, list) or not raw:
        raise StateMachineRegistryError(
            "state machine spec must contain a non-empty 'gate_states' list"
        )
    out: list[StateSpec] = []
    enum_members = {s.name for s in GateState}
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise StateMachineRegistryError(f"gate_states[{i}] must be a mapping")
        for key in ("name", "execution_allowed", "semantic"):
            if key not in entry:
                raise StateMachineRegistryError(f"gate_states[{i}] missing required field {key!r}")
        name = str(entry["name"])
        if name not in enum_members:
            raise StateMachineRegistryError(
                f"gate_states[{i}].name={name!r} is not a valid GateState member "
                f"(expected one of {sorted(enum_members)})"
            )
        out.append(
            StateSpec(
                name=name,
                execution_allowed=bool(entry["execution_allowed"]),
                semantic=str(entry["semantic"]),
            )
        )
    # Every GateState enum member must be specified.
    declared = {s.name for s in out}
    missing = enum_members - declared
    if missing:
        raise StateMachineRegistryError(f"gate_states missing GateState members: {sorted(missing)}")
    return out


def _parse_transitions(raw: Any, *, valid_states: frozenset[str]) -> list[Transition]:
    if not isinstance(raw, list) or not raw:
        raise StateMachineRegistryError(
            "state machine spec must contain a non-empty 'transitions' list"
        )
    out: list[Transition] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise StateMachineRegistryError(f"transitions[{i}] must be a mapping")
        required = (
            "id",
            "priority",
            "from_state",
            "trigger",
            "guard",
            "target",
            "execution_allowed",
            "tests",
        )
        for key in required:
            if key not in entry:
                raise StateMachineRegistryError(f"transitions[{i}] missing required field {key!r}")
        tid = str(entry["id"])
        if tid in seen_ids:
            raise StateMachineRegistryError(f"duplicate transition id: {tid!r}")
        seen_ids.add(tid)

        target = str(entry["target"])
        if target not in valid_states:
            raise StateMachineRegistryError(
                f"transitions[{i}] target={target!r} is not a valid state"
            )
        from_state = str(entry["from_state"])
        if from_state != "*" and from_state not in valid_states:
            raise StateMachineRegistryError(
                f"transitions[{i}] from_state={from_state!r} must be '*' or a valid state"
            )

        tests = entry["tests"]
        if not isinstance(tests, list) or not tests:
            raise StateMachineRegistryError(f"transitions[{i}] must bind ≥ 1 test (id={tid!r})")
        if not all(isinstance(t, str) and t for t in tests):
            raise StateMachineRegistryError(f"transitions[{i}] has a non-string or empty test id")

        invariant = entry.get("invariant")
        invariant_str = None if invariant is None else str(invariant)

        out.append(
            Transition(
                id=tid,
                priority=int(entry["priority"]),
                from_state=from_state,
                trigger=str(entry["trigger"]),
                guard=str(entry["guard"]),
                target=target,
                execution_allowed=bool(entry["execution_allowed"]),
                invariant=invariant_str,
                tests=tuple(tests),
            )
        )
    return out


def _parse_global_invariant(raw: Any) -> GlobalInvariant:
    if not isinstance(raw, dict):
        raise StateMachineRegistryError("'global_invariant' must be a mapping")
    for key in ("statement", "enforced_in", "tests"):
        if key not in raw:
            raise StateMachineRegistryError(f"global_invariant missing required field {key!r}")
    enforced_in_raw = raw["enforced_in"]
    if not isinstance(enforced_in_raw, list) or not enforced_in_raw:
        raise StateMachineRegistryError("global_invariant.enforced_in must be a non-empty list")
    sites: list[tuple[str, str]] = []
    for j, site in enumerate(enforced_in_raw):
        if not isinstance(site, dict) or "path" not in site or "symbol" not in site:
            raise StateMachineRegistryError(
                f"global_invariant.enforced_in[{j}] must contain path + symbol"
            )
        sites.append((str(site["path"]), str(site["symbol"])))
    tests = raw["tests"]
    if not isinstance(tests, list) or not tests:
        raise StateMachineRegistryError("global_invariant.tests must be a non-empty list")
    return GlobalInvariant(
        statement=str(raw["statement"]),
        enforced_in=tuple(sites),
        tests=tuple(str(t) for t in tests),
    )
