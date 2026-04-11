"""Governance layer — machine-readable invariant + state-machine registries.

This subpackage loads and validates two top-level YAML artifacts:

* ``INVARIANTS.yaml`` — the invariant registry (task A1).
* ``STATE_MACHINE.yaml`` — the formal gate state-machine specification (task A2).

Both registries are consumed by CI meta-tests to guarantee that every
registered contract has at least one bound pytest node id and that
every referenced file exists on disk.

Public API:

* :class:`Invariant`, :class:`InvariantRegistry`, :func:`load_registry`
* :class:`Transition`, :class:`StateMachineSpec`, :func:`load_state_machine`
* typed error classes :class:`InvariantRegistryError`,
  :class:`StateMachineRegistryError`.
"""

from __future__ import annotations

from neurophase.governance.completeness import (
    CompletenessAuditor,
    CompletenessCheckResult,
    CompletenessReport,
    run_completeness,
)
from neurophase.governance.invariants import (
    DEFAULT_REGISTRY_PATH,
    EnforcementSite,
    HonestNamingContract,
    Invariant,
    InvariantRegistry,
    InvariantRegistryError,
    InvariantSeverity,
    load_registry,
)
from neurophase.governance.state_machine import (
    DEFAULT_STATE_MACHINE_PATH,
    GlobalInvariant,
    StateMachineRegistryError,
    StateMachineSpec,
    StateSpec,
    Transition,
    load_state_machine,
)

__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "DEFAULT_STATE_MACHINE_PATH",
    "CompletenessAuditor",
    "CompletenessCheckResult",
    "CompletenessReport",
    "EnforcementSite",
    "GlobalInvariant",
    "HonestNamingContract",
    "Invariant",
    "InvariantRegistry",
    "InvariantRegistryError",
    "InvariantSeverity",
    "StateMachineRegistryError",
    "StateMachineSpec",
    "StateSpec",
    "Transition",
    "load_registry",
    "load_state_machine",
    "run_completeness",
]
