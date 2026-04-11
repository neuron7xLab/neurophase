"""Governance layer — machine-readable invariant registry.

This subpackage loads and validates ``INVARIANTS.yaml`` at the
repository root. The registry is consumed by a CI meta-test to
guarantee that every registered invariant has at least one bound
pytest node id and that every referenced test file actually exists.

Public API:

* :class:`Invariant` — typed record of a single invariant.
* :class:`InvariantRegistry` — loaded registry with helpers.
* :class:`HonestNamingContract` — typed record of an honest-naming contract.
* :func:`load_registry` — default loader that resolves ``INVARIANTS.yaml``.
"""

from __future__ import annotations

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

__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "EnforcementSite",
    "HonestNamingContract",
    "Invariant",
    "InvariantRegistry",
    "InvariantRegistryError",
    "InvariantSeverity",
    "load_registry",
]
