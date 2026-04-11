"""Loader and typed records for ``INVARIANTS.yaml`` — governance task A1.

The registry file at the repository root is hand-maintained and
load-bearing. This module provides:

* A schema validator that raises :class:`InvariantRegistryError` at
  load time if any required field is missing or malformed.
* Frozen typed records (:class:`Invariant`, :class:`HonestNamingContract`,
  :class:`EnforcementSite`) so consumers cannot silently mutate them.
* A default loader :func:`load_registry` that resolves the canonical
  path ``<repo-root>/INVARIANTS.yaml`` from the package location.

A CI meta-test (``tests/test_invariants_registry.py``) imports this
module and verifies:

1. The YAML file is present and parseable.
2. Every ``Invariant`` has ``len(tests) ≥ 1``.
3. Every referenced test file exists on disk.
4. Every invariant ``id`` is unique across the registry.
5. Every enforcement-site path exists on disk.
6. Every doc path exists on disk.

No runtime enforcement — the registry is consumed by CI and by
future governance tools only. It does not run inside the hot gate
path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

import yaml

#: Canonical path of ``INVARIANTS.yaml`` relative to the package root.
DEFAULT_REGISTRY_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent.parent / "INVARIANTS.yaml"
)

#: Supported registry schema version.
_SCHEMA_VERSION: Final[int] = 1


class InvariantRegistryError(ValueError):
    """Raised when the registry fails schema validation."""


class InvariantSeverity(Enum):
    """Two-tier severity classification.

    * ``HARD`` — violating the invariant must block execution
                (``I₁``, ``I₂``, ``I₃``, ``B₁``).
    * ``ADVISORY`` — violating the invariant routes execution to a
                non-permissive but semantically distinct state
                (``I₄`` — `UNNECESSARY`).
    """

    HARD = "hard"
    ADVISORY = "advisory"


@dataclass(frozen=True)
class EnforcementSite:
    """A (path, symbol) pointer to where an invariant is enforced."""

    path: str
    symbol: str


@dataclass(frozen=True)
class Invariant:
    """Typed record of a single invariant entry in ``INVARIANTS.yaml``."""

    id: str
    symbol: str
    statement: str
    severity: InvariantSeverity
    enforced_in: tuple[EnforcementSite, ...]
    tests: tuple[str, ...]
    docs: tuple[str, ...]
    introduced_in: str


@dataclass(frozen=True)
class HonestNamingContract:
    """Typed record of an honest-naming contract entry."""

    id: str
    statement: str
    enforced_in: tuple[EnforcementSite, ...]
    tests: tuple[str, ...]
    docs: tuple[str, ...]


@dataclass(frozen=True)
class InvariantRegistry:
    """Full loaded registry.

    Attributes
    ----------
    version
        Schema version read from the YAML file.
    invariants
        Tuple of :class:`Invariant` records, in file order.
    honest_naming
        Tuple of :class:`HonestNamingContract` records, in file order.
    path
        Absolute path the registry was loaded from.
    """

    version: int
    invariants: tuple[Invariant, ...]
    honest_naming: tuple[HonestNamingContract, ...]
    path: Path

    def by_id(self, invariant_id: str) -> Invariant:
        """Return the invariant with the given ``id``.

        Raises
        ------
        KeyError
            If no invariant with that ``id`` exists.
        """
        for inv in self.invariants:
            if inv.id == invariant_id:
                return inv
        raise KeyError(f"unknown invariant id: {invariant_id!r}")

    def all_test_ids(self) -> tuple[str, ...]:
        """Return every pytest node id bound to any invariant or contract."""
        out: list[str] = []
        for inv in self.invariants:
            out.extend(inv.tests)
        for hn in self.honest_naming:
            out.extend(hn.tests)
        return tuple(out)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_registry(path: Path | str | None = None) -> InvariantRegistry:
    """Load and validate ``INVARIANTS.yaml``.

    Parameters
    ----------
    path
        Optional override for the registry path. Defaults to
        ``DEFAULT_REGISTRY_PATH``.

    Returns
    -------
    InvariantRegistry
        The parsed, schema-validated registry.

    Raises
    ------
    InvariantRegistryError
        If the file is missing, unparseable, or schema-invalid.
    """
    resolved = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    if not resolved.is_file():
        raise InvariantRegistryError(f"registry not found at {resolved}")

    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InvariantRegistryError(f"registry at {resolved} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise InvariantRegistryError(f"registry at {resolved} must be a mapping at the top level")

    version = raw.get("version")
    if version != _SCHEMA_VERSION:
        raise InvariantRegistryError(
            f"registry version mismatch: expected {_SCHEMA_VERSION}, got {version!r}"
        )

    inv_raw = raw.get("invariants", [])
    if not isinstance(inv_raw, list) or not inv_raw:
        raise InvariantRegistryError("registry must contain a non-empty 'invariants' list")

    seen_ids: set[str] = set()
    invariants: list[Invariant] = []
    for i, entry in enumerate(inv_raw):
        inv = _parse_invariant(entry, index=i)
        if inv.id in seen_ids:
            raise InvariantRegistryError(f"duplicate invariant id: {inv.id!r} at index {i}")
        seen_ids.add(inv.id)
        invariants.append(inv)

    hn_raw = raw.get("honest_naming", [])
    if hn_raw is None:
        hn_raw = []
    if not isinstance(hn_raw, list):
        raise InvariantRegistryError("'honest_naming' must be a list if provided")

    honest_naming: list[HonestNamingContract] = []
    for i, entry in enumerate(hn_raw):
        honest_naming.append(_parse_honest_naming(entry, index=i))

    return InvariantRegistry(
        version=version,
        invariants=tuple(invariants),
        honest_naming=tuple(honest_naming),
        path=resolved,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _parse_invariant(entry: Any, *, index: int) -> Invariant:
    if not isinstance(entry, dict):
        raise InvariantRegistryError(f"invariants[{index}] must be a mapping")
    required = (
        "id",
        "symbol",
        "statement",
        "severity",
        "enforced_in",
        "tests",
        "docs",
        "introduced_in",
    )
    for key in required:
        if key not in entry:
            raise InvariantRegistryError(f"invariants[{index}] missing required field {key!r}")

    severity_raw = entry["severity"]
    try:
        severity = InvariantSeverity(severity_raw)
    except ValueError as exc:
        raise InvariantRegistryError(
            f"invariants[{index}] has invalid severity {severity_raw!r}; "
            f"expected one of {[s.value for s in InvariantSeverity]}"
        ) from exc

    tests = entry["tests"]
    if not isinstance(tests, list) or not tests:
        raise InvariantRegistryError(f"invariants[{index}] must bind ≥ 1 test (id={entry['id']!r})")
    if not all(isinstance(t, str) and t for t in tests):
        raise InvariantRegistryError(f"invariants[{index}] has a non-string or empty test id")

    docs = entry["docs"]
    if not isinstance(docs, list):
        raise InvariantRegistryError(f"invariants[{index}] docs must be a list")

    return Invariant(
        id=str(entry["id"]),
        symbol=str(entry["symbol"]),
        statement=str(entry["statement"]),
        severity=severity,
        enforced_in=tuple(_parse_sites(entry["enforced_in"], index=index)),
        tests=tuple(tests),
        docs=tuple(str(d) for d in docs),
        introduced_in=str(entry["introduced_in"]),
    )


def _parse_sites(raw: Any, *, index: int) -> list[EnforcementSite]:
    if not isinstance(raw, list) or not raw:
        raise InvariantRegistryError(f"invariants[{index}] enforced_in must be a non-empty list")
    sites: list[EnforcementSite] = []
    for j, site in enumerate(raw):
        if not isinstance(site, dict) or "path" not in site or "symbol" not in site:
            raise InvariantRegistryError(
                f"invariants[{index}].enforced_in[{j}] must contain path + symbol"
            )
        sites.append(EnforcementSite(path=str(site["path"]), symbol=str(site["symbol"])))
    return sites


def _parse_honest_naming(entry: Any, *, index: int) -> HonestNamingContract:
    if not isinstance(entry, dict):
        raise InvariantRegistryError(f"honest_naming[{index}] must be a mapping")
    required = ("id", "statement", "enforced_in", "tests", "docs")
    for key in required:
        if key not in entry:
            raise InvariantRegistryError(f"honest_naming[{index}] missing required field {key!r}")
    tests = entry["tests"]
    if not isinstance(tests, list) or not tests:
        raise InvariantRegistryError(f"honest_naming[{index}] must bind ≥ 1 test")
    return HonestNamingContract(
        id=str(entry["id"]),
        statement=str(entry["statement"]),
        enforced_in=tuple(_parse_sites(entry["enforced_in"], index=index)),
        tests=tuple(str(t) for t in tests),
        docs=tuple(str(d) for d in entry["docs"]),
    )
