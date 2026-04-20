"""Ablation policy loader (final_1/final_2)."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

DEFAULT_ABLATION_POLICY_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent.parent / "ABLATION_POLICY.yaml"
)
_CRITICAL_ID_RE: Final[re.Pattern[str]] = re.compile(r"^G[1-9]\d*$")


class AblationPolicyError(ValueError):
    """Raised when ABLATION_POLICY.yaml is malformed or unverifiable."""


@dataclass(frozen=True)
class AblationPolicy:
    version: int
    mutation_suite: str
    critical_elements: tuple[str, ...]
    test_registry: dict[str, str]

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise AblationPolicyError("version must be positive")
        if not self.mutation_suite.strip():
            raise AblationPolicyError("mutation_suite must be non-empty")
        if not self.critical_elements:
            raise AblationPolicyError("critical_elements must be non-empty")
        for elem in self.critical_elements:
            if not elem.strip():
                raise AblationPolicyError("critical_elements entries must be non-empty")
        if set(self.test_registry.keys()) != set(self.critical_elements):
            raise AblationPolicyError("test_registry keys must match critical_elements exactly")


def load_ablation_policy(path: Path | None = None) -> AblationPolicy:
    policy_path = path or DEFAULT_ABLATION_POLICY_PATH
    if not policy_path.is_file():
        raise AblationPolicyError(f"ablation policy not found: {policy_path}")

    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AblationPolicyError("ablation policy root must be a mapping")
    expected_keys = {"version", "mutation_suite", "critical_elements", "test_registry"}
    if set(raw.keys()) != expected_keys:
        raise AblationPolicyError(f"ablation policy keys must be exactly {sorted(expected_keys)}")

    version = raw.get("version")
    mutation_suite = raw.get("mutation_suite")
    critical = raw.get("critical_elements")
    test_registry = raw.get("test_registry")
    if not isinstance(version, int):
        raise AblationPolicyError("version must be an integer")
    if not isinstance(mutation_suite, str):
        raise AblationPolicyError("mutation_suite must be a string")
    if not isinstance(critical, list) or not all(isinstance(x, str) for x in critical):
        raise AblationPolicyError("critical_elements must be a list[str]")
    if len(critical) == 0:
        raise AblationPolicyError("critical_elements must be non-empty")
    if any(_CRITICAL_ID_RE.match(elem) is None for elem in critical):
        raise AblationPolicyError("critical_elements must use G<number> identifiers")
    if not isinstance(test_registry, dict):
        raise AblationPolicyError("test_registry must be a mapping")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in test_registry.items()):
        raise AblationPolicyError("test_registry must be dict[str, str]")

    policy = AblationPolicy(
        version=version,
        mutation_suite=mutation_suite,
        critical_elements=tuple(critical),
        test_registry=dict(test_registry),
    )
    if set(policy.test_registry.keys()) != set(policy.critical_elements):
        raise AblationPolicyError("test_registry keys must match critical_elements exactly")

    base = policy_path.parent
    suite_path = base / policy.mutation_suite
    if not suite_path.is_file():
        raise AblationPolicyError(f"mutation suite missing: {policy.mutation_suite}")
    suite_text = suite_path.read_text(encoding="utf-8")
    try:
        suite_tree = ast.parse(suite_text)
    except SyntaxError as exc:
        raise AblationPolicyError(
            f"mutation suite is not valid Python: {policy.mutation_suite}"
        ) from exc
    test_names = {node.name for node in ast.walk(suite_tree) if isinstance(node, ast.FunctionDef)}
    missing = []
    for elem in policy.critical_elements:
        node_id = policy.test_registry[elem]
        prefix = f"{policy.mutation_suite}::"
        if not node_id.startswith(prefix):
            raise AblationPolicyError(
                f"registry entry for {elem} must point into {policy.mutation_suite}"
            )
        func_name = node_id.split("::", 1)[1]
        if func_name not in test_names:
            missing.append(elem)
            continue
        if not func_name.startswith(f"test_{elem}_"):
            raise AblationPolicyError(
                f"registry entry for {elem} must map to test_{elem}_..., got {func_name}"
            )
    if missing:
        raise AblationPolicyError(f"critical elements without mutation tests: {missing[:5]}")
    return policy
