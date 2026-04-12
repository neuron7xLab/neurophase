"""Package-metadata contract: runtime vs research dependency split.

The kernel promise is that ``pip install neurophase`` gives you the
fail-closed cognition runtime without the 500 MB scientific stack.
This test asserts that promise at the ``pyproject.toml`` level so it
cannot silently regress.

Research dependencies (mne, pandas, neurodsp, PyWavelets, scikit-learn,
networkx) must appear only under
``[project.optional-dependencies].research`` — never under
``[project].dependencies``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

RESEARCH_DEPS = frozenset(
    {
        "mne",
        "pandas",
        "neurodsp",
        "pywavelets",
        "scikit-learn",
        "networkx",
    }
)
KERNEL_DEPS = frozenset({"numpy", "scipy", "pyyaml"})


def _strip_version(spec: str) -> str:
    """Normalise a PEP 508-ish requirement string to the bare package name."""
    # Handle PEP 508 markers after ';' and PEP 440 version specifiers.
    head = spec.split(";", 1)[0].strip()
    for sep in ("<=", ">=", "==", "~=", "!=", "<", ">", "@"):
        if sep in head:
            head = head.split(sep, 1)[0].strip()
    return head.strip().lower()


def _load() -> dict:  # type: ignore[type-arg]
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_pyproject_exists() -> None:
    assert PYPROJECT.exists(), f"pyproject.toml not found at {PYPROJECT}"


def test_kernel_deps_are_core() -> None:
    project = _load()["project"]
    core_names = {_strip_version(d) for d in project["dependencies"]}
    missing = KERNEL_DEPS - core_names
    assert not missing, f"kernel dependency absent from [project].dependencies: {sorted(missing)}"


def test_research_deps_are_not_core() -> None:
    project = _load()["project"]
    core_names = {_strip_version(d) for d in project["dependencies"]}
    leaked = RESEARCH_DEPS & core_names
    assert not leaked, (
        f"research dependencies must not be in [project].dependencies; "
        f"found: {sorted(leaked)}. Move them to "
        f"[project.optional-dependencies].research."
    )


def test_research_extra_exists_and_contains_all_research_deps() -> None:
    project = _load()["project"]
    extras = project.get("optional-dependencies", {})
    assert "research" in extras, (
        "pyproject.toml must declare [project.optional-dependencies].research"
    )
    research_names = {_strip_version(d) for d in extras["research"]}
    missing = RESEARCH_DEPS - research_names
    assert not missing, (
        f"research extra is missing dependencies: {sorted(missing)}. "
        f"Add them to [project.optional-dependencies].research."
    )


def test_dev_extra_pulls_research_stack() -> None:
    """``pip install .[dev]`` must be a superset of ``[research]``.

    Otherwise the full pytest suite (ds003458 loaders, Ricci metrics,
    calibration grids) would not run in a clean dev environment.
    """
    project = _load()["project"]
    extras = project.get("optional-dependencies", {})
    assert "dev" in extras
    dev_names = {_strip_version(d) for d in extras["dev"]}
    missing = RESEARCH_DEPS - dev_names
    assert not missing, (
        f"dev extra must carry the research stack so the full pytest suite runs; "
        f"missing: {sorted(missing)}."
    )
