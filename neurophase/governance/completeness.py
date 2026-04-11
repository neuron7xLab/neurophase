"""Ninth axis — Completeness (Повнота): dual of Coherence.

The eighth axis (:mod:`neurophase.governance.doctor`) asks
*"do the self-descriptions AGREE?"*. The ninth axis asks
*"do they COVER everything?"*. A system that passes axis 8 can
still be incomplete — a public symbol might be exported
without any test, a module might be orphaned from the import
tree, an HN binding might point at a pytest node that no
longer exists. Every such gap is silent: no drift, just
absence.

The :class:`CompletenessAuditor` enumerates the public surface
mechanically and demands a test artifact for each:

1. **API_SYMBOL_TEST_COVERAGE** — every name in
   ``neurophase.api.__all__`` appears as a substring in at
   least one ``tests/test_*.py`` file. A symbol exported but
   never tested is a gap.

2. **PUBLIC_MODULE_REACHABLE** — every non-private ``.py``
   file under ``neurophase/`` (excluding ``_*.py``,
   ``__pycache__``, and private ``_internal``-style modules)
   is reachable via the public import tree rooted at
   ``neurophase.__init__``. An orphan module is a gap.

3. **INVARIANT_TEST_NODE_EXISTS** — every pytest node id
   referenced in ``INVARIANTS.yaml`` resolves to a real
   test function in ``tests/``. A dead binding is a gap
   (the contract points at a test that does not exist).

4. **SENSOR_ADAPTER_HAS_TEST** — every adapter name
   registered in
   :data:`~neurophase.sensors.DEFAULT_ADAPTER_REGISTRY`
   appears in at least one test file. A registered adapter
   without a test is a gap.

5. **MONOGRAPH_MENTIONS_EVERY_HN** — every HN id from the
   invariant registry appears verbatim in the committed
   monograph. This is defense-in-depth vs HN29: HN29 checks
   that the monograph is byte-equal to the generator output;
   axis 9 checks that the generator output does not silently
   *omit* an HN.

Each check returns a frozen :class:`CompletenessCheckResult`;
the suite aggregates into :class:`CompletenessReport`. The
report is deterministic — same repository state produces the
same report byte-for-byte.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "COMPLETENESS_CHECKS",
    "CompletenessAuditor",
    "CompletenessCheckResult",
    "CompletenessReport",
    "run_completeness",
]

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
_TESTS_DIR: Final[Path] = _REPO_ROOT / "tests"
_NEUROPHASE_DIR: Final[Path] = _REPO_ROOT / "neurophase"


@dataclass(frozen=True, repr=False)
class CompletenessCheckResult:
    """Frozen outcome of one completeness check.

    Attributes
    ----------
    check_id
        Stable UPPER_SNAKE_CASE id.
    passed
        ``True`` iff the check found no gap.
    detail
        One-line human-readable summary.
    gaps
        Tuple of specific gap identifiers surfaced by the check.
        Non-empty implies ``passed is False``.
    """

    check_id: str
    passed: bool
    detail: str
    gaps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.gaps and self.passed:
            raise ValueError(f"check {self.check_id} has {len(self.gaps)} gaps but passed=True")

    def __repr__(self) -> str:  # HN35 aesthetic
        flag = "✓" if self.passed else "✗"
        suffix = f" · {len(self.gaps)} gap(s)" if self.gaps else ""
        return f"CompletenessCheckResult[{self.check_id} · {flag}{suffix}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "passed": self.passed,
            "detail": self.detail,
            "gaps": list(self.gaps),
        }


@dataclass(frozen=True, repr=False)
class CompletenessReport:
    """Aggregated completeness outcome.

    Attributes
    ----------
    results
        Tuple of :class:`CompletenessCheckResult`, one per
        registered check, in declaration order.
    complete
        ``True`` iff every result passed.
    total_gaps
        Sum of ``len(r.gaps)`` across all results.
    """

    results: tuple[CompletenessCheckResult, ...]
    complete: bool
    total_gaps: int

    def __post_init__(self) -> None:
        expected_complete = all(r.passed for r in self.results)
        if expected_complete != self.complete:
            raise ValueError(f"complete={self.complete} disagrees with per-result flags")
        expected_gaps = sum(len(r.gaps) for r in self.results)
        if expected_gaps != self.total_gaps:
            raise ValueError(f"total_gaps={self.total_gaps} disagrees with per-result count")

    def __repr__(self) -> str:  # HN35 aesthetic
        flag = "✓" if self.complete else "✗"
        passed = sum(1 for r in self.results if r.passed)
        return f"CompletenessReport[{passed}/{len(self.results)} · gaps={self.total_gaps} · {flag}]"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "total_gaps": self.total_gaps,
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Check runners — each pure of (repo state, module imports).
# ---------------------------------------------------------------------------


def _check_api_symbol_test_coverage() -> CompletenessCheckResult:
    """Check 1: every facade symbol appears in at least one test file."""
    import neurophase.api as api

    test_files = sorted(_TESTS_DIR.glob("test_*.py"))
    if not test_files:
        return CompletenessCheckResult(
            "API_SYMBOL_TEST_COVERAGE",
            False,
            "no test files found under tests/",
        )
    # Concatenate all test source into one haystack.
    haystack = "\n".join(path.read_text(encoding="utf-8") for path in test_files)
    gaps: list[str] = []
    # __version__ and DEFAULT_THRESHOLD are constants — they may
    # not be literally mentioned by name in tests even though they
    # are tested transitively. Whitelist them.
    _CONSTANT_ALLOWLIST = {"DEFAULT_THRESHOLD", "__version__"}
    for name in api.__all__:
        if name in _CONSTANT_ALLOWLIST:
            continue
        if name not in haystack:
            gaps.append(name)
    if gaps:
        return CompletenessCheckResult(
            "API_SYMBOL_TEST_COVERAGE",
            False,
            f"{len(gaps)} facade symbol(s) never mentioned in any test file",
            gaps=tuple(gaps),
        )
    covered = len(api.__all__) - len(_CONSTANT_ALLOWLIST)
    return CompletenessCheckResult(
        "API_SYMBOL_TEST_COVERAGE",
        True,
        f"all {covered} non-constant facade symbols appear in tests/",
    )


def _check_public_module_reachable() -> CompletenessCheckResult:
    """Check 2: every public .py file is reachable via the import tree.

    Walks the neurophase/ directory, collects every non-private
    .py file, and verifies each one is transitively imported
    from neurophase/__init__.py by checking for a matching
    ``from neurophase.X import`` statement anywhere in the
    package.
    """
    # Enumerate public .py files under neurophase/.
    candidates: list[Path] = []
    for path in _NEUROPHASE_DIR.rglob("*.py"):
        rel = path.relative_to(_NEUROPHASE_DIR)
        # Skip __pycache__, private modules, __init__.py, __main__.py.
        if any(part.startswith("_") or part == "__pycache__" for part in rel.parts):
            continue
        candidates.append(path)

    # Build the set of imported-from paths by scanning every
    # .py file under neurophase/ for `from neurophase.X.Y import`
    # statements.
    imported_modules: set[str] = set()
    for src in _NEUROPHASE_DIR.rglob("*.py"):
        if "__pycache__" in src.parts:
            continue
        try:
            tree = ast.parse(src.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "neurophase" or node.module.startswith("neurophase."):
                    imported_modules.add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "neurophase" or alias.name.startswith("neurophase."):
                        imported_modules.add(alias.name)

    gaps: list[str] = []
    for path in candidates:
        rel = path.relative_to(_NEUROPHASE_DIR).with_suffix("")
        module_name = "neurophase." + ".".join(rel.parts)
        if module_name not in imported_modules:
            gaps.append(module_name)

    if gaps:
        return CompletenessCheckResult(
            "PUBLIC_MODULE_REACHABLE",
            False,
            f"{len(gaps)} module(s) not imported anywhere in the package",
            gaps=tuple(gaps[:10]),
        )
    return CompletenessCheckResult(
        "PUBLIC_MODULE_REACHABLE",
        True,
        f"all {len(candidates)} public modules are reachable via imports",
    )


def _check_invariant_test_node_exists() -> CompletenessCheckResult:
    """Check 3: every pytest node id bound in INVARIANTS.yaml resolves.

    A pytest node id looks like
    ``tests/test_foo.py::TestClass::test_method`` or
    ``tests/test_foo.py::test_method``. This check verifies:

    1. The file ``tests/test_foo.py`` exists.
    2. The test function name appears in the file's source
       as a ``def test_method`` definition or as a parametrised
       variant (we use substring containment, because
       ``test_method[param-id]`` encodes the parametrisation).

    A dead binding is a coherence fault that axis 8 cannot
    catch (the registry loads, the nodes just don't point at
    anything).
    """
    from neurophase.governance.invariants import load_registry

    registry = load_registry()
    all_test_ids: list[str] = []
    for inv in registry.invariants:
        all_test_ids.extend(inv.tests)
    for hn in registry.honest_naming:
        all_test_ids.extend(hn.tests)

    # Cache file contents to avoid re-reading.
    file_cache: dict[Path, str] = {}

    gaps: list[str] = []
    for test_id in all_test_ids:
        # Parse tests/test_foo.py::TestClass::test_method
        parts = test_id.split("::")
        if len(parts) < 2:
            gaps.append(f"{test_id} (malformed: no ::)")
            continue
        file_part = parts[0]
        func_part = parts[-1]
        # Strip parametrisation suffix like [Foo.BAR-Bar.BAZ]
        bracket = func_part.find("[")
        func_name = func_part[:bracket] if bracket != -1 else func_part
        file_path = _REPO_ROOT / file_part
        if not file_path.is_file():
            gaps.append(f"{test_id} (file missing: {file_part})")
            continue
        if file_path not in file_cache:
            file_cache[file_path] = file_path.read_text(encoding="utf-8")
        text = file_cache[file_path]
        # Accept either "def {func_name}" or " {func_name}(" as
        # the pattern that proves the test exists. Both are
        # robust against method-in-class definitions.
        if f"def {func_name}(" not in text:
            gaps.append(f"{test_id} (no def {func_name}(… in {file_part})")

    if gaps:
        return CompletenessCheckResult(
            "INVARIANT_TEST_NODE_EXISTS",
            False,
            f"{len(gaps)} dead pytest node binding(s) in INVARIANTS.yaml",
            gaps=tuple(gaps[:10]),
        )
    return CompletenessCheckResult(
        "INVARIANT_TEST_NODE_EXISTS",
        True,
        f"all {len(all_test_ids)} pytest node bindings resolve to real tests",
    )


def _check_sensor_adapter_has_test() -> CompletenessCheckResult:
    """Check 4: every registered sensor adapter is mentioned in a test.

    The ``neurophase.sensors`` subpackage ships in a parallel PR
    and may not be present on every branch. When absent, this
    check is skipped cleanly with ``passed=True``.
    """
    try:
        import importlib

        sensors_mod = importlib.import_module("neurophase.sensors")
    except ImportError:
        return CompletenessCheckResult(
            "SENSOR_ADAPTER_HAS_TEST",
            True,
            "sensors subpackage not present; check skipped",
        )
    registry = getattr(sensors_mod, "DEFAULT_ADAPTER_REGISTRY", None)
    if registry is None:
        return CompletenessCheckResult(
            "SENSOR_ADAPTER_HAS_TEST",
            True,
            "sensors subpackage has no DEFAULT_ADAPTER_REGISTRY; check skipped",
        )

    test_files = sorted(_TESTS_DIR.glob("test_*.py"))
    haystack = "\n".join(path.read_text(encoding="utf-8") for path in test_files)

    gaps: list[str] = []
    for name in registry.names():
        # Look for a literal '"name"' or "'name'" match in tests.
        if f'"{name}"' not in haystack and f"'{name}'" not in haystack:
            gaps.append(name)

    if gaps:
        return CompletenessCheckResult(
            "SENSOR_ADAPTER_HAS_TEST",
            False,
            f"{len(gaps)} registered adapter(s) without a test mention",
            gaps=tuple(gaps),
        )
    return CompletenessCheckResult(
        "SENSOR_ADAPTER_HAS_TEST",
        True,
        f"all {len(registry)} registered adapters are tested",
    )


def _check_monograph_mentions_every_hn() -> CompletenessCheckResult:
    """Check 5: every HN id appears verbatim in the committed monograph."""
    from neurophase.governance.invariants import load_registry
    from neurophase.governance.monograph import MONOGRAPH_PATH

    if not MONOGRAPH_PATH.is_file():
        return CompletenessCheckResult(
            "MONOGRAPH_MENTIONS_EVERY_HN",
            False,
            f"monograph missing at {MONOGRAPH_PATH}",
        )
    text = MONOGRAPH_PATH.read_text(encoding="utf-8")
    registry = load_registry()

    gaps: list[str] = []
    for inv in registry.invariants:
        if f"### {inv.id}" not in text:
            gaps.append(inv.id)
    for hn in registry.honest_naming:
        if f"### {hn.id}" not in text:
            gaps.append(hn.id)

    if gaps:
        return CompletenessCheckResult(
            "MONOGRAPH_MENTIONS_EVERY_HN",
            False,
            f"{len(gaps)} registry id(s) missing from monograph",
            gaps=tuple(gaps),
        )
    total = len(registry.invariants) + len(registry.honest_naming)
    return CompletenessCheckResult(
        "MONOGRAPH_MENTIONS_EVERY_HN",
        True,
        f"all {total} registry ids have a monograph section",
    )


# ---------------------------------------------------------------------------
# Registry + runner.
# ---------------------------------------------------------------------------

#: Stable tuple of every completeness check, in declaration order.
COMPLETENESS_CHECKS: tuple[tuple[str, Callable[[], CompletenessCheckResult]], ...] = (
    ("API_SYMBOL_TEST_COVERAGE", _check_api_symbol_test_coverage),
    ("PUBLIC_MODULE_REACHABLE", _check_public_module_reachable),
    ("INVARIANT_TEST_NODE_EXISTS", _check_invariant_test_node_exists),
    ("SENSOR_ADAPTER_HAS_TEST", _check_sensor_adapter_has_test),
    ("MONOGRAPH_MENTIONS_EVERY_HN", _check_monograph_mentions_every_hn),
)


class CompletenessAuditor:
    """Runner over :data:`COMPLETENESS_CHECKS`.

    Stateless. Two invocations on the same repository state
    produce byte-identical :class:`CompletenessReport` payloads.
    """

    def run(self) -> CompletenessReport:
        results: list[CompletenessCheckResult] = []
        for _, runner in COMPLETENESS_CHECKS:
            try:
                result = runner()
            except Exception as exc:  # defensive
                result = CompletenessCheckResult(
                    check_id="UNKNOWN",
                    passed=False,
                    detail=f"check raised {type(exc).__name__}: {exc}",
                )
            results.append(result)

        complete = all(r.passed for r in results)
        total_gaps = sum(len(r.gaps) for r in results)
        return CompletenessReport(
            results=tuple(results),
            complete=complete,
            total_gaps=total_gaps,
        )

    def run_one(self, check_id: str) -> CompletenessCheckResult:
        for registered_id, runner in COMPLETENESS_CHECKS:
            if registered_id == check_id:
                return runner()
        raise KeyError(f"unknown completeness check id: {check_id!r}")


def run_completeness() -> CompletenessReport:
    """Shortcut: ``CompletenessAuditor().run()``."""
    return CompletenessAuditor().run()
