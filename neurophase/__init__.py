"""neurophase — canonical fail-closed cognition kernel.

Phase synchronization as execution gate: brain and market modelled as coupled
Kuramoto oscillators sharing order parameter R(t).

Four invariants that cannot be overridden:

    I₁: R(t) < θ              ⇒ execution_allowed = False
    I₂: bio-sensor absent     ⇒ execution_allowed = False
    I₃: R(t) invalid / OOR    ⇒ execution_allowed = False
    I₄: stillness             ⇒ execution_allowed = False  (action_unnecessary)

See ``docs/theory/scientific_basis.md`` and
``docs/theory/stillness_invariant.md`` for the full derivation.

Import boundary (kernelization v1)
----------------------------------

The root package is deliberately thin. ``import neurophase`` loads only
``__version__`` and a lazy attribute accessor. No heavy scientific
dependencies (``mne``, ``pandas``, ``pywt``, ``neurodsp``, ``networkx``,
``scikit-learn``) are pulled in at module load time.

**Runtime consumers** should import from the blessed façade::

    from neurophase.api import create_pipeline, StreamingPipeline, ExecutionGate

**Research / experiments** live under dedicated subpackages and pull heavy
deps locally::

    from neurophase.experiments.ds003458_analysis import run_analysis

Backward-compat symbols (``KLRConfig``) remain reachable through this
module but are resolved on first access via :pep:`562` ``__getattr__``,
so they do not trigger eager loading of the KLR / reset substack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.4.0"

__all__ = [
    "KLRConfig",
    "__version__",
]

if TYPE_CHECKING:
    from neurophase.reset import KLRConfig as KLRConfig


# PEP 562 lazy attribute accessor. New code should import from the subpackage
# (e.g. ``from neurophase.reset import KLRConfig``); this accessor exists so
# ``from neurophase import KLRConfig`` keeps working without forcing
# ``reset/`` to load at package import time.
_LAZY_BACKWARD_COMPAT: dict[str, tuple[str, str]] = {
    "KLRConfig": ("neurophase.reset", "KLRConfig"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_BACKWARD_COMPAT.get(name)
    if target is None:
        raise AttributeError(f"module 'neurophase' has no attribute {name!r}")
    module_name, attr_name = target
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value  # cache so subsequent access is O(1)
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
