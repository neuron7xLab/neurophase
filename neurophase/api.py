"""Public façade — the single blessed import path for downstream consumers.

This module re-exports the small set of symbols that form the
**stable public API** of `neurophase`. Everything else in the
package is internal and subject to change without a major version
bump.

The façade is deliberately thin. It does **not** wrap anything in
additional abstraction layers — every symbol re-exported here is
the same frozen dataclass or function you would get by importing
it from its canonical module. The façade exists so that a
downstream consumer can write:

.. code-block:: python

    from neurophase.api import create_pipeline, explain_decision

and be guaranteed that:

1. Those symbols are the recommended entry points.
2. Their signatures will not change in a patch release.
3. Everything they need to build a minimally useful pipeline is
   reachable from this single import.

What is **not** exported here (internal / advanced):

* Per-module implementation details (``_DeterministicStillnessDetector``,
  ``_RollingStats``, etc.).
* Governance loaders (``load_registry``, ``load_state_machine``) — those
  are CI meta-test machinery, not product API.
* Calibration / H1 / F2 helpers that are specifically for research
  workflows rather than deployment. They live in their own subpackages
  and callers should import directly from them.

If you are writing a downstream library or service, prefer
``neurophase.api``. If you are writing a research notebook that
needs to poke at a specific detector or surrogate generator,
importing from the internal modules is fine — you just get no API
stability guarantees.
"""

from __future__ import annotations

from typing import Any

from neurophase.data.temporal_validator import TimeQuality
from neurophase.explain import (
    Contract,
    DecisionExplanation,
    ExplanationStep,
    Verdict,
    explain_decision,
    explain_gate,
)
from neurophase.gate.execution_gate import (
    DEFAULT_THRESHOLD,
    ExecutionGate,
    GateDecision,
    GateState,
)
from neurophase.gate.stillness_detector import StillnessDetector, StillnessState
from neurophase.runtime.pipeline import (
    DecisionFrame,
    PipelineConfig,
    StreamingPipeline,
)

__version__: str


def create_pipeline(**kwargs: Any) -> StreamingPipeline:
    """Convenience factory: build a :class:`StreamingPipeline` in one call.

    Equivalent to ``StreamingPipeline(PipelineConfig(**kwargs))``.
    The keyword arguments are forwarded to :class:`PipelineConfig`
    verbatim; see that class for the full configuration surface.

    Examples
    --------
    >>> from neurophase.api import create_pipeline
    >>> p = create_pipeline(threshold=0.65, warmup_samples=4)
    >>> frame = p.tick(timestamp=0.0, R=0.9, delta=0.01)
    >>> frame.gate_state.name
    'DEGRADED'   # warmup
    """
    return StreamingPipeline(PipelineConfig(**kwargs))


# Re-export __version__ from the package root.
from neurophase import __version__ as _pkg_version  # noqa: E402

__version__ = _pkg_version


__all__ = [
    "DEFAULT_THRESHOLD",
    "Contract",
    "DecisionExplanation",
    "DecisionFrame",
    "ExecutionGate",
    "ExplanationStep",
    "GateDecision",
    "GateState",
    "PipelineConfig",
    "StillnessDetector",
    "StillnessState",
    "StreamingPipeline",
    "TimeQuality",
    "Verdict",
    "__version__",
    "create_pipeline",
    "explain_decision",
    "explain_gate",
]
