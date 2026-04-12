"""Public API freeze test.

Freezes the exact set of symbols exposed by ``neurophase.api.__all__`` as a
contract. The list is duplicated here on purpose: if someone adds or
removes a symbol on the façade, this test forces them to amend the
contract explicitly — at which point ``docs/PUBLIC_API.md`` must be
updated in the same commit.

Every symbol is also asserted to:

* resolve to a live runtime object via ``getattr(neurophase.api, name)``
* be re-exported verbatim (no wrapping) from its stated canonical module

The second property makes the façade a pure aliasing layer and prevents
it from growing parallel implementations.
"""

from __future__ import annotations

import importlib

import pytest

EXPECTED_API_SURFACE: frozenset[str] = frozenset(
    {
        # meta
        "__version__",
        # pipeline / orchestrator
        "StreamingPipeline",
        "PipelineConfig",
        "DecisionFrame",
        "RuntimeOrchestrator",
        "OrchestratorConfig",
        "OrchestratedFrame",
        "create_pipeline",
        # gate
        "ExecutionGate",
        "GateDecision",
        "GateState",
        "DEFAULT_THRESHOLD",
        "StillnessDetector",
        "StillnessState",
        # policy
        "ActionPolicy",
        "PolicyConfig",
        "ActionIntent",
        "ActionDecision",
        # regime inference
        "RegimeClassifier",
        "RegimeThresholds",
        "RegimeState",
        "RegimeLabel",
        "RegimeTransitionTracker",
        "RegimeTransitionMatrix",
        "TransitionEvent",
        # explanation
        "Contract",
        "DecisionExplanation",
        "ExplanationStep",
        "Verdict",
        "explain_decision",
        "explain_gate",
        # data quality
        "TimeQuality",
    }
)


CANONICAL_ORIGIN: dict[str, str] = {
    "StreamingPipeline": "neurophase.runtime.pipeline",
    "PipelineConfig": "neurophase.runtime.pipeline",
    "DecisionFrame": "neurophase.runtime.pipeline",
    "RuntimeOrchestrator": "neurophase.runtime.orchestrator",
    "OrchestratorConfig": "neurophase.runtime.orchestrator",
    "OrchestratedFrame": "neurophase.runtime.orchestrator",
    "ExecutionGate": "neurophase.gate.execution_gate",
    "GateDecision": "neurophase.gate.execution_gate",
    "GateState": "neurophase.gate.execution_gate",
    "DEFAULT_THRESHOLD": "neurophase.gate.execution_gate",
    "StillnessDetector": "neurophase.gate.stillness_detector",
    "StillnessState": "neurophase.gate.stillness_detector",
    "ActionPolicy": "neurophase.policy.action",
    "PolicyConfig": "neurophase.policy.action",
    "ActionIntent": "neurophase.policy.action",
    "ActionDecision": "neurophase.policy.action",
    "RegimeClassifier": "neurophase.analysis.regime",
    "RegimeThresholds": "neurophase.analysis.regime",
    "RegimeState": "neurophase.analysis.regime",
    "RegimeLabel": "neurophase.analysis.regime",
    "RegimeTransitionTracker": "neurophase.analysis.regime_transitions",
    "RegimeTransitionMatrix": "neurophase.analysis.regime_transitions",
    "TransitionEvent": "neurophase.analysis.regime_transitions",
    "Contract": "neurophase.explain",
    "DecisionExplanation": "neurophase.explain",
    "ExplanationStep": "neurophase.explain",
    "Verdict": "neurophase.explain",
    "explain_decision": "neurophase.explain",
    "explain_gate": "neurophase.explain",
    "TimeQuality": "neurophase.data.temporal_validator",
}


def test_api_surface_is_exactly_frozen() -> None:
    import neurophase.api as api

    actual = frozenset(api.__all__)

    added = actual - EXPECTED_API_SURFACE
    removed = EXPECTED_API_SURFACE - actual

    assert not added and not removed, (
        "neurophase.api.__all__ drifted from the frozen public surface.\n"
        f"Added (update docs/PUBLIC_API.md + this test): {sorted(added)}\n"
        f"Removed (requires minor version bump + CHANGELOG entry): {sorted(removed)}"
    )


def test_every_api_symbol_resolves() -> None:
    import neurophase.api as api

    for name in api.__all__:
        assert hasattr(api, name), (
            f"neurophase.api.__all__ lists {name!r} but the symbol is not set"
        )
        assert getattr(api, name) is not None, f"neurophase.api.{name} is None"


@pytest.mark.parametrize(
    "symbol,origin_module",
    sorted(CANONICAL_ORIGIN.items()),
)
def test_symbol_is_verbatim_reexport(symbol: str, origin_module: str) -> None:
    """Each façade symbol must be the **same object** as in its canonical module.

    This prevents the façade from quietly growing a parallel wrapper
    implementation: ``neurophase.api.ExecutionGate`` *is*
    ``neurophase.gate.execution_gate.ExecutionGate``.
    """
    import neurophase.api as api

    facade_obj = getattr(api, symbol)
    origin = importlib.import_module(origin_module)
    canonical_obj = getattr(origin, symbol)

    assert facade_obj is canonical_obj, (
        f"neurophase.api.{symbol} is not the verbatim re-export of "
        f"{origin_module}.{symbol}; the façade must not wrap."
    )


def test_version_matches_package() -> None:
    import neurophase
    import neurophase.api as api

    assert api.__version__ == neurophase.__version__


def test_create_pipeline_returns_streaming_pipeline() -> None:
    from neurophase.api import StreamingPipeline, create_pipeline

    p = create_pipeline()
    assert isinstance(p, StreamingPipeline)
