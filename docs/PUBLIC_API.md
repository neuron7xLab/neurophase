# neurophase — Public API

**Status.** Frozen public surface as of kernelization v1.
**Import path.** ``neurophase.api``
**Stability guarantee.** Symbols listed here do not change signature in a
patch release; semantic breakage requires a minor version bump and a note
in ``CHANGELOG.md``.

A downstream consumer who only wants to build, run, and audit a phase gate
should never need to import from a submodule path. Everything required
for that flow is re-exported from ``neurophase.api`` verbatim — no
wrapping, no additional abstraction layer.

```python
from neurophase.api import create_pipeline, explain_decision
```

---

## The 32 blessed symbols

### Pipeline & orchestrator — 7

| Symbol | Kind | Module of origin |
|---|---|---|
| `StreamingPipeline` | class | `neurophase.runtime.pipeline` |
| `PipelineConfig` | dataclass | `neurophase.runtime.pipeline` |
| `DecisionFrame` | dataclass | `neurophase.runtime.pipeline` |
| `RuntimeOrchestrator` | class | `neurophase.runtime.orchestrator` |
| `OrchestratorConfig` | dataclass | `neurophase.runtime.orchestrator` |
| `OrchestratedFrame` | dataclass | `neurophase.runtime.orchestrator` |
| `create_pipeline` | function | `neurophase.api` |

### Gate — 6

| Symbol | Kind | Module of origin |
|---|---|---|
| `ExecutionGate` | class | `neurophase.gate.execution_gate` |
| `GateDecision` | dataclass | `neurophase.gate.execution_gate` |
| `GateState` | enum | `neurophase.gate.execution_gate` |
| `DEFAULT_THRESHOLD` | constant | `neurophase.gate.execution_gate` |
| `StillnessDetector` | class | `neurophase.gate.stillness_detector` |
| `StillnessState` | enum | `neurophase.gate.stillness_detector` |

### Policy — 4

| Symbol | Kind | Module of origin |
|---|---|---|
| `ActionPolicy` | class | `neurophase.policy.action` |
| `PolicyConfig` | dataclass | `neurophase.policy.action` |
| `ActionIntent` | dataclass | `neurophase.policy.action` |
| `ActionDecision` | dataclass | `neurophase.policy.action` |

### Regime inference — 7

| Symbol | Kind | Module of origin |
|---|---|---|
| `RegimeClassifier` | class | `neurophase.analysis.regime` |
| `RegimeThresholds` | dataclass | `neurophase.analysis.regime` |
| `RegimeState` | dataclass | `neurophase.analysis.regime` |
| `RegimeLabel` | enum | `neurophase.analysis.regime` |
| `RegimeTransitionTracker` | class | `neurophase.analysis.regime_transitions` |
| `RegimeTransitionMatrix` | dataclass | `neurophase.analysis.regime_transitions` |
| `TransitionEvent` | dataclass | `neurophase.analysis.regime_transitions` |

### Explanations — 6

| Symbol | Kind | Module of origin |
|---|---|---|
| `Contract` | class | `neurophase.explain` |
| `DecisionExplanation` | dataclass | `neurophase.explain` |
| `ExplanationStep` | dataclass | `neurophase.explain` |
| `Verdict` | enum | `neurophase.explain` |
| `explain_decision` | function | `neurophase.explain` |
| `explain_gate` | function | `neurophase.explain` |

### Time / data quality — 1

| Symbol | Kind | Module of origin |
|---|---|---|
| `TimeQuality` | enum | `neurophase.data.temporal_validator` |

### Meta — 1

| Symbol | Kind | Module of origin |
|---|---|---|
| `__version__` | string | `neurophase.__init__` |

---

## What is deliberately *not* on the façade

| Surface | Rationale |
|---|---|
| `neurophase.calibration.*` | Offline threshold / stillness grid-search; research workflow, not runtime. |
| `neurophase.benchmarks.*` | Synthetic ground-truth generators; for validation, not live consumption. |
| `neurophase.experiments.*` | ds003458 analyses and related scripts; pull mne/pandas locally. |
| `neurophase.governance.*` | YAML loaders for INVARIANTS / CLAIMS / STATE_MACHINE. Used by the boot Doctor and CI meta-tests, not by runtime consumers. |
| `neurophase.intel.*`, `neurophase.indicators.*` | Crypto / BTC-specific request validators and scalar indicators. Domain adapters, not generic runtime. |
| `neurophase.risk.*` | EVT, multifractal, sizing. Trading-specific, downstream of the gate. |
| `neurophase.reset.*` | KLR / reset substack. Adjacent subsystem, lives behind its own API; backward-compat ``KLRConfig`` is reachable via PEP 562 lazy access on the root package. |
| `neurophase.metrics.*` | Phase-locking and related statistics. Useful to researchers, not to gate consumers. |
| `neurophase.validation.*` | Null-model harness + surrogate generators. Not required for live gate decisions. |
| `neurophase.sync.*` | Coupled brain–market simulators. Research infrastructure. |
| `neurophase.sensors.*` | EEG capture / synthetic sensors. Bridge layer territory (see PHASE 6). |
| `neurophase.audit.decision_ledger`, `neurophase.audit.replay` | Ledger and replay remain **importable** but are not on the public façade; they are embedded by `StreamingPipeline` and `RuntimeOrchestrator`. Integrators who need direct ledger access should import from the subpackage and acknowledge the internal-API nature of that contract. |

Nothing on this list is hidden or forbidden — they are simply not part of
the stability promise made by ``neurophase.api``. Importing from the
subpackage path is the documented way to reach them.

---

## Façade integrity guarantees

These properties are enforced by automated tests so the surface cannot
silently expand or regress:

* ``tests/test_import_surface.py`` — ``import neurophase.api`` does not
  pull ``mne`` / ``pandas`` / ``pywt`` / ``neurodsp`` / ``networkx`` /
  ``sklearn`` into ``sys.modules``.
* ``tests/test_public_api.py`` — the set of symbols exposed by
  ``neurophase.api.__all__`` equals the list above exactly (no silent
  additions, no silent removals).
* ``governance.completeness.API_FACADE_SURFACE`` (Doctor check #6) —
  every symbol in ``__all__`` resolves to a real runtime object.

If you add a symbol to ``neurophase.api``, you must update both this
document and ``tests/test_public_api.py`` in the same commit. If you
remove a symbol, you must bump the minor version and add an explicit
``CHANGELOG.md`` entry.
