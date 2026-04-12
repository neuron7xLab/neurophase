# neurophase — Runtime Canonical Frame

**Status.** Authoritative contract as of kernelization v1.
**Schema version.** `1.0.0` (`neurophase.contracts.CANONICAL_FRAME_SCHEMA_VERSION`)
**Typed envelope.** `neurophase.runtime.orchestrator.OrchestratedFrame`
**Serialization contract.** `neurophase.contracts.as_canonical_dict` +
`neurophase.contracts.validate_canonical_dict`

---

## The rule

Every live-runtime observation emitted by `neurophase` is an
`OrchestratedFrame`. Every **serialized** copy of that observation
— ledger entry, replay input, observatory export, downstream-adapter
message — is a **canonical dict** conforming to the schema in this
document.

Alternative payload shapes are explicitly banned on the runtime path.
See §5 for how the three adjacent frame-shaped types relate.

---

## 1 — Why one envelope

The PHASE 1 kernelization audit found four competing frame types
co-existing on the package:

| Type | Module | Role |
|---|---|---|
| `DecisionFrame` | `runtime/pipeline.py` | inner pipeline record per tick |
| `OrchestratedFrame` | `runtime/orchestrator.py` | full runtime envelope |
| `NeuralFrame` | `oscillators/neural_protocol.py` | **ingress** payload from a bio-sensor |
| `KLRFrame` | internal to `reset/` | parallel-subsystem payload |

That plurality is closure blocker **C2**. A downstream adapter, a
ledger, and an observatory collector cannot all agree on "what a frame
is" if there are four candidates. PHASE 4 resolves it by declaring a
single canonical envelope and making the others explicitly adjacent:

* `OrchestratedFrame` is the canonical typed runtime envelope.
* `DecisionFrame` is the **inner pipeline record** attached to
  `OrchestratedFrame.pipeline_frame`. It is not a top-level frame.
* `NeuralFrame` is **ingress** data (bio-sensor reading) — it flows
  *into* a pipeline tick, not *out of* it.
* `KLRFrame` is an **advisory subsystem** record; its effect on the
  runtime surfaces only through the three explicit `klr_*` keys
  on `DecisionFrame` (see §3).

---

## 2 — The schema

Schema version `1.0.0`. 22 keys, all at the top level. Never nested.
Every value is `None`, `str`, `int`, `float`, or `bool`.

### Meta (1)

| Key | Type | Notes |
|---|---|---|
| `schema_version` | str | Must equal `1.0.0` exactly. |

### Tick identity (2)

| Key | Type | Notes |
|---|---|---|
| `tick_index` | int | Zero-based monotonic counter within a session. |
| `timestamp` | float | Caller-supplied tick timestamp in seconds. |

### Raw inputs (2, nullable)

| Key | Type | Notes |
|---|---|---|
| `R` | float \| null | Joint order parameter. `None` when the caller passes no R. |
| `delta` | float \| null | Circular distance. `None` when the caller passes no delta. |

### Temporal integrity — B1 (2)

| Key | Type | Notes |
|---|---|---|
| `time_quality` | str | `TimeQuality` enum name (`WARMUP`, `HEALTHY`, `DEGRADED`, …). |
| `temporal_reason` | str | Human-readable reason string from the temporal validator. |

### Stream regime — B2 + B6 (3)

| Key | Type | Notes |
|---|---|---|
| `stream_regime` | str | `StreamRegime` enum name. |
| `stream_reason` | str | Stream detector reason. |
| `stream_fault_rate` | float | Observed fault rate in the rolling window. |

### Gate — I₁–I₄ (3)

| Key | Type | Notes |
|---|---|---|
| `gate_state` | str | `GateState` enum name (`READY`, `BLOCKED`, `DEGRADED`, `SENSOR_ABSENT`, `UNNECESSARY`). |
| `gate_reason` | str | Reason string — always ledger-visible. |
| `execution_allowed` | bool | True only when `gate_state == READY`. |

### Ledger tip — F1 (1, nullable)

| Key | Type | Notes |
|---|---|---|
| `ledger_record_hash` | str \| null | SHA256 of the appended ledger record, or `None` when no ledger is attached. |

### Regime inference — G1 (4, nullable)

Nullable because the regime layer is skipped when `R` or `delta` is
`None` (the gate has already vetoed that tick via `DEGRADED`).

| Key | Type | Notes |
|---|---|---|
| `regime_label` | str \| null | `RegimeLabel` enum name. |
| `regime_confidence` | float \| null | Confidence in `[0, 1]`. |
| `regime_warm` | bool \| null | True after the classifier's warmup window. |
| `regime_reason` | str \| null | Reason string. |

### Policy — I1 (1, nullable)

| Key | Type | Notes |
|---|---|---|
| `action_intent` | str \| null | `ActionIntent` enum name. `None` when the regime layer was skipped. |

### KLR subsystem (3, nullable)

Optional domain extension. **Advisory only**: these fields never widen
`execution_allowed`.

| Key | Type | Notes |
|---|---|---|
| `klr_decision` | str \| null | KLR summary (e.g. `"HOLD"`, `"RESET_PROPOSED"`). |
| `klr_ntk_rank_delta` | float \| null | Change in NTK rank proxy across the tick. |
| `klr_warning` | str \| null | One-line warning surface (`"ERROR"` if the KLR call raised). |

Total: 22 keys.

---

## 3 — Serialization and validation

```python
from neurophase.api import create_pipeline, RuntimeOrchestrator, OrchestratorConfig, PipelineConfig, PolicyConfig
from neurophase.contracts import as_canonical_dict, validate_canonical_dict

orch = RuntimeOrchestrator(OrchestratorConfig(pipeline=PipelineConfig(), policy=PolicyConfig()))
frame = orch.tick(timestamp=0.0, R=0.9, delta=0.01)

payload = as_canonical_dict(frame)       # dict
validate_canonical_dict(payload)         # raises SchemaValidationError on drift

import json
wire = json.dumps(payload, sort_keys=True)
```

`validate_canonical_dict` is the inverse contract. It enforces:

1. Exactly the declared set of keys (unexpected or missing keys raise).
2. `schema_version == CANONICAL_FRAME_SCHEMA_VERSION` (strict equality).
3. Each value is either `None` (for declared nullable keys) or an
   instance of the declared primitive type.
4. `bool` is not silently coerced into numeric fields.

Violations raise `SchemaValidationError` (`ValueError` subclass) —
callers should catch at the ingress boundary and reject the frame,
not crash.

---

## 4 — Replay safety

Two `RuntimeOrchestrator` objects with the same config fed the same
`(timestamp, R, delta)` sequence emit byte-identical canonical dict
sequences. This is covered by
`tests/test_canonical_frame.py::test_two_orchestrators_emit_identical_canonical_sequences`.

Ledger replay remains the authoritative byte-level replay mechanism
(`neurophase.audit.replay.replay_ledger`); the canonical dict is the
consumer-friendly projection of the same fact.

---

## 5 — Adjacent shapes that are *not* canonical

| Shape | Where it lives | Why it is not the canonical frame |
|---|---|---|
| `DecisionFrame` | `runtime/pipeline.py` | Inner pipeline record. Reachable via `OrchestratedFrame.pipeline_frame`. Exported on `neurophase.api` because downstream code often inspects it directly, but the full runtime envelope is always `OrchestratedFrame`. |
| `NeuralFrame` | `oscillators/neural_protocol.py` | Bio-sensor **ingress** payload (`sensor_status`, `phase`, `timestamp`). Flows *into* the bridge / pipeline, not out. Will be consumed by `bridges/eeg_ingress` in PHASE 6. |
| `KLRFrame` | internal to `reset/` | Parallel-subsystem record. Its effect on the runtime surfaces only via the three `klr_*` fields on `DecisionFrame`. |
| `OrchestratedFrame.to_json_dict()` | `runtime/orchestrator.py` | Backward-compat nested-dict projection. Retained because other tests / consumers depend on it. `as_canonical_dict` is the new blessed flat + versioned projection. |

---

## 6 — Version bump policy

* **Patch (`1.0.0 → 1.0.1`).** Documentation-only or non-observable
  internal changes. Typically does not require a version bump.
* **Minor (`1.0.0 → 1.1.0`).** Adding an optional nullable field with
  a safe default. Consumers on `1.0.0` can still read `1.1.0` frames
  as long as they ignore the unknown key — but `validate_canonical_dict`
  will refuse (strict equality), so consumers must bump first. This is
  intentional: silent schema drift is a worse failure mode than a
  refused frame.
* **Major (`1.0.0 → 2.0.0`).** Renaming, removing, or retyping any
  existing field. Requires a CHANGELOG entry, an updated
  `docs/RUNTIME_CANONICAL_FRAME.md`, an updated
  `tests/test_canonical_frame.py`, and coordinated consumer upgrades.

Every bump requires the same PR to update:

* `neurophase.contracts.CANONICAL_FRAME_SCHEMA_VERSION`
* `_FIELDS` in `neurophase/contracts/frame.py`
* this document
* `CHANGELOG.md`
