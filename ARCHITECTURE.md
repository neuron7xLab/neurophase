# neurophase — Architecture Overview (v0.4.0)

*Single-page map of the system for readers landing in the repo for
the first time. For the full doctrine and task map see
[`docs/EVOLUTION_BOARD.md`](docs/EVOLUTION_BOARD.md) and
[`docs/TASK_MAP.md`](docs/TASK_MAP.md).*

---

## 1. What `neurophase` is

A disciplined decision-and-research system that treats the brain
and the market as **coupled Kuramoto oscillators sharing a single
order parameter** `R(t)`. The core operational statement is a
5-state gate enforcing **four invariants** — `R(t) < θ`, sensor
absence, `R(t)` invalidity, and stillness — plus a **temporal
precondition** (`B₁`) that makes every downstream phase claim
physically meaningful.

`neurophase` is **not** a trading bot. It is the typed, tested,
reproducible substrate on which trading policy, research
experiments, and falsification protocols can be layered without
contaminating the core invariants.

---

## 2. Five-state gate

| State | `execution_allowed` | Semantic | Invariant |
|---|---|---|---|
| `SENSOR_ABSENT` | ❌ | bio-sensor unavailable | `I₂` |
| `DEGRADED`      | ❌ | `R(t)` invalid or **temporal precondition failed** | `I₃` ∪ `B₁` |
| `BLOCKED`       | ❌ | `R(t) < threshold` (desynchronized) | `I₁` |
| `UNNECESSARY`   | ❌ | `R(t) ≥ threshold` but dynamics are still | `I₄` |
| `READY`         | ✅ | synchronized + active — **the only permissive state** | — |

The global invariant `execution_allowed=True ⇒ state=READY` is
enforced at `GateDecision.__post_init__` and verified by the A2
formal state-machine specification in
[`STATE_MACHINE.yaml`](STATE_MACHINE.yaml).

---

## 3. Layered architecture

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  CALLER / RESEARCH EXPERIMENT / POLICY LAYER                 │
│  (not shipped here — Program I lives above the pipeline)     │
│                                                              │
└──────────────────────┬───────────────────────────────────────┘
                       │ DecisionFrame (typed runtime envelope)
┌──────────────────────┴───────────────────────────────────────┐
│  E1  StreamingPipeline             neurophase/runtime/       │
│       - single tick() entry point                            │
│       - owns all state                                       │
│       - parameter fingerprint for audit                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
      ┌────────────────┼───────────────────┬────────────────┐
      │                │                   │                │
┌─────┴───────┐ ┌──────┴────────┐ ┌────────┴──────┐ ┌───────┴──────┐
│ B1 Temporal │ │ B2+B6 Stream  │ │ 5-state Gate  │ │ F1 Ledger    │
│ Validator   │ │ Detector      │ │ +Stillness(I₄)│ │ (append-only │
│ (per-sample)│ │ (stream-level)│ │               │ │ SHA256 chain)│
│  neurophase/│ │               │ │  neurophase/  │ │              │
│  data/      │ │  neurophase/  │ │  gate/        │ │  neurophase/ │
│             │ │  data/        │ │               │ │  audit/      │
└─────────────┘ └───────────────┘ └───────────────┘ └──────────────┘
                              │
                       ┌──────┴──────────────┐
                       │                     │
         ┌─────────────┴──────┐ ┌────────────┴───────────┐
         │ CoupledBrainMarket │ │ PredictionError +      │
         │ System (Kuramoto)  │ │ ExecutiveMonitor       │
         │                    │ │                        │
         │  neurophase/sync/  │ │  neurophase/analysis/  │
         └────────────────────┘ │  neurophase/state/     │
                                └────────────────────────┘
```

Sideways, the **scientific and governance** layers sit **outside**
the hot path but enforce everything above at CI time:

```
┌───────────────────────────────────────────────────────────────┐
│  A1 + A2 GOVERNANCE                                           │
│    INVARIANTS.yaml  (5 invariants + 14 honest-naming HN)      │
│    STATE_MACHINE.yaml  (5 states + 8 transitions)             │
│    neurophase/governance/  (loaders + CI meta-tests)          │
├───────────────────────────────────────────────────────────────┤
│  C1 + C2 + C3 FALSIFICATION                                   │
│    NullModelHarness  (Phipson–Smyth smoothed p-values)        │
│    Surrogate generators  (phase_shuffle / cyclic_shift / bb)  │
│    PLV significance retrofit + HeldOutSplit enforcement       │
│    neurophase/validation/  +  neurophase/metrics/plv.py       │
├───────────────────────────────────────────────────────────────┤
│  H1 GROUND TRUTH                                              │
│    PhaseCouplingGenerator  (closed-form PLV at c ∈ {0, 1})    │
│    neurophase/benchmarks/                                     │
├───────────────────────────────────────────────────────────────┤
│  D1 CALIBRATION                                               │
│    calibrate_gate_threshold  (Youden-J on H1 traces)          │
│    Train/test split + generalization gap                      │
│    neurophase/calibration/                                    │
├───────────────────────────────────────────────────────────────┤
│  F3 DETERMINISM CERTIFICATION                                 │
│    Same-input → byte-identical ledger                         │
│    tests/test_determinism_certification.py                    │
└───────────────────────────────────────────────────────────────┘
```

---

## 4. Four invariants + 14 honest-naming contracts

**Hard invariants** (violation → `execution_allowed = False`):

- `I₁` · `R(t) < threshold ⇒ execution_allowed = False`
- `I₂` · sensor absent ⇒ `execution_allowed = False`
- `I₃` · `R(t)` invalid / NaN / OOR ⇒ `execution_allowed = False`
- `B₁` · non-VALID `time_quality` ⇒ `execution_allowed = False` (precondition, routed through `I₃`)

**Advisory invariant**:

- `I₄` · stillness ⇒ `execution_allowed = False` (distinct state `UNNECESSARY`)

**Honest-naming contracts** (HN1–HN14) — enforced by the A1 CI
meta-test. Each contract binds a load-bearing semantic rule to ≥ 1
pytest node id. If a PR removes or renames a bound test without
updating `INVARIANTS.yaml`, CI fails.

See [`INVARIANTS.yaml`](INVARIANTS.yaml) for the full registry.

---

## 5. Module map

| Package | Purpose | Key symbols |
|---|---|---|
| `neurophase.core` | Kuramoto primitives, phase extraction, order parameter | `KuramotoNetwork`, `order_parameter` |
| `neurophase.sync` | Coupled brain × market system (eq. 8.1) | `CoupledBrainMarketSystem`, `CoupledStep` |
| `neurophase.gate` | 5-state execution gate + stillness detector | `ExecutionGate`, `GateState`, `StillnessDetector` |
| `neurophase.data` | **B1/B2/B6** temporal validity + stream regime | `TemporalValidator`, `TemporalStreamDetector`, `TimeQuality`, `StreamRegime` |
| `neurophase.analysis` | Prediction-error monitor (Friston/Clark) | `PredictionErrorMonitor`, `CognitiveState` |
| `neurophase.state` | Executive function monitor (EEG β, HRV, error burst) | `ExecutiveMonitor`, `PacingDirective` |
| `neurophase.validation` | **C1/C2/C3** null models + PLV significance | `NullModelHarness`, `cyclic_shift`, `plv_significance`, `HeldOutSplit` |
| `neurophase.benchmarks` | **H1** synthetic ground truth | `generate_phase_coupling`, `PhaseCouplingConfig` |
| `neurophase.calibration` | **D1** threshold calibration | `calibrate_gate_threshold`, `ThresholdGrid` |
| `neurophase.audit` | **F1** append-only SHA256-chained ledger | `DecisionTraceLedger`, `verify_ledger`, `fingerprint_parameters` |
| `neurophase.runtime` | **E1/E2** streaming pipeline + DecisionFrame | `StreamingPipeline`, `PipelineConfig`, `DecisionFrame` |
| `neurophase.governance` | **A1/A2** invariant + state-machine registries | `load_registry`, `load_state_machine` |
| `neurophase.metrics` | PLV, Hurst, entropy, Ricci curvature, ISM | `plv`, `hurst_dfa`, `shannon_entropy`, `ollivier_ricci` |
| `neurophase.risk` | EVT/POT sizing + MF-DFA | `size_position`, `mfdfa` |
| `neurophase.oscillators` | Market oscillators + neural bridge protocol | `MarketOscillators`, `NeuralPhaseExtractor` |
| `neurophase.indicators` | QILM + FMN | `compute_qilm`, `compute_fmn` |
| `neurophase.intel` | BTC Field Order intel layer | `BTCFieldOrderRequest`, `validate_request` |
| `neurophase.agents` | π-calculus agents | `PiAgent`, `PiRule` |

---

## 6. Reproducibility contract

The system ships a **bit-deterministic** reproducibility contract:

> Same seeds + same inputs ⇒ byte-identical decision trace ledger files
> across two independent runs of the full pipeline.

Certified end-to-end by
[`tests/test_determinism_certification.py`](tests/test_determinism_certification.py)
(13 tests spanning 6 pillars). No downstream claim — calibration,
significance, regime classification — is trusted without the ledger
tip hash matching across two replays of the same configuration.

---

## 7. What `neurophase` deliberately does NOT do

- **Claim scientific significance without null-model confrontation.**
  Every p-value in the library is Phipson–Smyth smoothed and routed
  through `NullModelHarness` with a seeded surrogate generator.
- **Train learned models over hand-picked thresholds.** D1 is a
  transparent grid search with an explicit train/test split and a
  reported generalization gap. There is no silent hyperparameter
  tuning.
- **Compute "free energy".** The library exports `free_energy_proxy(δ) = ½·δ²`
  and nothing else. The honest-naming contract HN1 forbids claiming
  the full variational functional.
- **Repair time.** `TemporalValidator` reports faults, it never
  interpolates, resamples, or estimates drift. HN2.
- **Emit a decision without temporal validity.** The runtime
  pipeline's single entry point refuses to reach the gate for any
  non-HEALTHY stream regime.
- **Ship a policy layer.** Position sizing, action throttling, and
  external side effects are Program I tasks and must be layered
  above `StreamingPipeline`, never inside it.

---

## 8. Kernelization v1 — closure-ready cognition kernel

Shipped on `feat/kernelization-v1`, the repository now exposes a
canonical fail-closed cognition kernel with a closure-ready bridge
architecture. The additions are structural, not another research layer.

### 8.1 New subpackages

| Path | Role | Doc |
|---|---|---|
| `neurophase/contracts/` | Versioned on-wire schema for the runtime frame. `CANONICAL_FRAME_SCHEMA_VERSION = "1.0.0"`. | `docs/RUNTIME_CANONICAL_FRAME.md` |
| `neurophase/bridges/` | Typed, fail-closed ingress + egress: `EegIngress`, `MarketIngress`, `ClockSync`, `FrameMux`, `DownstreamAdapter`. | `docs/BRIDGE_CONTRACTS.md` |
| `neurophase/observatory/` | Outbound witness / event export for an external collector. `OBSERVATORY_SCHEMA_VERSION = "1.0.0"`. | `docs/OBSERVATORY_EXPORT.md` |

### 8.2 Import surface

* `import neurophase` no longer pulls the scientific stack. The root
  package now exposes only `__version__` plus a PEP 562 lazy
  accessor (`KLRConfig` backward-compat). Heavy modules load only when
  their subpackage is reached directly.

---

## 9. Governance closure layer (HN39/HN40)

- `neurophase.governance.checklist.governance_closure_valid()` is the
  canonical fail-closed predicate for governance readiness.
- `ExecutionGate` evaluates this predicate before any `READY` output;
  when governance artifacts are missing/invalid, the gate returns
  `BLOCKED` even if `R(t)` exceeds threshold.
- `owner_manifest.yaml` now includes explicit per-artifact ownership
  records for critical governance files.
* `neurophase.api` is the blessed public façade — 32 load-bearing
  runtime symbols, pinned by `tests/test_public_api.py`,
  documented at `docs/PUBLIC_API.md`.
* `pyproject.toml` is split:
  `pip install neurophase` → kernel-only (`numpy + scipy + PyYAML`);
  `pip install neurophase[research]` → full scientific stack;
  `pip install neurophase[dev]` → research + pytest/mypy/ruff.
  Enforced by `tests/test_pyproject_boundary.py`.

### 8.3 Canonical envelope

`OrchestratedFrame` is the single canonical **typed** runtime envelope.
`contracts.as_canonical_dict` is its single canonical **on-wire**
projection (22-key flat schema, all primitives).
`DecisionFrame`, `NeuralFrame`, `KLRFrame` are adjacent shapes: inner
pipeline record, ingress payload, and advisory-subsystem record
respectively — not runtime frames. Formalised in
`docs/RUNTIME_CANONICAL_FRAME.md` §5.

### 8.4 Gate-first execution law

`docs/GATE_FIRST_EXECUTION.md`: no downstream action derives from
anything other than a gate-approved `OrchestratedFrame`. Four
structural enforcements (runtime, `ActionPolicy`, `DownstreamAdapter`,
`FrameMux`) plus `tests/test_gate_first_execution.py`.

### 8.5 Closure-ready causal path

```
FrameMux → RuntimeOrchestrator → OrchestratedFrame
         → as_canonical_dict → DownstreamAdapter
```

Runs end-to-end in `tests/test_closure_path.py`; byte-identical under
replay. Honest name: **closure-ready causal path**, not "closed loop".
A true closed loop would need a live write-back channel from the
transport into the policy — **not** implemented and the documentation
does not claim otherwise. See `docs/CLOSURE_PATH.md` §"Future full
closed-loop target".

### 8.6 Closure blockers — status

| # | Blocker | Status |
|---|---|---|
| C1 | No `bridges/` layer | ✅ closed in phase 6 |
| C2 | No canonical `RuntimeFrame` | ✅ closed in phase 4 |
| C3 | `neosynaptex_adapter` is advisory | acknowledged; kept advisory by design |
| C4 | `import neurophase` not clean-env-safe | ✅ closed in phase 2 |
| C5 | Research deps treated as core | ✅ closed in phase 5 |
| C6 | Action intent not ledger-bound | partial: canonical frame carries `action_intent` + `ledger_record_hash`; deeper co-signing is a future phase |
| C7 | Reset substack is parallel runtime | accepted scope; documented in `RUNTIME_RESEARCH_SEPARATION.md` |
| C8 | No observatory export boundary | ✅ closed in phase 9 |

### 8.7 Test suite contracts

| Suite | Assertions | Role |
|---|---|---|
| `tests/test_import_surface.py` | 18 (subprocess-isolated) | kernel stays lean |
| `tests/test_public_api.py` | 34 | public façade is frozen |
| `tests/test_canonical_frame.py` | 14 | canonical envelope contract |
| `tests/test_pyproject_boundary.py` | 5 | dep split is honest |
| `tests/test_bridge_layer.py` | 28 | ingress + egress fail-closed |
| `tests/test_gate_first_execution.py` | 8 | no bypass |
| `tests/test_closure_path.py` | 4 | full chain + replay |
| `tests/test_observatory_export.py` | 10 | outbound contract |

---

## 9. Prior state (v0.4.0, pre-kernelization)

- **1354+ tests green** when v0.4.0 was tagged, `mypy --strict` clean,
  `ruff` clean.
- **5 invariants + 14 honest-naming contracts** enforced by CI
  meta-tests.
- **8 state-machine transitions** in the formal spec, each test-bound.
- **13 merged PRs** from v0.3.0 through v0.4.0 covering:
  StillnessDetector (I₄), TemporalValidator (B1), Invariant registry
  (A1), NullModelHarness (C1+C2), Decision ledger (F1), PLV
  significance retrofit (C3), Determinism certification (F3),
  State-machine spec (A2), H1 synthetic ground truth, D1 calibration,
  Stream detector (B2+B6), Runtime pipeline (E1+E2).

See [`CHANGELOG.md`](CHANGELOG.md) for the ordered release history.
