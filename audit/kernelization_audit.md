# neurophase — Kernelization Audit

**Date:** 2026-04-12
**Auditor:** Claude (Opus 4.6) under NEUROPHASE EXECUTION PROTOCOL v1.0
**Target state:** `canonical fail-closed cognition kernel with closure-ready live bridge architecture`
**Current state:** `research-rich repository` — runtime primitives coexist with four classes of
research artefacts on a flat namespace.

This audit records only facts established from the current working tree
(`main @ 5115f1d` + dirty branch `feat/kernelization-v1`). No claims are
extrapolated.

---

## 1. Systemic shape

| Axis | Count |
|---|---|
| `neurophase/` Python modules | 122 |
| `tests/` Python modules | 103 |
| Top-level package re-exports (`__init__.py`) | 443 LoC, ~150 symbols |
| Blessed façade (`neurophase/api.py`) | 150 LoC, 28 symbols |
| Subpackages | 20 |
| Modules with top-level heavy-dep imports | 17 (see §5) |
| Distinct runtime frame types | 4 (see §4) |

---

## 2. Canonical runtime path (what the gate actually needs)

The minimum set of modules required to take a raw `(R, delta, timestamp)`
tuple and emit a gate-approved decision with audit record:

```
core/phase.py           compute_phase, preprocess_signal
core/order_parameter.py order_parameter
core/kuramoto.py        KuramotoNetwork (not on hot path, reference impl)

data/temporal_validator.py  TemporalValidator, TimeQuality, TemporalError
data/stream_detector.py     TemporalStreamDetector

gate/execution_gate.py     ExecutionGate, GateDecision, GateState
gate/stillness_detector.py StillnessDetector
gate/direction_index.py    direction_index
gate/emergent_phase.py     detect_emergent_phase

analysis/regime.py              RegimeClassifier, RegimeState
analysis/regime_transitions.py  RegimeTransitionTracker

policy/action.py         ActionPolicy, ActionIntent, ActionDecision
explain.py               explain_decision, explain_gate

runtime/pipeline.py      StreamingPipeline, PipelineConfig, DecisionFrame
runtime/orchestrator.py  RuntimeOrchestrator, OrchestratorConfig, OrchestratedFrame

audit/decision_ledger.py DecisionTraceLedger, LedgerVerification
audit/replay.py          replay_ledger
audit/session_manifest.py

validation/surrogates.py   cyclic_shift, time_reversal (minimal subset used at runtime)

governance/invariants.py  (load INVARIANTS.yaml at boot)
governance/state_machine.py (load STATE_MACHINE.yaml at boot)
```

**Total kernel footprint:** ~20 modules out of 122. Everything else is
research, calibration, audit-adjacent, or domain-specific intel.

---

## 3. Research-only components (must move off kernel surface)

These compute evidence but do not participate in a live gate decision:

| Path | Role | Heavy deps |
|---|---|---|
| `experiments/ds003458_analysis.py` | PLV verdict on EEG | mne, pandas, scipy |
| `experiments/ds003458_delta_analysis.py` | δ-power xcorr | mne, pandas, scipy |
| `experiments/ds003458_scp_analysis.py` | SCP xcorr | mne, pandas, scipy |
| `experiments/ds003458_trial_lme.py` | Trial-level LME | mne, pandas, statsmodels-like |
| `experiments/synthetic_plv_validation.py` | Synthetic PLV sweep | scipy |
| `data/ds003458_loader.py` | BIDS loader | mne, pandas |
| `data/eeg_preprocessor.py` | Phase extraction | mne, scipy |
| `benchmarks/neural_phase_generator.py` | Synthetic truth | scipy |
| `benchmarks/phase_coupling.py` | Synthetic truth | scipy |
| `benchmarks/stochastic_market_sim.py` | GBM market proxy | scipy |
| `benchmarks/ppc_analytical.py` | Ott-Antonsen PPC | scipy |
| `benchmarks/parameter_sweep.py` | Calibration grid search | - |
| `calibration/threshold.py` | Youden-J grid search | scipy |
| `calibration/stillness.py` | Stillness param grid | scipy |
| `metrics/ricci.py` | Forman/Ollivier curvature | networkx, scipy |
| `metrics/trial_theta_lme.py` | Trial-level LME | pandas, scipy |
| `metrics/scp.py` | SCP extractor | scipy |
| `metrics/delta_power.py` | δ-power extractor | scipy |
| `metrics/iplv.py` | iPLV (advanced variant) | scipy |
| `risk/mfdfa.py` | Multifractal DFA | scipy |
| `risk/evt.py` | GPD / EVT | scipy |

**Observation:** `metrics/plv.py`, `metrics/rayleigh.py`, `metrics/effect_size.py`,
`metrics/hurst.py`, `metrics/asymmetry.py`, `metrics/entropy.py` are thin and stay
on the kernel. `metrics/ricci.py`, `metrics/trial_theta_lme.py`, `metrics/scp.py`,
`metrics/delta_power.py`, `metrics/iplv.py` are research-specific.

---

## 4. Competing frame types

Four concurrent runtime frame definitions exist:

| Type | Module | Fields (abridged) |
|---|---|---|
| `DecisionFrame` | `runtime/pipeline.py` | timestamp, R, delta, gate_state, gate_decision, explanation, regime |
| `OrchestratedFrame` | `runtime/orchestrator.py` | wraps DecisionFrame + action_intent + replay metadata |
| `NeuralFrame` | `oscillators/neural_protocol.py` | sensor_status, phase, timestamp (bio-sensor payload) |
| `KLRFrame` | `reset/gamma_witness.py` (via `KLRPipeline`) | gamma, witness_report, reset state |

`NeuralFrame` is upstream (ingress). `DecisionFrame` and `OrchestratedFrame` are
adjacent — `OrchestratedFrame` composes a `DecisionFrame`. `KLRFrame` is parallel
to the main pipeline and lives inside `reset/` — it does not flow into
`DecisionFrame`.

**Not present:** a single load-bearing `RuntimeFrame` with audit metadata,
serialization, replay determinism, and ingress↔egress parity. The closest
approximation is `OrchestratedFrame`, but it is not used by the audit ledger and
does not carry `source_ids`, `coupling/coherence metrics`, or `gamma_state` in a
uniform schema.

---

## 5. Eager import + dependency hazards

`neurophase/__init__.py` (443 LoC) re-exports from every subpackage **including
`experiments/`**. Concretely:

```python
from neurophase.experiments.ds003458_analysis import run_analysis as ...
from neurophase.experiments.ds003458_delta_analysis import ...
from neurophase.experiments.ds003458_scp_analysis import ...
```

Each of those modules imports `mne` and `pandas` at module top. Result:

* `import neurophase` eagerly loads mne + pandas + scipy + neurodsp on
  every consumer, including thin ones that only want `ExecutionGate`.
* `pyproject.toml` lists mne, pandas, neurodsp, scikit-learn, networkx,
  PyWavelets, scipy as **core** dependencies. No `[project.optional-dependencies]`
  segregates research deps.
* `experiments/__init__.py` imports `ds003458_trial_lme` at package load.

Modules with top-level heavy imports (17, confirmed by grep):

```
analysis/prediction_error.py   (scipy)
benchmarks/ppc_analytical.py   (scipy)
core/phase.py                  (scipy) — on kernel path; acceptable
data/ds003458_loader.py        (mne, pandas) — research
data/eeg_preprocessor.py       (mne, scipy) — research
experiments/ds003458_*.py      (mne, pandas, scipy) — research (× 4)
metrics/delta_power.py         (scipy)
metrics/entropy.py             (scipy) — on kernel path; acceptable
metrics/ricci.py               (networkx, scipy)
metrics/scp.py                 (scipy)
metrics/trial_theta_lme.py     (pandas, scipy)
risk/evt.py                    (scipy)
sync/coupled_brain_market.py   (scipy) — research simulator
sync/market_phase.py           (scipy)
```

`scipy` is effectively required by the kernel (Hilbert transform in `core/phase.py`,
KS stats in `metrics/entropy.py`, surrogate tests). Keeping it core is defensible;
mne/pandas/neurodsp/networkx/statsmodels-adjacent code is not.

---

## 6. Boundary violations (what breaks the kernel contract)

1. **`__init__.py` re-exports research symbols.** 443 LoC of eager loads.
   Fix: trim to runtime surface, move experiments/benchmarks behind lazy import
   or remove from top-level `__init__.py`.
2. **`api.py` duplicates `__init__.py` without superseding it.** Downstream
   consumers can still write `from neurophase import run_ds003458_analysis` and
   pay the mne import cost. The façade is not load-bearing because the package
   root exposes the same wide surface.
3. **`experiments/__init__.py` eagerly imports `ds003458_trial_lme`** (mne +
   pandas). `from neurophase.experiments import X` for any X triggers mne load.
4. **No `bridges/` layer.** Ingress adapters live inline in
   `StreamingPipeline.tick()` and `RuntimeOrchestrator`. There is no named
   contract for `eeg_ingress`, `market_ingress`, `clock_sync`, `frame_mux`, or
   `downstream_execution_adapter`.
5. **Multiple frame types; no canonical.** `DecisionFrame`,
   `OrchestratedFrame`, `NeuralFrame`, `KLRFrame` coexist without a single
   canonical `RuntimeFrame` that unifies ingress metadata, phase state, gate
   verdict, and audit identity.
6. **`reset/` is parallel runtime.** `KLRPipeline` has its own 20-module
   substack (curriculum, calibrator, ntk_monitor, plasticity_injector,
   neosynaptex_adapter, …) that does not flow into `DecisionFrame` or the
   audit ledger. It is load-bearing for the γ-witness story, not for the live
   gate. Needs an explicit contract or a move under `research/` if it is not
   on the closure path.
7. **`pyproject.toml` treats research deps as core.** `mne`, `pandas`,
   `neurodsp`, `PyWavelets`, `scikit-learn`, `networkx` are listed in
   `[project].dependencies`. A kernel consumer cannot install without them.
8. **`governance/`, `intel/`, `indicators/`, `sensors/` are orthogonal.**
   `governance/` loads INVARIANTS/CLAIMS/STATE_MACHINE yaml — load-bearing at
   boot, but framed as audit meta-tests. `intel/btc_field_order.py` is a
   domain-specific request validator. `indicators/{fmn, qilm}.py` are crypto
   pipeline components. `sensors/` is EEG capture infra. None of these have a
   stated kernel role; all are on the top-level package surface.

---

## 7. Load-bearing artefacts

| Artefact | Role | Keep |
|---|---|---|
| `INVARIANTS.yaml` | 26 machine-readable contracts, each bound to ≥1 pytest | kernel |
| `CLAIMS.yaml` | 5 scientific claims with DOI evidence | kernel |
| `STATE_MACHINE.yaml` | 8 gate transitions, CI-verified | kernel |
| `py.typed` | public PEP 561 marker | kernel |
| `audit/decision_ledger.py` | SHA256 append-only chain | kernel |
| `audit/replay.py` | deterministic replay | kernel |
| `governance/doctor.py` | 11-check boot self-test | kernel |
| `results/*.json` | null results archive — scientific honesty | research |
| `data/ds003458/` | raw EEG (not in tree, gitignored) | external |

---

## 8. Demo / example-only components

* `__main__.py` (249 LoC) — CLI doctor + info + version, safe on kernel.
* `explain.py` — used by both kernel and research; stays.
* `intel/`, `indicators/` — crypto-specific; not on the generic cognition
  kernel path. Either move to a `research/crypto/` subpackage or declare
  them as a second-class extension.

---

## 9. Kernel surface (what survives Phase 2)

Provisional kernel inventory after kernelization:

```
core/                              phase primitives
  phase.py, order_parameter.py, kuramoto.py

contracts/            (NEW, Phase 3)
  frame.py           canonical RuntimeFrame
  schema.py          serialization + validation
  invariants.py      (thin re-export of governance.invariants)

runtime/                           execution loop
  pipeline.py, orchestrator.py, memory_audit.py

bridges/              (NEW, Phase 4)
  eeg_ingress.py, market_ingress.py
  clock_sync.py, frame_mux.py
  downstream_execution_adapter.py

gate/                              decision surface (unchanged)
  execution_gate.py, stillness_detector.py,
  direction_index.py, emergent_phase.py

analysis/regime{,_transitions}.py  regime inference
policy/action.py                   action intent synthesis

data/                              data-integrity layer (runtime)
  temporal_validator.py, stream_detector.py

metrics/                           kernel metrics only
  plv.py, rayleigh.py, effect_size.py, hurst.py, asymmetry.py, entropy.py

validation/                        surrogates for gate self-test
  surrogates.py, null_model.py

audit/                             ledger
  decision_ledger.py, replay.py, session_manifest.py

governance/                        yaml-backed contracts
  invariants.py, state_machine.py, claims.py, doctor.py, completeness.py,
  reproducibility.py, resistance.py, monograph.py

observatory/         (NEW, Phase 6)
  witness_export.py                outbound γ-witness contract
  event_bus.py                     outbound event contract

research/            (NEW, Phase 2)   isolated, opt-in, heavy deps ok
  experiments/                     ds003458_* , synthetic_plv_validation
  data_loaders/                    ds003458_loader, eeg_preprocessor
  benchmarks/                      neural_phase_generator, phase_coupling,
                                   stochastic_market_sim, ppc_analytical,
                                   parameter_sweep
  calibration/                     threshold, stillness
  metrics_ext/                     ricci, scp, delta_power, trial_theta_lme, iplv
  risk/                            evt, mfdfa, sizer
  sync/                            coupled_brain_market, market_phase
  sensors/                         recording, registry, synthetic
  klr/                             20-module reset substack (renamed from reset/)
  intel/                           btc_field_order
  indicators/                      fmn, qilm
```

---

## 10. Import hazards to repair (Phase 2 scope)

* Delete eager experiment imports from `neurophase/__init__.py`.
* Delete `ds003458_trial_lme` re-export from `neurophase/experiments/__init__.py`.
* Move or gate `data/ds003458_loader.py` + `data/eeg_preprocessor.py` behind
  a research extra (`data` subpackage keeps only the temporal layer).
* Add `[project.optional-dependencies].research`: mne, pandas, neurodsp,
  PyWavelets, scikit-learn, networkx.
* Leave `scipy` and `PyYAML` as core (used by kernel paths).
* Add `tests/test_import_surface.py` (Phase 8) that asserts `import neurophase`
  and `import neurophase.api` succeed with **only** core deps installed.

---

## 11. DoD check — PHASE 1

| Requirement | Status |
|---|---|
| List of system nodes | ✅ §2, §9 |
| List of boundary violations | ✅ §6 (8 items) |
| List of import hazards | ✅ §5, §10 |
| List of kernel-surface files | ✅ §2 (20 modules enumerated) |
| Research / demo separation | ✅ §3, §8 |
| Frame-type conflict inventory | ✅ §4 (4 types) |

**Phase 1 DoD: passed.** Artefact `audit/kernelization_audit.md` committed.

---

## 12. Closure blockers (what prevents honest live closure today)

A "closure blocker" is a fact that would make the statement
*"neurophase can close a live `ingress → gate → downstream` loop"* a lie.

| # | Blocker | Evidence |
|---|---|---|
| C1 | **No `bridges/` layer.** Ingress and egress adapters are inlined in `StreamingPipeline.tick()`. There is no typed contract for EEG ingress, market ingress, clock sync, frame mux, or downstream adapter. | absent from tree |
| C2 | **No canonical `RuntimeFrame`.** Four competing frame types (`DecisionFrame`, `OrchestratedFrame`, `NeuralFrame`, `KLRFrame`) without a single envelope carrying timestamp, tick index, temporal quality, regime, gate verdict, action intent, and ledger tip. | §4 |
| C3 | **`neosynaptex_adapter` is advisory, not live.** `reset/neosynaptex_adapter.py` is a read-only γ-witness. Calling it a closed-loop bus would be a lie. | audit-read |
| C4 | **`import neurophase` is not clean-env-safe.** `__init__.py` → `core.phase` → `pywt` at import stage. Any consumer without `pywt` gets `ModuleNotFoundError` before they can even touch the kernel. | reproduced: `neurophase/core/phase.py:29 import pywt` |
| C5 | **Research and runtime share one dependency list.** `mne`, `pandas`, `neurodsp`, `PyWavelets`, `scikit-learn`, `networkx` are core deps. Downstream runtime consumers pay ~500 MB of research baggage. | `pyproject.toml[project].dependencies` |
| C6 | **Action intent is not ledger-bound.** `policy/action.py` emits `ActionIntent`, but the audit ledger records `DecisionFrame` — not the emitted action. Gate verdict and downstream intent have no co-signed record. | `audit/decision_ledger.py` |
| C7 | **Reset substack is parallel runtime, not integrated.** `reset/KLRPipeline` runs its own 20-module stack that never produces a `DecisionFrame` or writes to the audit ledger. | §6.6 |
| C8 | **No observatory export boundary.** Outbound γ-witness / event export to external collectors has no typed contract. Hooks exist (`reset/gamma_witness.py`) but are internal. | absent |

---

## 13. Verdict

| Surface | Status |
|---|---|
| **Kernel surface** (see §9 provisional list) | present but polluted; ~20 modules on the real hot path |
| **Non-kernel surface** (research, experiments, benchmarks, calibration, KLR, intel, indicators, sensors, a subset of metrics and risk) | present, conflated with kernel on the top-level namespace |
| **Closure blockers** (§12) | 8 distinct, each addressable in Phases 2–9 |
| **Package boundaries** | violated in 8 ways (§6), dominant defect is C4 (pywt) + C5 (dep leak) |
| **Canonical runtime frame** | missing (C2) |
| **Live bridge layer** | missing (C1) |
| **Gate-first enforcement** | partial — gate exists; action path bypass not yet audited |
| **Closure-ready causal chain** | not demonstrable yet; no end-to-end test that chains ingress → frame → gate → intent → ledger |

**Direct verdict.** `neurophase` is a broad research repository with a real
but buried kernel spine. It is *not* a canonical fail-closed cognition kernel
today. It *is* within reach of one — the missing pieces are a thin import
surface, a canonical frame, a bridges layer, and a closure-path integration
test. None of these require inventing new science; they require cutting
scope, hardening contracts, and removing boundary violations.

Target state after full Phases 2–10 execution:
`closure-ready live cognition kernel`, not "fully closed-loop cognitive machine".

---

## 14. Known residual risk

* The audit maps the *current* tree. Two files in `sensors/` (`recording.py`,
  `synthetic.py`) have not been opened; their role is assumed from path and
  `__init__.py`.
* `reset/` substack is large (20 modules) and its closure relationship to the
  main `StreamingPipeline` is not fully mapped; Phase 2 must decide
  "kernel vs research" for each module individually, not as a bulk move.
* Some `governance/` modules (`monograph.py`, `resistance.py`) may be
  documentation tooling rather than runtime contracts; will revisit in Phase 9.
* The ds003458 live download is still running in `/tmp/neurophase/`; this has
  no bearing on kernel structure but is noted for context.
