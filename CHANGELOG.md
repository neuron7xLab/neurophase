# Changelog

All notable changes to neurophase are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to semantic versioning.

## [Unreleased] — Interpretability + batch API + public façade + CLI

### Added — Interpretability layer (HN19)

- `neurophase/explain.py` — structured `DecisionExplanation`
  replacing opaque `reason: str` fields.
  - `Contract` enum: `B1 / I1 / I2 / I3 / I4 / READY`.
  - `Verdict` enum: `PASS / FAIL / SKIPPED`.
  - `ExplanationStep` frozen dataclass with typed `contract`,
    `verdict`, `observed`, `detail`.
  - `DecisionExplanation` frozen dataclass with `final_state`,
    `execution_allowed`, `causal_contract` (the first FAIL step
    in the chain, or READY), `chain`, `summary`. Deterministic;
    `as_text()` tree renderer; flat `to_dict()` for JSON pipelines.
  - `explain_decision(frame)` walks the priority order B₁ → I₂ →
    I₃ → I₁ → I₄ and emits one step per layer with structured
    pass/fail/skipped verdicts. No string parsing.
  - `explain_gate(decision)` convenience wrapper for bare
    `GateDecision` consumers (unit tests, direct gate calls).
- `tests/test_explain.py` — 17 new tests including determinism,
  flat JSON contract, causal-root marking, replay-bit-identity.

### Added — Vectorized batch API (HN20)

- `StreamingPipeline.tick_batch(frame)` — DataFrame in, DataFrame out.
  Processes `(timestamp, R, delta)` rows through the same stateful
  layers as serial `tick()` calls and emits one output row per input.
  When a ledger is attached, the resulting file is
  **byte-identical** to the serial path — strongest possible parity
  contract.
- `tests/test_batch_pipeline.py` — 13 new tests including column
  contract, semantic parity with the serial path, state-preservation
  across multiple `tick_batch` calls, NaN/missing handling,
  ledger byte-identity.

### Added — Public API façade + CLI (HN21)

- `neurophase/api.py` — single blessed import path re-exporting the
  minimal stable public surface: `StreamingPipeline`, `PipelineConfig`,
  `DecisionFrame`, `DecisionExplanation`, `Contract`, `Verdict`,
  `ExecutionGate`, `GateState`, `GateDecision`, `StillnessDetector`,
  `StillnessState`, `TimeQuality`, `create_pipeline`,
  `explain_decision`, `explain_gate`, `__version__`. Every symbol is
  identity-equal to its canonical module; the façade never wraps.
- `neurophase/__main__.py` — `python -m neurophase` CLI with four
  subcommands:
  - `version` — print installed version.
  - `demo [--ticks N]` — run a short synthetic pipeline and print
    gate states line by line.
  - `verify-ledger <path>` — verify a decision ledger's SHA256
    chain; exit code 0 iff verified.
  - `explain-ledger <path>` — emit one JSONL `DecisionExplanation`
    per ledger record for postmortem inspection.
- `tests/test_api_and_cli.py` — 11 new tests including façade
  contract (`__all__` frozen set, identity-equal symbols, version
  parity), `create_pipeline` round-trip, and CLI smoke tests for
  all four subcommands (clean and tampered ledger paths).

### INVARIANTS.yaml — HN19 / HN20 / HN21

Three new honest-naming contracts registered and bound to the
strongest tests of each layer. Enforced by the A1 CI meta-test.

### Stats

- **703 tests** green (up from 662).
- **106 source files** pass `mypy --strict`.
- **21 honest-naming contracts** (HN1–HN21) CI-bound.

## [Unreleased] — D2 stillness calibration + F2 replay engine

### Added — D2 (PR follows)

- `neurophase/calibration/stillness.py` — `calibrate_stillness_parameters`
  grid search over `(eps_R, eps_F, delta_min, window)` against H1
  synthetic quiet / active traces. Train/test split with
  generalization-gap reporting, parameter fingerprint, frozen
  `StillnessCalibrationReport`, JSON-serializable. 25 new tests in
  `tests/test_calibration_stillness.py` including determinism,
  fingerprint distinguishes grids, and physical-reality checks
  (tight `eps_R` rejects drifting R; loose params admit quiet
  traces).
- `INVARIANTS.yaml` — HN17 registered and bound to 8 tests.

### Added — F2 replay engine (same PR)

- `neurophase/audit/replay.py` — `replay_ledger()` non-destructive
  byte-level replay engine. Given an original ledger + input stream
  + `PipelineConfig`, re-runs the full pipeline into a scratch
  file and verifies the scratch is byte-identical to the original.
  Any config drift (e.g. threshold change) surfaces as a record-0
  divergence because `parameter_fingerprint` differs. Never writes
  to the original path.
- `ReplayInput` tuple mirrors `StreamingPipeline.tick` signature.
- `ReplayResult` frozen dataclass with `ok`, `n_records`,
  `original_tip_hash`, `replayed_tip_hash`, `first_divergent_index`,
  `scratch_path`, `reason`.
- Tampered original ledger is detected BEFORE replay starts
  (F1's `verify_ledger` is re-run as a safety check).
- 13 new tests in `tests/test_replay_engine.py` including happy
  path, config-drift divergence, tampered-original rejection, and
  full `CoupledBrainMarketSystem → ledger → replay` round-trip.
- `INVARIANTS.yaml` — HN18 registered and bound to 8 tests.

### Added — A3 (PR #25, merged separately)

- `tests/test_invariant_matrix.py` — **25 new tests** enforcing the
  safety-proof complement to every per-module test.
  Cartesian sweep of ~3 744 distinct gate-input cells against a
  pure analytical predictor encoded from `STATE_MACHINE.yaml`
  verbatim, plus 8 priority-ordering tests and 5 pipeline
  reachability tests.
- `docs/theory/invariant_matrix.md` — formal documentation.
- `INVARIANTS.yaml` — HN16 registered and bound to 18 tests.

### Stats

- **662 tests** green (up from 598).
- **100 source files** pass `mypy --strict`.
- **18 honest-naming contracts** (HN1–HN18) CI-bound.

## [Unreleased] — A3 cross-module invariant matrix + HN16

### Added — A3 cross-module invariant matrix (PR follows)

- `tests/test_invariant_matrix.py` — **25 new tests** enforcing the
  safety-proof complement to every per-module test.
  - **Analytical predictor**: a pure function encoding
    `STATE_MACHINE.yaml` verbatim (`B₁ > I₂ > I₃ > I₁ > I₄` priority
    order). No dependency on `ExecutionGate`; readable as the formal
    gate specification.
  - **Full Cartesian sweep**: ~3 744 distinct cells over
    `(time_quality, sensor_present, R, δ, has_stillness, forced_stillness)`.
    Every cell checked: `ExecutionGate.evaluate` output must equal
    predictor output. A single disagreement fails CI.
  - **Reachability**: every `GateState` member reached by at least
    one matrix cell; parametrized per-state for specific failure
    messages.
  - **Priority ordering**: eight tests each construct inputs that
    satisfy *two* failing conditions simultaneously and verify the
    higher-priority one wins (e.g. `temporal=GAPPED + sensor=False`
    → `DEGRADED` not `SENSOR_ABSENT`).
  - **State-machine × matrix consistency**: loads `STATE_MACHINE.yaml`
    and verifies every declared transition target is reached by ≥ 1
    matrix cell; every `execution_allowed=True` transition targets
    `READY`.
  - **Pipeline reachability**: complementary liveness proof driving
    the full `StreamingPipeline` (B1 → B2+B6 → gate) with a natural
    tick sequence and asserting every `GateState` is reached —
    `DEGRADED` via bad R, `BLOCKED` via low R, `READY` via healthy
    high R, `UNNECESSARY` via still high R, `SENSOR_ABSENT` via
    direct gate call (the pipeline has no sensor hook by design).
- `docs/theory/invariant_matrix.md` — formal documentation of the
  matrix: why it exists, the analytical predictor, what it covers,
  what it deliberately does not, and how to extend it.
- `INVARIANTS.yaml` — honest-naming contract **HN16** registered
  and bound to 18 strongest matrix tests. Enforced by the A1 CI
  meta-test.

### Why A3 matters (Karpathy / Sutskever-mode)

Every other test file in the suite probes one axis at a time. A
future refactor of `ExecutionGate._classify_ready` that accidentally
lets a single `(time_quality, sensor_present, R, δ, stillness)`
combination slip from `UNNECESSARY` to `READY` would pass every
isolated test — because none of them touches that specific
combination. A3 sweeps the full cross-product and compares the
live gate against a pure analytical predictor encoded from
`STATE_MACHINE.yaml`. This is the load-bearing *"a proof that
something cannot silently fail"* test for the gate-level semantic
surface.

### Stats

- **624 tests** green (up from 598).
- **96 source files** pass `mypy --strict`.
- **16 honest-naming contracts** (HN1–HN16) CI-bound.

## [Unreleased] — Research-grade bibliography + HN15 honest-citation contract

### Added

- `docs/theory/neurophase_elite_bibliography.md` — canonical,
  DOI-annotated, evidence-labelled source list: **24 real peer-reviewed
  sources** in S/A/B tiers with a full traceability matrix mapping
  every load-bearing claim to a module + test + falsification
  criterion.
- `docs/theory/hierarchical_status_bibliography.md` — compact
  companion.
- `docs/validation/evidence_labeling_style_guide.md` — four-tier
  evidence taxonomy (Established / Strongly Plausible / Tentative /
  Unsupported-Weak).
- `docs/validation/integration_readiness_protocol.md` — seven-phase
  release-readiness protocol wired to existing CI (no bespoke
  "governance kernel").
- `tests/test_bibliography_contract.py` — **25 CI tests** enforcing
  HN15: no fabricated future-dated citations, elite bibliography has
  ≥ 15 real DOI anchors, cross-references intact. Fails the build
  on any re-introduction of "Friston/Clark 2026", "Ming/Wharton
  2026", "NIH 2026", or "Capital-Weighted Kuramoto WG 2026".
- `INVARIANTS.yaml` — honest-naming contract **HN15** registered
  and bound to the 7 strongest bibliography tests.

### Changed — Fake-citation cleanup

Every reference to the fabricated "Friston/Clark 2026", "Ming/Wharton
2026", "NIH 2026", "Capital-Weighted Kuramoto WG 2026", and "Clark
2026" has been replaced with its real dated counterpart across
`docs/science_basis.md`, `docs/theory/neurophase_scientific_basis.md`,
`docs/theory/scientific_basis.md`, and `README.md`. Real sources:
Friston 2010, Clark 2013 / 2016, Haken 1983, Kelso 1995, Strogatz
2003, Miyake 2000, Arnsten 2009, Shenhav-Botvinick-Cohen 2013,
Cavanagh-Frank 2014, Lachaux 1999, Thayer-Lane 2000,
Shaffer-Ginsberg 2017, Phipson-Smyth 2010, Theiler 1992.

- `neurophase/metrics/asymmetry.py` — `_is_effectively_constant`
  signature tightened from `np.ndarray` to `npt.NDArray[np.float64]`.
- `neurophase/metrics/ricci.py` — `_local_distribution` return type
  tightened to `tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]`.
- `README.md` — "Citations" table now carries explicit DOI anchors
  and an evidence-status column.

### Rejected (from incoming PR #21, explicitly not merged)

The upstream bibliography work arrived bundled with theatrical
scaffolding: an "Ω governance kernel" that replaced the CI with a
single `python omega_governance_kernel.py` call, a "singularity
manifest", a "π-core self-improvement loop", an "evidence oracle"
with a broken `|cos(π·(1−x))|` non-monotonicity, fabricated
`FINAL_PRODUCTION_SIGNOFF` docs claiming "24/24 passed" before any
test had run, and — most importantly — the branch was cut from a
stale `main` predating the 13 v0.4.0 PRs, so merging it as-is would
have deleted ~9 000 lines of real B1/B2/B6/I₄/A1/A2/C1/C2/C3/D1/E1/F1/F3
code and 15 test files. This PR cherry-picks only the genuinely
valuable real content; everything theatrical is rejected. The
existing `pytest / ruff / mypy --strict` CI gates remain untouched.

### Stats

- **598 tests** green (up from 573 in v0.4.0).
- **95 source files** pass `mypy --strict`.
- **15 honest-naming contracts** (HN1–HN15) CI-bound.
- Bibliography: **24 real peer-reviewed sources**, ≥ 15 unique DOI anchors.

## [0.4.0] — 2026-04-11

Systems Evolution Board release. Installs the complete governance
+ falsification + reproducibility stack on top of v0.3.0. Four
invariants now enforced uniformly, 14 honest-naming contracts
CI-bound, the 5-state gate formally specified, and a single-path
streaming pipeline deployable end-to-end.

### Added — Governance (Program A)

- `docs/EVOLUTION_BOARD.md` — six-role identity contract + 10-question
  task-maturity gate + priority equation + anti-weakness enforcement.
- `docs/TASK_MAP.md` — 25 ranked tasks in Tier 1–5 structure with
  dependency gates.
- `INVARIANTS.yaml` — machine-readable invariant registry (5 hard
  invariants + 14 honest-naming contracts) with a CI meta-test that
  validates every binding.
- `STATE_MACHINE.yaml` — formal 5-state + 8-transition specification
  with strict priority ordering and per-transition test bindings.
- `neurophase/governance/` — typed loaders for both registries.
- `ARCHITECTURE.md` — single-page system map.

### Added — Temporal integrity (Program B)

- `neurophase/data/temporal_validator.py` (B1) — per-sample temporal
  contract check with 7-way `TimeQuality` enum: `VALID`, `GAPPED`,
  `STALE`, `REVERSED`, `DUPLICATE`, `WARMUP`, `INVALID`. Integrated
  into `ExecutionGate` as a pre-check that routes through `DEGRADED`.
- `neurophase/data/stream_detector.py` (B2 + B6) — stream-level
  `TemporalStreamDetector` with rolling fault-rate classification and
  hysteresis. `StreamRegime` enum: `WARMUP`, `HEALTHY`, `DEGRADED`,
  `OFFLINE`.
- `docs/theory/time_integrity.md` — formal derivation of the four
  temporal contracts with state-machine diagram.

### Added — Scientific validation (Program C)

- `neurophase/validation/null_model.py` (C1) — seeded deterministic
  `NullModelHarness` with Phipson–Smyth `(1+k)/(1+n)` smoothed
  p-values (never exactly zero for finite n).
- `neurophase/validation/surrogates.py` (C2) — three surrogate
  generators (`phase_shuffle`, `cyclic_shift`, `block_bootstrap`)
  with explicit null-hypothesis contracts.
- `neurophase/metrics/plv.py` (C3) — retrofit: `plv_significance`
  now delegates to `NullModelHarness`; new `HeldOutSplit` +
  `plv_on_held_out` make in-sample PLV architecturally impossible.

### Added — Calibration (Program D)

- `neurophase/benchmarks/phase_coupling.py` (H1) — synthetic
  phase-coupling generator with closed-form PLV ground truth at
  `coupling_strength ∈ {0, 1}`.
- `neurophase/calibration/threshold.py` (D1) — `calibrate_gate_threshold`
  with Youden-J grid search, train/test split, generalization-gap
  reporting, and parameter-fingerprinted reports.

### Added — Auditability (Program F)

- `neurophase/audit/decision_ledger.py` (F1) — append-only
  SHA256-chained `DecisionTraceLedger`; `verify_ledger` detects six
  orthogonal tamper mutation classes.
- `tests/test_determinism_certification.py` (F3) — 13-test end-to-end
  certification that same-input produces byte-identical ledgers
  across independent runs.

### Added — Runtime (Program E)

- `neurophase/runtime/pipeline.py` (E1 + E2) — `StreamingPipeline` +
  `DecisionFrame` + `PipelineConfig`. Composes `B1 → B2+B6 →
  ExecutionGate(+stillness) → optional F1 ledger` into a single
  stateful `tick()` entry point with parameter fingerprinting and
  byte-identical replay.

### Changed

- `ExecutionGate.evaluate` gains an optional `time_quality` parameter.
  Non-`VALID` time quality routes through `DEGRADED` (step 0 — strict
  priority above every downstream check).
- `CoupledBrainMarketSystem.run()` DataFrame now includes a `delta`
  column so downstream stillness / prediction-error consumers can
  drive the full `(R, δ)` stream without recomputing it.
- `plv_significance` now uses `NullModelHarness`. The naive unsmoothed
  p-value estimator is gone. `n_surrogates` floor lifted to 10.
- Invariant count in every public surface (docstrings, README, docs)
  updated from "Three" to **"Four"** (I₁–I₄).

### Stats

- **573 tests** green (up from 321 at v0.3.0).
- **94 source files** pass `mypy --strict`.
- **14 honest-naming contracts** (HN1–HN14) CI-bound.
- **13 PRs** shipped (#10 → #22) on top of the v0.3.0 baseline.

## [Unreleased]

### Added

**StillnessDetector + fourth invariant `I₄` + fifth gate state `UNNECESSARY`**

- `neurophase/gate/stillness_detector.py` — новий детектор `I₄`:
  `StillnessDetector` з rolling-window criterion (три кляузи:
  `max |dR/dt| < ε_R`, `max |dF_proxy/dt| < ε_F`, `max δ < δ_min`),
  warmup-семантика через `ACTIVE` (ніколи не `SENSOR_ABSENT`),
  опціональна hysteresis через `hold_steps`, `StillnessDecision` з
  повним provenance (dR/dt_max, dF_proxy/dt_max, delta_max,
  window_filled, reason), `free_energy_proxy(δ) = ½·δ²` як чесний
  геометричний surrogate (ніколи не full variational free energy).
- `neurophase/gate/execution_gate.py` — розширено до **5 станів**:
  `READY / BLOCKED / SENSOR_ABSENT / DEGRADED / UNNECESSARY`.
  `ExecutionGate.__init__` приймає опціональний `stillness_detector`;
  `ExecutionGate.evaluate` приймає опціональний `delta`. Strict
  evaluation order: sensor → R invalid → R<θ → stillness layer.
  Missing/invalid `δ` → `READY` (never `DEGRADED`). `GateDecision`
  інваріант `execution_allowed=True ⇒ state=READY` розширений на
  `UNNECESSARY` — enum включено в `__post_init__` перевірку.
  `GateDecision.stillness_state` — новий optional provenance field.
- `neurophase/sync/coupled_brain_market.py` — `run()` тепер повертає
  додаткову колонку `delta` (circular distance), щоб downstream
  `StillnessDetector` / `PredictionErrorMonitor` могли споживати
  `(R, δ)` прямо з DataFrame. `CoupledStep` extended accordingly.
- `tests/test_stillness_detector.py` — **55 тестів**, включно зі всіма
  5 обов'язковими (`test_still_when_all_criteria_met`,
  `test_active_when_R_changing`, `test_active_during_warmup`,
  `test_active_when_delta_too_large`, `test_rejects_invalid_inputs`)
  плюс: clause-wise isolation, ε-boundary behavior, window-wide vs
  last-sample differential test (core design claim), dt-scaling,
  F_proxy chain-rule verification, config validation, reset
  semantics, frozen-dataclass invariant, reason-string contract,
  hysteresis residency lock, window-size sweep, long-horizon noise
  stability.
- `tests/test_execution_gate.py` — розширено на повне `I₄` покриття:
  `test_returns_unnecessary_when_ready_but_still`,
  `test_still_is_not_blocked`, `test_five_gate_states_exhaustive`,
  `test_ready_when_stillness_detector_not_configured`,
  `test_ready_when_delta_missing_for_stillness_layer`, плюс priority
  tests (upstream invariants override `I₄`) і parametrized
  `GateDecision` invariant test включно з `UNNECESSARY`.
- `tests/test_stillness_pipeline.py` — **4 end-to-end тести** повного
  pipeline `CoupledBrainMarketSystem → (R, δ) → StillnessDetector →
  ExecutionGate`: converges-to-UNNECESSARY at high coupling,
  blocks-at-zero-coupling, stillness-state-reported-when-layer-runs,
  DataFrame-contract sanity.
- `docs/theory/stillness_invariant.md` — формальна деривація `I₄`:
  motivation, honest free-energy derivation, three-clause criterion з
  доказом що кожна кляуза незалежно необхідна, доказ що window-wide
  max — єдиний правильний оператор (last-sample та EMA failure
  modes), justification warmup → ACTIVE (not SENSOR_ABSENT), п'ять
  станів gate з semantic distinctions, опційна hysteresis, три
  worked counter-examples (oscillatory R, biased phase-lock at K=10,
  micro-noise), falsification hook.
- `docs/theory/scientific_basis.md` — Section 5 переписана з
  "Three" на "Four Invariants", додано повний `I₄` блок з
  посиланням на `stillness_invariant.md`.
- `examples/stillness_demo.py` — runnable demo, що інтегрує
  `CoupledBrainMarketSystem(K=50)` 500 RK4 steps через повний
  5-стан gate і друкує гістограму станів + перше досягнення кожного
  (на `seed=11`: 10 BLOCKED → 18 READY → 472 UNNECESSARY).

**Coupled brain–market Kuramoto system + prediction-error monitor + formal scientific basis**

- `neurophase/sync/coupled_brain_market.py` — новий модуль
  `CoupledBrainMarketSystem`, що реалізує рівняння 8.1 R&D звіту
  (Fioriti & Chinnici, 2012): brain ∪ market оператори діляться
  **одним** order parameter `R(t)`, RK4 на детерміністичному дрейфі,
  Euler–Maruyama на шумі, опціональна затримка `τ` через кільцевий
  буфер середніх фаз підпопуляцій. `run(n_steps)` повертає
  `pandas.DataFrame` зі схемою `t, R, psi_brain, psi_market,
  execution_allowed`.
- `tests/test_coupled_brain_market.py` — 23 тести, включно з усіма
  обов'язковими: `test_R_is_shared_between_brain_and_market`,
  `test_synchronizes_at_high_K`, `test_gate_blocks_when_R_below_threshold`,
  `test_equations_match_8_1_numerically` (RK4 до 1e-12),
  `test_delay_reduces_synchronization`, `test_noise_sigma_bounded_output`.
- `neurophase/analysis/prediction_error.py` — новий модуль
  `PredictionErrorMonitor`: Friston/Clark prediction error як circular
  distance `δ(t) = arccos(cos(ψ_brain − ψ_market))`, похідний
  `R_proxy = (1 + cos δ)/2`, three-band cognitive state
  (SYNCHRONIZED / DIVERGING / SURRENDERED), session archive через
  `history() -> pd.DataFrame`.
- `tests/test_prediction_error.py` — 15 тестів: zero error, maximum
  error, surrendered state, history schema, плюс validation, reset,
  монотонність `R_proxy` в δ.
- `docs/theory/scientific_basis.md` — формальний науковий базис: 6
  секцій (Theoretical Foundation / Neuroscience Evidence / Financial
  Evidence / Falsifiable Prediction / Gate Invariant / References),
  28 джерел, включно з R&D звітом і всіма цитатами з README.

**Cognitive-safety science basis and executive monitor**

- `docs/science_basis.md` — теоретичне обґрунтування архітектури через три
  нейрокогнітивні механізми 2026 року (cognitive surrender, executive
  function under stress, cognitive processing speed). Додано мапінг кожного
  механізму на компонент системи й falsifiable predictions.
- `docs/theory/neurophase_scientific_basis.md` — короткий науковий каркас
  (Predictive Brain / Cognitive Surrender / Individual Resilience) з одним
  спростовним прогнозом `PLV(EEG_beta, market_phase)` vs HRV/load.
- `neurophase/state/executive_monitor.py` — новий модуль `ExecutiveMonitor`:
  online-оцінка `OverloadIndex` з беспроводних каналів beta-power / HRV /
  error-burst, clip±4σ, scale-aware std floor, warmup-sentinel, strictly
  monotonic timestamps, `PacingDirective` (NORMAL / SLOW_DOWN / HARD_BLOCK /
  SENSOR_ABSENT), `VerificationStep` як structured friction.
- `tests/test_executive_monitor.py` — 32 тести: config validation, warmup,
  sensor-absent (None / NaN / ±inf), monotonic timestamps, classification
  bands, per-channel monotonicity, reset semantics, verification-step
  mapping, і falsifiable baseline property (monitor детектує injected
  stress-burst ДО появи поведінкових помилок).

### Changed

- `README.md` — секція *The Hypothesis* переписана під predictive-processing
  формулювання (brain = predictions, market = reality, `R(t)` = accumulated
  prediction error) + додана таблиця цитувань на 5 ключових джерел з
  посиланням на повний референсний лист у
  `docs/theory/scientific_basis.md`.
- `pyproject.toml` — `pandas>=2.0` перенесено у основні залежності
  (`CoupledBrainMarketSystem.run` і `PredictionErrorMonitor.history`
  повертають `pandas.DataFrame`).

## [0.3.0] — 2026-04-11

Full integration of the π-system / Neuron7X / BTC Field Order research
archive into a single typed, tested package. Five phases shipped in five
merged pull requests. 176 tests, 30 source files, mypy `--strict` clean.

### Added

**Phase A — math substrate (PR #1, commit 7f3db7b)**
- `core/phase.py` — Hilbert + Daubechies D4 wavelet denoising + adaptive
  `R_threshold(t) = mean + k·σ`.
- `core/kuramoto.py` — RK4 integrator with optional integer delays
  `τ_ij`, Gaussian phase noise `ξ_i(t)`, and liquidity-modulated
  coupling `K(t) = K₀·L(t)`.
- `core/order_parameter.py` — `R(t)·exp(iΨ)` for 1-D snapshots and 2-D
  trajectories.
- `metrics/entropy.py` — Shannon / Tsallis / Rényi with Freedman–Diaconis
  adaptive binning and `ΔH(t)` phase-transition signal.
- `metrics/ricci.py` — Ollivier (Wasserstein-1) and Forman curvature on
  weighted graphs + weighted mean `κ̄`.
- `metrics/hurst.py` — R/S and DFA Hurst estimators with Huber regression.
- `metrics/ism.py` — Information-Structural Metric
  `ISM = η · H'(t) / ⟨κ̄²⟩_T`.

**Phase B — emergent trigger + direction + indicators (PR #2, commit 5f5f552)**
- `gate/emergent_phase.py` — 4-condition criterion
  `R > 0.75 ∧ ΔH_S < −0.05 ∧ κ̄ < −0.1 ∧ ISM ∈ [0.8, 1.2]`.
- `gate/direction_index.py` — `DI = w_s·Skew + w_c·Δ_curv + w_b·Bias`
  resolving to `Direction.{LONG, SHORT, FLAT}`.
- `metrics/asymmetry.py` — skewness, excess kurtosis, topological
  `Δ_curv` across bull / bear subgraphs.
- `indicators/qilm.py` — Quantum Integrated Liquidity Metric (Neuron7X).
- `indicators/fmn.py` — Flow Momentum Network `tanh(w₁·OB + w₂·CVD/N)`.

**Phase C — risk (PR #3, commit 1999a22)**
- `risk/evt.py` — Peaks-Over-Threshold GPD fit + closed-form
  `VaR_p = u + (σ/ξ)·[(α/ζ)^(−ξ) − 1]` and
  `CVaR_p = (VaR_p + σ − ξu) / (1 − ξ)`. Exponential limit for
  `|ξ| < 1e-8`. Honest errors for `ξ ≥ 1`.
- `risk/mfdfa.py` — Multifractal Detrended Fluctuation Analysis with
  Huber log–log slopes; returns full `h(q)` spectrum and the
  multifractal instability index (spectrum width).
- `risk/sizer.py` — composite position sizer:
  `fraction = min(max_leverage, (risk_per_trade / CVaR) · scale_R · scale_m)`
  where `scale_R = (R − θ)/(1 − θ)` and
  `scale_m = max(1 − γ · mfdfa_instability, 0)`. Stateless, strictly
  validated `RiskProfile`.

**Phase D — agents + intel (PR #4, commit 108d627)**
- `agents/pi_agent.py` — π-calculus agent skeleton with `PiRule`,
  `AgentEfficiency` (`Sharpe + λ·Stability`), `MarketContext`,
  `SemanticMemory` (cosine retrieval), and `PiAgent.step()` A/B cycle
  (`mutate` / `repair` / `clone` / `learn`).
- `intel/btc_field_order.py` — BTC Market Intelligence Field Order v3.2
  strictly-typed payload builder: `SpotBlock`, `DerivativesBlock`,
  `OrderBookBlock`, `WhaleEvent`, `OnchainBlock`, `Scenario`,
  `BTCFieldOrderRequest`. `validate_request()` emits soft hygiene
  warnings; `build_signal_scan_payload()` serialises per section 8 of
  the protocol. No network, no LLM calls, no secrets.

**Phase E — neural bridge + market oscillators + theory (PR #5, commit aaf2b1d)**
- `oscillators/market.py` — three-channel phase bundle
  (`φ_price`, `φ_volume`, `φ_volatility`) via the shared Hilbert +
  wavelet pipeline plus rolling realized volatility.
- `oscillators/neural_protocol.py` — abstract bridge contract.
  `SensorStatus.{LIVE, ABSENT, DEGRADED}`, `NeuralFrame`,
  `NeuralPhaseExtractor` (runtime-checkable Protocol),
  `NullNeuralExtractor` (honest-absent default).
- `docs/theory/sensory_basis.md` — neurophysiological backing for the
  pupil / HRV / EEG α/β neural oscillators: four-circuit value model
  (V4 → VTA → vmPFC → dlPFC), Tobii / OpenBCI / Polar bridge contracts,
  DRD2 / COMT / DAT1 genetic moderators, cognitive-control strategies.
  References: Preuschoff 2011, Joshi 2016, Lo & Repin 2002,
  Jensen & Mazaheri 2010.
- `tests/test_integration.py` — end-to-end walk from Kuramoto physics
  through gate, emergent detector, direction index, and sizer on
  synthetic data.

### Changed

- **Package reorganisation:** moved `plv.py`, `execution_gate.py`, and
  `test_core.py` into the namespaced `neurophase.*` package. The flat
  top-level imports from 0.1 no longer exist; use
  `from neurophase import …`.
- **README refresh** — new architecture diagram covering all 8
  sub-packages, updated status table, typed usage example showing the
  full pipeline composition.
- **Quality bar** — ruff (E / F / W / I / N / UP / B / C4 / SIM / RUF),
  ruff format, mypy `--strict`, and pytest enforced on every commit via
  `.github/workflows/ci.yml` matrix on py3.11 + py3.12.

### Invariants

- **I1:** `R(t) < θ ⇒ execution_allowed = False` — enforced in
  `GateDecision.__post_init__` (cannot be bypassed by construction).
- **I2:** PLV computed on held-out data only — documented in
  `metrics.plv.plv_significance` docstring.
- **I3:** bio-sensor absent ⇒ `SensorStatus.ABSENT` with empty frame —
  enforced at the `oscillators.neural_protocol` boundary. No synthetic
  fallback anywhere in the package.

## [0.1.0] — 2026-04-11

Initial flat-layout scaffold: `plv.py`, `execution_gate.py`,
`test_core.py`, and the first README + animated hero SVG.
