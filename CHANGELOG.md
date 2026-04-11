# Changelog

All notable changes to neurophase are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to semantic versioning.

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
