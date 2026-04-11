# Changelog

All notable changes to neurophase are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to semantic versioning.

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
