<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero_kuramoto.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/hero_kuramoto.svg">
  <img alt="neurophase — Kuramoto phase synchronization animation" src="docs/assets/hero_kuramoto.svg" width="100%">
</picture>

# neurophase

**Phase synchronization as execution gate: a Kuramoto model of the market–trader system.**

[![status](https://img.shields.io/badge/status-experimental-blueviolet?style=flat-square)](#status)
[![invariants](https://img.shields.io/badge/invariants-4_hard_%2B_B%E2%82%81-critical?style=flat-square)](#invariants)
[![tests](https://img.shields.io/badge/tests-1354-brightgreen?style=flat-square)](tests/)
[![mypy](https://img.shields.io/badge/mypy-strict-1F5082?style=flat-square)](pyproject.toml)
[![doctor](https://img.shields.io/badge/doctor-11%2F11-00C853?style=flat-square)](#governance)
[![license](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat)](https://scipy.org/)
[![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat)](https://pytest.org/)
[![ruff](https://img.shields.io/badge/ruff-D7FF64?style=flat)](https://docs.astral.sh/ruff/)

</div>

---

## Abstract

**neurophase** treats a biological trader and a financial market as coupled
phase oscillators and computes a joint Kuramoto order parameter `R(t)` that
quantifies the accumulated prediction error between them. When `R(t)` falls
below a calibrated threshold, the system is desynchronized and execution is
blocked at the type boundary. The predicate

```
PLV( φ_neural , φ_market ) > 0    on held-out intraday horizons
```

is binary and falsifiable on the first run against human-in-the-loop data.
The library ships the complete measurement stack, a five-state execution
gate whose invariants are enforced at construction time, an append-only
audit ledger with byte-identical replay, and a governance layer of
machine-readable contracts covering invariants, claims, and state
transitions.

---

## Hypothesis

<table>
<tr>
<td width="50%" valign="top">

```
 Order parameter R(t)
 1.0 ┤          ╭──────╮      ╭──── θ
     │          │      │      │
 0.8 ┤       ╭──╯      ╰──╮   │
     │       │            │   │
 0.6 ┤ ──────┼────────────┼───┼──── gate
     │       │            │   │
 0.4 ┤    ╭──╯            ╰───╯
     │    │
 0.2 ┤╭───╯
     ││
 0.0 ┼┴──────────────────────────→ t
       09:30       12:00       16:00

          trade          silence
          window         window
```

</td>
<td width="50%" valign="top">

**neurophase** models the trader's nervous system and the financial market as
coupled Kuramoto oscillators sharing a single order parameter `R(t)`.

Biological channels (EEG α/β, HRV, pupil) supply predictions; the market
supplies realised dynamics. `R(t)` measures the accumulated prediction error
between the two populations as phase desynchronization
(Friston 2010; Clark 2013, 2016; Fioriti & Chinnici 2012).

When `R(t) < threshold` the gate blocks execution. When `R(t)` is sufficient
but the joint dynamics are still — `|dR/dt| < ε`, `|dF_proxy/dt| < ε`,
`δ < δ_min` over a rolling window — execution is marked `UNNECESSARY`:
no new information justifies action (invariant I₄,
see [`docs/theory/stillness_invariant.md`](docs/theory/stillness_invariant.md)).

The falsification predicate is `PLV(EEG_β, market_phase) > 0` on held-out
intraday horizons. The "cognitive surrender" product claim (structured
anti-offloading friction, Miyake 2000 + Arnsten 2009) is currently graded
*Strongly Plausible* rather than *Established*; see
[`docs/validation/evidence_labeling_style_guide.md`](docs/validation/evidence_labeling_style_guide.md).

```
              1   N
R(t)·e^{iΨ} = ─  Σ  e^{iθ_k(t)}
              N  k=1
```

</td>
</tr>
</table>

### Evidence base (DOI-anchored)

| Claim | Status | Source |
|-------|--------|--------|
| `R(t)` forecasts market critical points | Strongly Plausible | Fioriti & Chinnici (2012) |
| Brain as predictive engine (free-energy principle) | Established | Friston (2010) [`10.1038/nrn2787`]; Clark (2013) [`10.1017/S0140525X12000477`]; Clark (2016) *Surfing Uncertainty* |
| Executive function decomposition | Established | Miyake et al. (2000) [`10.1006/cogp.1999.0734`] |
| Stress impairs prefrontal control | Established | Arnsten (2009) [`10.1038/nrn2648`] |
| Expected value of cognitive control | Strongly Plausible | Shenhav, Botvinick & Cohen (2013) [`10.1016/j.neuron.2013.07.007`] |
| Frontal-theta as conflict/control signal | Established | Cavanagh & Frank (2014) [`10.1016/j.tics.2014.04.012`] |
| Phase-locking value (PLV) methodology | Established | Lachaux et al. (1999) |
| Phipson–Smyth smoothed p-values | Established | Phipson & Smyth (2010) [`10.2202/1544-6115.1585`] |
| HRV normative ranges (RMSSD / HF / SDNN) | Established | Shaffer & Ginsberg (2017) [`10.3389/fpubh.2017.00258`] |
| Anti-offloading architecture as product intervention | Strongly Plausible (context-bound) | Mechanism from Miyake + Arnsten; product claim requires A/B validation |
| Capital-weighted Kuramoto as crowding signal | Tentative | Working hypothesis; no peer-reviewed validation yet |

Full DOI-anchored reference list + traceability matrix: [`docs/theory/neurophase_elite_bibliography.md`](docs/theory/neurophase_elite_bibliography.md).
Compact companion: [`docs/theory/hierarchical_status_bibliography.md`](docs/theory/hierarchical_status_bibliography.md).
Evidence-labeling policy: [`docs/validation/evidence_labeling_style_guide.md`](docs/validation/evidence_labeling_style_guide.md).
System-scale evidence chain: [`docs/theory/scientific_basis.md`](docs/theory/scientific_basis.md).

---

## Falsification predicate

The system admits a single, binary, pre-registered predicate:

```
PLV( φ_neural , φ_market ) > 0    on held-out intraday horizons
```

where

```
         | mean[ exp(i·(φ_x − φ_y)) ] |
PLV  =   ─────────────────────────────     ∈ [0, 1]

    PLV = 0   random phase difference
    PLV = 1   perfect phase locking
```

Significance is assessed with `N = 1000` random cyclic shifts of `φ_y`,
which preserves the autocorrelation of the second signal while destroying
its cross-signal phase relationship (Lachaux et al. 1999). A PLV held-out
split prevents in-sample leakage (enforced by `HeldOutViolation`). A
rejection disconfirms the hypothesis on the public record in a single
commit; a confirmation is a falsifiable structural result.

---

## Invariants

The gate enforces four hard invariants and one temporal precondition.
All five are checked at construction time inside
`GateDecision.__post_init__`; constructing a decision with
`execution_allowed = True` while the resolved state is anything other
than `READY` raises `ValueError`.

<table>
<tr><th width="4%">#</th><th width="36%">Invariant</th><th width="60%">Mechanism</th></tr>
<tr>
<td align="center"><code>I₁</code></td>
<td><code>R(t) &lt; θ  ⇒  execution_allowed = False</code></td>
<td>Enforced in <code>GateDecision.__post_init__</code> — constructing a permissive decision while not <code>READY</code> raises <code>ValueError</code>.</td>
</tr>
<tr>
<td align="center"><code>I₂</code></td>
<td>bio-sensor absent ⇒ <code>execution_allowed = False</code></td>
<td>No synthetic fallback is substituted for missing hardware. Gate returns <code>SENSOR_ABSENT</code>; the silent default is the only non-permissive option.</td>
</tr>
<tr>
<td align="center"><code>I₃</code></td>
<td><code>R(t)</code> invalid / NaN / OOR ⇒ <code>execution_allowed = False</code></td>
<td>Failed <code>R(t)</code> computations are never silently coerced — gate returns <code>DEGRADED</code>.</td>
</tr>
<tr>
<td align="center"><code>I₄</code></td>
<td>stillness (<code>|dR/dt|&lt;ε</code> ∧ <code>|dF_proxy/dt|&lt;ε</code> ∧ <code>δ&lt;δ_min</code>) ⇒ <code>execution_allowed = False</code></td>
<td>Rolling-window <code>StillnessDetector</code> in <code>neurophase.gate</code>. Gate returns <code>UNNECESSARY</code>: no new information justifies action. See <a href="docs/theory/stillness_invariant.md"><code>stillness_invariant.md</code></a>.</td>
</tr>
</table>

---

## Architecture

```
                    ┌──────────────────────────────────────────────────────────────┐
                    │                       neurophase                             │
                    │       Kuramoto-gated execution engine (25 modules)           │
                    └───────────────────────────┬──────────────────────────────────┘
                                                │
        ┌──────────────┬────────────┬───────────┴──────────┬──────────────┐
        │              │            │                      │              │
   ┌────▼─────┐  ┌─────▼────┐  ┌───▼─────────┐  ┌────────▼───────┐  ┌──▼────────┐
   │OSCILLATORS│  │   CORE   │  │   METRICS   │  │   INDICATORS   │  │   SYNC    │
   │           │  │          │  │  (14 modules)│  │                │  │           │
   │ market    │  │ phase    │  │ plv · iplv   │  │ qilm           │  │ coupled   │
   │ neural    │  │ kuramoto │  │ entropy      │  │ fmn            │  │  _brain   │
   │  _protocol│  │ order    │  │ ricci · hurst│  │                │  │  _market  │
   └─────┬─────┘  │  _param  │  │ ism · asym   │  └────────────────┘  │ market    │
         │        └────┬─────┘  │ scp · delta  │                      │  _phase   │
         │             │        │ rayleigh     │                      └─────┬─────┘
         │             │        │ plv_verdict  │                            │
         │             │        └──────┬───────┘                            │
         │             │               │                                    │
         └──────────┐  └────────┐      │      ┌────────────────────────────┘
                    │           │      │      │
                    ▼           ▼      ▼      ▼
              ┌─────────────────────────────────────────────────────────┐
              │                                                         │
              │                     G  A  T  E                          │
              │    ┌─────────────────────────────────────────────┐      │
              │    │         execution_gate (I₁ · I₂ · I₃)      │      │
              │    │         stillness_detector (I₄)             │      │
              │    │         emergent_phase (4-criterion)        │      │
              │    │         direction_index (DI)                │      │
              │    └─────────────────────────────────────────────┘      │
              │                                                         │
              └──────────┬──────────────────────┬──────────────────────┘
                         │                      │
         ┌───────────────┤                      ├────────────────┐
         │               │                      │                │
    ┌────▼─────┐   ┌─────▼──────┐   ┌──────────▼────┐   ┌──────▼──────┐
    │   DATA   │   │    RISK    │   │    RUNTIME    │   │    AUDIT    │
    │          │   │            │   │               │   │             │
    │ temporal │   │ evt (GPD)  │   │ pipeline      │   │ ledger      │
    │  _valid  │   │ mfdfa      │   │ orchestrator  │   │ replay      │
    │ stream   │   │ sizer      │   │ memory_audit  │   │ manifest    │
    │  _detect │   │            │   │               │   │ (SHA-256)   │
    └──────────┘   └────────────┘   └───────┬───────┘   └─────────────┘
                                            │
        ┌───────────────┬───────────────────┼───────────────┬──────────────┐
        │               │                   │               │              │
   ┌────▼─────┐   ┌─────▼──────┐   ┌───────▼───────┐  ┌───▼──────┐  ┌───▼────────┐
   │  AGENTS  │   │   INTEL    │   │     RESET     │  │ ANALYSIS │  │ GOVERNANCE │
   │          │   │            │   │    (KLR)      │  │          │  │            │
   │ π-agent  │   │ btc_field  │   │ pipeline      │  │ predict  │  │ invariants │
   │ semantic │   │  _order    │   │ controller    │  │  _error  │  │ claims     │
   │  _memory │   │            │   │ γ-witness     │  │ regime   │  │ state_mach │
   └──────────┘   └────────────┘   │ plasticity    │  │ regime   │  │ doctor     │
                                   │ ensemble      │  │  _trans  │  │ monograph  │
                                   │ ntk_monitor   │  └──────────┘  └────────────┘
                                   │ curriculum    │
                                   └───────────────┘
                                            │
              ┌─────────────────────────────┼──────────────────────────┐
              │                             │                          │
        ┌─────▼──────┐             ┌────────▼──────┐          ┌───────▼─────┐
        │ VALIDATION │             │  BENCHMARKS   │          │ CALIBRATION │
        │            │             │               │          │             │
        │ null_model │             │ phase_coupling│          │ threshold   │
        │ surrogates │             │ neural_phase  │          │  (Youden-J) │
        │ (Phipson-  │             │ stochastic    │          │ stillness   │
        │  Smyth)    │             │  _market      │          │  (grid)     │
        └────────────┘             │ ppc_analytic  │          └─────────────┘
                                   │ param_sweep   │
                                   └───────────────┘
```

<br>

<table>
<tr>
<td align="center" width="12%"><b>Package</b></td>
<td align="center" width="26%"><b>Module</b></td>
<td align="center" width="6%"><b>State</b></td>
<td align="center" width="56%"><b>Purpose</b></td>
</tr>

<tr><td colspan="4" align="center"><b>— PHYSICS KERNEL —</b></td></tr>
<tr><td rowspan="3"><code>core</code></td><td><code>phase.py</code></td><td>implemented</td><td>Hilbert + Daubechies D4 wavelet denoising + adaptive R_thr</td></tr>
<tr><td><code>kuramoto.py</code></td><td>implemented</td><td>RK4 integrator · delays τ<sub>ij</sub> · noise ξ<sub>i</sub> · liquidity K(t)</td></tr>
<tr><td><code>order_parameter.py</code></td><td>implemented</td><td>R(t)·e<sup>iΨ</sup> = (1/N)·Σ e<sup>iθ<sub>k</sub></sup></td></tr>
<tr><td rowspan="2"><code>sync</code></td><td><code>coupled_brain_market.py</code></td><td>implemented</td><td>Coupled Kuramoto system — brain × market phase locking</td></tr>
<tr><td><code>market_phase.py</code></td><td>implemented</td><td>Market oscillator extraction from price series</td></tr>
<tr><td rowspan="2"><code>oscillators</code></td><td><code>market.py</code></td><td>implemented</td><td>Price · log-volume · realized volatility → φ<sub>market</sub></td></tr>
<tr><td><code>neural_protocol.py</code></td><td>implemented</td><td>Tobii / OpenBCI / Polar bridge Protocol — invariant <b>I₂</b></td></tr>

<tr><td colspan="4" align="center"><b>— EXECUTION GATE —</b></td></tr>
<tr><td rowspan="4"><code>gate</code></td><td><code>execution_gate.py</code></td><td>implemented</td><td>5-state gate · invariants I₁–I₃ + B₁ enforced at <code>__post_init__</code></td></tr>
<tr><td><code>stillness_detector.py</code></td><td>implemented</td><td>Rolling-window stillness (I₄) — |dR/dt| < ε ∧ |dF/dt| < ε ∧ δ < δ<sub>min</sub></td></tr>
<tr><td><code>emergent_phase.py</code></td><td>implemented</td><td>4-condition criterion: R ∧ ΔH ∧ κ̄ ∧ ISM</td></tr>
<tr><td><code>direction_index.py</code></td><td>implemented</td><td>DI = w<sub>s</sub>·Skew + w<sub>c</sub>·Δ<sub>curv</sub> + w<sub>b</sub>·Bias</td></tr>

<tr><td colspan="4" align="center"><b>— TEMPORAL INTEGRITY —</b></td></tr>
<tr><td rowspan="4"><code>data</code></td><td><code>temporal_validator.py</code></td><td>implemented</td><td>B₁ precondition — GAPPED / STALE / REVERSED / DUPLICATE detection</td></tr>
<tr><td><code>stream_detector.py</code></td><td>implemented</td><td>B₂/B₆ — regime classification: HEALTHY / WARMUP / STALE</td></tr>
<tr><td><code>ds003458_loader.py</code></td><td>implemented</td><td>OpenNeuro BIDS loader (Cavanagh 2021)</td></tr>
<tr><td><code>eeg_preprocessor.py</code></td><td>implemented</td><td>FMθ extraction, D4 denoising pipeline</td></tr>

<tr><td colspan="4" align="center"><b>— METRICS STACK —</b></td></tr>
<tr><td rowspan="14"><code>metrics</code></td><td><code>plv.py</code></td><td>implemented</td><td>Phase Locking Value + held-out surrogate test</td></tr>
<tr><td><code>iplv.py</code></td><td>implemented</td><td>Imaginary PLV — volume-conduction guard + PPC (Vinck 2010)</td></tr>
<tr><td><code>entropy.py</code></td><td>implemented</td><td>Shannon · Tsallis · Rényi + Freedman–Diaconis + ΔH</td></tr>
<tr><td><code>ricci.py</code></td><td>implemented</td><td>Ollivier (Wasserstein-1) + Forman + weighted mean κ̄</td></tr>
<tr><td><code>hurst.py</code></td><td>implemented</td><td>R/S + DFA with Huber regression</td></tr>
<tr><td><code>ism.py</code></td><td>implemented</td><td>ISM = η·H'(t) / ⟨κ̄²⟩<sub>T</sub> — information-to-curvature ratio</td></tr>
<tr><td><code>asymmetry.py</code></td><td>implemented</td><td>Skewness · kurtosis · topological Δ<sub>curv</sub></td></tr>
<tr><td><code>delta_power.py</code></td><td>implemented</td><td>Delta envelope extraction (1–4 Hz)</td></tr>
<tr><td><code>delta_price_xcorr.py</code></td><td>implemented</td><td>Cross-correlation: δ-power × price change</td></tr>
<tr><td><code>scp.py</code></td><td>implemented</td><td>Slow Cortical Potential (0.01–0.1 Hz)</td></tr>
<tr><td><code>trial_theta_lme.py</code></td><td>implemented</td><td>Trial-LME theta power (Toma method replication)</td></tr>
<tr><td><code>rayleigh.py</code></td><td>implemented</td><td>Rayleigh test — effect-size gate for phase uniformity</td></tr>
<tr><td><code>plv_verdict.py</code></td><td>implemented</td><td>Dual-surrogate verdict: Rayleigh + Bessel + held-out</td></tr>
<tr><td colspan="3" align="center"><i>(13 modules)</i></td></tr>
<tr><td rowspan="2"><code>indicators</code></td><td><code>qilm.py</code></td><td>implemented</td><td>Quantum Integrated Liquidity Metric (Neuron7X)</td></tr>
<tr><td><code>fmn.py</code></td><td>implemented</td><td>Flow Momentum Network — tanh(w₁·OB + w₂·CVD/N)</td></tr>

<tr><td colspan="4" align="center"><b>— RISK & SIZING —</b></td></tr>
<tr><td rowspan="3"><code>risk</code></td><td><code>evt.py</code></td><td>implemented</td><td>POT/GPD fit + closed-form VaR / CVaR</td></tr>
<tr><td><code>mfdfa.py</code></td><td>implemented</td><td>Multifractal DFA + instability index</td></tr>
<tr><td><code>sizer.py</code></td><td>implemented</td><td>Composite position sizer — CVaR cap · scale<sub>R</sub> · scale<sub>m</sub></td></tr>

<tr><td colspan="4" align="center"><b>— RUNTIME & AUDIT —</b></td></tr>
<tr><td rowspan="3"><code>runtime</code></td><td><code>pipeline.py</code></td><td>implemented</td><td>StreamingPipeline + DecisionFrame — tick-by-tick execution</td></tr>
<tr><td><code>orchestrator.py</code></td><td>implemented</td><td>Tick orchestration — data → physics → gate → action</td></tr>
<tr><td><code>memory_audit.py</code></td><td>implemented</td><td>Memory tracking + bounds enforcement</td></tr>
<tr><td rowspan="3"><code>audit</code></td><td><code>decision_ledger.py</code></td><td>implemented</td><td>Append-only SHA-256 chained decision trace</td></tr>
<tr><td><code>replay.py</code></td><td>implemented</td><td>Bit-deterministic replay engine (F₃ certification)</td></tr>
<tr><td><code>session_manifest.py</code></td><td>implemented</td><td>Session metadata + integrity verification</td></tr>

<tr><td colspan="4" align="center"><b>— KLR RESET SYSTEM —</b></td></tr>
<tr><td rowspan="6"><code>reset</code></td><td><code>pipeline.py</code></td><td>implemented</td><td>KLR full reset orchestration — state transitions + metrics</td></tr>
<tr><td><code>controller.py</code></td><td>implemented</td><td>Ketamine-Like Reset: plasticity injection + refractory</td></tr>
<tr><td><code>gamma_witness.py</code></td><td>implemented</td><td>External γ-verification (neosynaptex adapter, NEO-I₁/I₂)</td></tr>
<tr><td><code>ensemble.py</code></td><td>implemented</td><td>Multi-model ensemble averaging + passive learner</td></tr>
<tr><td><code>ntk_monitor.py</code></td><td>implemented</td><td>Neural Tangent Kernel rank monitor — loss landscape</td></tr>
<tr><td colspan="3" align="center"><i>(21 modules in reset/)</i></td></tr>

<tr><td colspan="4" align="center"><b>— VALIDATION & CALIBRATION —</b></td></tr>
<tr><td rowspan="2"><code>validation</code></td><td><code>null_model.py</code></td><td>implemented</td><td>NullModelHarness — Phipson–Smyth p = (1+k)/(1+n)</td></tr>
<tr><td><code>surrogates.py</code></td><td>implemented</td><td>cyclic_shift · phase_shuffle · time_reversal · block_bootstrap</td></tr>
<tr><td rowspan="2"><code>calibration</code></td><td><code>threshold.py</code></td><td>implemented</td><td>Youden-J calibration with explicit train/test split</td></tr>
<tr><td><code>stillness.py</code></td><td>implemented</td><td>Stillness parameter grid search (ε<sub>R</sub>, ε<sub>F</sub>, δ<sub>min</sub>)</td></tr>
<tr><td rowspan="5"><code>benchmarks</code></td><td><code>phase_coupling.py</code></td><td>implemented</td><td>Controlled coupling with closed-form PLV at c ∈ {0, 1}</td></tr>
<tr><td><code>neural_phase_generator.py</code></td><td>implemented</td><td>Kuramoto ODE for synthetic EEG phase</td></tr>
<tr><td><code>stochastic_market_sim.py</code></td><td>implemented</td><td>GBM + neural EMA tracking</td></tr>
<tr><td><code>ppc_analytical.py</code></td><td>implemented</td><td>Analytical PPC + theoretical PLV</td></tr>
<tr><td><code>parameter_sweep.py</code></td><td>implemented</td><td>Grid search harness for coupling sweeps</td></tr>

<tr><td colspan="4" align="center"><b>— GOVERNANCE & INTELLIGENCE —</b></td></tr>
<tr><td rowspan="4"><code>governance</code></td><td><code>invariants.py</code></td><td>implemented</td><td>INVARIANTS.yaml loader — 26 machine-readable contracts</td></tr>
<tr><td><code>claims.py</code></td><td>implemented</td><td>CLAIMS.yaml — hypothesis → theory → fact promotion rules</td></tr>
<tr><td><code>state_machine.py</code></td><td>implemented</td><td>STATE_MACHINE.yaml — 8 transitions, CI-verified exhaustive</td></tr>
<tr><td><code>doctor.py</code></td><td>implemented</td><td>11-axis completeness checker (reachability, resistance, determinism)</td></tr>
<tr><td><code>agents</code></td><td><code>pi_agent.py</code></td><td>implemented</td><td>π-calculus: mutation / repair / clone / learn + semantic memory</td></tr>
<tr><td><code>intel</code></td><td><code>btc_field_order.py</code></td><td>implemented</td><td>BTC Field Order v3.2 structured LLM payload (no network)</td></tr>
<tr><td rowspan="3"><code>analysis</code></td><td><code>prediction_error.py</code></td><td>implemented</td><td>Friston/Clark prediction-error monitor</td></tr>
<tr><td><code>regime.py</code></td><td>implemented</td><td>TRENDING / COMPRESSING / FLASH / FLASH-SUPPRESS taxonomy</td></tr>
<tr><td><code>regime_transitions.py</code></td><td>implemented</td><td>Regime transition logic + hysteresis</td></tr>
<tr><td rowspan="2"><code>state</code></td><td><code>executive_monitor.py</code></td><td>implemented</td><td>EEG β · HRV · error-burst detection → OverloadIndex</td></tr>
<tr><td><code>klr_reset.py</code></td><td>implemented</td><td>Ketamine-Like Reset state machine</td></tr>
<tr><td><code>explain</code></td><td><code>explain.py</code></td><td>implemented</td><td>Causal decision explanation engine — contracts → steps → verdict</td></tr>
</table>

---

## Execution Gate — 5-State Machine

> *Priority-ordered evaluation: T₀ → T₇. First match wins. Only `READY` permits execution.*

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                         evaluate(R, θ, ...)                             │
  └────────┬─────────────────────────────────────────────────────────────────┘
           │
           │ T₀: time_quality ≠ VALID (B₁)
           ├──────────────────────────────────────▶ ┌──────────────────────┐
           │                                        │     DEGRADED         │
           │ T₂: R is NaN / None / OOR (I₃)        │   allowed = False    │
           ├──────────────────────────────────────▶ │   reason: temporal / │
           │                                        │          R invalid   │
           │                                        └──────────────────────┘
           │
           │ T₁: sensor_present = False (I₂)
           ├──────────────────────────────────────▶ ┌──────────────────────┐
           │                                        │   SENSOR_ABSENT      │
           │                                        │   allowed = False    │
           │                                        └──────────────────────┘
           │
           │ T₃: R < θ (I₁)
           ├──────────────────────────────────────▶ ┌──────────────────────┐
           │                                        │     BLOCKED          │
           │                                        │   allowed = False    │
           │                                        └──────────────────────┘
           │
           │ R ≥ θ
           │
           │ T₇: stillness_active = STILL (I₄)
           ├──────────────────────────────────────▶ ┌──────────────────────┐
           │                                        │   UNNECESSARY        │
           │                                        │   allowed = False    │
           │                                        │   (no new info)      │
           │                                        └──────────────────────┘
           │
           │ T₄/T₅/T₆: all checks passed
           └──────────────────────────────────────▶ ┌──────────────────────┐
                                                    │      READY           │
                                                    │   allowed = True     │
                                                    │   (execute)          │
                                                    └──────────────────────┘
```

<br>

<div align="center">

| State | `execution_allowed` | Invariant | Semantics |
| :--- | :---: | :---: | :--- |
| `READY` | `True` | — | R(t) ≥ θ, bio-sensor present, dynamics active |
| `BLOCKED` | `False` | I₁ | R(t) < θ; joint system desynchronized |
| `SENSOR_ABSENT` | `False` | I₂ | Bio-sensor unavailable; execution suppressed |
| `DEGRADED` | `False` | I₃ / B₁ | R(t) invalid or temporal quality failed |
| `UNNECESSARY` | `False` | I₄ | Dynamics stationary; no new information |

**Global invariant:** `execution_allowed = True  ⇒  state = READY` — enforced at `GateDecision.__post_init__`

</div>

---

## Lifecycle — Signal Flow

> *Every tick: raw input → temporal guard → physics → gate → decision → audit chain.*

```
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║                          SIGNAL LIFECYCLE                               ║
  ╚══════════════════════════════════╤════════════════════════════════════════╝
                                     │
         ┌───────────────────────────┴───────────────────────────┐
         │                   RAW INPUT                           │
         │     market price   ·   EEG α/β   ·   HRV   ·   pupil│
         └───────────────────────────┬───────────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  B₁: TEMPORAL       │  GAPPED → block
                          │      VALIDATOR      │  STALE  → block
                          │  (precondition)     │  VALID  → pass
                          └──────────┬──────────┘
                                     │ VALID
                          ┌──────────▼──────────┐
                          │  B₂: STREAM         │  WARMUP → wait
                          │      DETECTOR       │  STALE  → block
                          │  (regime class.)    │  HEALTHY→ pass
                          └──────────┬──────────┘
                                     │ HEALTHY
         ┌───────────────────────────▼───────────────────────────┐
         │              PHYSICS KERNEL                           │
         │                                                       │
         │  oscillators → coupled Kuramoto → R(t)·e^{iΨ}        │
         │  dθ_k/dt = ω_k + (K/N)·Σ sin(θ_j − θ_k) + ξ_i     │
         │                                                       │
         │  metrics: PLV · entropy · Ricci · Hurst · ISM         │
         └───────────────────────────┬───────────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  EXECUTION GATE     │
                          │  (5-state, I₁–I₄)   │  ──→ BLOCKED / DEGRADED
                          │                     │  ──→ SENSOR_ABSENT
                          │  evaluate(R, θ, ..) │  ──→ UNNECESSARY
                          └──────────┬──────────┘
                                     │ READY
         ┌───────────────────────────▼───────────────────────────┐
         │              DECISION LAYER                           │
         │                                                       │
         │  emergent_phase(R, ΔH, κ̄, ISM)   →  is_emergent?    │
         │  direction_index(skew, curv, bias) →  LONG / SHORT   │
         │  size_position(R, θ, CVaR, MFDFA)  →  fraction       │
         └───────────────────────────┬───────────────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  F₁: AUDIT LEDGER   │  append-only
                          │  SHA-256 chain       │  bit-deterministic
                          │  replay-certified    │  no silent drops
                          └─────────────────────┘
```

---

## Governance

> *Every claim traces to DOI. Every invariant has a CI-bound test. Every transition is exhaustive.*

<div align="center">

| Contract Layer | File | Entries | CI-Enforced |
| :--- | :--- | :---: | :---: |
| **Invariants** | `INVARIANTS.yaml` | 26 | meta-test: unbound invariant → fail |
| **Claims** | `CLAIMS.yaml` | 5 | status must match citation count |
| **State Machine** | `STATE_MACHINE.yaml` | 8 transitions | exhaustiveness verified in test |
| **Honest Naming** | `INVARIANTS.yaml` (HN*) | 26+ | naming contract violations → fail |
| **Doctor** | `governance/doctor.py` | 11 axes | reachability + resistance + determinism |

</div>

**Claim promotion ladder:**
```
hypothesis (0–1 citations)  →  theory (2 citations)  →  fact (3+ citations, no contradiction)
```

---

## Synthetic Validation

Until bio-sensor hardware arrives, the falsification pipeline runs on synthetic data where ground-truth PLV is known.

```bash
python -m neurophase.experiments.synthetic_plv_demo
```

Expected behaviour across coupling strengths `K ∈ [0.5, 4.0]`:

```
  PLV
  1.0 ┤                        ╭─────────●     ← phase-locked regime
      │                   ╭────╯
  0.8 ┤              ╭────╯
      │           ╭──╯
  0.6 ┤        ╭──╯                      ╭──── hypothesis: PLV > 0
      │     ╭──╯
  0.4 ┤   ╭─╯
      │  ╭╯
  0.2 ┤ ╭╯
      │╭╯
  0.0 ●───────────────────────────────────── ← desynchronized regime
      0.5  1.0  1.5  2.0  2.5  3.0  3.5  4.0   K
```

---

## Install & Run

```bash
git clone https://github.com/neuron7xLab/neurophase
cd neurophase
pip install -e ".[dev]"

# Optional — enable the external γ-verification witness (NEO-I1/NEO-I2).
# Adds `neosynaptex` as an advisory channel on KLRFrame.witness_report;
# without it KLRPipeline keeps working and witness_report stays None.
pip install -e ".[dev,witness]"

ruff check neurophase tests
mypy neurophase        # --strict, 122 source files, 0 errors
pytest -q              # 1354 passed, 6 skipped
python -m neurophase doctor  # 11/11 axes, exit 0
```

<br>

```python
# Full pipeline — physics → gate → emergent → direction → sizer
import numpy as np
from neurophase import (
    KuramotoNetwork, order_parameter,
    ExecutionGate, detect_emergent_phase, direction_index,
    RiskProfile, size_position,
    Direction,
)

# 1. Physics — high-coupling Kuramoto network
omega = np.linspace(-0.3, 0.3, 20)
net = KuramotoNetwork(omega, coupling=5.0, dt=0.05, seed=0)
trajectory = net.run(n_steps=600)
R = order_parameter(trajectory[-1]).R   # ~0.97

# 2. Execution gate (invariant I1)
gate = ExecutionGate(threshold=0.65)
assert gate.evaluate(R).execution_allowed

# 3. Emergent phase — 4-condition criterion
emergent = detect_emergent_phase(R=R, dH=-0.08, kappa=-0.15, ism=1.0)
assert emergent.is_emergent

# 4. Direction from skewness + curvature asymmetry + bias
di = direction_index(skew=0.5, curv=0.2, bias=0.1)
assert di.direction is Direction.LONG

# 5. Position sizing — CVaR budget × sync × multifractal
size = size_position(R=R, threshold=0.65, cvar=0.05,
                     multifractal_instability_value=0.1,
                     profile=RiskProfile(max_leverage=3.0))
print(size.fraction, size.reason)
```

---

## Status & Research Results

<div align="center">

### Engineering

| Metric | Value |
| :--- | ---: |
| Python source modules | 122 |
| Test cases | 1 354 |
| mypy --strict errors | 0 |
| ruff violations | 0 |
| Doctor axes passing | 11 / 11 |
| Invariant contracts (`INVARIANTS.yaml`) | 26 |
| Scientific claims (`CLAIMS.yaml`) | 5 |
| State-machine transitions (CI-exhaustive) | 8 |
| Determinism certification | bit-identical across 6 pillars |

### Empirical results (ds003458 — Cavanagh 2021, 23 subjects)

| Analysis | Metric | N | Significant | Verdict |
| :--- | :--- | :---: | :---: | :--- |
| FMθ (4–8 Hz) vs market | PLV held-out | 17 | 0 / 17 | NULL — frequency mismatch (0.001 Hz vs 4–8 Hz) |
| Delta (1–4 Hz) × price | xcorr | 23 | 2 / 23 | NULL — mixed signs, no systematic effect |
| SCP (0.01–0.1 Hz) × reward | xcorr | 23 | 0 / 23 | NULL — no signal detected |
| Trial theta power | LME (Toma) | 23 | p = 0.935 | NULL — deterministic rewards |
| Synthetic PLV bridge | PPC sweep | — | confirmed | Methodology verified at known c ∈ {0, 1} |

Null results are committed verbatim to `results/`. The hypothesis survives; the
dataset does not confirm it. The next empirical target is a stochastic reward
dataset (Torres et al. or equivalent).

</div>

### Open hardware dependency

Bio-sensor adapters implementing `NeuralPhaseExtractor` are declared as a
`Protocol` with a `NullNeuralExtractor` contract fixture; concrete
implementations (Tobii → pupil phase, OpenBCI → EEG phase, Polar → HRV) are
out of scope for this repository and gated by physical hardware availability.
Scientific grounding and the bridge contract are documented in
[`docs/theory/sensory_basis.md`](docs/theory/sensory_basis.md).

---

## KLR — Ketamine-Like Reset

The KLR subsystem (21 modules) implements a controlled reset protocol that
dislodges pathological attractor lock-in by injecting structured noise into
the coupling matrix and safely committing only validated improvements. The
mechanism is inspired by ketamine's documented restoration of neural
plasticity under established stress conditions; the biological reference
motivates the architecture but does not enter any quantitative claim:

```
  NORMAL ──→ MONITORING ──→ TRIGGERED ──→ INJECTING ──→ REFRACTORY ──→ NORMAL
                │                              │
                │  NTK rank declining           │  plasticity restored
                │  ensemble diverging           │  γ-witness confirmed
                └──────────────────────────────┘
```

| Component | Role |
| :--- | :--- |
| `controller.py` | State transitions: NORMAL → MONITORING → TRIGGERED → INJECTING → REFRACTORY |
| `plasticity_injector.py` | Temporal noise injection calibrated to loss landscape curvature |
| `ntk_monitor.py` | Neural Tangent Kernel rank — early warning for loss landscape degeneration |
| `gamma_witness.py` | External γ-verification via neosynaptex (advisory, NEO-I₁/I₂) |
| `ensemble.py` | Multi-model consensus — rank-based passive learner |
| `curriculum.py` | Curriculum learning scheduler — progressive difficulty |
| `deterministic_oracle.py` | Noise-free reference model for calibration |

When installed with `pip install -e ".[witness]"`, the neosynaptex γ-witness provides external verification. Without it, the system degrades gracefully — `witness_report = None`, all invariants hold.

---

## Physics Kernel

> *Every signal traces back to peer-reviewed science. Every clamp traces back to a law.*

- **Kuramoto, Y.** (1984). *Chemical Oscillations, Waves, and Turbulence.* Springer.
- **Gidea, M. & Katz, Y.** (2018). *Topological data analysis of financial time series.* Physica A.
- **Lachaux, J.-P., Rodriguez, E., Martinerie, J., Varela, F.** (1999). *Measuring phase synchrony in brain signals.* Human Brain Mapping, 8(4), 194–208.
- **Vasylenko, Y.** (2026). *Phase Synchronization as Execution Gate in Human-Market Systems.* [in preparation]

---

<div align="center">

<br>

### Engineering summary

```
102 source modules · 1354 tests · 26 invariant contracts
4 hard invariants + B₁ · 11-axis doctor certification
bit-deterministic replay · Phipson–Smyth p-value estimator
```

**neuron7xLab** — Poltava, Ukraine — 2026.
Released under the MIT License.

</div>
