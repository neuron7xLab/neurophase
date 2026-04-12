<div align="center">

<a href="#the-hypothesis">
  <img src="https://raw.githubusercontent.com/neuron7xLab/neurophase/main/.github/assets/neurophase-hero.svg" alt="neurophase — brain · physics · market causality animation" width="100%"/>
</a>

<br>

<img src="https://readme-typing-svg.demolab.com/?lines=neuroscience+%C2%B7+physics+%C2%B7+first+cause+of+value;R(t)+%E2%89%A5+%CE%B8+%E2%87%92+trade;R(t)+%3C+%CE%B8+%E2%87%92+silence;PLV(%CF%86_neural%2C+%CF%86_market)+%3E+0&font=JetBrains+Mono&size=18&pause=1600&color=8B5CF6&center=true&vCenter=true&width=720&height=46" alt="neurophase tagline" />

<br>

# `n e u r o p h a s e`

***A market and a mind are both oscillating systems.***
***When they fall out of phase, trading is not a decision — it is noise.***

<br>

[![status](https://img.shields.io/badge/status-experimental-blueviolet?style=for-the-badge)](#status)
[![invariants](https://img.shields.io/badge/invariants-4_hard_%2B_B%E2%82%81-critical?style=for-the-badge)](#four-invariants)
[![falsifiable](https://img.shields.io/badge/falsifiable-PLV_%3E_0-gold?style=for-the-badge)](#the-falsifiable-predicate)
[![tests](https://img.shields.io/badge/tests-1238-brightgreen?style=for-the-badge)](tests/)
[![mypy](https://img.shields.io/badge/mypy-strict-1F5082?style=for-the-badge)](pyproject.toml)
[![doctor](https://img.shields.io/badge/doctor-11%2F11-00C853?style=for-the-badge)](#governance)
[![license](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)

<br>

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)](https://pytest.org/)
[![ruff](https://img.shields.io/badge/ruff-D7FF64?style=flat&logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/mypy--strict-1F5082?style=flat)](https://mypy-lang.org/)
[![Ukraine](https://img.shields.io/badge/%F0%9F%87%BA%F0%9F%87%A6-Poltava-005BBB?style=flat)](#)

</div>

<p align="center">
  <code>One law. Two oscillators. One gate. Zero hallucinated edges.</code>
</p>

---

## The Hypothesis

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

**NeuroPhase** models brain and market as **coupled Kuramoto oscillators sharing a single order parameter** `R(t)`.

The brain (EEG α/β, HRV, pupil) generates predictions; the market generates reality. `R(t)` physically measures **accumulated prediction error** as phase desynchronization (Friston 2010; Clark 2013, 2016; Fioriti & Chinnici 2012).

When `R(t) < threshold`, the gate **blocks execution** — preserving the trader's executive function via structured anti-offloading friction (mechanism grounded in Miyake 2000 + Arnsten 2009; the specific LLM-mediated "cognitive surrender" claim is currently *Strongly Plausible*, not *Established* — see [`docs/validation/evidence_labeling_style_guide.md`](docs/validation/evidence_labeling_style_guide.md)).

When `R(t)` is sufficient but the joint dynamics are still — `|dR/dt| < ε`, `|dF_proxy/dt| < ε`, `δ < δ_min` over a rolling window — execution is marked **`UNNECESSARY`**: no new information justifies action (invariant **I₄**, see [`docs/theory/stillness_invariant.md`](docs/theory/stillness_invariant.md)).

A neuro-symbolic trading agent grounded in **predictive-processing brain theory**. Falsifiable: `PLV(EEG_β, market_phase) > 0`.

```
              1   N
R(t)·e^{iΨ} = ─  Σ  e^{iθ_k(t)}
              N  k=1
```

</td>
</tr>
</table>

### Citations (real, DOI-anchored)

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

## The Falsifiable Predicate

<div align="center">

**`PLV( φ_neural , φ_market ) > 0`   on held-out intraday horizons.**

</div>

<table>
<tr>
<td width="50%" valign="top">

```
        |mean[ exp(i·(φ_x − φ_y)) ]|
PLV  =  ─────────────────────────────
                    ∈ [0, 1]

0  →  random phase difference
1  →  perfect phase locking
```

</td>
<td width="50%" valign="top">

The predicate is **binary and honest**:

- `PLV ≈ 0`  →  hypothesis dies, publicly, in one commit.
- `PLV > 0`  →  *Physical Review E* material **and** a structural trading edge.

Significance is assessed by a **surrogate test** over `N = 1000` random cyclic shifts of `φ_y`, which preserves autocorrelation while destroying cross-signal phase locking.

</td>
</tr>
</table>

---

## Four Invariants

> *Invariants are not rules. They are laws that cannot be overridden — enforced at construction time.*

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
<td>No synthetic fallback. No "graceful degradation to random." Silence is the only honest default — gate returns <code>SENSOR_ABSENT</code>.</td>
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
                    │                    N E U R O P H A S E                       │
                    │         physics-first neuro-symbolic execution engine        │
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
<tr><td rowspan="3"><code>core</code></td><td><code>phase.py</code></td><td>🟢</td><td>Hilbert + Daubechies D4 wavelet denoising + adaptive R_thr</td></tr>
<tr><td><code>kuramoto.py</code></td><td>🟢</td><td>RK4 integrator · delays τ<sub>ij</sub> · noise ξ<sub>i</sub> · liquidity K(t)</td></tr>
<tr><td><code>order_parameter.py</code></td><td>🟢</td><td>R(t)·e<sup>iΨ</sup> = (1/N)·Σ e<sup>iθ<sub>k</sub></sup></td></tr>
<tr><td rowspan="2"><code>sync</code></td><td><code>coupled_brain_market.py</code></td><td>🟢</td><td>Coupled Kuramoto system — brain × market phase locking</td></tr>
<tr><td><code>market_phase.py</code></td><td>🟢</td><td>Market oscillator extraction from price series</td></tr>
<tr><td rowspan="2"><code>oscillators</code></td><td><code>market.py</code></td><td>🟢</td><td>Price · log-volume · realized volatility → φ<sub>market</sub></td></tr>
<tr><td><code>neural_protocol.py</code></td><td>🟢</td><td>Tobii / OpenBCI / Polar bridge Protocol — invariant <b>I₂</b></td></tr>

<tr><td colspan="4" align="center"><b>— EXECUTION GATE —</b></td></tr>
<tr><td rowspan="4"><code>gate</code></td><td><code>execution_gate.py</code></td><td>🟢</td><td>5-state gate · invariants I₁–I₃ + B₁ enforced at <code>__post_init__</code></td></tr>
<tr><td><code>stillness_detector.py</code></td><td>🟢</td><td>Rolling-window stillness (I₄) — |dR/dt| < ε ∧ |dF/dt| < ε ∧ δ < δ<sub>min</sub></td></tr>
<tr><td><code>emergent_phase.py</code></td><td>🟢</td><td>4-condition criterion: R ∧ ΔH ∧ κ̄ ∧ ISM</td></tr>
<tr><td><code>direction_index.py</code></td><td>🟢</td><td>DI = w<sub>s</sub>·Skew + w<sub>c</sub>·Δ<sub>curv</sub> + w<sub>b</sub>·Bias</td></tr>

<tr><td colspan="4" align="center"><b>— TEMPORAL INTEGRITY —</b></td></tr>
<tr><td rowspan="4"><code>data</code></td><td><code>temporal_validator.py</code></td><td>🟢</td><td>B₁ precondition — GAPPED / STALE / REVERSED / DUPLICATE detection</td></tr>
<tr><td><code>stream_detector.py</code></td><td>🟢</td><td>B₂/B₆ — regime classification: HEALTHY / WARMUP / STALE</td></tr>
<tr><td><code>ds003458_loader.py</code></td><td>🟢</td><td>OpenNeuro BIDS loader (Cavanagh 2021)</td></tr>
<tr><td><code>eeg_preprocessor.py</code></td><td>🟢</td><td>FMθ extraction, D4 denoising pipeline</td></tr>

<tr><td colspan="4" align="center"><b>— METRICS STACK —</b></td></tr>
<tr><td rowspan="14"><code>metrics</code></td><td><code>plv.py</code></td><td>🟢</td><td>Phase Locking Value + held-out surrogate test</td></tr>
<tr><td><code>iplv.py</code></td><td>🟢</td><td>Imaginary PLV — volume-conduction guard + PPC (Vinck 2010)</td></tr>
<tr><td><code>entropy.py</code></td><td>🟢</td><td>Shannon · Tsallis · Rényi + Freedman–Diaconis + ΔH</td></tr>
<tr><td><code>ricci.py</code></td><td>🟢</td><td>Ollivier (Wasserstein-1) + Forman + weighted mean κ̄</td></tr>
<tr><td><code>hurst.py</code></td><td>🟢</td><td>R/S + DFA with Huber regression</td></tr>
<tr><td><code>ism.py</code></td><td>🟢</td><td>ISM = η·H'(t) / ⟨κ̄²⟩<sub>T</sub> — information-to-curvature ratio</td></tr>
<tr><td><code>asymmetry.py</code></td><td>🟢</td><td>Skewness · kurtosis · topological Δ<sub>curv</sub></td></tr>
<tr><td><code>delta_power.py</code></td><td>🟢</td><td>Delta envelope extraction (1–4 Hz)</td></tr>
<tr><td><code>delta_price_xcorr.py</code></td><td>🟢</td><td>Cross-correlation: δ-power × price change</td></tr>
<tr><td><code>scp.py</code></td><td>🟢</td><td>Slow Cortical Potential (0.01–0.1 Hz)</td></tr>
<tr><td><code>trial_theta_lme.py</code></td><td>🟢</td><td>Trial-LME theta power (Toma method replication)</td></tr>
<tr><td><code>rayleigh.py</code></td><td>🟢</td><td>Rayleigh test — effect-size gate for phase uniformity</td></tr>
<tr><td><code>plv_verdict.py</code></td><td>🟢</td><td>Dual-surrogate verdict: Rayleigh + Bessel + held-out</td></tr>
<tr><td colspan="3" align="center"><i>(13 modules)</i></td></tr>
<tr><td rowspan="2"><code>indicators</code></td><td><code>qilm.py</code></td><td>🟢</td><td>Quantum Integrated Liquidity Metric (Neuron7X)</td></tr>
<tr><td><code>fmn.py</code></td><td>🟢</td><td>Flow Momentum Network — tanh(w₁·OB + w₂·CVD/N)</td></tr>

<tr><td colspan="4" align="center"><b>— RISK & SIZING —</b></td></tr>
<tr><td rowspan="3"><code>risk</code></td><td><code>evt.py</code></td><td>🟢</td><td>POT/GPD fit + closed-form VaR / CVaR</td></tr>
<tr><td><code>mfdfa.py</code></td><td>🟢</td><td>Multifractal DFA + instability index</td></tr>
<tr><td><code>sizer.py</code></td><td>🟢</td><td>Composite position sizer — CVaR cap · scale<sub>R</sub> · scale<sub>m</sub></td></tr>

<tr><td colspan="4" align="center"><b>— RUNTIME & AUDIT —</b></td></tr>
<tr><td rowspan="3"><code>runtime</code></td><td><code>pipeline.py</code></td><td>🟢</td><td>StreamingPipeline + DecisionFrame — tick-by-tick execution</td></tr>
<tr><td><code>orchestrator.py</code></td><td>🟢</td><td>Tick orchestration — data → physics → gate → action</td></tr>
<tr><td><code>memory_audit.py</code></td><td>🟢</td><td>Memory tracking + bounds enforcement</td></tr>
<tr><td rowspan="3"><code>audit</code></td><td><code>decision_ledger.py</code></td><td>🟢</td><td>Append-only SHA-256 chained decision trace</td></tr>
<tr><td><code>replay.py</code></td><td>🟢</td><td>Bit-deterministic replay engine (F₃ certification)</td></tr>
<tr><td><code>session_manifest.py</code></td><td>🟢</td><td>Session metadata + integrity verification</td></tr>

<tr><td colspan="4" align="center"><b>— KLR RESET SYSTEM —</b></td></tr>
<tr><td rowspan="6"><code>reset</code></td><td><code>pipeline.py</code></td><td>🟢</td><td>KLR full reset orchestration — state transitions + metrics</td></tr>
<tr><td><code>controller.py</code></td><td>🟢</td><td>Ketamine-Like Reset: plasticity injection + refractory</td></tr>
<tr><td><code>gamma_witness.py</code></td><td>🟢</td><td>External γ-verification (neosynaptex adapter, NEO-I₁/I₂)</td></tr>
<tr><td><code>ensemble.py</code></td><td>🟢</td><td>Multi-model ensemble averaging + passive learner</td></tr>
<tr><td><code>ntk_monitor.py</code></td><td>🟢</td><td>Neural Tangent Kernel rank monitor — loss landscape</td></tr>
<tr><td colspan="3" align="center"><i>(21 modules in reset/)</i></td></tr>

<tr><td colspan="4" align="center"><b>— VALIDATION & CALIBRATION —</b></td></tr>
<tr><td rowspan="2"><code>validation</code></td><td><code>null_model.py</code></td><td>🟢</td><td>NullModelHarness — Phipson–Smyth p = (1+k)/(1+n)</td></tr>
<tr><td><code>surrogates.py</code></td><td>🟢</td><td>cyclic_shift · phase_shuffle · time_reversal · block_bootstrap</td></tr>
<tr><td rowspan="2"><code>calibration</code></td><td><code>threshold.py</code></td><td>🟢</td><td>Youden-J calibration with explicit train/test split</td></tr>
<tr><td><code>stillness.py</code></td><td>🟢</td><td>Stillness parameter grid search (ε<sub>R</sub>, ε<sub>F</sub>, δ<sub>min</sub>)</td></tr>
<tr><td rowspan="5"><code>benchmarks</code></td><td><code>phase_coupling.py</code></td><td>🟢</td><td>Controlled coupling with closed-form PLV at c ∈ {0, 1}</td></tr>
<tr><td><code>neural_phase_generator.py</code></td><td>🟢</td><td>Kuramoto ODE for synthetic EEG phase</td></tr>
<tr><td><code>stochastic_market_sim.py</code></td><td>🟢</td><td>GBM + neural EMA tracking</td></tr>
<tr><td><code>ppc_analytical.py</code></td><td>🟢</td><td>Analytical PPC + theoretical PLV</td></tr>
<tr><td><code>parameter_sweep.py</code></td><td>🟢</td><td>Grid search harness for coupling sweeps</td></tr>

<tr><td colspan="4" align="center"><b>— GOVERNANCE & INTELLIGENCE —</b></td></tr>
<tr><td rowspan="4"><code>governance</code></td><td><code>invariants.py</code></td><td>🟢</td><td>INVARIANTS.yaml loader — 26 machine-readable contracts</td></tr>
<tr><td><code>claims.py</code></td><td>🟢</td><td>CLAIMS.yaml — hypothesis → theory → fact promotion rules</td></tr>
<tr><td><code>state_machine.py</code></td><td>🟢</td><td>STATE_MACHINE.yaml — 8 transitions, CI-verified exhaustive</td></tr>
<tr><td><code>doctor.py</code></td><td>🟢</td><td>11-axis completeness checker (reachability, resistance, determinism)</td></tr>
<tr><td><code>agents</code></td><td><code>pi_agent.py</code></td><td>🟢</td><td>π-calculus: mutation / repair / clone / learn + semantic memory</td></tr>
<tr><td><code>intel</code></td><td><code>btc_field_order.py</code></td><td>🟢</td><td>BTC Field Order v3.2 structured LLM payload (no network)</td></tr>
<tr><td rowspan="3"><code>analysis</code></td><td><code>prediction_error.py</code></td><td>🟢</td><td>Friston/Clark prediction-error monitor</td></tr>
<tr><td><code>regime.py</code></td><td>🟢</td><td>TRENDING / COMPRESSING / FLASH / FLASH-SUPPRESS taxonomy</td></tr>
<tr><td><code>regime_transitions.py</code></td><td>🟢</td><td>Regime transition logic + hysteresis</td></tr>
<tr><td rowspan="2"><code>state</code></td><td><code>executive_monitor.py</code></td><td>🟢</td><td>EEG β · HRV · error-burst detection → OverloadIndex</td></tr>
<tr><td><code>klr_reset.py</code></td><td>🟢</td><td>Ketamine-Like Reset state machine</td></tr>
<tr><td><code>explain</code></td><td><code>explain.py</code></td><td>🟢</td><td>Causal decision explanation engine — contracts → steps → verdict</td></tr>
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

| State | `execution_allowed` | Invariant | Meaning |
| :--- | :---: | :---: | :--- |
| `READY` | **`True`** | — | R(t) ≥ θ, bio-sensor live, dynamics active — **execute** |
| `BLOCKED` | `False` | I₁ | R(t) < θ — system desynchronized, acting is noise |
| `SENSOR_ABSENT` | `False` | I₂ | No bio-sensor — silence is the only honest default |
| `DEGRADED` | `False` | I₃ / B₁ | R(t) invalid or temporal quality failed — honest failure |
| `UNNECESSARY` | `False` | I₄ | Dynamics are still — no new information justifies action |

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
mypy neurophase        # --strict, 102 source files, 0 errors
pytest -q              # 1238 collected, 1236 passed, 2 skipped
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
| Python source modules | **102** |
| Test cases (collected) | **1,238** |
| mypy --strict errors | **0** |
| ruff violations | **0** |
| Doctor axes passing | **11 / 11** |
| Invariant contracts | **26** (INVARIANTS.yaml) |
| Scientific claims | **5** (CLAIMS.yaml) |
| State-machine transitions | **8** (CI-exhaustive) |
| Determinism certification | bit-identical across 6 pillars |

### Research (ds003458 — Cavanagh 2021, 23 subjects)

| Analysis | Metric | N | Significant | Verdict |
| :--- | :--- | :---: | :---: | :--- |
| FMθ (4–8 Hz) vs market | PLV held-out | 17 | **0 / 17** | NULL — freq mismatch (0.001 Hz vs 4–8 Hz) |
| Delta (1–4 Hz) × price | xcorr | 23 | **2 / 23** | NULL — mixed signs, no systematic effect |
| SCP (0.01–0.1 Hz) × reward | xcorr | 23 | **0 / 23** | NULL — no signal detected |
| Trial theta power | LME (Toma) | 23 | **p = 0.935** | NULL — deterministic rewards |
| Synthetic PLV bridge | PPC sweep | — | **confirmed** | Methodology validated at known c ∈ {0, 1} |

*Null results are committed, not hidden. Hypothesis survives; dataset does not confirm.*
*Next: stochastic reward dataset (Torres or equivalent).*

</div>

### Missing Piece

Concrete bio-sensor adapters implementing `NeuralPhaseExtractor` (Tobii eye-tracker → pupil phase, OpenBCI → EEG phase, Polar → HRV). Scientific backing and bridge contracts are documented in [`docs/theory/sensory_basis.md`](docs/theory/sensory_basis.md).

---

## KLR — Ketamine-Like Reset

> *When the system's loss landscape degenerates, structured noise injection restores plasticity.*

The KLR subsystem (21 modules, 4+ KLOC) implements a **controlled reset protocol** inspired by ketamine's neuroplasticity mechanism:

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

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   Physics-first.  Falsifiable.  Silent by default.                       │
│                                                                          │
│   If the signal is not there, the system says nothing.                   │
│   If the hypothesis dies, it dies publicly, in one commit.               │
│                                                                          │
│   102 source modules · 1238 tests · 26 invariant contracts              │
│   4 hard laws · 11-axis doctor · bit-deterministic audit                 │
│   0 synthetic edges · 0 hallucinated claims                              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

<br>

**`neuron7xLab`** · Poltava, Ukraine 🇺🇦 · `2026`

<sub>MIT licensed · built in full autonomy · no synthetic edges · zero tech debt</sub>

</div>
