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
[![invariants](https://img.shields.io/badge/invariants-3_hard-critical?style=for-the-badge)](#three-invariants)
[![falsifiable](https://img.shields.io/badge/falsifiable-PLV_%3E_0-gold?style=for-the-badge)](#the-falsifiable-predicate)
[![tests](https://img.shields.io/badge/tests-246-brightgreen?style=for-the-badge)](tests/)
[![mypy](https://img.shields.io/badge/mypy-strict-1F5082?style=for-the-badge)](pyproject.toml)
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

The brain (EEG α/β, HRV, pupil) generates predictions; the market generates reality. `R(t)` physically measures **accumulated prediction error** as phase desynchronization (Friston/Clark, 2026; Fioriti & Chinnici, 2012).

When `R(t) < threshold`, the gate **blocks execution** — preserving the trader's executive function instead of enabling **cognitive surrender** (Ming/Wharton, 2026).

A neuro-symbolic trading agent grounded in **predictive-processing brain theory**. Falsifiable: `PLV(EEG_β, market_phase) > 0`.

```
              1   N
R(t)·e^{iΨ} = ─  Σ  e^{iθ_k(t)}
              N  k=1
```

</td>
</tr>
</table>

### Citations

| Claim | Source |
|-------|--------|
| `R(t)` forecasts market critical points | Fioriti & Chinnici (2012) |
| Brain as predictive engine | Friston / Clark (2026) |
| Cognitive surrender under AI | Ming / Wharton (2026) |
| EEG Kuramoto synchronization | Nguyen et al. (2020) |
| Capital-weighted Kuramoto | Capital-Weighted Kuramoto WG (2026) |

Full reference list and evidence chain: [`docs/theory/scientific_basis.md`](docs/theory/scientific_basis.md).

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

## Three Invariants

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
<td>PLV computed on <b>held-out</b> data only. No in-sample claims.</td>
<td>Separate train / test split; <code>plv_significance</code> operates only on the test window.</td>
</tr>
<tr>
<td align="center"><code>I₃</code></td>
<td>If bio-sensor unavailable ⇒ state <code>SENSOR_ABSENT</code>.</td>
<td>No synthetic fallback. No "graceful degradation to random." Silence is the only honest default.</td>
</tr>
</table>

---

## Architecture

```
                          ┌─────────────────────────────────────────┐
                          │             N E U R O P H A S E         │
                          │    physics-first execution infrastructure │
                          └────────────┬────────────────────────────┘
                                       │
  ┌──────────────┬─────────────┬───────┴──────┬───────────────┬─────────────┐
  │              │             │              │               │             │
┌─▼──────────┐ ┌─▼────────┐ ┌──▼──────────┐ ┌─▼───────────┐ ┌─▼─────────┐ ┌─▼───────┐
│ OSCILLATORS│ │   CORE   │ │   METRICS   │ │  INDICATORS │ │    GATE   │ │  RISK   │
│            │ │          │ │             │ │             │ │           │ │         │
│ market     │ │ phase    │ │ plv         │ │ qilm        │ │ execution │ │ evt     │
│ neural     │ │ kuramoto │ │ entropy     │ │ fmn         │ │  emergent │ │ mfdfa   │
│  _protocol │ │ order_pa │ │ ricci       │ │             │ │  direction│ │ sizer   │
└────────────┘ └──────────┘ │ hurst       │ └─────────────┘ └───────────┘ └─────────┘
                            │ ism         │
                            │ asymmetry   │
                            └─────────────┘
                                       │
                        ┌──────────────┴─────────────┐
                        │                            │
                  ┌─────▼──────┐               ┌─────▼──────┐
                  │   AGENTS   │               │    INTEL   │
                  │            │               │            │
                  │  π-agent   │               │ btc_field  │
                  │  memory    │               │  _order    │
                  └────────────┘               └────────────┘
```

<br>

<table>
<tr>
<td align="center" width="14%"><b>Package</b></td>
<td align="center" width="28%"><b>Module</b></td>
<td align="center" width="8%"><b>State</b></td>
<td align="center" width="50%"><b>Purpose</b></td>
</tr>
<tr><td rowspan="3"><code>core</code></td><td><code>phase.py</code></td><td>🟢</td><td>Hilbert + Daubechies D4 wavelet denoising + adaptive R_thr</td></tr>
<tr><td><code>kuramoto.py</code></td><td>🟢</td><td>RK4 integrator · delays τ<sub>ij</sub> · noise ξ<sub>i</sub> · liquidity K(t)</td></tr>
<tr><td><code>order_parameter.py</code></td><td>🟢</td><td>R(t)·e<sup>iΨ</sup> = (1/N)·Σ e<sup>iθ<sub>k</sub></sup></td></tr>
<tr><td rowspan="7"><code>metrics</code></td><td><code>plv.py</code></td><td>🟢</td><td>Phase Locking Value + surrogate significance test (I2)</td></tr>
<tr><td><code>entropy.py</code></td><td>🟢</td><td>Shannon · Tsallis · Rényi + Freedman–Diaconis + ΔH</td></tr>
<tr><td><code>ricci.py</code></td><td>🟢</td><td>Ollivier (Wasserstein-1) + Forman + weighted mean κ̄</td></tr>
<tr><td><code>hurst.py</code></td><td>🟢</td><td>R/S + DFA with Huber regression</td></tr>
<tr><td><code>ism.py</code></td><td>🟢</td><td>ISM = η·H'(t) / ⟨κ̄²⟩<sub>T</sub></td></tr>
<tr><td><code>asymmetry.py</code></td><td>🟢</td><td>Skewness · kurtosis · topological Δ<sub>curv</sub></td></tr>
<tr><td colspan="3" align="center"><i>(6 modules)</i></td></tr>
<tr><td rowspan="2"><code>indicators</code></td><td><code>qilm.py</code></td><td>🟢</td><td>Quantum Integrated Liquidity Metric (Neuron7X)</td></tr>
<tr><td><code>fmn.py</code></td><td>🟢</td><td>Flow Momentum Network — tanh(w₁·OB + w₂·CVD/N)</td></tr>
<tr><td rowspan="3"><code>gate</code></td><td><code>execution_gate.py</code></td><td>🟢</td><td>Hard <code>R(t) &lt; θ</code> block — invariant <b>I1</b></td></tr>
<tr><td><code>emergent_phase.py</code></td><td>🟢</td><td>4-condition criterion: R ∧ ΔH ∧ κ̄ ∧ ISM</td></tr>
<tr><td><code>direction_index.py</code></td><td>🟢</td><td>DI = w<sub>s</sub>·Skew + w<sub>c</sub>·Δ<sub>curv</sub> + w<sub>b</sub>·Bias</td></tr>
<tr><td rowspan="3"><code>risk</code></td><td><code>evt.py</code></td><td>🟢</td><td>POT/GPD fit + closed-form VaR / CVaR</td></tr>
<tr><td><code>mfdfa.py</code></td><td>🟢</td><td>Multifractal DFA + instability index</td></tr>
<tr><td><code>sizer.py</code></td><td>🟢</td><td>Composite position sizer — CVaR cap · scale<sub>R</sub> · scale<sub>m</sub></td></tr>
<tr><td><code>agents</code></td><td><code>pi_agent.py</code></td><td>🟢</td><td>π-calculus: mutation / repair / clone / learn + semantic memory</td></tr>
<tr><td><code>intel</code></td><td><code>btc_field_order.py</code></td><td>🟢</td><td>BTC Field Order v3.2 structured LLM payload (no network)</td></tr>
<tr><td rowspan="2"><code>oscillators</code></td><td><code>market.py</code></td><td>🟢</td><td>Price · log-volume · realized volatility → φ</td></tr>
<tr><td><code>neural_protocol.py</code></td><td>🟢 contract</td><td>Tobii / OpenBCI / Polar bridge Protocol — invariant <b>I3</b></td></tr>
</table>

---

## Execution Gate — State Machine

```
              sensor_present = False
           ┌───────────────────────────────┐
           │                               │
           │                               ▼
           │                     ┌───────────────────┐
           │                     │   SENSOR_ABSENT   │
           │                     │   allowed = F     │
           │                     └───────────────────┘
           │
    ┌──────┴──────┐      R ∈ [0, θ)       ┌───────────────────┐
    │             │ ────────────────────▶│      BLOCKED      │
    │  evaluate() │                      │   allowed = F     │
    │             │ ◀────────────────────└───────────────────┘
    └──────┬──────┘      R ∈ [θ, 1]
           │
           │ R ∈ [θ, 1]        ┌───────────────────┐
           └──────────────────▶│       READY       │
                               │   allowed = T     │
                               └───────────────────┘

       R is NaN / out-of-range
           ┌─────────────────────────────┐
           ▼                             │
   ┌───────────────────┐                 │
   │     DEGRADED      │ ◀───────────────┘
   │   allowed = F     │
   └───────────────────┘
```

<br>

<div align="center">

| state            | `execution_allowed` | meaning                                                   |
| :--------------- | :-----------------: | :-------------------------------------------------------- |
| `READY`          |        `True`       | `R(t) ≥ θ`, bio-sensor live, trader locked to market flow |
| `BLOCKED`        |       `False`       | `R(t) < θ`, system is desynchronized — no trades          |
| `SENSOR_ABSENT`  |       `False`       | bio-sensor unavailable — silent by default                |
| `DEGRADED`       |       `False`       | `R(t)` is NaN or out of range — honest failure            |

</div>

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

ruff check neurophase tests
mypy neurophase        # --strict, 30 source files, 0 errors
pytest -q              # 176 passed
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

## Status

<div align="center">

| Layer                     | Module                           | State                        |
| :------------------------ | :------------------------------- | :--------------------------- |
| Kuramoto integrator       | `core.kuramoto`                  | `🟢 implemented`              |
| Phase extraction          | `core.phase`                     | `🟢 implemented`              |
| PLV + surrogate test      | `metrics.plv`                    | `🟢 implemented`              |
| Entropy (S/T/R) + Δ       | `metrics.entropy`                | `🟢 implemented`              |
| Ricci curvature           | `metrics.ricci`                  | `🟢 implemented`              |
| Hurst R/S + DFA           | `metrics.hurst`                  | `🟢 implemented`              |
| ISM                       | `metrics.ism`                    | `🟢 implemented`              |
| Asymmetry (skew/kurt/Δκ)  | `metrics.asymmetry`              | `🟢 implemented`              |
| Execution gate (I1)       | `gate.execution_gate`            | `🟢 implemented`              |
| Emergent phase detector   | `gate.emergent_phase`            | `🟢 implemented`              |
| Direction index           | `gate.direction_index`           | `🟢 implemented`              |
| QILM / FMN indicators     | `indicators.*`                   | `🟢 implemented`              |
| EVT POT/GPD + VaR/CVaR    | `risk.evt`                       | `🟢 implemented`              |
| Multifractal DFA          | `risk.mfdfa`                     | `🟢 implemented`              |
| Composite position sizer  | `risk.sizer`                     | `🟢 implemented`              |
| π-calculus agents         | `agents.pi_agent`                | `🟢 implemented`              |
| BTC Field Order v3.2      | `intel.btc_field_order`          | `🟢 implemented`              |
| Market oscillators        | `oscillators.market`             | `🟢 implemented`              |
| Neural-bridge protocol    | `oscillators.neural_protocol`    | `🟢 contract + NullExtractor` |
| Tobii / OpenBCI adapter   | downstream                       | `⚪ requires hardware`        |

</div>

**Missing piece:** concrete bio-sensor adapters implementing `NeuralPhaseExtractor` (Tobii eye-tracker → pupil phase, OpenBCI → EEG phase, Polar → HRV). Scientific backing and bridge contracts are documented in [`docs/theory/sensory_basis.md`](docs/theory/sensory_basis.md).

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
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Physics-first.  Falsifiable.  Silent by default.          │
│                                                             │
│   If the signal is not there, the system says nothing.      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

<br>

**`neuron7xLab`** · Poltava, Ukraine 🇺🇦 · `2026`

<sub>MIT licensed · built in full autonomy · no synthetic edges</sub>

</div>
