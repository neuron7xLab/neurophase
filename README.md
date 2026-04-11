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
[![tests](https://img.shields.io/badge/core_tests-16-brightgreen?style=for-the-badge)](test_core.py)
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

Market oscillators (price, volume, realized volatility) and the trader's nervous system (EEG α/β bands, HRV, pupil dilation) can be modelled as **a single Kuramoto network**.

The **order parameter** `R(t) ∈ [0, 1]` measures their mutual phase synchronization in real time:

```
              1   N
R(t)·e^{iΨ} = ─  Σ  e^{iθ_k(t)}
              N  k=1
```

**Claim:** when `R(t) < θ_critical`, the trader is desynchronized from the market. Execution in this state is statistically lossy.

The system blocks it — not by rule, but by **physics**.

</td>
</tr>
</table>

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
                          │      phase synchronization as gate      │
                          └────────────┬────────────────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
  ┌───────▼────────┐          ┌────────▼────────┐          ┌────────▼────────┐
  │   OSCILLATORS  │          │      SYNC       │          │      GATE       │
  │                │          │                 │          │                 │
  │  market.py  ─┐ │          │  kuramoto.py    │          │ execution_gate  │
  │             ├─┼─────────▶│  order_param.py │─────────▶│   .py           │
  │  neural.py  ─┘ │          │  plv.py    ◀────┼──┐       │                 │
  └────────────────┘          └─────────────────┘  │       │  READY          │
                                                   │       │  BLOCKED        │
                                   ┌───────────────┘       │  SENSOR_ABSENT  │
                                   │                       │  DEGRADED       │
                          ┌────────▼────────┐              └────────┬────────┘
                          │    ANALYSIS     │                       │
                          │                 │                       ▼
                          │  falsification  │              execution_allowed
                          │      .py        │                   ∈ {T, F}
                          └─────────────────┘
```

<br>

<table>
<tr>
<td align="center" width="22%"><b>Module</b></td>
<td align="center" width="26%"><b>Path</b></td>
<td align="center" width="14%"><b>Status</b></td>
<td align="center" width="38%"><b>Purpose</b></td>
</tr>
<tr><td><code>PLV</code></td><td><code>plv.py</code></td><td>🟢 live</td><td>Phase Locking Value + surrogate-shift significance test</td></tr>
<tr><td><code>GATE</code></td><td><code>execution_gate.py</code></td><td>🟢 live</td><td>Hard <code>R(t) &lt; θ</code> gate with invariant enforcement</td></tr>
<tr><td><code>TESTS</code></td><td><code>test_core.py</code></td><td>🟢 16 cases</td><td>Order parameter · PLV · Kuramoto · Gate · Falsification</td></tr>
<tr><td><code>MARKET</code></td><td><code>oscillators/market.py</code></td><td>🟡 scaffold</td><td>Price · volume · σ → instantaneous phase</td></tr>
<tr><td><code>NEURAL</code></td><td><code>oscillators/neural.py</code></td><td>⚪ hardware gated</td><td>EEG · HRV · pupil → instantaneous phase (Tobii / OpenBCI bridge)</td></tr>
<tr><td><code>KURAMOTO</code></td><td><code>sync/kuramoto.py</code></td><td>🟡 scaffold</td><td>ODE solver, <i>N</i> coupled oscillators, coupling <i>K</i></td></tr>
<tr><td><code>ANALYSIS</code></td><td><code>analysis/falsification.py</code></td><td>🟡 scaffold</td><td>End-to-end PLV falsification pipeline + verdict</td></tr>
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

pytest test_core.py -v
```

<br>

```python
# minimal usage — the gate
from execution_gate import ExecutionGate

gate = ExecutionGate(threshold=0.65)

gate.evaluate(R=0.82)                          # READY          → allowed = True
gate.evaluate(R=0.41)                          # BLOCKED        → allowed = False
gate.evaluate(R=0.99, sensor_present=False)    # SENSOR_ABSENT  → allowed = False
gate.evaluate(R=float("nan"))                  # DEGRADED       → allowed = False
```

---

## Status

<div align="center">

| Component                | State                        |
| :----------------------- | :--------------------------- |
| PLV + surrogate test     | `🟢 implemented`              |
| Execution gate           | `🟢 implemented`              |
| Kuramoto ODE solver      | `🟡 reference scaffold`       |
| Market phase extractor   | `🟡 reference scaffold`       |
| Neural phase extractor   | `⚪ bio-sensor bridge`         |
| Live hardware experiment | `⚪ requires Tobii / OpenBCI`  |

</div>

**Missing piece:** a thin sensor bridge (Tobii eye-tracker → pupil phase, OpenBCI → EEG phase integrator). Two weeks of focused engineering once hardware is available.

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
