# kuramoto-trader

> *A market and a mind are both oscillating systems. When they fall out of phase, trading is not a decision — it is noise.*

---

## The Hypothesis

Market oscillators (price, volume, realized volatility) and the trader's nervous system (EEG α/β bands, HRV, pupil dilation) can be modelled as a single Kuramoto network.

The **order parameter** R(t) ∈ [0, 1] measures their mutual phase synchronization in real time:

```
R(t)·e^{iΨ(t)} = (1/N) Σ e^{iθ_k(t)}
```

**Claim:** when R(t) < θ_critical, the trader is desynchronized from the market. Execution in this state is statistically lossy. The system blocks it — not by rule, but by physics.

**Falsifiable predicate:** PLV(EEG, price_phase) > 0 on intraday horizons.  
If PLV ≈ 0 → hypothesis dies honestly.  
If PLV > 0 → publishable in *Physical Review E* and a structural trading edge.

---

## Architecture

```
kuramoto_trader/
├── oscillators/
│   ├── market.py       # Price · volume · vol → instantaneous phase
│   └── neural.py       # EEG · HRV · pupil → instantaneous phase
├── sync/
│   ├── kuramoto.py     # Kuramoto ODE solver (N oscillators, coupling K)
│   ├── order_param.py  # R(t) and Ψ(t) from phase vector
│   └── plv.py          # Phase Locking Value — the falsification metric
├── gate/
│   └── execution_gate.py  # Hard block when R(t) < threshold
└── analysis/
    └── falsification.py   # Full PLV significance pipeline
```

Three invariants that cannot be overridden:

| # | Invariant |
|---|-----------|
| 1 | `R(t) < θ` → `execution_allowed = False`. Always. |
| 2 | PLV computed on **held-out** data only. No in-sample claims. |
| 3 | If bio-sensor unavailable → system enters `SENSOR_ABSENT`. No synthetic fallback. |

---

## Status

| Component | State |
|-----------|-------|
| Market oscillator | ✅ implemented |
| Kuramoto solver | ✅ implemented |
| R(t) / PLV | ✅ implemented |
| Execution gate | ✅ implemented |
| Neural oscillator | 🔲 bio-sensor bridge (Tobii / OpenBCI) |
| Live experiment | 🔲 requires hardware |

**Missing piece:** a thin sensor bridge (Tobii eye-tracker → pupil phase, OpenBCI → EEG phase integrator). Estimated: two weeks of focused engineering once hardware is available.

---

## Synthetic Validation

Until hardware arrives, the falsification pipeline runs on synthetic data where ground-truth PLV is known:

```bash
python experiments/synthetic_plv_demo.py
```

Expected output: PLV recovery within 5% of ground truth across coupling strengths K ∈ [0.5, 4.0].

---

## Install

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Citation

Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence.*  
Gidea & Katz (2018). *Topological Data Analysis of Financial Time Series.*  
Vasylenko, Y. (2026). *Phase Synchronization as Execution Gate in Human-Market Systems.* [in preparation]

---

*neuron7xLab · Poltava, Ukraine · 2026*
