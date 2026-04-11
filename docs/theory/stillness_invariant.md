# The Fourth Invariant — `I₄: stillness ⇒ action_unnecessary`

*A formal derivation of the `StillnessDetector` criterion, the physical
meaning of the three clauses, and a sketch of the three counter-examples
that forced the window-wide formulation.*

---

## 1. Motivation

The first three gate invariants (`I₁ R < θ`, `I₂ sensor absent`,
`I₃ R invalid`) share a common shape: they identify **impossibility**
of correct action. A fourth, orthogonal regime exists — one where
correct action is possible but would add no information. A trader
staring at a perfectly phase-locked book **should not trade**, not
because trading is lossy but because the next trade is predictable
from the current state and therefore gives no edge. A brain whose
internal model is already fully aligned with the world has no
prediction error to reduce by further action — it is in the Friston
limit of free-energy stationarity.

The fourth invariant is:

```
I₄:   stillness  ⇒  execution_allowed = False
```

It is enforced by `StillnessDetector` in the `neurophase.gate`
subpackage and surfaced in `ExecutionGate` as the fifth gate state
`UNNECESSARY`.

Crucially, `I₄` does **not** mean "the system is done." It means "the
current action would be uninformative." The detector re-fires every
update cycle, so the gate reopens the moment the system leaves the
quiet region.

---

## 2. Free energy, honestly

Friston's variational free energy is

```
F[q] = E_q[ln q(s) − ln p(o, s)]
```

for a recognition density `q(s)` over latent states and a generative
model `p(o, s)` over observations and latents. Computing this functional
requires a generative model we do not have — `neurophase` deliberately
avoids claiming to implement it.

We instead define a **geometric surrogate** that vanishes exactly when
the circular-distance prediction error vanishes:

```
F_proxy(t) = ½ · δ(t)²          where δ(t) ∈ [0, π]
```

`F_proxy` is a second-order Taylor expansion of any smooth, strictly
convex, monotone function of `δ` with a minimum at `δ = 0`. It is **not**
the full variational functional. The naming `free_energy_proxy` (never
`free_energy`) is load-bearing: the library promises only that
`F_proxy` is proportional to the local curvature of `δ`, not that it
equals any specific probabilistic quantity.

Its time derivative follows from the chain rule:

```
dF_proxy/dt = δ(t) · dδ/dt
```

This is the quantity used by the detector's second clause.

---

## 3. The criterion (formal statement)

Let `R(t) ∈ [0, 1]` be the joint Kuramoto order parameter and
`δ(t) ∈ [0, π]` the circular distance between the brain and market
mean phases. Let `τ_s = window · dt` be the stillness-window length.
Define

```
S(t) := {
  (max_{τ ∈ [t−τ_s, t]}  |dR/dt(τ)|   < ε_R)
  ∧ (max_{τ ∈ [t−τ_s, t]}  |dF_proxy/dt(τ)| < ε_F)
  ∧ (max_{τ ∈ [t−τ_s, t]}  δ(τ)       < δ_min)
}
```

`S(t)` is **true** ⇒ the stillness detector emits `STILL`. Otherwise
the state is `ACTIVE` and the reason string identifies the dominant
failing clause.

### 3.1 Why three clauses

Each clause has an independent physical failure mode, and the
conjunction is needed to reject all three:

1. **`|dR/dt| < ε_R`** rejects states where the joint synchronization
   is moving — adding or losing oscillators. Even if `δ` is small at
   the current sample, a moving `R` means the population is
   restructuring and future samples will carry new information.
2. **`|dF_proxy/dt| < ε_F`** rejects states where `δ` is nonzero and
   moving. This is the genuine prediction-error signal: if `δ · dδ/dt`
   is large, the brain's model of the market is actively updating, even
   if `δ` itself is still small.
3. **`max δ < δ_min`** rejects states where `δ` is stationary but
   biased — e.g., two populations phase-locked at a persistent offset
   `≈ Δω / (2 K R)`. A biased lock is a **physical lag**, not true
   stillness: acting on the current phase relation is statistically
   different from acting on the zero-lag limit. See
   `tests/test_stillness_pipeline.py::test_pipeline_converges_to_unnecessary_at_high_coupling`
   for the numerical illustration at `K = 50`.

### 3.2 Why window-wide, not last-sample

A single-sample criterion reads `|dR/dt(t)| < ε_R`. This is **trivially
fooled**: any oscillatory `R(t)` has instantaneous `|dR/dt| = 0` at
every extremum. The detector would emit `STILL` for every zero crossing
of a sinusoid. `tests/test_stillness_detector.py::TestWindowWideVsLastSample::test_window_wide_beats_last_sample`
encodes the concrete counter-example: an `R` ramp that freezes in the
final step has `|dR/dt_last| = 0` but window-wide max `= 0.02/dt`.

An EMA-smoothed criterion is equally fragile because it **averages
out** transient excursions. The detector must commit to `STILL` only
when *no* sample in the window exceeded the threshold — the
window-wide max is the only operator with this property.

### 3.3 Why warmup is `ACTIVE`, not `SENSOR_ABSENT`

During the first `window` samples the rolling buffer is not full and
the criterion cannot be evaluated. The detector returns `ACTIVE` with
a `warmup` reason. It does **not** fall back to `SENSOR_ABSENT`
because `SENSOR_ABSENT` encodes a hardware fact (`I₃`); polluting it
with detector-internal warmup would confuse `ExecutionGate` consumers
and make `I₂` untestable.

Callers who need to distinguish warmup from steady-state `ACTIVE` can
read `decision.window_filled` or inspect `decision.reason[:7]`.

---

## 4. The five gate states (post-`I₄`)

| State | Execution allowed | Semantic meaning | Driven by |
|---|---|---|---|
| `SENSOR_ABSENT` | ❌ | Hardware unavailable | `I₂` |
| `DEGRADED`      | ❌ | `R(t)` invalid       | `I₃` |
| `BLOCKED`       | ❌ | Desynchronized       | `I₁` |
| `UNNECESSARY`   | ❌ | Still — no new info  | `I₄` |
| `READY`         | ✅ | Synchronized + active | — |

The `GateDecision.__post_init__` invariant enforces that
`execution_allowed=True` is only constructible when `state=READY`.
Every other state, **including `UNNECESSARY`**, raises `ValueError` if
the caller tries to mark it permissive. This keeps `I₄` on the same
enforcement tier as the other three.

### 4.1 Why `UNNECESSARY` is distinct from `BLOCKED`

Both states forbid execution, so an implementation could collapse them
into one. We keep them separate because:

- **Different root causes.** `BLOCKED` points to the market (the
  population is desynchronized, external world is the cause).
  `UNNECESSARY` points to the coupling itself (synchronized, nothing
  to add).
- **Different recovery dynamics.** A `BLOCKED` regime is left by
  waiting for `R` to cross the threshold — a slow, externally-driven
  transition. An `UNNECESSARY` regime is left by any dynamic
  perturbation; it is intrinsically fragile.
- **Different downstream policy.** Risk layers may want to log
  `UNNECESSARY` samples to the session archive as "correctly abstained"
  events (positive reinforcement for the trader), while `BLOCKED`
  samples are logged as "correctly avoided" events (different
  statistical population).

---

## 5. Hysteresis (optional)

A discrete `STILL`/`ACTIVE` classifier can chatter near the boundary
when any of the three clauses is right at its ε. The optional
`hold_steps` parameter introduces a minimum residence time: once a
state is entered, the classifier cannot leave for at least `hold_steps`
updates. This never relaxes the criterion — it only delays the
transition. Reason strings carry a `"held: ..."` prefix when
hysteresis vetoes a raw change, so the underlying criterion remains
fully observable.

Hysteresis is **off by default** (`hold_steps=0`) because the core
three-clause criterion already rejects the instantaneous-zero failure
mode that motivates most hysteresis implementations.

---

## 6. Worked counter-examples

### 6.1 Oscillatory `R` fools the last-sample criterion

```python
R_hist = [0.80, 0.82, 0.84, 0.86, 0.88, 0.88]
```

At the last sample, `|dR/dt| = 0` — a last-sample criterion would
emit `STILL`. The window-wide max is `0.02 / dt`, so the window-wide
criterion correctly rejects it. Test:
`TestWindowWideVsLastSample::test_window_wide_beats_last_sample`.

### 6.2 Biased phase lock at `K = 10`

Two Kuramoto populations with `ω_brain ≈ 2π · 1 Hz` and
`ω_market ≈ 2π · 0.5 Hz` lock with a steady-state offset
`δ_∞ ≈ Δω / (2 K R) ≈ π / (20) ≈ 0.157 rad`. At `delta_min = 0.10`
the detector correctly rejects this as `ACTIVE` (biased lock ≠
stillness). At `K = 50` the lag collapses to `≈ 0.031` and the
detector fires. Test:
`tests/test_stillness_pipeline.py::test_pipeline_converges_to_unnecessary_at_high_coupling`.

### 6.3 High-frequency micro-noise below `ε_R`

A 500-sample trajectory with `σ = 1e-4` noise on both `R` and `δ`
produces `STILL` on the vast majority of samples once the buffer is
full — provided the window is large enough to smooth over individual
noise samples. Test:
`TestStability::test_long_horizon_with_noise_respects_eps`.

---

## 7. Falsification hook

`I₄` is falsifiable at the population level: in a session archive,
the rate of `UNNECESSARY` firings must **positively** correlate with
quiet phases of the market (e.g., low realized volatility, high
spread-narrowing), and the false-positive rate against independently
labeled "regime-break" events must stay bounded. A monitor that
classifies a regime break as `UNNECESSARY` is wrong and must be
removed. This is a soft scientific contract — it becomes a binary
test once a labeled dataset exists.

Secondary falsification: `UNNECESSARY`-gated sessions should produce
strictly zero PnL variance in backtest compared to unconditional
trading — because no trades are executed by construction. This is a
sanity check on the gate plumbing, not on `I₄` itself.

---

## 8. Code entry points

| Concept | File | Symbol |
|---|---|---|
| Detector | `neurophase/gate/stillness_detector.py` | `StillnessDetector`, `StillnessDecision`, `StillnessState`, `free_energy_proxy` |
| Gate integration | `neurophase/gate/execution_gate.py` | `ExecutionGate.evaluate(..., delta=...)`, `GateState.UNNECESSARY` |
| Full pipeline test | `tests/test_stillness_pipeline.py` | — |
| Unit tests | `tests/test_stillness_detector.py` | 40+ cases |
| Extended gate tests | `tests/test_execution_gate.py` | `I₄` coverage |
| Demo | `examples/stillness_demo.py` | runnable script |

See also the broader scientific basis in
[`docs/theory/scientific_basis.md`](scientific_basis.md) §5, which
lists `I₄` alongside `I₁`–`I₃` as a load-bearing invariant.
