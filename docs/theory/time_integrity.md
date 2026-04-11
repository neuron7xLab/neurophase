# The B1 Temporal Integrity Gate

*Precondition for every phase-based claim in `neurophase`. Implemented
in `neurophase/data/temporal_validator.py`. Integrated into
`ExecutionGate` as a pre-check that settles in `DEGRADED` on any
non-`VALID` input.*

---

## 1. Motivation

Phase is a time-dependent quantity. The Kuramoto order parameter
`R(t)`, the circular distance `δ(t)`, the stillness criterion, and
every PLV claim depend **implicitly** on the assumption that the input
timestamps are:

1. Finite (no NaN, no ±∞).
2. Strictly non-decreasing (no reversals).
3. Not separated by gaps larger than some application-level bound.
4. Not lagging an external reference clock by more than a staleness
   bound.

If any of these four contracts is silently violated, the entire
downstream phase pipeline still produces numerically valid output —
it is just not a phase any more. It is numerical noise with
phase-shaped output. **No unit test of the Kuramoto, stillness, or
PLV layers can catch this** because those layers accept any real-
valued input and return real-valued output by construction.

B1 closes that silent-failure mode by enforcing the four contracts
**upstream** of all phase computation, and by routing every violation
through the existing `DEGRADED` state — so temporal corruption is
surfaced through the same invariant chain as NaN / OOR `R(t)`.

---

## 2. The four temporal contracts

Let `ts_n` be the `n`-th incoming timestamp in seconds.

### Contract 1 — finiteness

```
∀ n:  ts_n ∈ ℝ ∧ isfinite(ts_n)
```

Violation → `TimeQuality.INVALID`.

### Contract 2 — monotonicity

```
∀ n ≥ 1:  ts_n ≥ ts_{n−1}
```

Strict inequality violation → `TimeQuality.REVERSED`. Equality
(`ts_n == ts_{n−1}`) is treated separately as `TimeQuality.DUPLICATE`
because it is a recoverable fault (the caller can drop the sample),
while a reversal is a clock-domain fault and must be surfaced.

### Contract 3 — gap bound

```
∀ n ≥ 1:  ts_n − ts_{n−1}  ≤  max_gap_seconds
```

Violation → `TimeQuality.GAPPED`. A gapped sample is **still
committed** to the validator's history so the stream can recover on
the next observation. The gate sees `DEGRADED` on this tick but the
validator does not stall.

### Contract 4 — staleness bound

```
∀ n:  reference_now − ts_n  ≤  max_staleness_seconds
```

Violation → `TimeQuality.STALE`. The staleness check is **opt-in**:
when the caller does not supply `reference_now`, the check is skipped
and `STALE` is unreachable. This keeps offline / batch replay
workflows free of wall-clock coupling.

---

## 3. State machine

```
    ┌──────────┐   finite?       ┌──────────┐
    │  ingest  │ ───── no ────▶ │  INVALID │
    └────┬─────┘                 └──────────┘
         │ yes
         ▼
    ┌──────────┐   ts < last?    ┌──────────┐
    │  compare │ ─────── yes ───▶│ REVERSED │
    └────┬─────┘                 └──────────┘
         │ ts == last →          ┌──────────┐
         │       ─────── yes ───▶│ DUPLICATE│
         │                        └──────────┘
         │ ts > last
         ▼
    ┌──────────┐   gap > max?    ┌──────────┐
    │  gap     │ ─────── yes ───▶│ GAPPED   │ (commits)
    └────┬─────┘                 └──────────┘
         │ no
         ▼
    ┌──────────┐   stale > max?  ┌──────────┐
    │ staleness│ ─────── yes ───▶│  STALE   │ (commits)
    └────┬─────┘                 └──────────┘
         │ no
         ▼
    ┌──────────┐   n_seen < W?   ┌──────────┐
    │  warmup  │ ─────── yes ───▶│ WARMUP   │ (commits)
    └────┬─────┘                 └──────────┘
         │ no
         ▼
      ┌──────┐
      │ VALID│ (commits)
      └──────┘
```

`INVALID`, `REVERSED`, and `DUPLICATE` do **not** commit to history
— they are rejected faults, not state advances. All other paths
commit the sample so the validator's internal state reflects the
most recent legal observation.

---

## 4. Integration with the five-state gate

B1 does **not** add a fifth invariant. It is a **precondition** for
`I₁`–`I₄`:

> "no phase without valid time"

`ExecutionGate.evaluate` accepts an optional `time_quality` argument.
The evaluation order becomes:

```
0. time_quality != VALID        → DEGRADED  (temporal: ...)
1. sensor_present == False      → SENSOR_ABSENT           (I₂)
2. R invalid / None / OOR       → DEGRADED                (I₃)
3. R < threshold                → BLOCKED                 (I₁)
4. R ≥ threshold:
   4.1 no stillness detector    → READY
   4.2 δ missing / invalid      → READY (stillness skipped)
   4.3 stillness ACTIVE         → READY
       stillness STILL          → UNNECESSARY             (I₄)
```

Step 0 is strictly the highest priority: a non-`VALID` stream means
we cannot even trust that `sensor_present` is a meaningful
measurement at this tick. This is why the temporal check dominates
every downstream invariant, not just `I₃`.

### Reason-string convention

Every `GateDecision` produced by the temporal pre-check carries a
reason string starting with `temporal: `. Downstream log-processing
tools can partition decisions into:

* `valid:`, `gapped:`, `stale:`, `reversed:`, `duplicate:`,
  `warmup:`, `invalid:` — emitted by `TemporalValidator`.
* `temporal: …` — emitted by `ExecutionGate` when it inherits a
  non-VALID decision.
* `active: …`, `still: …`, `held: …` — from `StillnessDetector`.

The first token of every reason string is stable and parseable.

---

## 5. What B1 deliberately does NOT do

B1 is the **hygiene** layer. It does not and must not:

* **Resample** (B4) — interpolating to a regular grid is a separate
  decision about signal processing, not a validity check.
* **Interpolate** (B4) — filling gaps changes the physics of the
  downstream phase computation; B1 refuses to make that choice.
* **Estimate clock drift** (B5) — that requires cross-stream
  comparison, out of scope for a single-stream validator.
* **Mask segments** (B7) — segment-level validity is a higher-level
  concept built on top of B1.
* **Run the policy layer** — B1 is upstream of `ExecutionGate`, not
  downstream. Policy sees only the end decision.

Every one of those is a separate task in `docs/TASK_MAP.md`.

---

## 6. Failure modes B1 actively catches

| Failure | Pre-B1 behavior | Post-B1 behavior |
|---|---|---|
| NaN timestamp mid-stream | Pipeline keeps running on corrupted phase history | `INVALID` → `DEGRADED` |
| Backward timestamp | Phase wraps unpredictably, downstream sees non-causal signal | `REVERSED` → `DEGRADED` |
| Duplicate tick | Spurious zero `dt` in `StillnessDetector` → division-by-floor, noisy `|dR/dt|` | `DUPLICATE` → `DEGRADED` |
| 60 s gap in a 1 Hz stream | `StillnessDetector` silently interprets stale samples as stationary | `GAPPED` → `DEGRADED` |
| Wall-clock drift > 10 min | Live decisions made on hours-old data | `STALE` → `DEGRADED` |
| Empty buffer on cold start | Pipeline attempts phase computation with no history | `WARMUP` → `DEGRADED` (until `warmup_samples` accumulate) |

Every row corresponds to at least one passing test in
`tests/test_temporal_validator.py`.

---

## 7. What B1 enables

Once B1 is merged, the following tasks become *safe to start*:

* **B2** — Gap/duplicate/stale packet detector at the stream level
  (wraps `TemporalValidator` in a packetized framing).
* **B6** — Time-quality state machine (regime transitions between
  `VALID` and the fault regimes, suitable for UI surfacing).
* **B8** — "No phase without valid time" enforcement layer (mandatory
  attachment of `time_quality` to every `evaluate` call — this is the
  runtime-mode contract).
* **C1**–**C8** — Null-model confrontation and PLV significance.
  Every one of these requires temporal validity upstream; without B1
  they would produce statistically meaningless surrogates.
* **E1**, **E4** — Online streaming pipeline and runtime quality
  flags. Both require a validated time contract.
* **F1**, **F3** — Decision ledger and determinism tests. A ledger
  entry must not be emitted for a corrupted-time tick.

---

## 8. Honest naming

`TemporalValidator` is a **validator**, not a fixer. It never
modifies its inputs. It never interpolates. It never estimates. It
only reports. This is load-bearing: any future module that wants to
*repair* time must do so explicitly and must not pretend to be B1.

The `TimeQuality` enum is a **closed** set — adding a new member is a
breaking change to every consumer. New fault modes must extend the
enum consciously, not by accident.
