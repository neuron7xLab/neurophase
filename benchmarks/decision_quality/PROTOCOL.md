# Decision-quality benchmark — protocol

## Intent

Measure whether the physio gate produces a **measurable operational
effect** on a real decision loop, not just whether it emits a signal.
This is the user-facing utility question: does gating on
`EXECUTE_ALLOWED` change interruption rate, premature switching,
overload onset, or recovery latency — compared to the same work
without the gate?

**This protocol is operator-run.** The pipeline has no built-in way
to measure "interruption rate on a coding sprint"; the operator
measures that externally (timer, self-rating, log) and the gate's
state stream is the covariate.

## Why a protocol and not a metric

Physio-utility benchmarking against a free-form human task has the
same circularity traps as the ds003458 ΔQ_gate benchmark:

- if you change the gate thresholds after seeing the effect, you
  fit the metric to the noise;
- if the comparison is single-arm (no "gate off" condition), you
  cannot attribute any effect to the gate;
- if the metric is redefined after the result, the result is
  rhetoric, not evidence.

This file nails the protocol **before** any data exists so the above
failure modes are pre-empted.

## The four candidate decision loops

Pick **one** per measurement block. Do not pool them.

1. **Coding sprint** (45–90 min of focused implementation).
   Metric: **self-reported interruption count** — how many times the
   operator switched away from the task. A switch is a context change
   > 30 s.
2. **Research session** (60 min).
   Metric: **premature-switching count** — how many times the
   operator abandoned a reading thread before finishing the current
   unit of content.
3. **Manual trading observation** (60 min, observation-only; no
   actual trade placement unless the operator has a separate proven
   risk layer).
   Metric: **impulsive-trigger count** — how many times the operator
   felt a pull to act impulsively (regardless of whether they did).
4. **Deep-focus block** (90 min of single-task work).
   Metric: **overload-onset latency** — wall-clock minutes until the
   operator first self-reports "tired / overloaded".

## Conditions

For each loop, run **two matched sessions** in counterbalanced order:

- **`gate_off`** — no physio feedback. The operator works normally.
- **`gate_on`** — the operator has a live physio HUD (e.g. a tmux
  pane tailing `neurophase.physio.live` events) and treats
  `ABSTAIN` / `EXECUTE_REDUCED` as an instruction to slow down or
  pause.

Both sessions must:
- use the same calibrated `profile`;
- be the same duration;
- start on the same day;
- alternate order across the benchmark (Day 1: `off, on`; Day 2:
  `on, off`; …) to cancel time-of-day confounds.

## Output

Under `benchmarks/decision_quality/<YYYY-MM-DD>/`:

```
benchmarks/decision_quality/2026-04-14/
  coding-sprint/
    gate_off/
      metrics.json         # {"interruption_count": N, "duration_min": T, ...}
      notes.md
    gate_on/
      metrics.json
      notes.md
      ledger.jsonl         # neurophase.physio.live --ledger-out
  RESULTS.md
```

`metrics.json` is small and self-describing. The operator owns the
count — the pipeline has no magic here.

`RESULTS.md` tabulates `gate_off` vs `gate_on` for each loop.

## Definition of Done

- [ ] At least **one** loop measured for **≥ 5 matched pairs** (10
  sessions total).
- [ ] Metric defined in `metrics.json` matches the metric in this
  protocol (no silent redefinitions).
- [ ] The `gate_on` ledgers replay green
  (`session_replay --strict` exit 0).
- [ ] `RESULTS.md` reports the raw per-pair numbers **and** the
  conclusion, including null / negative outcomes verbatim. No
  rescue stories.

## Honest limits

- Sample size will be small (single-subject, ~5-20 pairs). Stats
  will be descriptive, not inferential. This is a within-subject
  pilot, not a trial.
- Operator self-report metrics are noisy. That is explicitly
  accepted here — the bar is "does the gate produce a *measurable*
  effect in this specific operator's loop", not "does it generalise".
- A null result is a real result. The benchmark folder must contain
  it unchanged.
