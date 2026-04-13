# Operator-level metrics — baseline 2026-04-13

Three metrics that tell the repo whether the operator is drifting, not
whether the code is correct (tests and CI cover the latter):

| Metric | What it catches | Not what it catches |
|---|---|---|
| `recovery_latency` | how long the tree stays red before it goes green again | whether the red was legitimate |
| `first_push_green_rate` | whether local gate parity with CI is holding | whether the push was valuable |
| `time_to_admit` | how fast a failed assumption gets corrected instead of papered over | whether the correction was the right one |

These are **drift detectors**, not pagers. Review weekly, not
continuously. Re-measure with
`python scripts/ops_metrics.py --repo neuron7xLab/neurophase`.

## Baseline — 2026-04-13 UTC (08:21 → 16:29, 15 pushes)

```
first_push_green_rate: 60% (9/15)
recovery cycles: 3
  cycle 1: cb35a1ac → 3ae27f9d    diagnosis 118m15s, validation 6m34s, total 2h04m
  cycle 2: 3d65e817 → fd03d5b5    diagnosis  14m25s, validation 6m29s, total 20m54s
  cycle 3: f051ee47 → 590ec025    diagnosis   0m52s, validation 6m41s, total  7m33s
median diagnosis : 14m25s
median validation:  6m34s
```

### Interpretation (no spin)

* **Cycle 1** is a red flag: 4 consecutive failed CI pushes over ~2h.
  That pattern = local gate out of sync with CI (or no local gate was
  run). Root-cause was the session upstream of the hardening batch and
  is already closed — but the pattern is what the script exists to
  catch.
* **Cycle 2** (pip-audit / `neosynaptex` not on PyPI): the diagnosis
  time is honest — it took ~14 min to understand the root cause, then
  one clean push. No retry loop.
* **Cycle 3** (coverage gate `fail_under=78` vs CI-measured 77.98%):
  52 seconds diagnosis because the CI log spelled out the floor. One
  clean push. No retry loop.

### Drift thresholds (review if crossed next week)

* `first_push_green_rate < 50%` → local gate drifted or is being skipped
* `median diagnosis > 30 min` → either problems are getting harder or
  CI logs are too opaque; treat as a usability regression
* `any cycle > 3h` → treat as an incident postmortem, not a retry loop
* cycles where the same surface fails twice in a row → truth-discipline
  regression; the first correction was cosmetic, not root-cause

## Baseline — `neuron7xLab/neosynaptex`

```
window: 2026-04-07 → 2026-04-12 UTC, 5 pushes
first_push_green_rate: 100% (5/5)
recovery cycles: 0
```

No red→green cycles in the 5-day window. This is either genuinely
stable or low signal due to low push volume — re-measure after the
next substantial push batch before claiming stability.

## What this is **not**

* not `decision_quality` — no automated proxy for "did this commit
  advance the system" lives here. That still belongs in the promotion
  ladder (`CLAIMS.yaml`) and is human-judged.
* not a pager — nothing in this script pages anyone on red
* not a CI gate — this never blocks a merge; it only produces numbers
  for the operator's weekly review
