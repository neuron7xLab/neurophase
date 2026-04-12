# neurophase — Claude operating contract

## Identity

neurophase is a falsifiable physics-based execution gate treating brain and market
as coupled Kuramoto oscillators sharing order parameter R(t). The central research
question: is intelligence a property of regime (gamma ~ 1.0), not substrate?

Author: Yaroslav Vasylenko (solo, no institution, no grant).
This is both a research program and a production-grade runtime.

## Quality gates — every change MUST pass all before done

```
pytest -q                     # 1233+ tests, 0 failures
mypy --strict                 # zero errors, annotate from first line
ruff check && ruff format     # clean
python -m neurophase doctor   # 11/11 checks, exit 0
```

If doctor fails, the change is incomplete. No exceptions.

## Architecture in 30 seconds

- **5-state gate**: SENSOR_ABSENT | DEGRADED | BLOCKED | READY | UNNECESSARY
- **Only READY allows execution.** Enforced at `GateDecision.__post_init__`.
- **4 hard invariants** (I1-I3, B1) + 1 advisory (I4 stillness)
- **INVARIANTS.yaml**: 26 machine-readable contracts, each bound to >= 1 pytest
- **CLAIMS.yaml**: 5 scientific claims with DOI evidence and promotion rules
- **STATE_MACHINE.yaml**: 8 transitions, CI-verified exhaustive
- **Audit ledger**: append-only SHA256 chain, bit-deterministic replay

## Module map (what lives where)

| Path | Role |
|---|---|
| `core/` | Kuramoto primitives, R(t), phase extraction |
| `gate/` | ExecutionGate, StillnessDetector (I1-I4) |
| `sync/` | CoupledBrainMarketSystem |
| `data/` | TemporalValidator (B1), StreamDetector (B2/B6) |
| `metrics/` | PLV, IPLV, entropy, Hurst, SCP, delta xcorr |
| `validation/` | NullModelHarness, surrogates, Phipson-Smyth p-values |
| `benchmarks/` | Synthetic ground truth generators |
| `calibration/` | Youden-J threshold calibration with train/test split |
| `audit/` | DecisionTraceLedger, verify, replay |
| `runtime/` | StreamingPipeline, DecisionFrame |
| `governance/` | Invariant/claim/state-machine loaders, Doctor, monograph |
| `experiments/` | ds003458 analyses (PLV, delta, SCP, trial-LME) |
| `agents/` | pi-agent, semantic memory |
| `intel/` | BTC field order, QILM, FMN |

## Rules for code changes

1. **Never bypass invariants.** If I1-I4 block execution, that is correct behavior.
2. **New public module** -> register in its package `__init__.py` + `__all__`.
   Doctor check PUBLIC_MODULE_REACHABLE will catch orphans.
3. **New invariant** -> add to INVARIANTS.yaml with >= 1 test binding.
   CI meta-test will reject unbound invariants.
4. **New claim** -> add to CLAIMS.yaml with DOI evidence.
   Status must match evidence count (hypothesis=0-1, theory=1-2, fact=3+).
5. **Honest naming** -> never call F_proxy "free energy" (HN1).
   Never interpolate time (HN2). Never claim PLV on non-held-out data (HN8).
6. **Null results are committed.** 0/17 PLV, 0/23 SCP, 2/23 delta — all in results/.
   Never hide negative findings.

## Scientific status (as of 2026-04-12)

| Analysis | Result | Status |
|---|---|---|
| FM-theta PLV vs market phase | 0/17 significant | NULL — frequency mismatch |
| Delta (1-4 Hz) xcorr | 2/23 significant | NULL — no systematic effect |
| Trial-LME theta power | p=0.935 | NULL — deterministic rewards |
| SCP (0.01-0.1 Hz) xcorr | 0/23 significant | NULL — no signal |

ds003458 reward probabilities are deterministic random walks — no stochastic
signal for the brain to track in real time. Dataset limitation, not method failure.
Next: stochastic reward dataset (Torres or equivalent).

## What neurophase deliberately does NOT do

- Claim significance without null-model confrontation
- Train models over hand-picked thresholds (D1 is transparent grid search)
- Compute "free energy" (only F_proxy = 1/2 * delta^2, HN1)
- Repair or interpolate time (HN2)
- Ship a policy layer (position sizing lives above StreamingPipeline)

## Conventions

- Python 3.11+, strict typing everywhere
- Scientific unicode in comments ok (theta, psi, gamma) — ruff RUF002/003 ignored
- mixedCase for physics variables (R_min, dH_max) — ruff N802/806 ignored
- Line length 100
- Hypothesis property tests where applicable
- results/ contains timestamped JSON — never overwrite, only append new dates

## User preferences

- Language: Ukrainian
- Style: terse, no walls of text, max 3 bullets
- Autonomy: full — execute without asking, commit progress, report at end
- No Plan mode unless explicitly requested
- Zero tech debt: recursive improvement, all checks green, no shortcuts
- Always create PR, run CI, merge if 100% green
