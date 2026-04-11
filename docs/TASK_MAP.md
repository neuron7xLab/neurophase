# neurophase — Ranked Evolution Task Map (v1.0)

*Companion to [`EVOLUTION_BOARD.md`](EVOLUTION_BOARD.md). 25 ranked
tasks, Tier 1–5. Every Tier 1 task carries a full 10-field spec;
Tiers 2–5 are summarized and expanded when promoted.*

---

## Executive summary

| | | |
|---|---|---|
| **Most dangerous gap if ignored** | **B1 — Temporal Integrity Gate** | Silent invalidity: phase computed from non-monotonic or gapped time series is not a phase, it is numeric noise with phase-shaped output. |
| **Highest-ROI next task** | **B1 — Temporal Integrity Gate** | Unlocks every downstream program that depends on temporal validity: B2, B3, B6, E1, E4, F1, F3, and *all* of Program C. |
| **Highest-leverage governance task** | **A1 — Invariant registry** | Makes `I₁`–`I₄` machine-readable and bound to test identifiers; prevents future invariant erosion. |

---

## Tier 1 — Must build now (prevents silent invalidity)

### T1.1 — B1 · Temporal Integrity Gate

**Program.** B — Temporal & Data Integrity.
**Type.** temporal + semantic.
**Weakness addressed.** No enforcement that inputs to the phase layer
are temporally valid. `ExecutiveMonitor` checks local monotonicity only;
`PredictionErrorMonitor.update` accepts any `t`; there is no streaming
contract at the repo level. Phase is a time-dependent quantity and a
phase built on unaligned or reversed timestamps is physically meaningless.
**Why load-bearing.** Every downstream claim in Program C (null models,
PLV significance, regime tests) is void unless the input stream has
passed a temporal-validity check. Without B1, a `READY` or
`UNNECESSARY` decision can silently stand on corrupted time.
**Failure if omitted.** False `READY` / false `UNNECESSARY` decisions
on stale or reversed packets. No test can catch this without temporal
validation upstream.
**Scope.** `neurophase/data/temporal_validator.py` [new],
`neurophase/data/__init__.py` [new]. Integration hook via
`ExecutionGate` optional `time_quality` parameter. Monotonicity, gap,
stale, reversed detection.
**Non-scope.** Resampling (B4), interpolation (B4), clock drift (B5),
segment-validity masks (B7) — separate tasks.
**Implementation direction.** `TimeQuality` enum
{`VALID`, `GAPPED`, `STALE`, `REVERSED`, `WARMUP`} + `TemporalValidator`
class with rolling timestamp buffer, `validate(ts: float) → TemporalQualityDecision`.
Configurable `max_gap_seconds`, `max_staleness_seconds`, `reference_clock`.
`ExecutionGate.evaluate(..., time_quality=...)` refuses to leave
`DEGRADED` on non-VALID quality.
**Validation protocol.** Parametrized unit tests for each failure mode;
integration test that `ExecutionGate` forces `DEGRADED` on non-VALID
quality; differential test that the pre-B1 gate is unchanged when
`time_quality=None`.
**Definition of done.**
- [ ] `TemporalValidator` module exists with `TimeQuality`, `TemporalQualityDecision`, `TemporalValidator`, `TemporalError`.
- [ ] Monotonicity enforced; backward timestamps → `REVERSED`.
- [ ] Gap detection via `max_gap_seconds`.
- [ ] Stale detection via `max_staleness_seconds` vs reference clock.
- [ ] `ExecutionGate` accepts optional `time_quality`; non-VALID → `DEGRADED` with reason tagged `temporal:…`.
- [ ] ≥ 25 tests across all failure modes + full pipeline integration test.
- [ ] `mypy --strict` clean, `ruff` clean, full suite green.
- [ ] `docs/theory/time_integrity.md` written.
- [ ] Existing tests unchanged in behavior (backwards-compatible default).
**Unlocks.** B2, B3, B6, B8, E1, E4, F1, F3, *and every Program C task*.
**Anti-patterns forbidden in the same task.** resampling, interpolation,
drift estimation, policy layers.

### T1.2 — A1 · Invariant registry (INVARIANTS.yaml)

**Program.** A — Semantic & Invariant Governance.
**Type.** governance + infrastructural.
**Weakness addressed.** `I₁`–`I₄` live only in docstrings and in
`GateDecision.__post_init__`. There is no machine-readable registry
binding each invariant to its enforcing test(s), which means a future
refactor can silently weaken an invariant without breaking CI.
**Why load-bearing.** Makes the invariant contract queryable by tools,
CI, and external reviewers. Prevents silent invariant erosion.
**Failure if omitted.** A future PR removes / weakens an invariant and
only the human reviewer can catch it. Eventually, one slips through.
**Scope.** `INVARIANTS.yaml` at repo root, `neurophase/governance/`
package with a loader and a pytest plugin that fails CI if any
registered invariant has zero bound tests.
**Non-scope.** Natural-language generation of docs from the registry;
runtime-level invariant checks (A3 / A4 are separate).
**Implementation direction.** YAML schema: `id`, `statement`,
`enforced_in` (file + symbol), `tests` (list of pytest node ids),
`docs` (path), `version`. Loader returns typed `Invariant` dataclasses.
Test binding verified by a meta-test that imports the registry and
checks each `tests` entry exists in the pytest collection.
**Validation protocol.** Meta-test: every `Invariant` must have
`len(tests) ≥ 1` and every referenced test node id must resolve.
**Definition of done.**
- [ ] `INVARIANTS.yaml` exists with `I₁`, `I₂`, `I₃`, `I₄` registered.
- [ ] `neurophase/governance/invariants.py` loader.
- [ ] `tests/test_invariants_registry.py` meta-test.
- [ ] Each invariant bound to ≥ 1 existing test.
- [ ] CI fails if binding is broken.
**Unlocks.** A2, A3, A4, A5, A6, A7, A8 — the entire Program A.

### T1.3 — C1 + C2 · Null-model harness + surrogate generators

**Program.** C — Scientific Validation & Falsification.
**Type.** statistical + scientific.
**Weakness addressed.** The README promises `PLV > 0` as a
falsifiable predicate but the surrogate test is referenced only in
existing `neurophase.metrics.plv`; there is no harness that applies
null-model confrontation uniformly across claims, and there is no
standard surrogate generator suite.
**Why load-bearing.** Without a null-model harness, *any* significance
claim the system makes is publication-grade only if the reviewer
happens to write the surrogate. A shared harness enforces C1–C8.
**Failure if omitted.** Accidental data-snooping: a metric reports a
"significant" effect that would collapse under a properly seeded
surrogate.
**Scope.** `neurophase/validation/null_model.py`
(NullModelHarness), `neurophase/validation/surrogates.py`
(phase-shuffle, cyclic-shift, block-bootstrap, IAAFT-light). Seeded
via `np.random.Generator`.
**Non-scope.** Regime-conditioned falsification (C5),
cross-session replication (C7), claim registry (C8).
**Definition of done.**
- [ ] `NullModelHarness.test(statistic, surrogate_fn, n=1000, seed=…) → NullModelResult`.
- [ ] Three surrogate generators + contract tests that each one preserves the intended null hypothesis property.
- [ ] `PLV > 0` predicate retrofit onto the new harness.
- [ ] ≥ 20 tests including adversarial seed sweeps.
**Unlocks.** C3, C4, C5, C6.

### T1.4 — F1 · Decision trace ledger

**Program.** F — Auditability, Replay, Reproducibility.
**Type.** infrastructural + runtime.
**Weakness addressed.** Gate decisions are emitted but not persisted
in an append-only, tamper-evident form. Replay and postmortem are
impossible without this.
**Why load-bearing.** Doctrine #5 — an unreplayable result is not yet
a system result.
**Failure if omitted.** A live decision cannot be reproduced for
postmortem analysis; calibration drift goes undetected.
**Scope.** `neurophase/audit/decision_ledger.py` +
`DecisionTraceRecord` dataclass + SHA256 hash chain + `append(...)`,
`verify(path)`.
**Non-scope.** Replay engine (F2), determinism tests (F3 gets shipped
with F1 to give the ledger behavioral teeth).
**Definition of done.**
- [ ] Append-only JSONL ledger with `prev_hash + sha256(current_record)`.
- [ ] `verify(path)` returns bool + first broken index.
- [ ] Parameter fingerprint included in each record.
- [ ] ≥ 15 tests including tamper detection.
**Unlocks.** F2, F3, F4, F5, F7, F8.

---

## Tier 2 — Next leverage layer (unlocks programs)

| ID | Program · Task | Depends on | Unlocks |
|---|---|---|---|
| T2.1 | **B2** · Gap/duplicate/stale packet detector | B1 | B3, B6 |
| T2.2 | **B6** · Time-quality state machine | B1, B2 | B8, E4 |
| T2.3 | **A2** · Formal state-machine specification | A1 | A3, A4 |
| T2.4 | **C3** · PLV significance layer (held-out only) | C1, C2 | C4, C5 |
| T2.5 | **F3** · Determinism tests (same-input → same-decision) | F1 | F4, F7 |
| T2.6 | **D1** · Gate threshold calibration (grid + OOS) | B1, C3 | D2, D5, D7 |

## Tier 3 — Enabling infrastructure (runtime readiness)

| ID | Program · Task | Purpose |
|---|---|---|
| T3.1 | **E2** · `DecisionFrame` contract | Typed runtime envelope for every decision. |
| T3.2 | **E1** · Stateful streaming pipeline | Connects `CoupledBrainMarket` → gate without manual loops. |
| T3.3 | **D2** · Stillness parameter calibration (ε_R, ε_F, δ_min, window) | OOS-validated defaults. |
| T3.4 | **H1** · Synthetic phase-coupling generator (known-PLV ground truth) | Backs all of Program C. |
| T3.5 | **A3** · Cross-module invariant test matrix | Closes the A2 → test-coverage loop. |
| T3.6 | **J1** · Diagnostic telemetry schema | Makes runtime state queryable without PII. |

## Tier 4 — Advanced policy and intelligence

| ID | Program · Task | Note |
|---|---|---|
| T4.1 | **G1** · Regime taxonomy (TRENDING / COMPRESSING / REVERTING / CHAOTIC) | Baseline for G2–G8. |
| T4.2 | **I1** · Action policy layer above the gate | Only after B1, F1 in place. |
| T4.3 | **E3** · Runtime orchestrator | Only after E1, E2, F1. |
| T4.4 | **K1** · Session manifest schema | Locks dataset provenance. |
| T4.5 | **L1** · Memory-bounded rolling computations audit | Hot-path hardening. |

## Tier 5 — Long-horizon research expansion

| ID | Program · Task | Note |
|---|---|---|
| T5.1 | **G5** · Regime transition model | Requires G1–G4 shipped. |
| T5.2 | **H7** · Parameter sweep simulation lab | Requires D1/D2 calibrated. |
| T5.3 | **C8** · Claim registry (hypothesis → evidence → status) | Requires all of Program C mature. |
| T5.4 | **M1** · System invariants monograph | Documentation of the mature system, written from implementation truth. |

---

## Recommended execution order

```
T1.1 (B1)  ──┐
             ├──▶ T2.1 (B2) ──▶ T2.2 (B6) ──▶ T3.2 (E1) ──▶ T3.1 (E2) ──▶ T4.3 (E3)
T1.2 (A1) ──▶ T2.3 (A2) ──▶ T3.5 (A3)
T1.3 (C1+C2) ──▶ T2.4 (C3) ──▶ T3.4 (H1) ──▶ T2.6 (D1) ──▶ T3.3 (D2)
T1.4 (F1) ──▶ T2.5 (F3) ──▶ T4.4 (K1)
                                                     ↓
                                                 T4.1 (G1) ──▶ T4.2 (I1)
```

Dependency gates are **hard**: no Tier 2 task starts until every
Tier 1 it depends on is merged and green.

---

## The single most dangerous gap if ignored

**Temporal validity.** The system computes phase synchronization but
does not enforce that input timestamps are monotonic, gap-free, and
aligned. A desynchronization signal built on misaligned streams is
physically meaningless — it fails silently with no error.

## The single highest-ROI next task

**B1 — Temporal Integrity Gate**, specified as T1.1 above.
