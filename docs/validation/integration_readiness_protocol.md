# neurophase — INTEGRATION READINESS & CALIBRATION PROTOCOL v2026.04.11

**Objective:** Transform research framework into production-grade zero-trust system.  
**Gate:** 100% readiness required before release.

---

## PHASE 1: GLOBAL VALIDATION BASELINE

### 1.1 Functional Integrity
```bash
pytest tests/ -q --tb=short
```
Gate: no failing tests.

### 1.2 Type Safety
```bash
mypy neurophase/ --strict --no-implicit-optional --disallow-untyped-defs
```
Gate: 0 mypy errors.

### 1.3 Code Quality
```bash
ruff check neurophase/ --select E,F,W,C901 --line-length=100
```
Gate: 0 violations.

### 1.4 Documentation Completeness
- Every public function has docstring + type hints.
- Theory claim has source + evidence label.
- Product-critical claims have falsification rule.

---

## PHASE 2: METRIC CALIBRATION LOOP

### 2.1 Threshold Tuning (Time-split, no leakage)
1. Use historical archive (target n≥100 sessions).
2. Split chronologically 70/30 (no shuffle).
3. Fit thresholds on train, evaluate on holdout.

| Metric | Component | Target |
|---|---|---|
| `R(t)` | `execution_gate` | FAR minimized with bounded overblocking |
| PLV min | `metrics/plv` | ρ(PLV, accuracy) ≥ 0.30 |
| HRV min | `risk/hrv_proxy` (proposed) | stress sensitivity >70% |
| Theta threshold | `risk/evt` | specificity >75% |

Output artifact: `neurophase/config/calibration_results_YYYYMMDD.yaml`.

### 2.2 Signal Fusion Calibration
Composite score:

```python
risk_score = w_plv * PLV_norm + w_hrv * HRV_norm + w_load * Load_norm
```

- Normalize by robust scaling (median/IQR).
- Optimize weights on train; validate on holdout.
- Gate target: AUC(error_burst) > 0.70.

### 2.3 Decision Policy Calibration
- `ALLOW`: risk < 0.33
- `SLOW_DOWN`: 0.33–0.67 (verification step)
- `HARD_BLOCK`: > 0.67 (mandatory re-evaluation)

Targets:
- FAR < 10%
- hard-block < 5%
- over-friction complaints < 3%

---

## PHASE 3: ADAPTATION LAYER

### 3.1 Individual Baselines
- Collect 5–10 sessions before full adaptive gating.
- Build personalized baseline for PLV, HRV, latency, accuracy.

### 3.2 Drift Control
- Exponential update with half-life ≈14 days.
- No baseline updates during anomalous sessions.

### 3.3 Neurobiological Constraints
- Track circadian context.
- Include sleep/recovery context where available.
- Increase verification when HRV volatility is elevated.

---

## PHASE 4: FAILURE MODES & SAFE FALLBACKS

1. **Overblocking:** reduce hard-block sensitivity; prefer `SLOW_DOWN`.
2. **Underblocking:** tighten thresholds; require stronger verification.
3. **Signal drift:** run recalibration window + quality checks.
4. **Signal loss:** degrade gracefully to remaining channels.
5. **Extreme regime:** enter `SAFE_MODE` with stricter gates.

---

## PHASE 5: 100% INTEGRATION READINESS CHECKLIST

- [ ] `pytest -q` passes.
- [ ] `mypy neurophase --strict` passes.
- [ ] `ruff check .` passes.
- [ ] Claims carry evidence labels.
- [ ] Claim → module → test traceability present.
- [ ] Calibration artifacts versioned with date.
- [ ] Failure-mode fallbacks documented.
- [ ] Release sign-off completed (Research + Engineering + Product + Ops).

**Release status:** BLOCKED until all checks are green.

---

## PHASE 6: MAINTENANCE

- Quarterly review cadence (Apr/Jul/Oct/Jan).
- Trigger incident protocol if FAR > 20% or hard-block > 15% or accuracy drops > 10%.
- Recalibrate thresholds on detected drift > 10%.



---

## PHASE 7: CI/CD & GOVERNANCE ENFORCEMENT

All machine-checkable items in this protocol are enforced via
existing CI machinery. No bespoke "governance kernel" script is
required.

- **Functional + typing + lint**: `pytest`, `ruff check`,
  `ruff format --check`, `mypy --strict` — run on every PR on
  Python 3.11 and 3.12 (see `.github/workflows/ci.yml`).
- **Invariant registry meta-test**: `tests/test_invariants_registry.py`
  loads `INVARIANTS.yaml`, resolves every bound test node id, and
  fails CI if any binding is broken.
- **State-machine spec meta-test**: `tests/test_state_machine_spec.py`
  loads `STATE_MACHINE.yaml` and verifies every `GateState` member
  + every transition is test-bound.
- **Determinism certification**: `tests/test_determinism_certification.py`
  proves the full pipeline replays bit-identically — the foundation
  of any post-hoc evidence claim.
- **Reference to this protocol**: any PR that changes a product
  threshold, a calibration parameter, or an evidence-labelled claim
  must cite the relevant phase above in its description.

A green CI run on `main` is the only sign-off that counts.
