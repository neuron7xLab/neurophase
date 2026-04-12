# The A3 cross-module invariant matrix

*The safety-proof complement to the isolated per-module tests.
Load-bearing. Enforced by
[`tests/test_invariant_matrix.py`](../../tests/test_invariant_matrix.py)
under honest-naming contract **HN16** in
[`INVARIANTS.yaml`](../../INVARIANTS.yaml).*

---

## 1. Why A3 exists

Every other test file in the suite checks **one axis at a time**:

* `tests/test_execution_gate.py` probes `I₁` in isolation.
* `tests/test_temporal_validator.py` probes `B₁` in isolation.
* `tests/test_stillness_detector.py` probes `I₄` in isolation.
* `tests/test_state_machine_spec.py` verifies the static structure
  of `STATE_MACHINE.yaml` but does not exercise it at runtime.

This is enough to catch **regressions that break a single axis**.
It is **not** enough to catch regressions that live only in the
**cross-product** of two or more axes. A refactor of
`ExecutionGate._classify_ready` that accidentally lets a single
`(time_quality, sensor_present, R, δ, stillness)` combination slip
from `UNNECESSARY` to `READY` would pass every isolated test —
because none of them tests that specific combination.

A3 is the test that closes that gap. It enumerates the full
Cartesian product of upstream inputs and compares the live
`ExecutionGate.evaluate` output against a **pure analytical
predictor** that encodes `STATE_MACHINE.yaml` verbatim. A single
disagreement between the two fails CI.

**Design principle.** The A3 matrix addresses a stronger claim than
per-axis unit tests: *not only does the system work on the inputs
we thought to check, but no reachable combination of inputs can
produce a state the specification forbids.* This is the standard
test-oracle form for a finite-state specification.

A3 is that proof for the gate-level semantic surface.

---

## 2. The analytical predictor

```python
def predict_gate_state(inputs: GateInput) -> GateState:
    # Step 0 — B₁ temporal precondition.
    if inputs.time_quality is not None and inputs.time_quality is not VALID:
        return DEGRADED

    # Step 1 — I₂ sensor presence.
    if not inputs.sensor_present:
        return SENSOR_ABSENT

    # Step 2 — I₃ R validity.
    if not r_is_valid(inputs.R):
        return DEGRADED

    # Step 3 — I₁ threshold.
    if inputs.R < inputs.threshold:
        return BLOCKED

    # Step 4 — R ≥ threshold; stillness split.
    if not inputs.has_stillness_detector:
        return READY
    if not delta_is_valid_for_stillness(inputs.delta):
        return READY
    if inputs.forced_stillness is ACTIVE:
        return READY
    if inputs.forced_stillness is STILL:
        return UNNECESSARY
    return READY
```

This is the **formal specification** of the gate. It has no
dependency on any production module and can be read as the
answer to "what exactly is the gate supposed to do?". Reviewers
can read it in two minutes and verify it matches the priority
order declared in
[`STATE_MACHINE.yaml`](../../STATE_MACHINE.yaml):
`T0_temporal_invalid < T1_sensor_absent < T2_R_invalid <
T3_R_below_threshold < T4..T7_ready_and_stillness`.

---

## 3. What the matrix covers

| Axis | Values swept |
|---|---|
| `time_quality` | `None`, `VALID`, `GAPPED`, `STALE`, `REVERSED`, `DUPLICATE`, `WARMUP`, `INVALID` (8) |
| `sensor_present` | `True`, `False` (2) |
| `R` | `None`, `NaN`, `+inf`, `-0.1`, `1.1`, `0.0`, `0.30`, `0.50`, `θ − 0.01`, `θ`, `θ + 0.01`, `0.90`, `1.0` (13) |
| `δ` | `None`, `NaN`, `+inf`, `-0.1`, `π + 0.5`, `0.0`, `0.01`, `0.50`, `π` (9) |
| `has_stillness_detector` | `True`, `False` (2) |
| `forced_stillness` | `None`, `STILL`, `ACTIVE` (3) |

Raw product: `8 · 2 · 13 · 9 · 2 · 3 = 11 232` cells. After
deduplication (stillness outcome is only meaningful when the
detector is attached), the enumerator yields **~3 744 distinct
cells**, each evaluated by both the live gate and the predictor.

## 4. The reachability complement

Cross-product safety (section 3) proves that **every input cell
maps to the right state**. The reachability suite proves the
dual: **every state is actually reached by at least one cell**,
and — separately — **every state is reachable through a natural
`StreamingPipeline` tick sequence** (not just a direct
`ExecutionGate.evaluate` call).

The two halves together are the full safety × liveness proof:

| | Safety | Liveness |
|---|---|---|
| **Point-wise** | `TestCrossProduct::test_every_cell_matches_predictor` | `TestReachability::test_each_state_has_at_least_one_cell` |
| **Pipeline-driven** | `TestPipelineReachability::test_*_reachable_via_*` | (same) |

---

## 5. Priority-ordering tests

The priority order `B₁ > I₂ > I₃ > I₁ > I₄` is tested by
**constructing inputs designed to satisfy two failing conditions
at once** and verifying the higher-priority one wins:

| Test | Both conditions | Expected winner |
|---|---|---|
| `test_temporal_dominates_sensor_absent` | `time_quality=GAPPED` + `sensor_present=False` | `DEGRADED` (B₁) |
| `test_temporal_dominates_r_invalid` | `time_quality=STALE` + `R=NaN` | `DEGRADED` (B₁, reason="temporal") |
| `test_temporal_dominates_r_below_threshold` | `time_quality=REVERSED` + `R=0.10` | `DEGRADED` (B₁, reason="temporal") |
| `test_sensor_dominates_r_invalid` | `sensor_present=False` + `R=NaN` | `SENSOR_ABSENT` (I₂) |
| `test_sensor_dominates_r_below_threshold` | `sensor_present=False` + `R=0.1` | `SENSOR_ABSENT` (I₂) |
| `test_r_invalid_dominates_r_below_threshold` | `R=None` (which is neither valid nor below-threshold) | `DEGRADED` (I₃) |
| `test_r_below_threshold_dominates_stillness` | `R=0.3` + forced `STILL` | `BLOCKED` (I₁, not UNNECESSARY) |
| `test_stillness_only_applies_above_threshold` | `R=0.99` + forced `STILL` | `UNNECESSARY` (I₄) |

Each one catches a **different** family of regression:

* If step 0 moves after step 1 in `evaluate`, `test_temporal_dominates_sensor_absent` fails.
* If step 2 forgets to check `None`, `test_r_invalid_dominates_r_below_threshold` fails.
* If step 4's stillness layer runs before step 3, `test_r_below_threshold_dominates_stillness` fails.

---

## 6. What the matrix does **not** cover

A3 is **not**:

* A test of the stillness detector's internal rolling-window math
  — that's `tests/test_stillness_detector.py` (55 tests).
* A test of the temporal validator's per-sample contracts — that's
  `tests/test_temporal_validator.py`.
* A test of the null-model harness — that's `tests/test_null_model_harness.py`.
* A physical / calibration claim. A3 is purely about the
  cross-product semantics of the gate state machine.

A3 assumes the **per-module** tests have already proven each
individual layer's correctness; it then proves the **composition**
behaves as the specification demands. If a per-module test is
weakened, A3 may still pass because it only forces each layer
through regimes, not through internal invariants.

---

## 7. How to extend

Adding a new gate state (e.g. a hypothetical `SATURATED`) requires
updating **three places simultaneously**:

1. `GateState` enum in `neurophase/gate/execution_gate.py`.
2. `STATE_MACHINE.yaml` with the new state + the new transition.
3. The analytical predictor in `tests/test_invariant_matrix.py`
   + a reachability test for the new state.

The A3 meta-binding in `INVARIANTS.yaml::HN16` guarantees that
**step 3 cannot be skipped** — the A1 CI meta-test will break if
the matrix file is modified without re-binding.

This is the whole point of the cross-module matrix: it raises the
cost of silently weakening the gate from "break one assertion in
one file" to "break the cross-product of a formal predictor".

---

*Document version:* aligned with the PR that shipped A3 on top of
v0.4.0 + HN15. See [`CHANGELOG.md`](../../CHANGELOG.md) for the
ordered release history.
