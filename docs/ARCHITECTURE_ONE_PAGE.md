# neurophase — architecture in one page

*60-second tour for a reviewer landing in the repo for the first
time. For the full doctrine see
[`docs/EVOLUTION_BOARD.md`](EVOLUTION_BOARD.md); for the ranked task
map see [`docs/TASK_MAP.md`](TASK_MAP.md); for the visual state
machine see
[`docs/diagrams/gate_state_machine.md`](diagrams/gate_state_machine.md).*

---

## What it is

A disciplined **phase-synchronization decision system**. Brain and
market are modelled as coupled Kuramoto oscillators; `R(t)` measures
their joint order parameter. When `R(t)` drops below threshold,
execution is blocked *by physics*, not by rule. Built as a
typed Python library with a 5-state gate, a temporal precondition,
a null-model harness, OOS-validated calibration, a byte-identical
decision ledger, and a runtime streaming pipeline.

## What it is NOT

* **Not** a trading bot. No policy layer, no sizing, no execution side effects.
* **Not** a free-energy implementation. `F_proxy = ½·δ²` is a geometric surrogate (HN1).
* **Not** a real-time drift repair layer. `TemporalValidator` reports, never fixes (HN2).
* **Not** an ML framework. Doctrine forbids learned models before baseline falsification.

---

## The five states

| State | Execution | Meaning | Invariant |
|---|---|---|---|
| `READY` | ✅ | synchronized and actionable | — |
| `BLOCKED` | ❌ | `R < θ` — desynchronized | I₁ |
| `UNNECESSARY` | ❌ | `R ≥ θ` but dynamics still — no new information | I₄ |
| `SENSOR_ABSENT` | ❌ | bio-sensor unavailable | I₂ |
| `DEGRADED` | ❌ | `R` invalid / temporal precondition failed | I₃ ∪ B₁ |

Strict priority order inside `ExecutionGate.evaluate`:

```
B₁ (time_quality != VALID)  →  DEGRADED
I₂ (sensor absent)          →  SENSOR_ABSENT
I₃ (R invalid / OOR)        →  DEGRADED
I₁ (R < threshold)          →  BLOCKED
I₄ (stillness)              →  UNNECESSARY
otherwise                   →  READY
```

**Only `READY` produces `execution_allowed = True`.** Always. Forever.
Enforced at `GateDecision.__post_init__` and at the type boundary;
proven exhaustively by the A3 cross-module invariant matrix.

---

## The single blessed import path

```python
from neurophase.api import (
    create_pipeline,    # factory
    explain_decision,   # interpretability
    DecisionFrame,      # runtime envelope
    GateState,          # 5-state enum
    Contract,           # B1/I1/I2/I3/I4/READY
    Verdict,            # PASS/FAIL/SKIPPED
)

pipeline = create_pipeline(threshold=0.65, warmup_samples=4)
frame = pipeline.tick(timestamp=0.0, R=0.9, delta=0.01)
explanation = explain_decision(frame)
print(explanation.causal_contract)   # → Contract.READY / I1 / etc.
```

Or use the CLI:

```console
$ python -m neurophase demo --ticks 16
$ python -m neurophase verify-ledger path/to/decisions.jsonl
$ python -m neurophase explain-ledger path/to/decisions.jsonl
```

## The hot path

```
raw (timestamp, R, δ)
        ↓
TemporalValidator                       ← B₁ per-sample time contract
        ↓
TemporalStreamDetector                  ← B₂+B₆ stream-level regime
        ↓
ExecutionGate + StillnessDetector       ← I₁ · I₂ · I₃ · I₄
        ↓
DecisionFrame                           ← runtime envelope
        ↓
DecisionTraceLedger (optional)          ← F₁ append-only SHA256 chain
        ↓
DecisionExplanation                     ← causal chain, deterministic
```

Everything above is a thin composition layer in `StreamingPipeline.tick()`.
No global state, no hidden coupling, no policy.

---

## The enforcement stack (sideways, CI-bound)

| Program | Artefact | CI meta-test |
|---|---|---|
| A1 governance | [`INVARIANTS.yaml`](../INVARIANTS.yaml) — 22 contracts | `tests/test_invariants_registry.py` |
| A2 state machine | [`STATE_MACHINE.yaml`](../STATE_MACHINE.yaml) — 8 transitions | `tests/test_state_machine_spec.py` |
| A3 cross-module | analytical predictor vs live gate | `tests/test_invariant_matrix.py` |
| C1–C3 falsification | null-model harness + surrogates + PLV held-out | `tests/test_null_model_harness.py` |
| D1 + D2 calibration | threshold + stillness Youden-J grid | `tests/test_calibration_*.py` |
| F1 + F2 + F3 audit | append-only ledger + replay engine + byte-identical determinism | `tests/test_decision_ledger.py` + `test_replay_engine.py` + `test_determinism_certification.py` |
| H1 ground truth | synthetic phase coupling with closed-form PLV | `tests/test_benchmarks_phase_coupling.py` |
| HN15 bibliography | research-grade, DOI-anchored, no fake citations | `tests/test_bibliography_contract.py` |

No green → no merge.

---

## Stats at a glance

- **703+** tests green on every PR
- **106** source files pass `mypy --strict`
- **22** honest-naming contracts CI-bound
- **5** gate states, **8** transitions, **1** permissive
- **0** fabricated citations, **0** learned models, **0** hidden state

---

## Where to read next

* [`ARCHITECTURE.md`](../ARCHITECTURE.md) — longer architecture doc with module map
* [`docs/EVOLUTION_BOARD.md`](EVOLUTION_BOARD.md) — governance doctrine
* [`docs/TASK_MAP.md`](TASK_MAP.md) — 25 ranked tasks in Tier 1–5 structure
* [`docs/theory/scientific_basis.md`](theory/scientific_basis.md) — system-level scientific grounding
* [`docs/theory/neurophase_elite_bibliography.md`](theory/neurophase_elite_bibliography.md) — 24 DOI-anchored peer-reviewed sources
* [`docs/theory/invariant_matrix.md`](theory/invariant_matrix.md) — A3 safety proof
* [`docs/theory/stillness_invariant.md`](theory/stillness_invariant.md) — I₄ formal derivation
* [`docs/theory/time_integrity.md`](theory/time_integrity.md) — B₁ temporal contract
* [`docs/diagrams/gate_state_machine.md`](diagrams/gate_state_machine.md) — visual state diagrams
