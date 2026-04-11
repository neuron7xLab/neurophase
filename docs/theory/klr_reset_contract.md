# KLR Reset Contract (Tentative)

## Purpose
KLR is an **advisory adaptive reconfiguration subsystem** for escaping pathological attractors. It is **not** a gate-permission authority.

## State machine
`IDLE -> PREPARED -> DISINHIBITED -> PLASTICITY_OPEN -> CONSOLIDATING -> STABLE | ROLLBACK`

No commit path may bypass `_validate_and_commit`.

## Lock-in score
`score = w_error*error + w_persistence*persistence + w_diversity*(1-diversity) + w_improvement*(1-improvement)`

Current calibrated constants (synthetic scope):
- `w_error=0.30`
- `w_persistence=0.29`
- `w_diversity=0.23`
- `w_improvement=0.18`
- `threshold=0.7631`

## Calibration honesty
- Scope: `synthetic_archive`
- Label source: heuristic rule, legacy `ground_truth` compatibility
- Expert labels: not available
- Evidence status: **Tentative**

## Validation and rollback semantics
Commit requires:
1. contract-valid post-state,
2. finite gamma,
3. relapse below threshold,
4. explicit validation phase success.

Any failure collapses to `ROLLBACK` (fail-closed).

## What KLR does not claim
- no clinical equivalence,
- no established biological detector,
- no external replication complete,
- no widening of core execution permissions.

## Invariants
- checkpoint exists before mutation phases,
- no commit without validation,
- rollback restores checkpointed safe state,
- deterministic under fixed `(state, metrics, curriculum)` with deterministic oracle.

## Anti-patterns
- treating heuristic labels as expert truth,
- silent `None` frozen-mask usage,
- docs/config threshold drift,
- mutable checkpoint corruption,
- claim promotion above Tentative without external replication.

## External γ-verification witness (NEO-I1, NEO-I2)

The KLR pipeline exposes an **advisory** γ-verification channel backed
by the external [`neosynaptex`](https://github.com/neuron7xLab/neosynaptex)
integrating mirror. It runs alongside — never instead of — the internal
`_gamma()` row-entropy probe. The witness is wired into every
`KLRPipeline.tick()` via `GammaWitness.observe(active_state)` and its
report is attached to `KLRFrame.witness_report` (or `None` if the
witness is disabled via `KLRPipeline(..., enable_witness=False)` or if
the optional `neosynaptex` dependency is not installed).

### Projection

`NeosynaptexResetAdapter` projects a `SystemState` onto a three-scalar
vector consumed by `neosynaptex` as a single domain (`klr_reset`):

- `ntk_rank`      — normalized NTK-rank proxy (plasticity headroom).
- `frozen_ratio`  — fraction of frozen nodes.
- `usage_entropy` — normalized Shannon entropy of the usage distribution.

Topology signal: non-frozen node count. Thermodynamic cost:
`1 − ntk_rank`. Both are clamped away from zero to keep downstream
log-transforms finite.

### Report schema

```python
@dataclass(frozen=True)
class GammaWitnessReport:
    gamma_external: float  # 0.0 during warmup, finite post-warmup
    phase:          str    # WARMUP | METASTABLE | CONVERGING | ...
    coherence:      float  # 0.0 during warmup
    verdict:        str    # COHERENT | INCOHERENT | INSUFFICIENT_DATA
```

### Invariants

- **NEO-I1 (read-only).** `GammaWitness.observe()` never mutates the
  `SystemState` it inspects. `NeosynaptexResetAdapter.update()` is the
  only write path into the witness and performs only reads on the
  passed-in state.
- **NEO-I2 (advisory-only).** No verdict emitted by the witness — in
  particular `INCOHERENT` and `INSUFFICIENT_DATA` — may alter
  `KLRFrame.decision`, any gate state, or propagate exceptions out of
  `KLRPipeline.tick()`. Any internal failure inside `neosynaptex` makes
  the witness degrade silently (`phase=WARMUP`,
  `verdict=INSUFFICIENT_DATA`, `gamma_external=0.0`) and keeps the
  pipeline running.

### Non-claims

- The witness **does not replace** the internal `_gamma()` probe.
- The witness **does not gate execution** — `ExecutionGate`, the KLR
  controller, and every `I₁`–`I₄` / `KLR-I₁`–`KLR-I₄` invariant are
  unaffected by its output.
- With a single-domain projection, `coherence` is inherently a data
  quality signal rather than a regime indicator; multi-domain witness
  configurations are out of scope for this contract.
