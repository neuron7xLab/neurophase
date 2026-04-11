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
