# neurophase — Gate-First Execution Law

**Rule.** No downstream action may be derived from anything other than
a gate-approved `OrchestratedFrame`.

**Mechanism.** Four structural enforcements + a test contract.

## Structural enforcement

1. **The runtime produces only `OrchestratedFrame`.**
   `RuntimeOrchestrator.tick()` always returns a frame in which
   `pipeline_frame.gate.execution_allowed` reflects the full
   I₁–I₄ invariant stack. There is no path from a raw `(R, delta)`
   to a decision that skips the gate.

2. **`ActionPolicy` never widens the gate.**
   `neurophase.policy.action.ActionPolicy.decide(frame, regime)`
   emits `ActionIntent.HOLD` whenever `frame.execution_allowed` is
   `False`, regardless of regime confidence. There is no override.

3. **`DownstreamAdapter.dispatch(frame)` refuses non-READY frames.**
   Before the transport callable is invoked, the adapter asserts
   `frame.execution_allowed is True`. A failing assertion raises
   `DownstreamAdapterError`; the transport is never called.

4. **`FrameMux` is fail-closed on ingress.**
   Any driver exception, `None` return, or wrong-type payload, and
   any neural–market clock drift, aborts `poll()` before it can yield
   a canonical tick. The kernel never sees partial data.

## Reject-path semantics

Every rejection has a ledger-visible reason string. The five
`GateState` values carry the full reject taxonomy:

| State | Reject condition | Enforced by |
|---|---|---|
| `SENSOR_ABSENT` | bio-sensor missing (I₂) | `ExecutionGate` + `NeuralPhaseExtractor` |
| `DEGRADED` | temporal / stream / warmup fault (B1, B2, B6) | `TemporalValidator`, `TemporalStreamDetector` |
| `BLOCKED` | `R(t) < θ` (I₁) or `R` out of range (I₃) | `ExecutionGate` |
| `UNNECESSARY` | stillness regime active (I₄) | `StillnessDetector` |
| `READY` | all of the above passed | `ExecutionGate` |

Only `READY` permits `ActionPolicy` and `DownstreamAdapter` to act.
Every other state produces a frame whose `ledger_record_hash` (when a
ledger is attached) makes the rejection auditable end-to-end.

## What the law forbids

* **Bypass execution from raw metrics.** There is no public call that
  takes a bare `R` and returns a decision. `StreamingPipeline.tick()`
  is the earliest entry point and it already runs the gate.
* **Silent promotion of an invalid input.** NaN / None R routes
  through `DEGRADED`. The orchestrator never returns an
  `execution_allowed=True` frame for an invalid input.
* **"Best-effort" transport calls.** `DownstreamAdapter` does not
  implement retry, deduplication, or partial success. If the transport
  raises, the caller sees `DownstreamAdapterError`; it is the caller's
  responsibility to decide what to do with the failed dispatch —
  without bypassing the adapter.

## Enforcement

`tests/test_gate_first_execution.py` — 8 assertions:

* warmup / low-R / None-R / NaN-R all fail-closed;
* `ActionPolicy` emits `HOLD` when the gate refuses;
* `DownstreamAdapter.dispatch` is called zero times for non-READY
  frames;
* `FrameMux.poll` surfaces every ingress / clock failure as
  `BridgeError`.
