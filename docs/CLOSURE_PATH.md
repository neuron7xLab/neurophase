# neurophase — Closure-Ready Causal Path

**Honest name.** Closure-ready causal path. Not "closed loop."
**Status.** End-to-end demonstrable as of kernelization v1.

## What the path is

One-way typed chain, driven by `FrameMux.poll` and terminated by
`DownstreamAdapter.dispatch`:

```
driver       ┐
             │ bridges.ingress.{EegIngress, MarketIngress}
neural ──────┤─────────▶ NeuralSample
market ──────┘─────────▶ MarketTick
                         │
                         │ bridges.clock_sync.ClockSync
                         ▼
                       MuxedTick (timestamp, R, delta, source_ids)
                         │
                         │ runtime.orchestrator.RuntimeOrchestrator.tick
                         ▼
                       OrchestratedFrame (canonical typed envelope)
                         │  — gate (I₁..I₄)
                         │  — regime (G1)
                         │  — policy (I1)
                         │  — ledger append (F1)
                         ▼
                       contracts.as_canonical_dict  →  dict v1.0.0
                         │
                         │ validate_canonical_dict (strict)
                         ▼
                       bridges.downstream_execution.DownstreamAdapter
                         │  — refuses non-READY frames
                         │  — forwards to transport
                         ▼
                       transport callable (owned by integrator)
```

## What the path is *not*

* **Not a closed loop.** There is no live two-way bus between the
  downstream receiver and the ingress adapters. Promoting to a true
  closed-loop system would require a control channel that feeds
  downstream replies back into the policy — that is not implemented
  and the documentation does not pretend otherwise.
* **Not a trading engine.** The path stops at the transport callable.
  Order management, retry, deduplication, accounting, risk sizing
  post-gate — all live above the adapter.
* **Not research-dep-coupled.** The full path is importable with
  kernel-only dependencies (`numpy`, `scipy`, `PyYAML`). No `mne`,
  `pandas`, `pywt`, `neurodsp`, `sklearn`, or `networkx` is loaded at
  any point in the chain.

## Why we can replay it

Every layer is deterministic given its input:

* `FrameMux.poll` has no hidden state.
* `RuntimeOrchestrator.tick` has no RNG, no clock, and no I/O.
* `as_canonical_dict` is a pure projection.
* `DownstreamAdapter.dispatch` stamps the exact same payload into the
  transport.

Two sessions driven by identical drivers produce byte-identical
canonical dict sequences. `tests/test_closure_path.py::
test_closure_path_is_deterministic_under_replay` asserts this.

## Future full closed-loop target

The missing pieces are all *outside* the kernel:

1. A control channel that surfaces transport acknowledgements back
   into `ActionPolicy` (would require a typed `AckFrame` and a
   feedback adapter — PHASE 11 material).
2. A live γ-witness bus that actively pings the γ-collector rather
   than the current advisory `neosynaptex_adapter` (PHASE 9 scope).
3. An ingress pacer so the mux can pull at a guaranteed rate rather
   than whenever the caller asks (PHASE 11 material).

None of these are available today. The repository will not claim
they are.

## Enforcement

`tests/test_closure_path.py` — 4 end-to-end assertions:

* the full chain runs without exception;
* replay is byte-identical;
* only READY frames reach the downstream;
* every dispatched payload carries the audit fields required by the
  canonical schema.
