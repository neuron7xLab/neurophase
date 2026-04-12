# neurophase — Bridge Contracts

**Module.** `neurophase.bridges`
**Role.** Typed, fail-closed boundary between the kernel and the outside world.
**Status.** Load-bearing as of kernelization v1.

The kernel (`runtime`, `gate`, `contracts`, `audit`) never reads raw
driver payloads directly. Every live ingress and egress flows through a
named bridge that validates shape, invariants, and timing, and raises
:class:`BridgeError` (or a subclass) on any deviation.

## Layout

| Module | Public surface | Role |
|---|---|---|
| `bridges/ingress.py` | `NeuralSample`, `MarketTick`, `EegIngress`, `MarketIngress` | Typed ingress payloads + driver-wrapping adapters. |
| `bridges/clock_sync.py` | `ClockSync`, `ClockDesyncError` | Neural–market clock drift tolerance. |
| `bridges/frame_mux.py` | `FrameMux`, `MuxedTick` | Composes both ingresses + clock sync into one `poll()`. |
| `bridges/downstream_execution.py` | `DownstreamAdapter`, `DownstreamAdapterError`, `DownstreamDispatchResult` | Gate-first egress. |
| `bridges/errors.py` | `BridgeError` | Base class for every bridge rejection. |

## Contract summary

### Ingress

* `NeuralSample` and `MarketTick` refuse non-finite values, out-of-range
  phases / R, and empty `source_id` at construction.
* `EegIngress.poll()` and `MarketIngress.poll()` raise on driver
  exception, `None` return, wrong return type, and `source_id` mismatch.
* Drivers are injected — never imported at module load. Bridges are
  safe to import in kernel-only environments.

### Clock sync

* `ClockSync(max_drift_seconds)` rejects non-positive tolerances at
  construction.
* `fuse(neural, market)` returns the market timestamp after asserting
  `|neural.ts - market.ts| ≤ max_drift_seconds`; otherwise raises
  `ClockDesyncError`.

### Frame mux

* `FrameMux.poll()` is the **only** permitted construction path for
  the `(timestamp, R, delta)` triple fed to
  `RuntimeOrchestrator.tick()`. It surfaces every ingress failure and
  every clock desync as `BridgeError`.
* No buffering, no retry, no lenient mode. Production callers that
  need retries must add them *above* the mux, never inside.

### Downstream execution

* `DownstreamAdapter.dispatch(frame)` is the **only** permitted egress
  path for a gate-approved frame.
* The adapter refuses to dispatch when `frame.execution_allowed` is
  `False` — the transport callable is never called.
* The outbound payload is produced by
  `neurophase.contracts.as_canonical_dict` and validated by
  `validate_canonical_dict` before the transport sees it.
* Transport exceptions are re-raised as `DownstreamAdapterError` (via
  `raise … from exc`), never swallowed.
* The adapter has no concept of orders, retries, or trading logic.
  Those concerns live **above** the adapter.

## Observer boundary

The `reset.neosynaptex_adapter` γ-witness remains an **advisory
observer**. Whatever it reports surfaces into `DecisionFrame.klr_*` as
optional fields. It is not a bridge. The bridge layer's mandate is
*live ingress* and *live egress*; observer read-outs that do not affect
the canonical envelope belong in PHASE 9 (`observatory` export), not
here.

## Enforcement

`tests/test_bridge_layer.py` — 28 assertions covering happy paths and
every failure mode for every bridge class.
