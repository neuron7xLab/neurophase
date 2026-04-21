# `tools/` — out-of-repo hardware bridges

This directory lives **outside the NeuroPhase kernel** (`neurophase/`).
Nothing in `tools/` is imported by `neurophase/`, and nothing in
`neurophase/` depends on anything here. Scripts in `tools/` exist only
to adapt real hardware producers onto the LSL contract that the kernel's
live consumer (`neurophase.physio.live`) already speaks.

## `polar_producer.py` — Polar H10 → LSL bridge

One standalone Python script that:

* scans for (or connects directly to) a Polar H10 over BLE (`bleak`);
* subscribes to the Bluetooth SIG **Heart Rate Service** (`0x180D`),
  **Heart Rate Measurement** characteristic (`0x2A37`);
* strictly parses each notification: `flags` byte → optional energy-
  expended field → zero or more little-endian `uint16` RR intervals;
* converts every RR value using the non-negotiable **`rr_ms = rr_raw * 1000 / 1024`**;
* pushes one LSL sample per RR interval as `[timestamp_s, rr_ms]` on a
  stream named `neurophase-rr` (default);
* emits the required **`[NaN, NaN]`** EOF sentinel exactly once on
  clean shutdown, guarded by a single-shot boolean (`SentinelGuard`).

The LSL outlet is shaped **identically** to what
`neurophase.physio.live` expects: `channel_count = 2`, `double64`,
`type = "RR"`. The contract is insured by the kernel-side test
`tests/test_physio_live.py::test_polar_producer_constants_match_kernel_contract`.

The SDK is not used. No reconnect logic. No ECG / raw streaming.

### Install

```bash
pip install -r tools/requirements.txt
```

Dependencies: `bleak` (BLE), `pylsl` (LSL). Both are pure-Python
wheels on Linux / macOS; Windows needs `WinRT` support built into
`bleak>=0.22`.

### Run the parser self-test (no BLE, no LSL, no hardware)

```bash
python tools/polar_producer.py --self-test
```

Covers five HRS payload fixtures, the two RR-conversion invariants
(512 → 500 ms, 1024 → 1000 ms), and the sentinel single-shot invariant.
Exits `0` on full pass, `6` otherwise.

### Run the end-to-end contract check (no hardware, but exercises LSL)

```bash
python tools/check_contract.py
```

This spawns the real neurophase live consumer, then pushes a
deterministic 20-sample sequence through `polar_producer.create_lsl_outlet`
(the same function the BLE path calls) and verifies the consumer
exits `0` after receiving the sentinel. Proves the LSL shape is
identical without needing a Polar H10.

### Run the real bridge (requires Polar H10 + Bluetooth adapter)

```bash
# Shell 1 — consumer
python -m neurophase.physio.live --stream-name neurophase-rr

# Shell 2 — producer (scan mode)
python tools/polar_producer.py --stream-name neurophase-rr

# Or targeted connect (skips the scan):
python tools/polar_producer.py --address AA:BB:CC:DD:EE:FF
```

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--stream-name` | `neurophase-rr` | LSL stream name (consumer must match) |
| `--address` | *(none)* | Explicit BLE address. Wins over `--name`; skips scan. |
| `--name` | *(none)* | Device-name substring filter. Default is `Polar H10`. |
| `--scan-timeout` | `10.0` | Seconds to scan before failing with `EXIT_NO_DEVICE`. |
| `--source-id` | derived | LSL `source_id` override; default is `polar-h10-rr-<address>`. |
| `--debug` | off | Include raw bytearray hex in `PACKET_REJECTED` events. |
| `--self-test` | — | Run parser + sentinel fixtures, exit 0 or 6. No BLE, no LSL. |
| `--version` | — | Print version and exit. |

### Exit-code table

| code | meaning |
|:---:|---|
| 0 | Orderly shutdown: `SIGINT` / `SIGTERM`, clean disconnect, EOF sentinel sent. |
| 1 | Device not found or ambiguous device selection. |
| 2 | BLE connect failure (`BleakError`). |
| 3 | HRS Measurement characteristic (`0x2A37`) missing on the device. |
| 4 | Unexpected peer-initiated disconnect **after** successful streaming. |
| 5 | Fatal LSL outlet / push failure. |
| 6 | `--self-test` failure. |

### Logged events (JSON-lines on stdout)

`SCAN_START`, `DEVICE_FOUND`, `DEVICE_SELECTED`, `DEVICE_SELECTED_BY_ADDRESS`,
`NO_DEVICE`, `AMBIGUOUS_DEVICE`, `SCAN_FAILED`, `LSL_OUTLET_CREATED`,
`LSL_OUTLET_FAILED`, `CONNECTING`, `CONNECTED`, `NOTIFY_STARTED`,
`RR_EMIT`, `PACKET_REJECTED`, `LSL_PUSH_FAILED`, `SHUTDOWN_REQUESTED`,
`DISCONNECTED`, `SENTINEL_SENT`, `SENTINEL_PUSH_FAILED`,
`CONNECT_FAILED`, `CHAR_MISSING`, `FATAL`, `EXIT`.

### Parser assumptions

* Heart Rate Service (`0x180D`) is a Bluetooth SIG standard and is the
  correct characteristic for Polar H10 RR intervals. The Polar SDK is
  **not** used. If Polar ships a firmware that exposes RR only over
  the SDK path in some future revision, this tool would need a
  fallback; that path is explicitly not implemented here.
* RR values are little-endian `uint16` in units of `1/1024 s`. One
  notification can carry zero, one, or several RR values; this tool
  emits one LSL sample per RR value.
* A truncated RR region (odd byte count) raises `HRSParseError` and
  the packet is dropped with a `PACKET_REJECTED` log event; the
  stream continues.
* RR values outside the physiological envelope `[300, 2000] ms` are
  dropped with `PACKET_REJECTED`; they never reach LSL. This mirrors
  the kernel's replay-side envelope in `neurophase/physio/replay.py`.

### Sentinel semantics

* `SentinelGuard` has one `emit_once()` method and one internal
  `_sent: bool` flag. The first call pushes `[NaN, NaN]`; every
  subsequent call is a no-op. This is enforced by the kernel-side
  test as well as the `--self-test` fixture.
* The sentinel is pushed on:
  * BLE peer-initiated disconnect (after successful streaming),
  * `SIGINT` / `SIGTERM`,
  * `bleak.BleakError` during connect (best-effort after outlet exists),
  * any fatal path that can still reach the outlet.
* It is **not** pushed if the producer exits before an outlet exists
  (device not found, scan failure). That is intentional — the consumer
  was never fed anything in that case.

### Hardware validation status

**Not performed in this repository.** No Polar H10 was available in
the environment where this tool was implemented. Layers validated:

* **A — static validation:** imports resolve; CLI parses; LSL outlet
  creates cleanly (exercised via `tools/check_contract.py`).
* **B — parser validation:** `python tools/polar_producer.py --self-test`
  returns `0`; 8/8 fixtures pass (5 HRS payload shapes + 2 RR
  conversion invariants + 1 sentinel invariant).
* **C — BLE integration validation:** *unperformed*. The tool has
  been exercised only against the "no Bluetooth adapter" failure
  path (which routes to `EXIT_NO_DEVICE` without a traceback).
* **D — end-to-end with real device:** *unperformed*.
* **D' — end-to-end with LSL, no hardware:** `tools/check_contract.py`
  passes (20 FRAMEs delivered, consumer exit 0, sentinel landed).

Any downstream user who has real hardware can run the `Run the real
bridge` commands above and verify; no repo changes are required.
