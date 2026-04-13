# NeuroPhase — quickstart

Fresh clone to first gated decision in under 10 minutes.

## Install

```bash
git clone https://github.com/neuron7xLab/neurophase
cd neurophase
pip install -e '.[dev]'          # kernel + scientific stack + LSL
pip install -r tools/requirements.txt   # Polar H10 bridge (bleak)
```

## 1. Replay-from-CSV demo (no hardware)

Runs the shipped synthetic replay sample through the physio gate end
to end. Exits 0, emits 4 distinct states.

```bash
python -m neurophase.physio.demo
```

## 2. True live demo without hardware (LSL loopback)

Two shells. Consumer in A, synthetic producer in B. Proves the live
path, the ledger, and offline replay all agree byte-for-byte.

```bash
# --- shell A ------------------------------------------------------
python -m neurophase.physio.live \
    --stream-name neurophase-rr \
    --max-frames 24 \
    --ledger-out /tmp/loopback.jsonl

# --- shell B (after shell A prints LISTENING) ---------------------
python -m neurophase.physio.live_producer \
    --stream-name neurophase-rr \
    --inter-sample-s 0.02

# --- after both exit ----------------------------------------------
python -m neurophase.physio.session_replay /tmp/loopback.jsonl --strict
# -> Parity: OK
```

## 3. Real hardware (Polar H10)

```bash
# --- shell A ------------------------------------------------------
python -m neurophase.physio.live \
    --stream-name neurophase-rr \
    --ledger-out artifacts/live_runs/$(date +%F)/baseline-calm/ledger.jsonl

# --- shell B ------------------------------------------------------
python tools/polar_producer.py --stream-name neurophase-rr
# (or --address AA:BB:CC:DD:EE:FF to skip scan)
```

See `docs/LIVE_SESSION_PROTOCOL.md` for the full session protocol.

## 4. Personal calibration

After 3+ baseline sessions recorded as ledgers:

```bash
python tools/calibrate_physio.py \
    --user-id alex-2026-04 \
    --out profiles/alex-2026-04.json \
    --ledger artifacts/live_runs/2026-04-13/baseline-calm/ledger.jsonl \
    --ledger artifacts/live_runs/2026-04-13/cognitive-load/ledger.jsonl \
    --ledger artifacts/live_runs/2026-04-13/recovery/ledger.jsonl \
    --note "chest strap, seated, 20C"

# Then run live with calibrated thresholds:
python -m neurophase.physio.live \
    --stream-name neurophase-rr \
    --profile profiles/alex-2026-04.json \
    --ledger-out artifacts/live_runs/$(date +%F)/work-session/ledger.jsonl
```

See `profiles/README.md` for calibration policy.

## 5. Run the full local CI gate (before any push)

```bash
ruff check neurophase tests
ruff format --check neurophase tests
mypy --strict neurophase
pytest tests/ -q
```

## What is shipped vs what requires the operator

Shipped, green in CI, runnable without hardware:

- Fail-closed physio kernel (replay + live, shared `PhysioSession`).
- True asynchronous live LSL ingress (`neurophase.physio.live`).
- Synthetic validation producer (`neurophase.physio.live_producer`).
- Polar H10 BLE → LSL bridge (`tools/polar_producer.py`).
- Session ledger + offline replay (byte-identical parity).
- Per-user calibration layer (profile + calibrator + gate mode).
- Adversarial fault suite (7 fault classes, subprocess-based).
- Quickstart + session protocol + CNS protocol + benchmark protocol.

Requires the operator + real hardware:

- Layer-D live run triplet (3 sessions × 12-15 min).
- A real per-user `<user_id>.json` calibration profile.
- CNS-protocol mode recordings across days.
- Decision-quality benchmark matched pairs.

Every one of those operator tasks plugs into the pipeline without any
repo change: the tooling is the same whether the data is synthetic or
real.

## What is NOT claimed

- Not a neural gate. FMθ utility on ds003458 is null (see `CLAIMS.yaml` C5).
- Not a medical device. HRV features are signal-quality indicators only.
- Not a trading-alpha signal. The gate is a fail-closed admission layer.
