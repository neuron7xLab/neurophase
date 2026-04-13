# Live session protocol — Layer D operator run-book

This protocol is for the operator with real Polar H10 hardware. It
specifies exactly what to capture per session, how to name the
artifacts, and what constitutes "done". The neurophase CI never runs
this protocol — it cannot — but every piece of tooling it invokes is
green in CI.

## Session triplet

The first-shipping Layer D evidence is three sessions of 12–15 minutes
each, run back-to-back on the same day where feasible:

1. **`baseline-calm`** — seated, low stimulation, no task. Establishes
   the user's resting signal-quality distribution.
2. **`cognitive-load`** — a real work task that demands sustained
   focus (coding block, deep reading, calculation). Stresses the
   signal.
3. **`recovery`** — five-minute cool-down after the load block:
   eyes closed, slow breathing, no phone. Establishes return to
   baseline.

Each session must produce a clean shutdown via the `[NaN, NaN]` EOF
sentinel. At least ONE session across the triplet must surface a
degraded episode (artifact, pause, RR spike, etc.) so the SENSOR_DEGRADED
branch is exercised on live data.

## Per-session artifacts

For each session, produce these files under
`artifacts/live_runs/<YYYY-MM-DD>/`:

```
artifacts/live_runs/2026-04-13/
  baseline-calm/
    producer.log            # stdout of tools/polar_producer.py
    consumer.log            # stdout of neurophase.physio.live
    ledger.jsonl            # --ledger-out from the consumer
    replay_report.json      # python -m neurophase.physio.session_replay --json
  cognitive-load/
    producer.log
    consumer.log
    ledger.jsonl
    replay_report.json
  recovery/
    producer.log
    consumer.log
    ledger.jsonl
    replay_report.json
  RUN_REPORT.md             # operator-authored summary, see below
```

`RUN_REPORT.md` fills the template below.

## Commands (copy-paste ready)

```bash
DATE=$(date +%F)
mkdir -p "artifacts/live_runs/${DATE}/baseline-calm"
mkdir -p "artifacts/live_runs/${DATE}/cognitive-load"
mkdir -p "artifacts/live_runs/${DATE}/recovery"

# Shell A (consumer) -- repeat per session with a different --ledger-out
python -m neurophase.physio.live \
    --stream-name neurophase-rr \
    --ledger-out "artifacts/live_runs/${DATE}/baseline-calm/ledger.jsonl" \
    2>&1 | tee "artifacts/live_runs/${DATE}/baseline-calm/consumer.log"

# Shell B (producer)
python tools/polar_producer.py \
    --stream-name neurophase-rr \
    2>&1 | tee "artifacts/live_runs/${DATE}/baseline-calm/producer.log"

# After each session: offline replay parity report
python -m neurophase.physio.session_replay \
    "artifacts/live_runs/${DATE}/baseline-calm/ledger.jsonl" \
    --json --strict \
    > "artifacts/live_runs/${DATE}/baseline-calm/replay_report.json"
```

## Definition of Done

The triplet is complete when ALL are true:

- [ ] `producer.log` contains `RR_EMIT` events with monotonic
  timestamps and plausible RR values (300–2000 ms).
- [ ] `consumer.log` contains a `SUMMARY` event with `clean_exit: true`.
- [ ] `ledger.jsonl` opens with `SESSION_HEADER` and closes with
  `SESSION_SUMMARY`.
- [ ] `replay_report.json.parity_ok == true` for all three sessions.
- [ ] Across the triplet, at least one `SENSOR_DEGRADED` frame or
  `STALL` event was emitted, AND at least one session reached
  `EXECUTE_ALLOWED` (or `EXECUTE_REDUCED`) for a sustained window.

## Honest limits

- Pipeline behaviour under "cognitive load" depends on the user's
  own physiology. The protocol does not claim that a particular gate
  state corresponds to a particular cognitive state — see
  `docs/CNS_PROTOCOL.md` for that mapping, and see `CLAIMS.yaml` C5
  for what is and is not established.
- This is a **physiological signal-quality** pipeline. It is not a
  medical instrument, a readiness biomarker, or a trading-alpha
  signal.
