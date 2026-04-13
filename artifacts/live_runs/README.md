# `artifacts/live_runs/` — where live-session artifacts land

Layout per session:

```
artifacts/live_runs/<YYYY-MM-DD>/
  <session-label>/
    producer.log
    consumer.log
    ledger.jsonl
    replay_report.json
  RUN_REPORT.md
```

## RUN_REPORT.md template

```markdown
# Live run — <YYYY-MM-DD>

## Hardware
- Device model     :
- Firmware         :
- LSL transport    :

## Operator notes
- User ID          :
- Context (room / posture / time-of-day) :
- Anything unusual :

## Sessions

### baseline-calm  (<HH:MM>, duration <N> min)
- declared mode     : stillness
- dominant gate     : EXECUTE_ALLOWED | EXECUTE_REDUCED | ABSTAIN | SENSOR_DEGRADED
- degraded episodes : 0 | N
- clean_exit        : true | false
- replay parity_ok  : true | false

### cognitive-load  (<HH:MM>, duration <N> min)
- declared mode     : focused_work | overload
- ...

### recovery  (<HH:MM>, duration <N> min)
- declared mode     : recovery
- ...

## Verdict vs expected gate behaviour (per docs/CNS_PROTOCOL.md)
- stillness       : [matched | diverged | N/A]
- focused_work    : [...]
- overload        : [...]
- recovery        : [...]

## What surprised you

Freeform. Anything that would inform the next calibration cycle or
test case.

## Attachments

- [ ] baseline-calm/ledger.jsonl
- [ ] baseline-calm/replay_report.json
- [ ] cognitive-load/ledger.jsonl
- [ ] cognitive-load/replay_report.json
- [ ] recovery/ledger.jsonl
- [ ] recovery/replay_report.json
```

## Rules

- **Never hand-edit ledgers.** The replay parity check would fail and
  the session would be unverifiable.
- **Keep raw logs.** Even when a session goes sideways, the logs are
  the record. Only delete if they contain something you would not
  want to commit publicly.
- **Don't cherry-pick.** If you record five sessions and three are
  messy, all five go into the run report.
