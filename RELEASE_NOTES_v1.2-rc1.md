# NeuroPhase v1.2-rc1 — Truth-aligned human-in-the-loop system

**Tag:** `v1.2-rc1`
**Date:** 2026-04-13
**Predecessor:** `v1.1` (true async live RR ingress + Polar H10 producer)

## Summary

v1.2-rc1 closes the loop from "live signal" to "replayable audit trail
plus per-user calibration plus adversarial-fault coverage". The story
neurophase now tells:

> Accept a real physiological stream. Gate fail-closed under every
> adversarial input we can synthesise. Calibrate thresholds per user
> from their own baselines. Record an immutable session ledger. Prove
> offline, frame-by-frame, that every recorded decision is reproducible.

None of that requires changing the neurophase kernel. None of it is
fake. The things it does not ship — real Polar H10 sessions, a real
`<user_id>.json`, CNS-protocol day-across-days data, decision-quality
benchmark pairs — are explicitly operator-run and have pre-committed
protocols documented in `docs/`.

## Highlights

- **Session ledger + offline replay** (`neurophase/physio/ledger.py`,
  `neurophase/physio/session_replay.py`). Every live session can be
  recorded to a JSONL file with a `SESSION_HEADER` + `SESSION_SUMMARY`
  envelope and re-executed offline through a fresh `PhysioSession`.
  The replayer asserts byte-identical gate decisions via
  `session_replay --strict`.
- **Per-user calibration layer** (`neurophase/physio/profile.py`,
  `neurophase/physio/calibration.py`,  `tools/calibrate_physio.py`).
  Profile schema is strict; calibrator is conservative (>= 3
  sessions, >= 128 healthy frames); calibration only *tightens* the
  default thresholds, never relaxes them.
- **Calibrated vs default gate modes** (`neurophase/physio/gate.py`).
  `PhysioGate.from_profile` is the single blessed constructor for
  calibrated mode; every threshold comes from one profile.
- **Adversarial fault suite** (`tools/fault_producer.py`,
  `tests/test_physio_faults.py`). Six injection modes + a clean
  reference, each a real subprocess test. Load-bearing invariant:
  no FRAME under any fault carries `execution_allowed=True` with
  `gate_state != EXECUTE_ALLOWED`.
- **Operator protocols** (`docs/LIVE_SESSION_PROTOCOL.md`,
  `docs/CNS_PROTOCOL.md`, `benchmarks/decision_quality/PROTOCOL.md`).
  Pre-committed specifications for Layer-D live runs, CNS-mode
  recordings, and decision-quality matched pairs. Nothing post-hoc.
- **Quickstart** (`QUICKSTART.md`). Fresh clone to first gated
  decision in under 10 minutes; all five layers (install / replay /
  live-loopback / real hardware / calibration) spelled out.

## Local CI gate

```
ruff check neurophase tests         All checks passed!
ruff format --check neurophase tests  268 files formatted
mypy --strict neurophase            148 source files OK
pytest tests/ -q                    1591 passed, 6 skipped
```

`pylsl` and `bleak` were objectively probed before adoption (install +
import + loopback smoke test, all exit 0 — see v1.1 commit trail).

## Claim posture unchanged

`CLAIMS.yaml` and `README.md` stand exactly as they did under v1.1
truth alignment:

- FMθ existence on ds003458: observation-level, not gating-relevant.
- FMθ utility for gating on ds003458: **null**.
- PLV-market coupling: historical / exploratory, demoted.
- Kernel = the strongest validated contribution.
- Physio replay / live = minimal truthful ingestion path.
- Polar H10 bridge = one real hardware adapter, out-of-repo, BLE
  standard HRS.

v1.2-rc1 adds capability; it does not promote any scientific claim.

## What requires the operator

- **Layer-D live runs:** three 12-15 min sessions per
  `docs/LIVE_SESSION_PROTOCOL.md`, producing
  `artifacts/live_runs/<DATE>/RUN_REPORT.md`.
- **Real per-user `<user_id>.json`:** via `tools/calibrate_physio.py`
  on the above ledgers.
- **CNS-protocol day-across-days data:** per `docs/CNS_PROTOCOL.md`.
- **Decision-quality benchmark:** matched pairs per
  `benchmarks/decision_quality/PROTOCOL.md`.

None of these require touching the repo kernel.

## Upgrade notes

- If you had a local pre-v1.2 clone with a running live session, the
  consumer CLI gained two optional flags: `--ledger-out PATH`,
  `--profile PATH`. Both default to "off" so nothing breaks.
- If you load a profile produced before `profile.py`'s
  `PROFILE_SCHEMA_VERSION` was bumped, the loader refuses it
  (fail-closed). No silent coercion.

## Fresh-clone acceptance check

A serious engineer should be able to:

1. `git clone … && cd neurophase && pip install -e '.[dev]' && pip install -r tools/requirements.txt`
2. `python -m neurophase.physio.demo` → exit 0, 4 states observed.
3. `python -m neurophase.physio.live --stream-name X --max-frames 24 --ledger-out /tmp/x.jsonl` + `python -m neurophase.physio.live_producer --stream-name X` in another shell → both exit 0.
4. `python -m neurophase.physio.session_replay /tmp/x.jsonl --strict` → `Parity: OK`, exit 0.
5. `pytest tests/ -q` → 1591 passed.

If any of the above fails on a clean checkout, it is a release-blocking
bug, not a limitation to document.

## Co-authors

Developed with Claude Opus 4.6 (1M context).
