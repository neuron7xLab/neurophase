# `profiles/` — per-user calibrated physio profiles

Each file here is one :class:`neurophase.physio.PhysioProfile`
serialised to JSON (`schema_version = "physio-profile-v1"`). One
profile represents one user at one calibration moment; re-calibrating
is a new file, not an in-place edit.

## Why profiles exist

The repo's illustrative admission thresholds
(`threshold_allow = 0.80`, `threshold_abstain = 0.50`) are defaults,
not deployment values. A user whose baseline HRV lives in a different
part of the distribution — or whose sensor setup yields a different
stability ceiling — will see the default gate over-abstain or
over-admit relative to their actual healthy state.

A profile fixes this by:

1. deriving *user-specific* admission thresholds from the user's own
   healthy-signal distribution (see
   `neurophase/physio/calibration.py` for the derivation rule);
2. capturing the empirical feature bands (RMSSD, stability,
   continuity, confidence) for audit;
3. pinning the `window_size` — a profile is only valid at the window
   it was calibrated under.

## How to produce a profile

Record 3-7 short baseline sessions in different nominal states
(morning / post-load / post-focus / recovery / …) using:

```bash
# live session with ledger
python -m neurophase.physio.live --stream-name neurophase-rr \
    --ledger-out artifacts/live_runs/<date>/baseline-morning.jsonl

# OR a replay CSV (e.g. existing examples/data/physio_replay_sample.csv)
```

Then feed them to the calibrator:

```bash
python tools/calibrate_physio.py \
    --user-id alex-2026-04 \
    --out profiles/alex-2026-04.json \
    --ledger artifacts/live_runs/2026-04-13/baseline-morning.jsonl \
    --ledger artifacts/live_runs/2026-04-13/baseline-post-load.jsonl \
    --ledger artifacts/live_runs/2026-04-13/baseline-post-focus.jsonl \
    --note "chest strap, seated, 20C ambient"
```

The calibrator fails closed (exit 1, `CalibrationError`) if:

* fewer than 3 baseline sessions,
* fewer than 32 healthy frames in any single session,
* fewer than 128 healthy frames in total.

A profile that survives those checks is saved as JSON here.

## How to use a profile

Pass `--profile` to the live consumer:

```bash
python -m neurophase.physio.live \
    --stream-name neurophase-rr \
    --profile profiles/alex-2026-04.json \
    --ledger-out artifacts/live_runs/<date>/work-session.jsonl
```

The consumer emits a `GATE_MODE` event at startup confirming
`mode: "calibrated"` and the `profile_user_id`. Without `--profile`
the gate stays in `default` mode.

## Policy

* **No anonymous profiles.** `user_id` is mandatory and opaque.
* **No partial overrides.** All thresholds come from one profile.
* **No silent fallback.** If `--profile PATH` points at a missing
  or schema-mismatched file, the consumer exits non-zero; it does
  NOT silently revert to defaults.
* **No in-place rotation.** Re-calibrating produces a new file.

This directory is checked into the repo for convenience; real
deployments usually keep user profiles in a private directory and
pass `--profile` absolute paths.
