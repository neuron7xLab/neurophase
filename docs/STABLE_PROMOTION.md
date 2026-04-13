# STABLE_PROMOTION.md — canonical gate for `v1.2-rc → v1.2`

This document is the **single source of truth** for promoting any
release-candidate tag (currently `v1.2-rc2`) to a stable tag (`v1.2`).
It is a hard checklist, not an aspirational sketch. Every item must
be independently verifiable from the repo state at the candidate
commit; if any item is not satisfied, the promotion is blocked.

There is no "mostly done" exit path. `v1.2` stable means every one
of the boxes below is green.

## Scope

Applies to: all production-grade stable tags for neurophase.
Does NOT apply to: rc tags, doc-only fixups, bug-fix commits.
Does NOT override: the claim-promotion ladder in `CLAIMS.yaml`
(hypothesis → theory → fact) — that is a separate, narrower gate.

---

## The promotion gate

### G1. Layer-D triplet completed

Three real-hardware live sessions must exist in `artifacts/live/`,
each with its own dated sub-directory, covering the modes defined in
`docs/LIVE_SESSION_PROTOCOL.md`:

- [ ] `artifacts/live/<date>/baseline-calm/` — ledger + logs
- [ ] `artifacts/live/<date>/focused-load/`  — ledger + logs
- [ ] `artifacts/live/<date>/recovery/`      — ledger + logs
- [ ] at least **one** clean shutdown via NaN sentinel
- [ ] at least **one** observed `SENSOR_DEGRADED` episode (proof
  that the degraded path fires on real signal, not only in the
  fault suite)
- [ ] zero kernel-source edits made during the triplet

Evidence: per-session `VERDICT.json` from
`scripts/run_layer_d_acceptance.sh`, plus a short operator-authored
`RUN_REPORT.md` at `artifacts/live/<date>/RUN_REPORT.md`.

### G2. Calibration profile exists

At least one real per-user calibration profile produced from real
baseline sessions:

- [ ] `profiles/<user_id>.json` exists
- [ ] passes `python tools/check_contract.py` + round-trips through
      `neurophase.physio.profile.load_profile`
- [ ] was generated via `tools/calibrate_physio.py` from ≥ 3 real
      baseline ledgers (not synthetic; not hand-edited)
- [ ] the live consumer accepts it via `--profile <path>` and
      emits a `GATE_MODE` event with `mode: "calibrated"` and the
      matching `profile_user_id`

Synthetic / hand-edited profiles do NOT satisfy G2. The spec is
explicit: all thresholds must flow from a calibrated profile, not
from manually-tuned defaults.

### G3. Replay ↔ live parity green

On the candidate commit:

- [ ] `pytest tests/test_physio_ledger.py tests/test_session_parity_matrix.py -q` → all pass
- [ ] `pytest tests/test_physio_live.py::TestProcessIndependence::test_live_session_round_trip_through_ledger -q` → pass
- [ ] one live-session ledger from the G1 triplet replays with
      `python -m neurophase.physio.session_replay <ledger> --strict --full-parity` → exit 0

### G4. Tools in CI

- [ ] `.github/workflows/ci.yml` runs `ruff check tools`, `ruff format --check ... tools`, and `mypy --strict tools` on every push and PR to `main`
- [ ] the **acceptance** suite (`pytest -m acceptance`) is a separate CI step
- [ ] the badge in `README.md` points to the live Actions status

### G5. Fresh-clone acceptance green

- [ ] `pytest -m acceptance` on the candidate commit exits 0
- [ ] the three sub-steps pass on a throwaway `git clone` of the
  same commit in a temp dir:
  - [ ] `python -m neurophase.physio.demo`
  - [ ] `python -m neurophase.physio.live` + `live_producer` + ledger → exit 0
  - [ ] `python -m neurophase.physio.session_replay --strict --full-parity`
  - [ ] `python tools/check_contract.py`

### G6. No open REAL_BUG

- [ ] `git log --grep="REAL_BUG" --since="rc tag date"` returns
      nothing unresolved
- [ ] no TODO / FIXME / HACK in source paths touched since rc
- [ ] `pytest tests/ -q` (default suite) → 0 failures, 0 unregistered xfails

### G7. Claims synced with evidence

Every claim in `CLAIMS.yaml` whose status was bumped since the
prior stable release must be backed by a citation that exists on
disk or on a public DOI resolver:

- [ ] `pytest tests/test_claim_registry.py tests/test_bibliography_contract.py -q` → pass
- [ ] every `supports: false` citation points at a real `results/*.json` OR a public reference
- [ ] no claim was moved UP the ladder without a new citation
- [ ] FMθ utility posture (C5) matches `docs/EEG_UTILITY_NEXT.md`:
      existence-positive, utility-null on ds003458, no new rescue
      analyses on that dataset

### G8. Decision-relevance benchmark — at least one operational result

Either a real operational effect OR a cleanly null verdict, recorded
on the operator's own live sessions:

- [ ] `benchmarks/decision_quality/<date>/` exists with at least one
      matched `gate_on` / `gate_off` pair per protocol in
      `benchmarks/decision_quality/PROTOCOL.md`
- [ ] `RESULTS.md` in that dir reports the raw per-pair numbers AND
      the verdict (positive / null / negative). No rescue stories.

If the operator has not yet completed a benchmark series, this gate
remains explicitly open. It **does not** silently pass.

---

## Running the gate

```bash
# 1. Parser + LSL + BLE acceptance (real hardware required for G1)
scripts/run_layer_d_acceptance.sh --session-label baseline-calm
scripts/run_layer_d_acceptance.sh --session-label focused-load
scripts/run_layer_d_acceptance.sh --session-label recovery

# 2. Calibration (G2)
python tools/calibrate_physio.py \
    --user-id <operator-id> \
    --out profiles/<operator-id>.json \
    --ledger artifacts/live/<date>/baseline-calm/ledger.jsonl \
    --ledger artifacts/live/<date>/focused-load/ledger.jsonl \
    --ledger artifacts/live/<date>/recovery/ledger.jsonl

# 3. Local CI-equivalent (G3, G4, G5, G6, G7)
ruff check neurophase tests tools
ruff format --check neurophase tests tools
python -m mypy --strict neurophase tools
pytest tests/ -q
pytest -m acceptance --no-cov
pytest tests/test_claim_registry.py tests/test_bibliography_contract.py -q

# 4. Decision-quality benchmark (G8)
# Operator-driven. See benchmarks/decision_quality/PROTOCOL.md.
```

## Promotion commit

Only after every box above is checked green:

```bash
git tag -a v1.2 -m "$(cat <<EOF
neurophase v1.2 — stable

G1 Layer-D triplet:      <link to artifacts>
G2 calibration profile:  profiles/<user_id>.json
G3 replay/live parity:   green on <commit>
G4 tools in CI:          green on <workflow run URL>
G5 fresh-clone:          green on <workflow run URL>
G6 no open REAL_BUG:     verified
G7 claims synced:        green on <commit>
G8 decision benchmark:   <verdict: positive / null / negative>

Everything else per STABLE_PROMOTION.md. No rescue language.
EOF
)"
git push origin v1.2
```

## Block / unblock

A single failing box blocks promotion. There is no "waive" button.
If a requirement is genuinely inapplicable (e.g. hardware breaks
mid-protocol), the runner captures the evidence and the rc cycle
continues with a `v1.2-rcN+1` tag. Do not promote while a gate is
open; do not promote against a red CI.
