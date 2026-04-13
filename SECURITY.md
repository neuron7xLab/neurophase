# Security Policy

## Scope

**neurophase** is an experimental runtime framework. It is not a
medical device, a trading system, or a clinical instrument. Security
concerns relevant to this repository are therefore limited to:

* **Code-level vulnerabilities** — injection, deserialisation,
  arbitrary-file-write, unsafe defaults, secret leaks.
* **Supply-chain integrity** — dependency tampering, compromised
  GitHub Actions, malicious pull requests.
* **Data-provenance integrity** — the session ledger's tamper-
  evidence contract (SHA-256 chain) must not be silently weakened.
* **Privacy** — physiological-data handling must honour the file-
  permission floor (0o600) documented in
  `neurophase/physio/ledger.py::LEDGER_FILE_MODE`.

## Supported versions

Only the latest `main` and the most recent `v1.*-rc` tag receive
security fixes. Older rc tags are not patched.

## Reporting a vulnerability

**Do not** open a public issue for a security-sensitive report.

1. **Prefer GitHub Private Vulnerability Reporting:** on the repo's
   Security tab, click *Report a vulnerability*.
2. **Or email:** the contact address in the repo's *About* section,
   with `[neurophase-security]` in the subject.
3. Include: affected version/commit, reproduction steps, scope of
   impact (what does the issue let an attacker do?), and proof-of-
   concept if safe to share.
4. **Expected response window:** acknowledgement within 7 days.
   Initial triage within 14 days.

## Disclosure policy

* Once a fix is available, a coordinated disclosure advisory is
  published on the repo's Security tab, referencing the fixing
  commit and the affected versions.
* The reporter is credited in the advisory unless they request
  anonymity.
* Silent patches are not acceptable: every security fix is a
  traceable commit with a visible reason.

## What is NOT a security issue

* Null results on ds003458 or any other research dataset.
* Disagreements with `CLAIMS.yaml` posture or
  `docs/EEG_UTILITY_NEXT.md`.
* Feature requests for new hardware adapters.
* Performance regressions that do not escalate privileges or leak
  data.

These belong in regular issues, not security reports.

## Mechanical guardrails already in place

* `ruff check` + `ruff format --check` + `mypy --strict` on every
  push / PR (`.github/workflows/ci.yml`).
* CodeQL static analysis (`.github/workflows/codeql.yml`).
* Weekly `pip-audit` (`.github/workflows/security.yml`).
* Dependabot for dependencies + GitHub Actions
  (`.github/dependabot.yml`).
* Tamper-evident SHA-256 audit ledger
  (`neurophase/audit/decision_ledger.py`).
* Immutable JSONL session ledger with strict parity replay
  (`neurophase/physio/ledger.py` + `session_replay.py`).
* Physio ledger files are chmod 0o600 regardless of umask.
* `scripts/run_layer_d_acceptance.sh` emits a machine-readable
  `VERDICT.json` per run.

A fail-closed posture is a hard rule; see `CLAUDE.md` and
`docs/STABLE_PROMOTION.md`.
