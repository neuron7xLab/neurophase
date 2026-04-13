# Contributing to neurophase

Welcome. The short version: read `CLAUDE.md` (the canonical operating
protocol), then open a PR that satisfies it.

## The quick path

1. **Fork + clone.**
2. **Install dev + live extras:**
   ```bash
   pip install -e '.[dev,witness]'
   pip install -r tools/requirements.txt  # only if touching tools/
   ```
3. **Enable pre-commit** (optional but strongly recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```
   This gives you `ruff` + `ruff format` + trailing-whitespace +
   merge-conflict-marker checks at commit time, before CI touches
   anything.
4. **Branch:** `git checkout -b your-feature-name`.
5. **Code + test.**
6. **Run the local gate (mandatory before push):**
   ```bash
   ruff check neurophase tests tools
   ruff format --check neurophase tests tools
   python -m mypy --strict neurophase
   python -m mypy --strict tools
   pytest tests/ -q
   ```
   All four must be green. No "almost green". No "skipping mypy this
   time".
7. **Open a PR** against `main`. The PR template will prompt you for
   the checklist defined in `CLAUDE.md::response_contract`.

## The rules

They live in **`CLAUDE.md`** at the repo root. This is not decorative
— it is the operating contract for every commit. Non-negotiables:

* **No fake-live behavior.** Replay is replay. Live is live.
* **No claim inflation.** Existence is not utility.
* **No contract guessing.** If a constant / sentinel / exit code
  exists in code, match it exactly.
* **No silent drift.** Every external artifact that depends on a
  repo constant gets a mechanical guard so a future change fails
  CI instead of drifting silently.
* **No untested fixes.** Every REAL_BUG fix ships with a regression
  test.
* **No "probably".** If something was not run, say it was not run.

## Classifying findings

When you audit code or respond to a review comment, classify each
finding into EXACTLY one of:

| Class | What to do |
|---|---|
| `REAL_BUG` | fix + regression test |
| `MISSING_GUARD` | add invariant + test |
| `DOC_DEBT` | update docs / claims / help text |
| `NON_ISSUE` | explain briefly; no code churn |

Do not mix classes. Do not use vague labels.

## What belongs on `main`

* Kernel code + its tests.
* Operator-side tooling (`tools/`) + its tests.
* Results JSONs committed verbatim (null or positive; never cherry-
  picked).
* Docs that mirror validated reality (no "promising", no "suggests
  utility" when utility is not established).

## What does NOT belong on `main`

* Rescue analyses on `ds003458`. See `docs/EEG_UTILITY_NEXT.md`.
* Fake "production-ready" adapters. See `CLAUDE.md` hard_rule 4.
* Commits that weaken the CI gate or delete tests.
* Any code that raises the claim ladder without new evidence.

## PR review

The author is responsible for the content of the PR matching
`CLAUDE.md::response_contract`: **RESULT / FILES / TESTS / CONTRACTS
/ LIMITATIONS / REPRO**. Reviews focus on:

1. does CI pass? (blocking)
2. does the PR respect the 4 forbidden classes above? (blocking)
3. does the change preserve shared-core semantics? (blocking)
4. is the commit message truthful about what was / was not done?
5. can a reviewer reproduce the result from the `REPRO` section?

## Stable release gating

The promotion of any rc to a stable tag is mechanical, not rhetorical.
See `docs/STABLE_PROMOTION.md` for the 8-gate checklist. No waivers.

## Questions / proposals

Open an issue using the right template:

* **Bug report** — something is broken or a guard is missing.
* **Feature request** — new capability; state scope + non-goals.
* **Research question** — a new-dataset or methodology proposal;
  must align with `docs/EEG_UTILITY_NEXT.md` paths.
* **Release blocker** — something that must be fixed before the
  next stable tag; must reference `docs/STABLE_PROMOTION.md` gate.

Thank you for the care. Ship truth.
