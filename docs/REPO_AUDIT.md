# Director-level repo audit — 2026-04-13

Canonical snapshot of the neurophase repository's engineering,
security, governance, and operational posture at commit `3ae27f9`
(pre-`v1.2-rc3`). Used to justify the ops-hardening batch that
follows this audit in the same commit stream.

## Engineering — strong

| Surface | State | Evidence |
|---|---|---|
| Kernel LOC | ~57 k across `neurophase/`, `tools/`, `tests/` | `find … | xargs wc -l` |
| TODO / FIXME | **0** in source | `grep -rn TODO\|FIXME neurophase tools` |
| Static typing | `mypy --strict` green on kernel + tools | CI step |
| Lint | `ruff check` + `ruff format --check` clean | CI step |
| Tests | 1700 passed, 6 skipped, markers `hardware` / `acceptance` | `pytest tests/ -q` |
| Determinism | replay / live parity, SHA-256-chained market ledger, 0o600 physio ledger | multiple |
| CI surface | kernel + tests + tools covered; acceptance as separate step | `.github/workflows/ci.yml` |

## Governance — strong

| Artefact | Role |
|---|---|
| `CLAUDE.md` | canonical operating contract (phases, hard_rules, decision_logic, response_contract) |
| `CLAIMS.yaml` | scientific claim registry with promotion ladder |
| `INVARIANTS.yaml` | 26 machine-readable invariants with CI-bound tests |
| `STATE_MACHINE.yaml` | 8 exhaustive transitions |
| `docs/STABLE_PROMOTION.md` | 8-gate checklist for rc → stable |
| `docs/EEG_UTILITY_NEXT.md` | research-track separation; rescue-analysis ban |
| `docs/LIVE_SESSION_PROTOCOL.md` | Layer-D triplet protocol |
| `docs/CNS_PROTOCOL.md` | four-mode operator protocol |
| `benchmarks/decision_quality/PROTOCOL.md` | pre-committed decision-quality metrics |
| `tests/README.md` | domain taxonomy index |

## Operational gaps — CLOSED in this commit

| Gap | Fix shipped |
|---|---|
| No `LICENSE` file (README claims MIT — legal vacuum) | `LICENSE` (MIT) |
| No `SECURITY.md` | `SECURITY.md` (scope, report path, disclosure policy) |
| No `CONTRIBUTING.md` | `CONTRIBUTING.md` (pointers to CLAUDE.md, local gate) |
| No `CODEOWNERS` | `.github/CODEOWNERS` with protocol-level guards |
| No Dependabot | `.github/dependabot.yml` (pip + actions + tools/requirements) |
| No CodeQL | `.github/workflows/codeql.yml` (python, security-and-quality queries, weekly + per-push) |
| No pip-audit | `.github/workflows/security.yml` (`pip-audit --strict`, weekly + per-push) |
| No PR template | `.github/pull_request_template.md` (response_contract + hard_rules + gate-touch checklist) |
| No issue templates | `bug_report`, `feature_request`, `research_question`, `release_blocker` (+ `config.yml` routing) |
| No pre-commit config | `.pre-commit-config.yaml` (ruff + ruff-format + hygiene hooks + opt-in mypy) |
| No CI concurrency control | `concurrency` block in `ci.yml` cancels in-flight on the same ref |
| CI missing `permissions` | `permissions: contents: read` added (least-privilege default) |

## Ops-hardening follow-up — 2026-04-13 (next-layer batch)

Shipped in this follow-up commit, each a distinct control with a
single clear purpose and no overlap with the controls above:

| Control | File | Guarantee |
|---|---|---|
| PR dependency-delta review | `.github/workflows/dependency-review.yml` | Fails a PR that introduces a dependency with a known advisory (any severity). Complements `security.yml` (installed-env audit) by catching vulns *before* merge instead of after. |
| GitHub Actions lint | `.github/workflows/actionlint.yml` | Semantic validation of workflow YAML on every change under `.github/workflows/**` — expression typos, shell-injection risks, invalid job keys. |
| OSSF Scorecard | `.github/workflows/scorecard.yml` | Weekly supply-chain / repo-posture telemetry. Results published as SARIF into code-scanning and to the OpenSSF site. **Posture signal, not a security guarantee.** |
| Coverage gate | `pyproject.toml` (`[tool.coverage]`) + `.github/workflows/ci.yml` | `pytest --cov` now enforces `fail_under = 77`. Cross-env baseline on 2026-04-13: local Py 3.12 = 78.89%, CI Py 3.11/3.12 = 77.98%. Floor pinned to 77 (the observed CI minimum, not the optimistic local number). Emits `coverage.xml` as a CI artifact. Catches major test-surface erosion, not cosmetic drift. |
| Acceptance artifact publication | `.github/workflows/ci.yml` | Default + acceptance pytest runs emit `--junit-xml`; CI uploads `coverage.xml` + both junit files as `ci-artifacts-py{3.11,3.12}` (30-day retention). |

### Explicit non-goals in this batch
- **No SLSA provenance.** Not justified at current release cadence.
- **No hardware acceptance in CI.** `scripts/run_layer_d_acceptance.sh`
  remains operator-gated — it requires a real Polar H10 + BT adapter.
  Only the synthetic / LSL-loopback slice of the acceptance chain is
  automated (`pytest -m acceptance`). This boundary is honest and
  enforced.
- **No coverage badge.** `coverage.xml` is the machine-readable source
  of truth; external trend tracking remains deferred.
- **No universal action SHA pinning.** Only `ossf/scorecard-action` is
  SHA-pinned (elevated-permission workflow). Other actions stay on
  major tags managed by Dependabot; pinning everything would add
  friction without a concrete supply-chain signal.

## Operational gaps — still open (not closed by this commit)

These require the repo owner to flip a toggle in GitHub settings;
they cannot be expressed in repo content and CI workflows MUST NOT
pretend to enforce them.

| Gap | Status | Note |
|---|---|---|
| Branch-protection rules on `main` | **GitHub-UI only** | Configure in Settings → Branches / Rules: require `CI / quality`, `CodeQL / Analyze (python)`, `Security (pip-audit) / pip-audit`, `Dependency Review / Review PR dependency delta` to pass; require 1 review (from CODEOWNERS); require linear history; disallow force-pushes and deletions. |
| Secret scanning + push protection | **GitHub-UI only** | Public repo default since 2024; verify in Settings → Code security. |
| Private vulnerability reporting | **GitHub-UI only** | Settings → Security → Enable "Private vulnerability reporting". |
| Dependabot security updates | **GitHub-UI only** | Separate toggle from the scheduled Dependabot PRs already configured in `.github/dependabot.yml`. |
| Release automation (release-drafter) | not adopted | Manual tag workflow via `STABLE_PROMOTION.md` is the current source of truth. Can be added if release cadence picks up. |

## Security surface — score

Using a loose OSSF-Scorecard-adjacent view:

| Criterion | Before audit | After this commit |
|---|---|---|
| License present | ❌ | ✅ MIT |
| Security-policy present | ❌ | ✅ `SECURITY.md` |
| Vulnerability scan (CodeQL) | ❌ | ✅ per-push + weekly |
| Dependency scan (pip-audit) | ❌ | ✅ per-push + weekly, strict mode |
| Dependency auto-updates | ❌ | ✅ Dependabot weekly |
| CODEOWNERS review routing | ❌ | ✅ protocol files explicitly covered |
| CI triggered on PR | ✅ | ✅ |
| Minimal `permissions:` in workflows | ❌ (default) | ✅ `contents: read` + elevation only where needed |
| Concurrency controls | ❌ | ✅ cancel-in-flight |
| PR template enforcing quality contract | ❌ | ✅ `response_contract` + `hard_rules` + `STABLE_PROMOTION` gates |
| Issue templates routing structured reports | ❌ | ✅ 4 templates + `config.yml` contact-link routing |
| Pre-commit hygiene | ❌ | ✅ opt-in, CI parity |

## Next-step recommendations (repo-level only; not part of this commit)

Configure in GitHub settings (not expressible in repo files):

1. **Branch protection on `main`:**
   - Require pull-request reviews before merging.
   - Require status checks to pass: `CI / quality`, `CodeQL / Analyze`, `Security / pip-audit`.
   - Require branches to be up-to-date before merging.
   - Require linear history.
   - Require signed commits (optional but strongly recommended).
   - Disallow force-pushes and deletions.

2. **Enable GitHub features:**
   - Secret scanning + push protection (Public repo: free, on by default since 2024).
   - Private vulnerability reporting (`Security` tab).
   - Dependabot security updates (separate toggle; complements the scheduled Dependabot PRs).

3. **At v2 maturity:**
   - OSSF Scorecard action as a nightly.
   - SLSA provenance via `slsa-framework/slsa-github-generator`.
   - Coverage publication via codecov or shields coverage badge.

## Summary

Kernel + governance are strong. The operational / security surface
had concrete, closable gaps. This commit closes them to the extent
repo-content can close them; the remaining items are repo-settings
toggles the owner applies in the GitHub UI and are documented above
so they are not forgotten.
