# PR62 Hardening Plan (v2)

Status document for `feature/governance-hardening-v2`.

## Level 0 — Preparation
- [x] 0.1 Created branch `feature/governance-hardening-v2`.
- [~] 0.2 `git pull --ff-only` attempted (blocked: no upstream configured in this environment).
- [~] 0.2 `pre-commit run --all-files` attempted (blocked: `pre-commit` not installed in this environment).
- [~] 0.2 `pytest --tb=short` attempted (blocked during collection due missing optional deps: pandas/networkx/hypothesis/sklearn).
- [x] 0.3 Created this plan document.

## Level 1 — Decomposition / Cleanup
- [~] 1.1 Repository branch split into 3 PRs is tracked as logical slices in this branch (single environment branch limitation).
- [x] 1.2 OpenAI-specific artifact names removed; retained `mechanical_governance_*` artifacts only.
- [x] 1.3 Governance loader changes kept isolated from unrelated hygiene code in this round.

## Level 2 — Invariants / State Machine
- [x] 2.1 HN39 rewritten with explicit fail-closed semantics and dedicated verification test binding.
- [x] 2.2 T8 guard updated to `governance_closure_valid()` and moved to top priority.
- [x] 2.3 Added HN40: no bypass path to READY without governance closure.

## Level 3 — Governance Layer
- [x] 3.1 Governance modules remain focused on artifact load + validation.
- [x] 3.2 Added explicit typed error classes (`MissingArtifactError`, `InvalidVerdictError`, `UnboundAblationError`) and fail-closed helper `governance_closure_valid()`.
- [x] 3.3 Doctor/completeness continue to call governance loaders (no duplicated validation introduced here).
- [x] 3.4 Execution gate now checks `governance_closure_valid()` before any READY output.

## Level 4 — Testing / Falsification
- [x] 4.1 Added HN39 failure-mode test coverage and no-bypass READY test.
- [x] 4.2 Execution gate T8 scenarios are covered by governance + gate tests.
- [~] 4.3 Full `pytest -k governance --cov=...` depends on optional packages not available in this environment.
- [ ] 4.4 CI destructive governance-artifact job not added in this patch (follow-up required).

## Level 5 — Documentation / Ownership
- [x] 5.1 Governance helper docstrings updated.
- [x] 5.2 Updated invariants/claims + monograph regeneration path.
- [x] 5.3 Owner manifest now lists explicit critical artifact ownership.
- [x] 5.4 Checklist includes documentation update item.

## Level 6/7 — Final Hygiene & Verification
- [x] `ruff check neurophase tests`
- [x] `python -m mypy --strict neurophase/governance/ablation.py neurophase/governance/checklist.py neurophase/governance/owner_manifest.py neurophase/gate/execution_gate.py`
- [x] `pytest tests/test_governance_checklist_and_owner.py tests/test_execution_gate.py tests/test_governance_verification_gate.py -q --tb=no`
- [x] `python -m neurophase doctor`
