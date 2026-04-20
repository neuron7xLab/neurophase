# OpenAI GPT-2026 Audit — 2026-04-19

## Status Summary

- Source checklist file: `openai_gpt_2026_checklist_2026-04-19.yaml`
- Required source items: **211**
- Current non-pass items (`fail`/`partial`): **0**
- Governance checklist verdict: **DONE**

## Critical Closures

- `gov_2` — claim_status_applied: **pass**
- `gov_5` — artifact_owner_declared: **pass**
- `final_1` — ablation policy: **pass**
- `final_2` — mutation proof binding: **pass**

## Closed-item table

| Item | Status | Evidence pointer |
|---|---|---|
| gov_2 | pass | `neurophase/governance/checklist.py::load_checklist` |
| gov_5 | pass | `neurophase/governance/owner_manifest.py::load_owner_manifest` |
| final_1 | pass | `ABLATION_POLICY.yaml::critical_elements` |
| final_2 | pass | `ABLATION_POLICY.yaml::test_registry` |

## Follow-up hardening evidence — 2026-04-20

- `neurophase/reset/calibrator.py` no longer calls
  `importlib.util.find_spec("sklearn.linear_model")`; guarded import now
  degrades to default weights when sklearn is absent.
- `neurophase/governance/ablation.py` now parses mutation suites with `ast.parse`
  and validates `test_registry` bindings against concrete `ast.FunctionDef`
  nodes, rejecting comment-string false positives.
- Formatting conformance restored for governance loaders/tests under
  `ruff format --check`.
