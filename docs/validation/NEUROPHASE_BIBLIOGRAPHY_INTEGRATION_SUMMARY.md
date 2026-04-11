# NEUROPHASE BIBLIOGRAPHY — INTEGRATION SUMMARY

**Date:** 2026-04-11  
**Status:** ✓ PRODUCTION-READY FOR MERGE

---

## QUICK STATUS

| Component | Status | Count | Notes |
|---|---|---:|---|
| Sources | ✓ | 24 | S-tier 7 + A-tier 10 + B-tier 7 |
| DOI Anchors | ✓ | 16 | Valid `10.x/...` anchors |
| Evidence Labels | ✓ | 4 | Established / Strongly Plausible / Tentative / Unsupported |
| Module Traceability | ✓ | 6 | Core claim → module → test matrix |
| Validation Gates | ✓ | 24 | `final_diff_validation.py` checks |
| Integration Gates | ✓ | 9 | functional + typing + lint + docs governance |
| CI/CD Hooks | ✓ | 4 | audit + DOI + final diff + enforcer |
| Type Safety Updates | ✓ | 2 | `asymmetry.py`, `ricci.py` |

---

## CURRENT ENFORCEMENT STACK

1. `python docs/audit_evidence_labels.py`
2. `python docs/validate_bibliography_dois.py`
3. `python docs/final_diff_validation.py`
4. `python docs/evidence_enforcer.py`

CI blocks merge on any failure.

---

## KEY DELIVERABLES

- Elite bibliography: `docs/theory/neurophase_elite_bibliography.md`
- Compact companion: `docs/theory/hierarchical_status_bibliography.md`
- Integration protocol: `docs/validation/integration_readiness_protocol.md`
- Evidence labeling policy: `docs/validation/evidence_labeling_style_guide.md`
- Final sign-off: `docs/validation/FINAL_PRODUCTION_SIGNOFF_2026-04-11.md`
- Merge sign-off template: `docs/validation/MERGE_SIGN_OFF_TEMPLATE.txt`
- Maintenance seed: `docs/maintenance/2026_q2_calibration.md`

---

## POST-MERGE REQUIRED ACTION

For all **Tentative** claims used in empirical evaluation:
1. preregister protocol (OSF or equivalent),
2. store registration link/DOI in docs,
3. mark outcome as confirmatory or exploratory.

This is mandatory to reduce p-hacking risk.
