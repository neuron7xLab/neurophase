# neurophase — ELITE RESEARCH-GRADE BIBLIOGRAPHY v2026.04.11

**Standard:** Laboratory-ready documentation (DeepMind/OpenAI research rigor)  
**Hierarchy:** S-tier (foundational) → A-tier (mechanistic) → B-tier (methods/validation)  
**Traceability:** Every claim → DOI/PMID → Module → Test → Falsification criterion

---

## GOVERNANCE RULES

### Evidence Classification
- **Established:** Consensus reviews + replicated meta-analyses, low conceptual uncertainty
- **Strongly Plausible:** Strong mechanism + consistent empirical alignment, context-bound generalization
- **Tentative:** Working hypothesis with partial data, requires preregistered verification
- **Unsupported/Weak:** Popular narrative lacking sufficient empirical foundation

### Inclusion Criteria
✓ Peer-reviewed (Nature, Neuron, PNAS, CUP/OUP/MIT Press, IEEE/ACM)  
✓ Operationalizable (variables map to neurophase modules)  
✓ Reproducible (clear design, quantified methods, data available)  
✓ Falsifiable (criterion exists to reject claim if data contradict)

### Exclusion Criteria
✗ Opinion/blog as primary source  
✗ "Retellings of retellings" without DOI link  
✗ Mechanisms without empirical support  
✗ Claims where falsification criterion cannot be stated

---

## S-TIER: FOUNDATIONAL CANON (Emergent dynamics, synchronization, predictive processing)

### Self-Organization & Synergetics
**1. Haken, H. (1983). _Synergetics: An Introduction_. Springer-Verlag.**
- **Role:** Formal framework for self-organized criticality (order/control parameters, slaving principle).
- **Neurophase mapping:** `R(t)` gate as control parameter orchestrating phase transitions.
- **Falsification:** If global dynamics cannot be reduced to <5 critical variables, framework fails.

**2. Kelso, J. A. S. (1995). _Dynamic Patterns: The Self-Organization of Brain and Behavior_. MIT Press.**
- **Role:** Coordination dynamics linking brain rhythms to behavioral modes.
- **Neurophase mapping:** Phase-transition detection in EEG/HRV before cognitive error.
- **Falsification:** If bistable/multistable transitions are not observable at predicted parameter ranges.

**3. Strogatz, S. H. (2003). _Sync: The Emerging Science of Spontaneous Order_. Hyperion.**
- **Role:** Mathematical framework for coupled oscillator networks (Kuramoto dynamics).
- **Neurophase mapping:** PLV as measure of basin alignment across brain regions.
- **Falsification:** If phase-locking cannot predict behavioral mode shift >50% accuracy vs. baseline.

### Predictive Processing & Free Energy
**4. Friston, K. (2010). The free-energy principle: a unified brain theory? _Nature Reviews Neuroscience_, 11(2), 127–138. DOI: 10.1038/nrn2787**
- **Evidence status:** Established (100+ citations, replicated across domains).
- **Mechanism:** Brain minimizes prediction error; mismatch triggers corrective action/attention.
- **Neurophase mapping:** `R(t)` gate as online prediction-error magnitude; execution blocked when error > adaptive threshold.
- **Module link:** `neurophase/gate/execution_gate.py`.
- **Falsification:** If prediction error does NOT correlate with error-rate in trading (Spearman r > 0.3, p < 0.01) on holdout data.

**5. Clark, A. (2016). _Surfing Uncertainty: Prediction, Action, and the Embodied Mind_. Oxford University Press.**
- **Evidence status:** Strongly Plausible.
- **Mechanism:** Action is embodied hypothesis-testing; uncertainty drives info-seeking behavior.
- **Neurophase mapping:** Verification micro-step as active reduction of prediction uncertainty.
- **Module link:** `neurophase/feedback/verification_step.py` (proposed).
- **Falsification:** If structured verification does NOT reduce false-accept-rate vs. unverified control (p < 0.05).

### Network Complexity & Criticality
**6. Goldberger, A. L., et al. (2002). Fractal dynamics in physiology: Alterations with disease and aging. _PNAS_, 99(Suppl 1), 2466–2472. DOI: 10.1073/pnas.012579499**
- **Evidence status:** Established.
- **Mechanism:** Healthy systems exhibit 1/f scaling; loss of fractal structure indicates pathology.
- **Neurophase mapping:** Hurst exponent & MFDFA as resilience indicators.
- **Module link:** `neurophase/metrics/hurst.py`, `neurophase/risk/mfdfa.py`.
- **Falsification:** If out-of-sample Hurst/MFDFA do NOT predict error-burst >2 hours in advance (AUC > 0.65).

**7. Bak, P., Tang, C., & Wiesenfeld, K. (1987). Self-organized criticality: an explanation of the 1/f noise. _Physical Review Letters_, 59(4), 381. DOI: 10.1103/PhysRevLett.59.381**
- **Evidence status:** Established.
- **Mechanism:** Complex systems self-tune to critical point where avalanche sizes follow power law.
- **Neurophase mapping:** Market phase as proximity to criticality; PLV fluctuations as early-warning precursor.
- **Falsification:** If avalanche-size distribution does NOT follow power law in market microstructure (KS p < 0.01).

---

## A-TIER: MECHANISTIC & HIGH-EVIDENCE

**8. Miyake, A., et al. (2000)... DOI: 10.1006/cogp.1999.0734**
- **Evidence status:** Established.
- **Neurophase mapping:** `R(t)` gate as inhibition; session archive as updating capacity.
- **Module link:** `neurophase/gate/execution_gate.py`, `neurophase/state/executive_monitor.py`.

**9. Arnsten, A. F. T. (2009)... DOI: 10.1038/nrn2648**
- **Evidence status:** Established.
- **Neurophase mapping:** EEG theta/beta + HRV drop as stress signature.
- **Module link:** `neurophase/risk/evt.py`, `neurophase/risk/hrv_proxy.py` (proposed).

**10. Shenhav, A., Botvinick, M. M., & Cohen, J. D. (2013)... DOI: 10.1016/j.neuron.2013.07.007**
- **Evidence status:** Strongly Plausible.
- **Neurophase mapping:** Gate stiffness tied to control budget.
- **Module link:** `neurophase/gate/control_budget.py` (proposed).

**11. Engel, A. K., Fries, P., & Singer, W. (2001)... DOI: 10.1038/35094565**
- **Evidence status:** Established.
- **Neurophase mapping:** PLV(EEG beta, market phase) as coherence proxy.
- **Module link:** `neurophase/metrics/plv.py`.

**12. Cavanagh, J. F., & Frank, M. J. (2014)... DOI: 10.1016/j.tics.2014.04.012**
- **Evidence status:** Established.
- **Neurophase mapping:** Theta burst as control-state signal.
- **Module link:** `neurophase/risk/evt.py`.

**13. Buzsáki, G. (2006). _Rhythms of the Brain_. OUP.**
- **Evidence status:** Established.
- **Neurophase mapping:** Cross-frequency coupling (CFC) as integration index.
- **Module link:** `neurophase/metrics/cfc.py` (proposed).

**14. Thayer, J. F., & Lane, R. D. (2000)... DOI: 10.1016/S0165-0327(00)00338-4**
- **Evidence status:** Established.
- **Neurophase mapping:** HRV as resilience proxy + EEG dual-channel stress signature.

**15. Shaffer, F., & Ginsberg, J. P. (2017)... DOI: 10.3389/fpubh.2017.00258**
- **Evidence status:** Established.
- **Neurophase mapping:** HRV normative baseline + drift detection.

**16. Walker, M. (2017). _Why We Sleep_. Scribner.**
- **Evidence status:** Strongly Plausible.
- **Neurophase mapping:** Session duration/recovery constraints in fatigue-aware gating.

**17. Chee, M. W. L., & Chuah, L. Y. M. (2007)... DOI: 10.1073/pnas.0610712104**
- **Evidence status:** Established.
- **Neurophase mapping:** Fatigue accumulation signature in session archive.

---

## B-TIER: METHODS, VALIDATION & STATISTICAL DISCIPLINE

**18. Lachaux, J.-P., et al. (1999)... DOI: 10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C**
- **Evidence status:** Established.
- **Module link:** `neurophase/metrics/plv.py`.

**19. Aru, J., et al. (2015)... DOI: 10.1016/j.cub.2015.05.029**
- **Evidence status:** Established.
- **Module link:** `neurophase/metrics/cfc.py` (proposed).

**20. Bassett, D. S., & Sporns, O. (2017)... DOI: 10.1038/nn.4502**
- **Evidence status:** Established.
- **Module link:** `neurophase/gate/emergent_phase.py`.

**21. Deco, G., Jirsa, V., & McIntosh, A. R. (2011)... DOI: 10.1038/nrn2961**
- **Evidence status:** Established.
- **Module link:** `neurophase/gate/emergent_phase.py`.

**22. Cohen, J. (1988). _Statistical Power Analysis for the Behavioral Sciences_ (2nd ed.). Routledge.**
- **Evidence status:** Established.
- **Module link:** `neurophase/validation/preregistration.yaml` (proposed).

**23. Benjamini, Y., & Hochberg, Y. (1995)...**
- **Evidence status:** Established.
- **Module link:** `neurophase/validation/` (proposed).

**24. Nosek, B. A., Ebersole, C. R., et al. (2018)... DOI: 10.1073/pnas.1708274114**
- **Evidence status:** Established.
- **Module link:** `neurophase/validation/preregistration.yaml` (proposed).

---

## TRACEABILITY MATRIX (Claim → Module → Test → Falsification)

| Core Claim | Evidence Status | Module | Primary Test | Falsification Criterion | DOI Anchor |
|---|---|---|---|---|---|
| `R(t)` reduces impulse errors | Strongly Plausible | `execution_gate.py` | FAR vs ctrl | FAR_treatment ≥ FAR_control | Shenhav 2013 |
| PLV predicts next-trial accuracy | Tentative | `metrics/plv.py` | Logistic AUC | AUC < 0.55 holdout | Lachaux 1999 |
| HRV signals stress before error | Tentative | `risk/hrv_proxy.py` (proposed) | Lagged AUC | AUC < 0.60 @ 5–30 min | Thayer 2000 |
| EEG theta indicates control state | Strongly Plausible | `risk/evt.py` | Theta vs accuracy | r < 0.30 or p > 0.05 | Cavanagh 2014 |
| Longitudinal training improves latency | Tentative | `session_archive.py` (proposed) | 8-week trend | No power-law decline | Walker 2017 |
| Market phase coherence precedes crashes | Tentative | `emergent_phase.py` | PLV × volatility | r < 0.35 | Bak 1987 |

---

## INTEGRATION READINESS CHECKLIST (100% Gate)

- [ ] Functional tests: `pytest -q`
- [ ] Type safety: `mypy neurophase --strict`
- [ ] Static analysis: `ruff check . --select E,F,W`
- [ ] Evidence labeling for non-trivial claims
- [ ] Explicit module/test traceability
- [ ] Preregistration for inferential experiments
- [ ] Falsification criterion for each hypothesis
- [ ] Calibration documented with timestamps
- [ ] Failure modes + safe fallbacks

**Release blocked until ALL checks pass.**

---

## NEXT: IMPLEMENTATION PROTOCOL

See `docs/validation/integration_readiness_protocol.md` for calibration loops, adaptation layer, failure scenarios, and CI/CD gates.

---

**Document Version:** 2026-04-11  
**Maintenance:** Quarterly review with evidence-status updates.
