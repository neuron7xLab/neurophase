# neurophase — research-grade bibliography

*Canonical, DOI-annotated, evidence-labelled source list for every
theoretical claim any part of the `neurophase` codebase makes.
Load-bearing. Governed by the Evolution Board doctrine in
[`docs/EVOLUTION_BOARD.md`](../EVOLUTION_BOARD.md) and cross-referenced
from [`scientific_basis.md`](scientific_basis.md),
[`neurophase_scientific_basis.md`](neurophase_scientific_basis.md),
and the compact companion
[`hierarchical_status_bibliography.md`](hierarchical_status_bibliography.md).*

---

## 1. Governance rules

### 1.1 Evidence classification (four-tier taxonomy)

* **Established** — consensus reviews + replicated meta-analyses,
  low conceptual uncertainty, safe to build product decisions on.
* **Strongly Plausible** — strong mechanism + consistent empirical
  alignment, bounded to the context in which it was measured;
  requires caution when generalising.
* **Tentative** — working hypothesis with partial data; requires a
  preregistered verification before any product claim can rest on it.
* **Unsupported / Weak** — popular narrative without sufficient
  empirical support; **never** a primary justification for a product
  decision.

Formal definitions and in-docs usage rules live in
[`docs/validation/evidence_labeling_style_guide.md`](../validation/evidence_labeling_style_guide.md).

### 1.2 Inclusion criteria (every entry must satisfy all five)

1. **Primary source** — peer-reviewed article, academic monograph,
   or official standard; not a blog post, not a retelling.
2. **Institutional weight** — Nature / Neuron / PNAS / Trends / IEEE /
   ACM / Cambridge / Oxford / MIT Press / Springer.
3. **Operationalisable** — the variables in the source map onto at
   least one `neurophase` module or contract.
4. **Reproducible** — clear design, quantified methods, testable
   claim.
5. **Falsifiable** — each claim admits a criterion that would
   mechanically reject it if the data contradicted it.

### 1.3 Exclusion criteria (automatic rejection)

* Opinion / blog as a **primary** source for a product claim.
* "Retelling of a retelling" without a DOI anchor.
* Mechanisms without empirical support.
* Claims for which no falsification criterion can be stated.
* **Fabricated / future-dated citations.** Any draft that references
  a future year before it has occurred, or a non-existent report,
  is a contract violation and must be removed on sight.

---

## 2. Hierarchy of status

| Tier | Name | Role in the system |
|---|---|---|
| **S** | Foundational canon | Ontology and base laws of dynamics — emergent self-organisation, synchronisation, predictive processing. |
| **A** | Mechanistic & high-evidence | Binds theory to concrete cognitive-control mechanisms (executive function, stress, rhythms, autonomic regulation). |
| **B** | Method & validation | Supplies the computable metrics, surrogate protocols, and statistical hygiene that make any of the above testable. |

---

## 3. S-tier — foundational canon

### Self-organisation and synergetics

**1. Haken, H. (1983).** *Synergetics: An Introduction.* Springer-Verlag.
- **Status.** Established.
- **Role.** Formal framework for self-organised criticality (order parameters, control parameters, slaving principle).
- **`neurophase` mapping.** `R(t)` as a control parameter orchestrating phase transitions; `ExecutionGate` as the permission boundary that emerges from those parameters.

**2. Kelso, J. A. S. (1995).** *Dynamic Patterns: The Self-Organization of Brain and Behavior.* MIT Press.
- **Status.** Established.
- **Role.** Coordination dynamics linking brain rhythms to behavioural modes; bistability and phase-transition topology.
- **`neurophase` mapping.** `CoupledBrainMarketSystem` phase transitions; `StillnessDetector` rejecting biased-lock regimes.

**3. Strogatz, S. H. (2003).** *Sync: The Emerging Science of Spontaneous Order.* Hyperion.
- **Status.** Established.
- **Role.** Canonical exposition of coupled-oscillator synchronisation.
- **`neurophase` mapping.** Kuramoto coupling in `neurophase.sync.coupled_brain_market`.

### Predictive processing and free energy

**4. Friston, K. (2010).** "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, **11**(2), 127–138. DOI: `10.1038/nrn2787`
- **Status.** Established.
- **Mechanism.** The brain minimises variational free energy; prediction-error signals drive action and attention.
- **`neurophase` mapping.** `R(t)` as an engineered proxy for prediction-error pressure; `F_proxy = ½·δ²` in `StillnessDetector` is a **geometric surrogate**, not the full variational functional (honest-naming contract HN1).
- **Modules.** `neurophase/gate/execution_gate.py`, `neurophase/gate/stillness_detector.py`.
- **Falsification.** If `δ(t)` from `PredictionErrorMonitor` does not correlate with error-rate in held-out sessions (Spearman *r* < 0.3, *p* > 0.01), the mapping is rejected.

**5. Clark, A. (2013).** "Whatever next? Predictive brains, situated agents, and the future of cognitive science." *Behavioral and Brain Sciences*, **36**(3), 181–204. DOI: `10.1017/S0140525X12000477`
- **Status.** Established.
- **Role.** Canonical formulation of predictive processing as a bridge between perception, action, and cognition.

**6. Clark, A. (2016).** *Surfing Uncertainty: Prediction, Action, and the Embodied Mind.* Oxford University Press.
- **Status.** Strongly Plausible (monograph consolidating a growing literature).
- **Role.** Action as embodied hypothesis-testing; uncertainty drives information-seeking behaviour.
- **`neurophase` mapping.** `VerificationStep` in `ExecutiveMonitor` as active reduction of prediction uncertainty before high-impact action.

### Network complexity and criticality

**7. Goldberger, A. L., et al. (2002).** "Fractal dynamics in physiology: Alterations with disease and aging." *PNAS*, **99**(Suppl 1), 2466–2472. DOI: `10.1073/pnas.012579499`
- **Status.** Established.
- **Mechanism.** Healthy physiological systems exhibit 1/f scaling; loss of fractal structure is a pathology signature.
- **`neurophase` mapping.** Hurst exponent and MFDFA as resilience diagnostics.
- **Modules.** `neurophase/metrics/hurst.py`, `neurophase/risk/mfdfa.py`.

**8. Bak, P., Tang, C., & Wiesenfeld, K. (1987).** "Self-organized criticality: an explanation of the 1/f noise." *Physical Review Letters*, **59**(4), 381. DOI: `10.1103/PhysRevLett.59.381`
- **Status.** Established.
- **Mechanism.** Complex systems self-tune to a critical point where avalanche sizes follow a power law.
- **`neurophase` mapping.** Market-phase proximity to criticality; PLV fluctuations as an early-warning precursor in the emergent-phase detector.

---

## 4. A-tier — mechanistic and high-evidence

**9. Miyake, A., et al. (2000).** "The unity and diversity of executive functions and their contributions to complex frontal-lobe tasks." *Cognitive Psychology*, **41**(1), 49–100. DOI: `10.1006/cogp.1999.0734`
- **Status.** Established.
- **Role.** Decomposes executive function into inhibition, updating, and shifting.
- **`neurophase` mapping.** `R(t)` gate ≈ inhibition; `ExecutiveMonitor` ≈ updating capacity under load.

**10. Arnsten, A. F. T. (2009).** "Stress signalling pathways that impair prefrontal cortex structure and function." *Nature Reviews Neuroscience*, **10**(6), 410–422. DOI: `10.1038/nrn2648`
- **Status.** Established.
- **Mechanism.** Stress catecholamines impair PFC-dependent executive control.
- **`neurophase` mapping.** EEG β / HRV drop as the stress-overload signature detected by `ExecutiveMonitor`.

**11. Shenhav, A., Botvinick, M. M., & Cohen, J. D. (2013).** "The expected value of control." *Neuron*, **79**(2), 217–240. DOI: `10.1016/j.neuron.2013.07.007`
- **Status.** Strongly Plausible.
- **Mechanism.** Cognitive control is allocated to maximise expected value; effort cost is real.
- **`neurophase` mapping.** Gate-stiffness calibration weighed against over-blocking cost — the Youden-J objective inside `calibrate_gate_threshold`.

**12. Engel, A. K., Fries, P., & Singer, W. (2001).** "Dynamic predictions: oscillations and synchrony in top-down processing." *Nature Reviews Neuroscience*, **2**(10), 704–716. DOI: `10.1038/35094565`
- **Status.** Established.
- **Role.** Functional significance of neural synchronisation.
- **`neurophase` mapping.** PLV(EEG β, market phase) as the principal coherence metric in `neurophase.metrics.plv`.

**13. Cavanagh, J. F., & Frank, M. J. (2014).** "Frontal theta as a mechanism for cognitive control." *Trends in Cognitive Sciences*, **18**(8), 414–421. DOI: `10.1016/j.tics.2014.04.012`
- **Status.** Established.
- **Mechanism.** Frontal theta bursts mark real-time control-state inference and conflict monitoring.
- **`neurophase` mapping.** Theta-band input into `ExecutiveMonitor` overload estimation.

**14. Buzsáki, G. (2006).** *Rhythms of the Brain.* Oxford University Press.
- **Status.** Established.
- **Role.** Canonical text on neural rhythms and the temporal organisation of brain computation.
- **`neurophase` mapping.** Multi-band orchestration of brain-side oscillators in `CoupledBrainMarketSystem`.

**15. Thayer, J. F., & Lane, R. D. (2000).** "A model of neurovisceral integration in emotion regulation and dysregulation." *Journal of Affective Disorders*, **61**(3), 201–216. DOI: `10.1016/S0165-0327(00)00338-4`
- **Status.** Established.
- **Mechanism.** Autonomic regulation and top-down cognitive control are functionally coupled.
- **`neurophase` mapping.** HRV as a second input channel into `ExecutiveMonitor` alongside EEG β and error-burst context.

**16. Shaffer, F., & Ginsberg, J. P. (2017).** "An overview of heart rate variability metrics and norms." *Frontiers in Public Health*, **5**, 258. DOI: `10.3389/fpubh.2017.00258`
- **Status.** Established.
- **Role.** Practical glossary of HRV metrics (RMSSD / HF / SDNN) and their normative ranges.
- **`neurophase` mapping.** Stable reference for HRV normalisation in `ExecutiveMonitor`.

**17. Helfrich, R. F., et al. (2018).** "Neural mechanisms of sustained attention are rhythmic." *Neuron*, **99**(4), 854–865. DOI: `10.1016/j.neuron.2018.07.032`
- **Status.** Established.
- **Mechanism.** Attention itself is rhythmically gated, not a continuous process.
- **`neurophase` mapping.** Justifies temporal sampling in the `StreamingPipeline` rather than continuous polling.

---

## 5. B-tier — methods, validation, statistical discipline

**18. Lachaux, J.-P., et al. (1999).** "Measuring phase synchrony in brain signals." *Human Brain Mapping*, **8**(4), 194–208. DOI: `10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C`
- **Status.** Established.
- **Role.** Canonical formalisation of the phase-locking value (PLV) and the cyclic-shift surrogate test.
- **`neurophase` mapping.** Exact algorithm implemented in `neurophase/metrics/plv.py`; surrogate in `neurophase/validation/surrogates.py::cyclic_shift`.

**19. Bassett, D. S., & Sporns, O. (2017).** "Network neuroscience." *Nature Neuroscience*, **20**(3), 353–364. DOI: `10.1038/nn.4502`
- **Status.** Established.
- **Role.** The brain as a dynamic graph; local signals integrated into global network states.
- **`neurophase` mapping.** Conceptual basis for reducing brain-side oscillators to their mean-field order parameter.

**20. Deco, G., Jirsa, V., & McIntosh, A. R. (2011).** "Emerging concepts for the dynamical organization of resting-state activity in the brain." *Nature Reviews Neuroscience*, **12**(1), 43–56. DOI: `10.1038/nrn2961`
- **Status.** Established.
- **Role.** Macro-level brain-state dynamics as a state-space interpretation of risk and stability regimes.
- **`neurophase` mapping.** State-machine semantics of the 5-state gate (READY / BLOCKED / UNNECESSARY / SENSOR_ABSENT / DEGRADED).

**21. Phipson, B., & Smyth, G. K. (2010).** "Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn." *Statistical Applications in Genetics and Molecular Biology*, **9**(1), Article 39. DOI: `10.2202/1544-6115.1585`
- **Status.** Established.
- **Role.** Derives the `(1 + k) / (1 + n)` smoothed p-value estimator used to avoid `p = 0` for finite permutation counts.
- **`neurophase` mapping.** Directly implemented in `neurophase/validation/null_model.py::NullModelHarness.test`; bound to honest-naming contract HN3.

**22. Theiler, J., et al. (1992).** "Testing for nonlinearity in time series: the method of surrogate data." *Physica D*, **58**(1–4), 77–94. DOI: `10.1016/0167-2789(92)90102-S`
- **Status.** Established.
- **Role.** Original formulation of phase-randomisation (IAAFT) surrogate tests.
- **`neurophase` mapping.** `neurophase/validation/surrogates.py::phase_shuffle`.

**23. Cohen, J. (1988).** *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Routledge.
- **Status.** Established.
- **Role.** Power planning and effect-size discipline before experiment execution.
- **`neurophase` mapping.** Preregistration requirement in
  [`docs/validation/integration_readiness_protocol.md`](../validation/integration_readiness_protocol.md).

**24. Benjamini, Y., & Hochberg, Y. (1995).** "Controlling the false discovery rate: a practical and powerful approach to multiple testing." *Journal of the Royal Statistical Society. Series B*, **57**(1), 289–300.
- **Status.** Established.
- **Role.** Canonical FDR control for multiple-comparison settings.
- **`neurophase` mapping.** Required whenever the validation harness reports ≥ 2 simultaneous tests.

---

## 6. Traceability matrix (claim → module → test → falsification)

| Core claim | Status | Module | Test or binding | Falsification criterion |
|---|---|---|---|---|
| `R(t) < θ` blocks execution (I₁) | Established | `neurophase/gate/execution_gate.py` | `tests/test_execution_gate.py::test_blocks_below_threshold` | A permissive decision with `R < θ` constructible → contract violation. |
| Non-VALID `time_quality` → `DEGRADED` (B₁) | Established | `neurophase/data/temporal_validator.py` | `tests/test_temporal_validator.py::TestGateIntegration` | Gate emits any non-`DEGRADED` state on a `GAPPED` / `STALE` / `REVERSED` input. |
| Stillness ⇒ `UNNECESSARY` (I₄) | Strongly Plausible | `neurophase/gate/stillness_detector.py` | `tests/test_stillness_detector.py::TestRequiredStillCriterion` | Window-wide criterion fails on the three worked counter-examples in `stillness_invariant.md`. |
| PLV rejects an independent pair under cyclic-shift | Established | `neurophase/metrics/plv.py`, `neurophase/validation/null_model.py` | `tests/test_plv_held_out.py::TestHarnessDelegation` | Rejection rate on H1 null ≥ α across seeds. |
| H1 endpoint ground truth: `c = 1 ⇒ PLV = 1` exactly | Established | `neurophase/benchmarks/phase_coupling.py` | `tests/test_benchmarks_phase_coupling.py::TestGroundTruthEndpoints` | Any seed producing PLV ≠ 1 exactly for `c = 1`. |
| D1 calibration report is OOS-validated, not post-hoc | Established | `neurophase/calibration/threshold.py` | `tests/test_calibration_threshold.py::TestCalibrationEndToEnd` | Best threshold selected on test split, or `generalization_gap` ≠ `train − test`. |
| Full pipeline replay is bit-deterministic | Established | `neurophase/runtime/pipeline.py`, `neurophase/audit/decision_ledger.py` | `tests/test_determinism_certification.py::TestEndToEndReplay` | Two runs with identical config produce divergent ledger bytes. |
| `R(t)` reduces impulse errors in human-in-the-loop traders | Strongly Plausible → Tentative (context-bound) | `neurophase/state/executive_monitor.py` | *Preregistration required* | FAR(gate on) ≥ FAR(gate off) on held-out sessions. |
| Personal resilience signature via longitudinal training | Tentative | *Session-archive contract (proposed)* | *Longitudinal study required* | No power-law decline in `decision_latency` under bounded accuracy. |
| Lock-in detection via weighted metrics combination | Tentative | *NeuroPhase KLR calibration protocol (2026-04-12)* | *Synthetic-archive calibration; external replication required* | AUC thresholding on weighted lock-in score with chronological validation. Contract: `docs/theory/klr_reset_contract.md`. Preregistration: OSF DOI pending publication. |
| Capital-weighted Kuramoto flags crowding | Tentative | Related: `neurophase/intel/btc_field_order.py` | *Preregistered validation required* | AUC vs. realised-drawdown < 0.60 OOS. |

The last three rows are deliberately labelled **Tentative** — they
are working hypotheses, not evidence. Any future PR that tries to
lift them to `Established` or `Strongly Plausible` must first point
at a preregistered study with OOS validation.

---

## 7. INTEGRATION READINESS CHECKLIST

These are the machine-checkable items that gate every merge into
`main`. The A1 CI meta-test (`tests/test_invariants_registry.py`)
enforces the bindings and the A2 state-machine meta-test
(`tests/test_state_machine_spec.py`) enforces the transitions.

- [ ] Every invariant `I₁`–`I₄` and `B₁` bound to ≥ 1 passing test.
- [ ] Every honest-naming contract `HN1`–`HN15` bound to ≥ 1 passing test.
- [ ] Every product-critical claim carries an explicit evidence label.
- [ ] Every `Tentative` claim carries either a preregistration link
  or a deferred-validation note — never a silent promotion to
  `Established`.
- [ ] `pytest` passes on the full test suite.
- [ ] `ruff check` + `ruff format --check` clean.
- [ ] `mypy --strict` clean on all source files.
- [ ] `tests/test_determinism_certification.py` (F3) passes on
  Python 3.11 **and** 3.12 in CI.
- [ ] No fabricated future-dated citations anywhere in `docs/`.

No promotional "24/24 checks passed" theatrics: the numbers are
whatever the CI run actually reports. The only artefact that matters
is a green run on `main`.

---

## 8. Honest-naming addendum specific to this document

* **No fabricated future citations.** Earlier drafts of `neurophase` docs contained the fabricated / fake / removed entries "Friston/Clark (2026)", "Ming/Wharton (2026)", and "NIH (2026)" — every one of those strings is a fabrication and has been removed, replaced with its real 1983 / 1995 / 2000 / 2009 / 2010 / 2013 / 2014 / 2016 counterpart. Any future draft that re-introduces a future-dated fabricated citation must be rejected on sight.
* **No "elite / DeepMind-rigor / S-tier" marketing.** The filename
  carries the word *elite* for historical reasons; it is not a
  certification. The contract is that every entry above is a real
  peer-reviewed source with a DOI that resolves today.
* **No "governance singularity", "Ω kernel", or "cognitive integrity
  system" language.** `neurophase` is a typed Python library with a
  5-state gate, a temporal precondition, a null-model harness, a
  calibration loop, and a decision ledger. It is not a singularity.

---

*Document version:* synchronised with the PR that re-integrated the
real bibliography and cleaned the fake "2026" citations. See
[`CHANGELOG.md`](../../CHANGELOG.md) for the ordered history.
