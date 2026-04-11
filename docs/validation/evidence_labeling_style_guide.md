# neurophase — evidence labeling style guide

**Purpose.** Every non-trivial claim in `docs/` must carry an explicit
evidence status. The taxonomy is enforced by review, by the
`INVARIANTS.yaml` registry (for honest-naming contracts), and by the
`docs/theory/neurophase_elite_bibliography.md` traceability matrix.

**Scope.** This guide covers prose documentation. Code-level
contracts (invariants, state-machine transitions, pipeline
guarantees) are enforced separately by the A1/A2 CI meta-tests.

---

## QUICK REFERENCE

```markdown
**[Established]** Statement with consensus support and replicated evidence.
**[Strongly Plausible]** Strong mechanism + consistent data, but bounded generalization.
**[Tentative]** Working hypothesis requiring preregistered validation.
**[Unsupported/Weak]** Popular narrative without sufficient empirical support.
```

---

## IMPLEMENTATION CHECKLIST

For each non-trivial claim:
- [ ] assign label (E / SP / T / U)
- [ ] add primary citation(s)
- [ ] state mechanism in 1–2 lines
- [ ] state limits/context dependence
- [ ] link to module/test when implemented
- [ ] define falsification criterion for tentative claims

For product-critical claims:
- [ ] must be `Established` or `Strongly Plausible`
- [ ] must include threshold rationale + calibration date
- [ ] must include validation plan (A/B or preregistered study)

---

## TEMPLATE SNIPPETS

### Established
```markdown
**[Established]** [Claim]. [Primary citations]. [Operational implication].
```

### Strongly Plausible
```markdown
**[Strongly Plausible]** [Claim]. Mechanism: [brief]. Limit: [context].
```

### Tentative
```markdown
**[Tentative]** [Hypothesis]. Validation required: [test + criterion].
```

### Unsupported/Weak
```markdown
**[Unsupported/Weak]** [Claim] — not suitable for product decisions.
```

---

## RELEASE ENFORCEMENT

Release is blocked if:
- product claim has no evidence label,
- tentative claim has no validation plan,
- unsupported claim is used as primary justification.

