# EEG_UTILITY_NEXT.md — what may follow, and what may not

This document closes the ambiguity between two very different
statements that have been confused in prior discussion:

> **(A) Existence:** FMθ at FCz co-varies trial-by-trial with
> reward-prediction-error magnitude on ds003458. 9 / 17 subjects
> individually significant, binomial vs chance p = 3.29e-8.
> Direction heterogeneous (5 ρ<0, 4 ρ>0).

> **(B) Utility:** FMθ at trial *t* predicts *better* decisions
> on trial *t+1* on ds003458. **NULL.** Three pre-registered
> metrics, quartile split + shuffled-θ null + BH-FDR, 0 / 17 on
> value_loss and accuracy, 2 / 17 on adaptation (compatible with
> chance across 51 subject-metric tests). Continuous regression
> follow-up on adaptation: mean ρ = +0.008, Wilcoxon p = 0.818,
> 0 / 17 FDR-significant.

(A) and (B) are the repo's final posture on ds003458. Both are
reported verbatim in `results/` (`ds003458_csd_20260413.json`,
`ds003458_delta_q_20260413.json`,
`ds003458_adaptation_regression_20260413.json`) and summarised in
`CLAIMS.yaml::C5`.

## Rule: no further rescue analyses on ds003458

The `ds003458` shipping track is **frozen**. Any new analysis on
that same dataset attempting to reverse or weaken (B) is a rescue
attempt and is forbidden on `main`. This rule exists because:

* the dataset has already been sliced by PLV (null), delta-power
  cross-corr (null), SCP cross-corr (null), trial-LME (null),
  Fz-Hilbert × |RPE| (null), FCz CSD existence (9/17), ΔQ_gate
  quartile utility (null), and adaptation continuous regression
  (null). The prior on finding new positive utility from the same
  480-trial behavioural data is very low.
* any further p-hacking would erode truth discipline without
  changing clinical or operational reality.
* repo time is finite. There are two honest paths forward (below);
  ds003458 is not one of them.

Exceptions:

* methodology fixes to existing analyses (e.g. a bug in the
  surrogate RNG) are allowed, provided the fix is documented as a
  fix, not as a new result.
* null results are always publishable here — they cost truth nothing.

Anything else touching ds003458 should be opened as a research-track
issue, not merged to `main`.

## The two permitted paths forward

### Path 1. New dataset screening, with pre-registered inclusion criteria

Before any new-dataset analysis can land, the dataset must clear a
hard screening pass. The screening is not a reviewer's taste; it is
the following explicit list:

1. **Stochastic reward signal.** The task must deliver reward on a
   timescale where the brain can plausibly track it, with
   non-trivial variance. ds003458's reward probabilities are
   deterministic random walks with very low trial-to-trial
   information content — that was a post-hoc diagnosis of why
   utility was null. The next dataset must not repeat this.
2. **Trial count per subject ≥ 400** (matching ds003458 so the
   quartile split has comparable statistical power).
3. **Continuous EEG at ≥ 250 Hz** with FCz or a comparable frontal-
   midline channel reconstructable via CSD.
4. **Trial-locked outcomes with machine-readable labels.** No PDF
   reverse-engineering of task timing.
5. **At least 20 subjects** (loose replication power for the
   existence benchmark at α = 0.05).
6. **Publicly available raw data.** BIDS-format preferred.
7. **A pre-registered hypothesis document** committed to the repo
   BEFORE any analysis code is run. The document must specify the
   primary metric, the surrogate null, the multiple-comparison
   correction, and the decision rule. Post-hoc amendments are
   labelled as such or rejected.

A dataset passing 1–7 opens a new experiment track under
`neurophase/experiments/<dataset-id>_*`. A dataset failing any of
1–7 does not.

Candidates that have been mentioned in discussion but not screened:

* Torres et al. (stochastic reward paradigm) — **unscreened**
* Miyakoshi ERP archive — **unscreened**

Screening them is the first commit on Path 1.

### Path 2. Live self-data, own body, own gate

Alternative to the dataset track: use real live sessions on the
operator's own physio stack and measure decision-quality effects
directly. No EEG. No claim about cortex. Just:

1. operator wears Polar H10 → `neurophase-rr` stream
2. consumer runs with `--profile profiles/<operator-id>.json`
3. operator works on a real decision loop (coding, research,
   trading observation, deep-focus block) per
   `benchmarks/decision_quality/PROTOCOL.md`
4. matched `gate_on` / `gate_off` pairs over N sessions
5. metrics committed verbatim, null or positive, to
   `benchmarks/decision_quality/<date>/RESULTS.md`

Path 2 is infrastructure-ready as of `v1.2-rc2`:

* `neurophase/physio/live.py` + `neurophase/physio/session_replay.py`
  (live + replay + parity)
* `tools/polar_producer.py` (real hardware bridge)
* `tools/calibrate_physio.py` + `profiles/` (per-user profile)
* `benchmarks/decision_quality/PROTOCOL.md` (pre-committed metrics)
* `scripts/run_layer_d_acceptance.sh` (Layer-D acceptance runner)

What Path 2 does NOT license, even on success:

* Any claim that the operator's physiology corresponds to EEG-
  derived FMθ. Path 2 is a HRV-based signal-quality gate, not a
  neural biomarker.
* Any clinical or cognitive-state interpretation.
* Any universal claim from a single-subject N. The benchmark is
  within-subject.

## Research / shipping separation

| Track | What lives here | Status |
|---|---|---|
| **shipping** | kernel, gate, ledger, replay, physio live, Polar bridge, calibration, CI, protocols | `v1.2-rc2` |
| **research** | ds003458 results (frozen), future dataset screening, future Path-1 experiments | open |

The research track does NOT block the shipping track. A stable
`v1.2` tag per `STABLE_PROMOTION.md` ships whether or not any new
EEG-utility result exists.

## Final posture in one line

> FMθ on ds003458 exists as a covariate. It is null as a utility
> signal for gating. The next slice is not another slice of
> ds003458; it is either a new screened dataset or live self-data.
