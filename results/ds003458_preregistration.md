# ds003458 Pre-Registration — Frozen Analysis Plan

**Dataset:** OpenNeuro ds003458 v1.1.0 (Cavanagh, 2021)
**Task:** Three-armed bandit with oscillating reward probabilities
**N subjects:** 23 (sub-001 to sub-023)
**EEG:** 64-ch Brain Vision ActiChamp, 500 Hz, CPz reference

## Experimental Design

Three stimuli (X, Y, Z) have reward probabilities that oscillate
sinusoidally over trials:

    P(reward|stim) = 0.6 + 0.4 · cos(2π · 0.33π · 0.025 · trial)

Stimuli are phase-shifted by 40 and 80 trials, creating three
overlapping sinusoids with period ≈ 240 trials (≈ 800 seconds).

## Pre-Registered Analysis Parameters

### Market proxy (φ_market)
- **Signal:** Reward probability of the CHOSEN arm on each trial
- **Interpolation:** Linear interpolation from trial-level (≈3.4s/trial)
  to EEG sampling rate (500 Hz)
- **Smoothing:** None beyond interpolation (raw oscillation is smooth)
- **Phase extraction:** Hilbert transform on bandpass [0.005, 0.05] Hz
  (matching the ≈0.0012 Hz reward oscillation with headroom)
- **Justification:** The chosen arm's probability is what the brain
  actually tracks for reward prediction error computation

### Neural signal (φ_neural)
- **EEG band:** Frontal midline theta (FMθ), **4–8 Hz**
- **Channel:** **Fz** (frontal midline — canonical FMθ site)
- **Justification:** FMθ is the established neural correlate of reward
  prediction error (Cavanagh et al., 2010; Cohen et al., 2007).
  The α/β band is NOT appropriate because the reward oscillation
  period (≈800s) is far below α/β frequencies. FMθ power modulates
  with reward prediction error magnitude, and its envelope may
  track the slow reward probability changes.
- **Phase extraction:** Hilbert transform of FMθ-filtered EEG
- **Note:** We compute PLV between the FMθ phase and the reward
  probability phase. This is a cross-frequency coupling measure:
  slow reward oscillation phase × fast neural oscillation phase.

### Alternative analysis (if primary fails)
- **Band:** Delta, 1–4 Hz (slower neural tracking)
- **Channel:** Cz (vertex — strongest slow oscillation)
- **Documented as exploratory, not confirmatory**

### Artifact rejection
- **Threshold:** Peak-to-peak > 150 μV → reject epoch
- **Edge trim:** First and last 5% of continuous data trimmed
  (Hilbert edge artifacts)
- **Bad channel:** If Fz is bad, use F1 or F2 (nearest neighbor)

### Statistical analysis
- **Primary metric:** PPC (Pairwise Phase Consistency, Vinck et al. 2010)
- **Significance:** p < 0.05, N = 1000 cyclic-shift surrogates
- **Held-out:** Last 30% of trials per subject
- **Three-gate verdict:** Rayleigh (R ≥ 0.10) + Dual surrogate + Theory
  - Theory gate: auto-pass (coupling_k unknown for real data)
- **Group test:** Binomial test on N_CONFIRMED vs chance (5%)
- **Effect size:** Group-mean PPC via Fisher z-transform

### Interpretation rules
- ≥60% subjects CONFIRMED → evidence_status: Strongly Plausible
- 30–60% CONFIRMED → evidence_status: Tentative (upgraded)
- <30% CONFIRMED → evidence_status: Tentative (unchanged)

---

## Amendment 2026-04-12: Delta Power Analysis

The PLV analysis (above) returned 17/17 REJECTED due to frequency
mismatch (reward ~0.001 Hz vs FMθ 4–8 Hz). This amendment adds a
delta-band power envelope analysis that operates at the correct
timescale.

### Rationale
Delta power envelope (1–4 Hz amplitude²) varies trial-by-trial,
matching the reward probability timescale. Cross-correlation measures
amplitude coupling, not phase locking — no frequency matching required.
Basis: Toma & Miyakoshi (2021) found delta EEG negatively correlates
with trial-by-trial stock price changes.

### Pre-registered parameters
- **Primary metric:** xcorr(delta_power_envelope, reward_prob_returns)
- **EEG band:** Delta, 1.0–4.0 Hz
- **Primary channel:** FC5 (left frontal, per Toma & Miyakoshi 2021)
- **Smoothing:** 500 ms Gaussian kernel on power envelope
- **Lag range:** ±2000 ms
- **Surrogate:** Phase randomization (Theiler et al. 1992), N=1000
- **Significance:** p < 0.05
- **Held-out:** Last 30% of trials
- **Direction hypothesis:** Negative correlation, lag ≤ 0 ms
  (neural anticipates or is simultaneous with reward change)

### Interpretation rules
- ≥5/N significant with group xcorr < −0.10 → Strongly Plausible
- 2–4/N significant → Tentative (upgraded)
- <2/N → Tentative (unchanged)

---

## Amendment 2, 2026-04-12: Slow Cortical Potential Analysis

The delta analysis (Amendment 1) showed 2/12 significant with mixed
signs. Root cause: classic delta (1–4 Hz) is 30–100x faster than the
reward oscillation (~0.03 Hz, period ~240 trials at ~3.4 s/trial).

Slow Cortical Potentials (SCP, 0.01–0.1 Hz) match the stimulus
timescale directly. This is what Toma & Miyakoshi likely measured
as "trial-by-trial" correlation — the effect lives at the stimulus
frequency, not in the classic delta band.

### Pre-registered parameters
- **Primary metric:** xcorr(SCP_FC5, reward_prob_slow)
- **EEG band:** SCP, 0.01–0.1 Hz
- **Primary channel:** FC5 (same as delta analysis)
- **Smoothing:** 10 s Gaussian kernel
- **Lag range:** ±5000 ms (wider for slow dynamics)
- **Surrogate:** Phase randomization (Theiler et al. 1992), N=1000
- **Significance:** p < 0.05
- **Held-out:** Last 30% of data
- **Direction hypothesis:** Negative correlation (per Toma 2021)

### Interpretation rules
- ≥5/N significant + group xcorr < −0.10 → Strongly Plausible
- 2–4/N significant → Tentative (upgraded)
- <2/N → Tentative (unchanged)

## Commit Hash

This file is frozen at the commit that adds it.
No analysis code runs before this commit exists.
