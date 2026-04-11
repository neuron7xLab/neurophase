# Sensory Basis of Financial Value — neurophysiological backing for the neural-side oscillator

*A synthesis for the neurophase hypothesis: pupil, HRV, and EEG α/β as phase-coupled oscillators measurable against the market.*

---

## Why this document exists

`neurophase` treats the trader's nervous system and the market as **a single Kuramoto network**. The market half is straightforward: price, volume, and realized volatility feed `neurophase.oscillators.market` through a Hilbert pipeline. The neural half is harder — it requires a hardware bridge (Tobii / OpenBCI / Muse / Emotiv) and a defensible choice of channels.

This document is the scientific spine for that choice. It collects the empirical results that make *pupil reflex*, *HRV*, and *EEG α/β* legitimate phase oscillators for reward-driven decision making, and it spells out the four neural circuits that translate sensory input into financial value.

---

## The four-circuit model

Value is never a raw sensory quantity. It is a **composition** of four sequentially engaged neural circuits:

| Stage | Circuit | Role | Measurable proxy |
|------:|---------|------|------------------|
| 1 | V4 — visual feature extraction | Detects perceptual salience (gloss, shape, motion) within ~100–200 ms | Early visual event-related potential, pupil latency |
| 2 | VTA → NAcc (dopaminergic reward) | Assigns affective weight; encodes reward prediction error | Pupil dilation, HRV dip, EEG β power |
| 3 | vmPFC — value integration | Blends learned, cultural, and context-dependent priors into a subjective price | EEG θ coherence, late ERP window |
| 4 | dlPFC — cognitive control | Regulates automatic valuation; the site of deliberate override | EEG α suppression, gaze redirection |

The hand-off V4 → VTA → vmPFC → dlPFC takes ~500 ms end-to-end. Phase-lock between the market's signal and this chain shows up in the **200–500 ms window** of the neural analytic signal.

---

## Pupil dilation as a phase carrier

- Constrictions and dilations are driven by **locus-coeruleus noradrenergic bursts**, which track reward prediction error on a ~250–400 ms timescale (Joshi et al., 2016).
- Fixation pupillometry is a reliable readout of **NAcc activation** during economic choice tasks (Preuschoff, 't Hart & Einhäuser, 2011).
- Pupil time series are low-dimensional and *already periodic enough* for a Hilbert-transform phase to be meaningful.

**Bridge contract:** a Tobii-class eye tracker at ≥ 120 Hz exposes pupil diameter as a 1-D real signal. `extract_market_phase`-style processing produces `φ_pupil` — a direct input to the joint Kuramoto network.

---

## Heart rate variability as a phase carrier

- HRV in the **0.15–0.4 Hz (HF band)** reflects parasympathetic regulation and follows emotional regulation with ~1–2 s latency.
- During losing trades traders show a systematic **HF suppression** (Lo & Repin, 2002) with a measurable phase signature against price return oscillations.
- HRV is the cheapest bio-channel to obtain: a chest strap or wrist-optical PPG at 1 kHz suffices.

**Bridge contract:** any Polar / Garmin / consumer PPG stream with inter-beat-interval export. Bandpass 0.04–0.4 Hz, Hilbert, phase.

---

## EEG α / β as phase carriers

- **α suppression (8–12 Hz) over parietal sites** indexes directed attention; its phase lags behind pupil activity by ~70–120 ms in decision tasks (Jensen & Mazaheri, 2010).
- **β increase (13–30 Hz) over motor cortex** precedes overt action by ~200 ms — the mechanical correlate of "I'm about to click."
- Both bands survive short-window Hilbert-phase extraction cleanly — neurophase's wavelet denoiser on a standardised signal is sufficient.

**Bridge contract:** OpenBCI Cyton or Ganglion at 250 Hz on 4 channels (Pz, Cz, C3, C4) is the minimum viable rig. Higher-density EEG is a nice-to-have, not a requirement.

---

## Why this coupling is not magical

The claim that market phase and neural phase can lock is **physical**, not mystical:

1. The trader's nervous system ingests market signals through the eyes.
2. V4 → VTA → vmPFC → dlPFC pipes those signals through a reward-weighted integrator.
3. Motor output (the click) feeds back into the market as delta volume.
4. Steps 1–3 are a *physical channel* with measurable impulse responses.
5. Any physical channel between two oscillating systems is a coupling term in the Kuramoto sense.

If the coupling is strong enough, `PLV(φ_neural, φ_market) > 0`. If it is not, the hypothesis dies honestly in one commit. That is the whole neurophase bet.

---

## Individual variability — a knob, not an obstacle

Three genetic polymorphisms modulate the strength of the reward circuit and therefore the expected neural coupling:

| Gene | Variant | Effect |
|------|---------|--------|
| `DRD2` | Taq1A A1+ | Stronger NAcc response; louder pupil and HRV signatures |
| `COMT` | Val158Met Met/Met | Stronger dlPFC override; cleaner late ERP |
| `DAT1` | 10R vs 9R | Modulates dopamine clearance rate |

These are **not** prerequisites for running neurophase. They are knobs for per-trader calibration once a real cohort is running. A stronger reward signature simply means the neural half of the PLV test wakes up earlier.

---

## Cognitive-control strategies as gates-inside-the-gate

Three strategies are documented to **attenuate** the automatic V4 → VTA pathway, making the dlPFC override louder:

1. **Cognitive reappraisal** — "think about the real economic value instead of the instinct."
2. **Attention refocusing** — "look at the risk side of the chart, not the P&L."
3. **Emotional distancing** — "evaluate as if for someone else."

Training a trader in any of these strategies *changes* the measured PLV because it changes the neural phase. The execution gate doesn't care why the phase lifts — it only cares that `R(t) ≥ θ`. But this is the mechanism by which practice and discipline **physically** improve the gate's permission rate.

---

## What's implemented vs what's pending

| Layer | Status |
|-------|--------|
| `oscillators.market` — price/volume/volatility phase pipeline | ✅ implemented |
| `oscillators.neural_protocol` — abstract contract | ✅ implemented |
| `NullNeuralExtractor` — honest-absent default | ✅ implemented |
| Tobii pupil adapter | 🔲 hardware-dependent |
| OpenBCI α/β adapter | 🔲 hardware-dependent |
| HRV (Polar / Garmin) adapter | 🔲 hardware-dependent |
| Per-trader DRD2/COMT/DAT1 calibration | 🔲 requires cohort study |

---

## Selected references

- Preuschoff, K., 't Hart, B. M., & Einhäuser, W. (2011). *Pupil dilation signals surprise: evidence for noradrenaline's role in decision making.* Frontiers in Neuroscience, 5:115.
- Joshi, S., Li, Y., Kalwani, R. M., & Gold, J. I. (2016). *Relationships between pupil diameter and neuronal activity in the locus coeruleus, colliculi, and cingulate cortex.* Neuron, 89(1), 221–234.
- Lo, A. W., & Repin, D. V. (2002). *The psychophysiology of real-time financial risk processing.* Journal of Cognitive Neuroscience, 14(3), 323–339.
- Jensen, O., & Mazaheri, A. (2010). *Shaping functional architecture by oscillatory alpha activity.* Frontiers in Human Neuroscience, 4:186.
- Vasylenko, Y. (2026). *Phase Synchronization as Execution Gate in Human-Market Systems* — in preparation.
