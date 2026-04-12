# Email to Toma & Miyakoshi — Data Request

**To:** mihai.toma@nexarchlab.com
**Subject:** Data request — replication of delta-price correlation (Toma & Miyakoshi 2021)

---

Dear Dr. Toma,

I am Yaroslav Vasylenko, an independent researcher in Kyiv, Ukraine, building neurophase — an open-source phase synchronization framework for neural-market coupling (github.com/neuron7xLab/neurophase). The system has 1200+ tests, bias-corrected PPC metrics (Vinck et al. 2010), and a three-gate independent verification protocol (Rayleigh + dual surrogate + analytical ground truth).

Your 2021 finding — negative correlation between frontal delta power (FC5) and trial-by-trial stock prices — is exactly the signal I am trying to replicate and extend. I ran a delta power analysis on OpenNeuro ds003458 (Cavanagh, 3-armed bandit with oscillating reward probabilities) as a proxy. Early results on 12/23 subjects show 2 significant (p < 0.05, phase-randomization surrogates), but with mixed signs (+0.23 and -0.17) and no systematic group effect. The reward probability oscillation (~0.001 Hz) is likely too slow and too different from real stock price dynamics to serve as a proper test.

I would like to request access to your raw EEG data with synchronized stock price time series from your 2021 experiment. Specifically, I need the continuous EEG (any standard format) and the trial-by-trial price stream at matched timestamps. I will run the same three-gate verification protocol and share all results openly — whether they replicate your finding or not.

The analysis code is public at github.com/neuron7xLab/neurophase. All results are pre-registered before analysis runs.

Best regards,
Yaroslav Vasylenko
neuron7xLab · Kyiv, Ukraine
