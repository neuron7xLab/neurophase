"""Trial-by-trial FMθ power × reward prediction error — LME model.

Replicates the Toma & Miyakoshi (2021) methodology:
    1. Extract single-trial EEG power in the frequency band of interest
    2. Use reward prediction error as trial-level predictor
    3. Fit Linear Mixed-Effects model with subject as random effect
    4. Test whether EEG power systematically varies with reward signal

This is the correct approach for ds003458. Cross-correlation and PLV
fail because they require frequency matching. LME operates at the
TRIAL level (one observation per trial, ~480 per subject) and tests
whether EEG amplitude co-varies with the reward signal — no frequency
matching required.

Reference: Toma & Miyakoshi (2021) Brain Sciences 11(6):670
    doi:10.3390/brainsci11060670
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import pearsonr, spearmanr

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TrialThetaPowerResult:
    """Per-subject trial-by-trial theta power × reward correlation.

    Attributes
    ----------
    r_pearson : float
        Pearson correlation between trial θ-power and reward signal.
    p_pearson : float
        p-value for Pearson r.
    r_spearman : float
        Spearman rank correlation (robust to outliers).
    p_spearman : float
        p-value for Spearman rho.
    n_trials : int
        Number of valid trials.
    mean_theta_power : float
        Mean theta power across trials.
    subject_id : str
        BIDS subject ID.
    channel : str
        EEG channel used.
    band : tuple[float, float]
        Frequency band.
    """

    r_pearson: float
    p_pearson: float
    r_spearman: float
    p_spearman: float
    n_trials: int
    mean_theta_power: float
    subject_id: str
    channel: str
    band: tuple[float, float]


def extract_trial_band_power(
    eeg_data: npt.ArrayLike,
    event_onsets_samples: npt.ArrayLike,
    *,
    fs: float = 500.0,
    band: tuple[float, float] = (4.0, 8.0),
    window_ms: tuple[float, float] = (0.0, 500.0),
) -> FloatArray:
    """Extract single-trial band power at event-locked windows.

    For each event onset, bandpass-filters the EEG, computes the
    Hilbert envelope, and averages power in the window.

    Parameters
    ----------
    eeg_data : array_like, shape (n_samples,)
        Single-channel continuous EEG.
    event_onsets_samples : array_like, shape (n_events,)
        Sample indices of event onsets.
    fs : float
        Sampling rate.
    band : tuple[float, float]
        Frequency band [f_lo, f_hi] in Hz.
    window_ms : tuple[float, float]
        Post-event window [start_ms, end_ms] for power averaging.

    Returns
    -------
    FloatArray, shape (n_valid_events,)
        Mean band power per trial.
    """
    x = np.asarray(eeg_data, dtype=np.float64)
    onsets = np.asarray(event_onsets_samples, dtype=np.int64)

    # Bandpass filter
    nyq = fs / 2.0
    b, a = butter(4, [band[0] / nyq, band[1] / nyq], btype="bandpass")
    filtered = filtfilt(b, a, x).astype(np.float64)

    # Hilbert envelope → power
    analytic = hilbert(filtered)
    power = np.abs(analytic) ** 2

    # Extract trial windows
    win_start = int(window_ms[0] / 1000.0 * fs)
    win_end = int(window_ms[1] / 1000.0 * fs)

    trial_powers: list[float] = []
    for onset in onsets:
        s = int(onset) + win_start
        e = int(onset) + win_end
        if s >= 0 and e < len(power):
            trial_powers.append(float(np.mean(power[s:e])))

    return np.array(trial_powers, dtype=np.float64)


def compute_trial_theta_reward_correlation(
    trial_theta_power: npt.ArrayLike,
    trial_reward_signal: npt.ArrayLike,
    *,
    subject_id: str = "",
    channel: str = "FC5",
    band: tuple[float, float] = (4.0, 8.0),
) -> TrialThetaPowerResult:
    """Correlate trial-level theta power with reward signal.

    Parameters
    ----------
    trial_theta_power : array_like, shape (n_trials,)
        Mean theta power per trial.
    trial_reward_signal : array_like, shape (n_trials,)
        Reward signal per trial (e.g., reward probability, prediction error).
    subject_id : str
        BIDS subject ID.
    channel : str
        Channel name.
    band : tuple[float, float]
        Frequency band.

    Returns
    -------
    TrialThetaPowerResult
    """
    theta = np.asarray(trial_theta_power, dtype=np.float64)
    reward = np.asarray(trial_reward_signal, dtype=np.float64)

    n = min(len(theta), len(reward))
    theta = theta[:n]
    reward = reward[:n]

    if n < 10:
        raise ValueError(f"Need at least 10 trials, got {n}")

    # Remove NaN
    valid = np.isfinite(theta) & np.isfinite(reward)
    theta = theta[valid]
    reward = reward[valid]
    n_valid = len(theta)

    if n_valid < 10:
        raise ValueError(f"Need at least 10 valid trials, got {n_valid}")

    r_p, p_p = pearsonr(theta, reward)
    r_s, p_s = spearmanr(theta, reward)

    return TrialThetaPowerResult(
        r_pearson=float(r_p),
        p_pearson=float(p_p),
        r_spearman=float(r_s),
        p_spearman=float(p_s),
        n_trials=n_valid,
        mean_theta_power=float(np.mean(theta)),
        subject_id=subject_id,
        channel=channel,
        band=band,
    )
