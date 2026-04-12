"""EEG preprocessor for phase extraction from ds003458.

Extracts neural phase φ_neural(t) and market-proxy phase φ_market(t)
from a single subject's data. Pre-registered parameters are hardcoded
as defaults to prevent post-hoc tuning.

Pre-registered choices:
    Neural: FMθ (4–8 Hz) at Fz
    Market: reward probability → bandpass [0.005, 0.05] Hz → Hilbert
    Edge trim: 5% on each end
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import hilbert

from neurophase.data.ds003458_loader import SubjectData, interpolate_to_eeg_rate

FloatArray = npt.NDArray[np.float64]

# Pre-registered parameters (frozen in ds003458_preregistration.md)
NEURAL_BAND: tuple[float, float] = (4.0, 8.0)  # FMθ
NEURAL_CHANNEL: str = "Fz"
MARKET_BAND: tuple[float, float] = (0.005, 0.05)  # reward oscillation
ARTIFACT_THRESHOLD_UV: float = 150.0
EDGE_TRIM_FRACTION: float = 0.05


@dataclass(frozen=True)
class PhaseExtractionResult:
    """Extracted phase pair from one subject.

    Attributes
    ----------
    phi_neural : FloatArray
        Instantaneous FMθ phase at Fz, trimmed.
    phi_market : FloatArray
        Instantaneous phase of reward probability, trimmed.
    n_samples : int
        Number of samples after trimming.
    fs : float
        Sampling rate.
    n_artifacts_rejected : int
        Number of samples zeroed due to artifact threshold.
    subject_id : str
        BIDS subject ID.
    """

    phi_neural: FloatArray
    phi_market: FloatArray
    n_samples: int
    fs: float
    n_artifacts_rejected: int
    subject_id: str


def extract_phases(
    subject_data: SubjectData,
    *,
    neural_band: tuple[float, float] = NEURAL_BAND,
    neural_channel: str = NEURAL_CHANNEL,
    market_band: tuple[float, float] = MARKET_BAND,
    artifact_threshold_uv: float = ARTIFACT_THRESHOLD_UV,
    edge_trim_fraction: float = EDGE_TRIM_FRACTION,
) -> PhaseExtractionResult:
    """Extract neural and market phases from one subject.

    Parameters
    ----------
    subject_data : SubjectData
        Loaded subject from DS003458Loader.
    neural_band : tuple[float, float]
        Bandpass for neural phase extraction.
    neural_channel : str
        EEG channel name for neural phase.
    market_band : tuple[float, float]
        Bandpass for market proxy phase extraction.
    artifact_threshold_uv : float
        Peak-to-peak artifact threshold in microvolts.
    edge_trim_fraction : float
        Fraction of samples to trim from each edge.

    Returns
    -------
    PhaseExtractionResult
    """
    raw = subject_data.raw.copy()
    fs = subject_data.fs

    # --- Neural phase ---

    # Pick channel (fall back to F1/F2 if Fz is bad)
    ch_names = raw.ch_names
    if neural_channel in ch_names:
        pick_ch = neural_channel
    elif "F1" in ch_names:
        pick_ch = "F1"
    elif "F2" in ch_names:
        pick_ch = "F2"
    else:
        raise ValueError(f"Channel {neural_channel} not found. Available: {ch_names[:10]}...")

    # Bandpass filter for FMθ
    raw_filt = raw.copy().filter(
        l_freq=neural_band[0],
        h_freq=neural_band[1],
        picks=[pick_ch],
        verbose=False,
    )

    # Extract single channel data
    neural_data = raw_filt.get_data(picks=[pick_ch])[0]  # shape (n_samples,)

    # Artifact rejection: zero out samples exceeding threshold
    threshold_v = artifact_threshold_uv * 1e-6  # μV → V
    artifacts = np.abs(neural_data) > threshold_v
    n_artifacts = int(np.sum(artifacts))
    neural_data[artifacts] = 0.0

    # Hilbert → phase
    analytic_neural = hilbert(neural_data)
    phi_neural = np.angle(analytic_neural).astype(np.float64)

    # --- Market phase ---

    # Interpolate reward probability to EEG rate
    total_dur = raw.times[-1]
    reward_continuous = interpolate_to_eeg_rate(
        subject_data.reward_prob_chosen,
        subject_data.trial_onsets_sec,
        fs,
        total_dur,
    )

    # Ensure same length as EEG
    n_eeg = len(phi_neural)
    n_market = len(reward_continuous)
    n_common = min(n_eeg, n_market)
    reward_continuous = reward_continuous[:n_common]
    phi_neural = phi_neural[:n_common]

    # Standardize before filtering
    reward_mean = float(np.mean(reward_continuous))
    reward_std = float(np.std(reward_continuous))
    if reward_std > 0:
        reward_norm = (reward_continuous - reward_mean) / reward_std
    else:
        reward_norm = reward_continuous - reward_mean

    # Bandpass the market signal
    # Use scipy FIR since neurodsp may struggle at very low frequencies
    from scipy.signal import butter, filtfilt

    nyq = fs / 2.0
    # For very low frequency bands, use a low-order filter
    low = market_band[0] / nyq
    high = market_band[1] / nyq
    # Clamp to valid range
    low = max(low, 1e-6)
    high = min(high, 0.99)
    if low >= high:
        # Band too narrow for this fs — skip filtering, use raw
        reward_filtered = reward_norm
    else:
        order = min(4, max(1, int(fs / (market_band[0] * 10))))
        b, a = butter(order, [low, high], btype="bandpass")
        reward_filtered = filtfilt(b, a, reward_norm).astype(np.float64)

    # Hilbert → phase
    analytic_market = hilbert(reward_filtered)
    phi_market = np.angle(analytic_market).astype(np.float64)

    # --- Edge trim ---
    n_trim = int(edge_trim_fraction * n_common)
    if n_trim > 0:
        phi_neural = phi_neural[n_trim:-n_trim]
        phi_market = phi_market[n_trim:-n_trim]

    # Final validation
    if not np.all(np.isfinite(phi_neural)):
        nan_count = int(np.sum(~np.isfinite(phi_neural)))
        phi_neural = np.nan_to_num(phi_neural, nan=0.0)
        n_artifacts += nan_count

    if not np.all(np.isfinite(phi_market)):
        phi_market = np.nan_to_num(phi_market, nan=0.0)

    return PhaseExtractionResult(
        phi_neural=phi_neural,
        phi_market=phi_market,
        n_samples=len(phi_neural),
        fs=fs,
        n_artifacts_rejected=n_artifacts,
        subject_id=subject_data.subject_id,
    )
