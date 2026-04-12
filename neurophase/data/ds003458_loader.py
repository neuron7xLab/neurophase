"""ds003458 BIDS loader — three-armed bandit with oscillating rewards.

Loads EEG (.set/.fdt EEGLAB format) and behavioral events from the
OpenNeuro ds003458 dataset. Constructs a market-proxy signal from
the reward probability of the chosen arm, interpolated to EEG sampling
rate.

Dataset: Cavanagh (2021), doi:10.18112/openneuro.ds003458.v1.1.0
Task: ThreeArmedBandit — 3 stimuli with sinusoidally oscillating
reward probabilities, 480 trials, 23 subjects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]

# Reward probability formula from BEH_BANDIT.m
_MEAN_INTERCEPT = 0.6
_AMPLITUDE = 0.4
_FREQ_PARAM = 0.33 * np.pi * 0.025  # angular frequency per trial


@dataclass(frozen=True)
class SubjectData:
    """Loaded data for one ds003458 subject.

    Attributes
    ----------
    subject_id : str
        BIDS subject ID (e.g. "sub-001").
    raw : mne.io.Raw
        Continuous EEG, 500 Hz, CPz reference.
    events_df : pd.DataFrame
        Events table from _events.tsv.
    reward_prob_chosen : FloatArray
        Reward probability of the chosen arm per trial (480 trials).
    trial_onsets_sec : FloatArray
        Onset time of each trial in seconds (from stimulus event).
    fs : float
        EEG sampling rate.
    """

    subject_id: str
    raw: mne.io.Raw
    events_df: pd.DataFrame
    reward_prob_chosen: FloatArray
    trial_onsets_sec: FloatArray
    fs: float


def _compute_reward_probabilities(n_trials: int = 480) -> FloatArray:
    """Compute the three reward probability curves (LO, MID, HI).

    Returns shape (3, n_trials) — rows are [LO, MID, HI] stimuli.
    """
    t = np.arange(1, n_trials + 100)  # extra headroom for shifts
    lo = _MEAN_INTERCEPT + _AMPLITUDE * np.cos(2 * _FREQ_PARAM * t)
    mid_shift = lo  # same formula, shifted
    hi_shift = lo

    # Phase-shifted versions (offsets from MATLAB code: 0, 40, 80)
    probs = np.stack([
        lo[:n_trials],
        mid_shift[40:40 + n_trials],
        hi_shift[80:80 + n_trials],
    ])
    return probs.astype(np.float64)


def _parse_chosen_arm(events_df: pd.DataFrame) -> FloatArray:
    """Extract reward probability of the chosen arm per trial.

    Maps stimulus positions (X=LO, Y=MID, Z=HI) and response
    direction to determine which arm was chosen on each trial.
    Falls back to the actual reward probability from the task design.
    """
    n_trials = 480
    probs = _compute_reward_probabilities(n_trials)

    # Parse events: each trial has Stimulus → Response → Feedback
    stim_events = events_df[
        events_df["trial_type"].str.startswith("Stimulus", na=False)
    ].reset_index(drop=True)

    resp_events = events_df[
        events_df["trial_type"].str.startswith("Response", na=False)
    ].reset_index(drop=True)

    fb_events = events_df[
        events_df["trial_type"].str.startswith("Feedback", na=False)
    ].reset_index(drop=True)

    actual_trials = min(len(stim_events), len(resp_events), len(fb_events), n_trials)

    # Map response to chosen position
    chosen_prob = np.full(actual_trials, np.nan, dtype=np.float64)

    for i in range(actual_trials):
        stim_str = str(stim_events.iloc[i]["trial_type"])
        resp_str = str(resp_events.iloc[i]["trial_type"])

        # Parse stimulus layout: "Stimulus (Left, Right, Up): X,Y,Z"
        # where X=1=LO, Y=2=MID, Z=3=HI
        try:
            stim_codes_str = stim_str.split(": ")[1]  # "X,Y,Z"
            stim_codes = stim_codes_str.split(",")
            # Position mapping: Left=0, Right=1, Up=2
            code_map = {"X": 0, "Y": 1, "Z": 2}  # LO=0, MID=1, HI=2

            # Parse response direction
            if "Left" in resp_str:
                pos_idx = 0  # Left position
            elif "Right" in resp_str:
                pos_idx = 1  # Right position
            elif "Up" in resp_str:
                pos_idx = 2  # Up position
            else:
                # No response or null — use middle prob as fallback
                chosen_prob[i] = float(probs[1, i])
                continue

            # Which stimulus code is at the chosen position?
            # Layout is (Left, Right, Up) → index into stim_codes
            # But stim_codes order matches the layout in the event string
            chosen_code = stim_codes[pos_idx].strip()
            stim_idx = code_map.get(chosen_code, 1)
            chosen_prob[i] = float(probs[stim_idx, i])

        except (IndexError, KeyError):
            # Fallback: use average probability
            chosen_prob[i] = float(np.mean(probs[:, i]))

    # Fill any remaining NaN with column mean
    nan_mask = np.isnan(chosen_prob)
    if np.any(nan_mask):
        chosen_prob[nan_mask] = float(np.nanmean(chosen_prob))

    return chosen_prob


class DS003458Loader:
    """Loads subjects from the ds003458 BIDS dataset."""

    def __init__(self, data_root: str | Path = "data/ds003458") -> None:
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_root}. "
                f"Download with: openneuro.download(dataset='ds003458', ...)"
            )

    def list_subjects(self) -> list[str]:
        """List available subject IDs."""
        subs = sorted(
            d.name for d in self.data_root.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        )
        return subs

    def load_subject(self, subject_id: str) -> SubjectData:
        """Load EEG and behavioral data for one subject.

        Parameters
        ----------
        subject_id : str
            BIDS subject ID, e.g. "sub-001".

        Returns
        -------
        SubjectData
        """
        sub_dir = self.data_root / subject_id / "eeg"
        if not sub_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {sub_dir}")

        # Load EEG (.set EEGLAB format)
        set_file = sub_dir / f"{subject_id}_task-ThreeArmedBandit_eeg.set"
        if not set_file.exists():
            raise FileNotFoundError(f"EEG file not found: {set_file}")

        raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
        fs = float(raw.info["sfreq"])

        # Load events
        events_file = sub_dir / f"{subject_id}_task-ThreeArmedBandit_events.tsv"
        events_df = pd.read_csv(events_file, sep="\t")

        # Compute reward probability of chosen arm
        reward_prob = _parse_chosen_arm(events_df)

        # Trial onset times (from stimulus events)
        stim_events = events_df[
            events_df["trial_type"].str.startswith("Stimulus", na=False)
        ]
        trial_onsets = np.asarray(
            stim_events["onset"].values[:len(reward_prob)], dtype=np.float64,
        )

        return SubjectData(
            subject_id=subject_id,
            raw=raw,
            events_df=events_df,
            reward_prob_chosen=reward_prob,
            trial_onsets_sec=trial_onsets,
            fs=fs,
        )


def interpolate_to_eeg_rate(
    trial_values: FloatArray,
    trial_onsets_sec: FloatArray,
    fs: float,
    total_duration_sec: float,
) -> FloatArray:
    """Interpolate trial-level values to continuous EEG sampling rate.

    Parameters
    ----------
    trial_values : FloatArray, shape (n_trials,)
        Value per trial (e.g., reward probability).
    trial_onsets_sec : FloatArray, shape (n_trials,)
        Onset time of each trial in seconds.
    fs : float
        Target sampling rate.
    total_duration_sec : float
        Total recording duration in seconds.

    Returns
    -------
    FloatArray, shape (n_samples,)
        Interpolated continuous signal at fs Hz.
    """
    n_samples = int(total_duration_sec * fs)
    t_continuous = np.arange(n_samples) / fs
    # Linear interpolation with edge clamping
    interpolated = np.interp(
        t_continuous,
        trial_onsets_sec,
        trial_values,
    )
    return interpolated.astype(np.float64)
