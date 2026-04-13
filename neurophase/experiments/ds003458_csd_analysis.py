"""ds003458 — CSD + Morlet wavelet analysis (Cavanagh 2010 method).

Third analysis on the same dataset. Replaces:

    1. PLV (4–8 Hz × 0.005–0.05 Hz)  — frequency mismatch, null.
    2. Fz Hilbert RPE                 — wrong preprocessing, null.

Current pipeline (per-subject):

    1. Load raw EEG with the existing loader.
    2. Drop EOG, apply ``standard_1005`` montage.
    3. Current Source Density via ``mne.preprocessing.compute_current_source_density``.
    4. Epoch FCz around feedback onset, [-200, 800] ms.
    5. Morlet TFR: freqs=arange(4, 9), n_cycles=3, power.
    6. Induced theta power = mean over 4–8 Hz × 200–500 ms.
    7. Q-learning RPE with α=0.1, initial expectation = 0.5:
           RPE(t) = reward(t) - V(t)
           V(t+1) = V(t) + α · RPE(t)
       (per stimulus — each of the three arms has its own V.)
    8. Spearman ρ of trial theta_power vs RPE across trials.
    9. 1000× surrogate shuffling RPE labels; two-sided p.

Group: one-sided binomial test of n_significant at α=0.05 vs chance.

Run:
    python -m neurophase.experiments.ds003458_csd_analysis
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np
import numpy.typing as npt
from scipy.stats import binomtest, spearmanr

from neurophase.data.ds003458_loader import DS003458Loader, SubjectData

FloatArray = npt.NDArray[np.float64]

NEURAL_CHANNEL: str = "FCz"
THETA_BAND: tuple[float, float] = (4.0, 8.0)
MORLET_FREQS: FloatArray = np.arange(4.0, 9.0)
MORLET_N_CYCLES: int = 3
EPOCH_SEC: tuple[float, float] = (-0.2, 0.8)
POWER_WINDOW_MS: tuple[float, float] = (200.0, 500.0)
N_SURROGATES: int = 1000
ALPHA: float = 0.05
Q_LEARNING_RATE: float = 0.1
Q_INIT: float = 0.5
CSD_SPHERE: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.095)

SKIP_SUBJECTS: frozenset[str] = frozenset(
    {"sub-006", "sub-017", "sub-020", "sub-021", "sub-022", "sub-023"}
)


@dataclass(frozen=True)
class TrialEvents:
    fb_onsets_samples: npt.NDArray[np.int64]
    reward: FloatArray           # per trial: 1 = Win, 0 = Loss
    chosen_stim: npt.NDArray[np.int64]  # per trial: 0/1/2 (LO/MID/HI)
    n_trials: int


def _parse_feedback_trials(subject: SubjectData) -> TrialEvents:
    """Walk event log, keep Stimulus→Response→Feedback triples only.

    Returns feedback onsets, binary reward, and chosen-stim index so a
    per-arm Q-value can be updated.
    """
    events = subject.events_df
    fs = subject.fs

    code_map = {"X": 0, "Y": 1, "Z": 2}  # LO=0, MID=1, HI=2
    pos_map = {"Left": 0, "Right": 1, "Up": 2}

    fb_onsets: list[float] = []
    rewards: list[float] = []
    chosen: list[int] = []

    pending_stim_codes: list[str] | None = None
    pending_pos: int | None = None

    for _idx, row in events.iterrows():
        tt = str(row["trial_type"])
        if tt.startswith("Stimulus"):
            try:
                stim_part = tt.split(": ", 1)[1]
                pending_stim_codes = [c.strip() for c in stim_part.split(",")]
                pending_pos = None
            except IndexError:
                pending_stim_codes = None
                pending_pos = None
        elif tt.startswith("Response") and pending_stim_codes is not None:
            for key, val in pos_map.items():
                if key in tt:
                    pending_pos = val
                    break
        elif tt.startswith("Feedback") and "Null" not in tt:
            if pending_stim_codes is None or pending_pos is None:
                pending_stim_codes = None
                pending_pos = None
                continue
            if "Win" in tt:
                r = 1.0
            elif "Loss" in tt:
                r = 0.0
            else:
                pending_stim_codes = None
                pending_pos = None
                continue
            try:
                chosen_code = pending_stim_codes[pending_pos]
                stim_idx = code_map.get(chosen_code, 1)
            except (IndexError, TypeError):
                stim_idx = 1
            fb_onsets.append(float(row["onset"]))
            rewards.append(r)
            chosen.append(stim_idx)
            pending_stim_codes = None
            pending_pos = None

    if not fb_onsets:
        return TrialEvents(
            fb_onsets_samples=np.zeros(0, dtype=np.int64),
            reward=np.zeros(0, dtype=np.float64),
            chosen_stim=np.zeros(0, dtype=np.int64),
            n_trials=0,
        )

    return TrialEvents(
        fb_onsets_samples=(np.asarray(fb_onsets, dtype=np.float64) * fs).astype(np.int64),
        reward=np.asarray(rewards, dtype=np.float64),
        chosen_stim=np.asarray(chosen, dtype=np.int64),
        n_trials=len(fb_onsets),
    )


def _q_learning_rpe(
    reward: FloatArray,
    chosen_stim: npt.NDArray[np.int64],
    *,
    alpha: float = Q_LEARNING_RATE,
    init: float = Q_INIT,
) -> FloatArray:
    """Per-arm delta-rule RPE: RPE(t) = r(t) − V_chosen(t), then update."""
    q = np.full(3, init, dtype=np.float64)
    rpe = np.empty_like(reward)
    for t, (r, a) in enumerate(zip(reward, chosen_stim, strict=False)):
        rpe[t] = r - q[a]
        q[a] = q[a] + alpha * rpe[t]
    return rpe


def _induced_theta_power_per_trial(
    raw_csd: mne.io.BaseRaw,
    fb_samples: npt.NDArray[np.int64],
    *,
    fs: float,
) -> FloatArray:
    """Epoch FCz around feedback; return mean θ power in 200–500 ms.

    The TFR epoch is widened with a 1 s edge buffer so the Morlet
    wavelets (n_cycles=3 at 4 Hz ⇒ ≥~0.75 s support each side) fit.
    The analysis window 200–500 ms is then indexed out of the wider
    TFR.
    """
    data_1d = raw_csd.get_data(picks=[NEURAL_CHANNEL])[0].astype(np.float64)
    n_samples = data_1d.shape[0]

    edge_buffer_sec = 1.0
    pre_sec = -EPOCH_SEC[0] + edge_buffer_sec
    post_sec = EPOCH_SEC[1] + edge_buffer_sec
    pre = int(round(pre_sec * fs))
    post = int(round(post_sec * fs))
    epoch_len = pre + post

    # Time axis of wide epoch (milliseconds relative to feedback).
    t_epoch_ms = (np.arange(epoch_len) - pre) / fs * 1000.0
    window_mask = (t_epoch_ms >= POWER_WINDOW_MS[0]) & (t_epoch_ms <= POWER_WINDOW_MS[1])
    freq_mask = (MORLET_FREQS >= THETA_BAND[0]) & (MORLET_FREQS <= THETA_BAND[1])

    epochs: list[FloatArray] = []
    kept_idx: list[int] = []
    for i, s in enumerate(fb_samples):
        a = int(s) - pre
        b = int(s) + post
        if a < 0 or b > n_samples:
            continue
        epochs.append(data_1d[a:b])
        kept_idx.append(i)

    if not epochs:
        return np.zeros(0, dtype=np.float64)

    # Shape (n_epochs, 1, epoch_len) for tfr_array_morlet.
    epoch_arr = np.stack(epochs, axis=0)[:, np.newaxis, :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tfr = mne.time_frequency.tfr_array_morlet(
            epoch_arr,
            sfreq=fs,
            freqs=MORLET_FREQS,
            n_cycles=MORLET_N_CYCLES,
            output="power",
            verbose=False,
        )
    # Shape: (n_epochs, 1, n_freqs, n_times)
    power = tfr[:, 0, :, :]
    band_power = power[:, freq_mask, :][:, :, window_mask]
    mean_power = band_power.mean(axis=(1, 2))

    # Return full-length array aligned to fb_samples order, with NaN for
    # trials whose epoch fell outside the record.
    out = np.full(len(fb_samples), np.nan, dtype=np.float64)
    out[np.asarray(kept_idx, dtype=np.int64)] = mean_power.astype(np.float64)
    return out


def _apply_csd(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Return a CSD-transformed copy of raw with standard_1005 montage.

    Drops any channel absent from the standard_1005 montage (EOG, SCR,
    EKG, etc.) rather than trusting per-dataset channel-type flags.
    """
    raw = raw.copy()
    montage = mne.channels.make_standard_montage("standard_1005")
    montage_names_upper = {n.upper() for n in montage.ch_names}
    drop = [c for c in raw.ch_names if c.upper() not in montage_names_upper]
    if drop:
        raw.drop_channels(drop)
    raw.set_montage(montage, match_case=False, on_missing="raise", verbose=False)
    return mne.preprocessing.compute_current_source_density(
        raw, sphere=CSD_SPHERE, verbose=False
    )


def _surrogate_p_value(
    theta_power: FloatArray,
    rpe: FloatArray,
    observed_rho: float,
    *,
    n_surrogates: int,
    rng: np.random.Generator,
) -> float:
    shuffled = rpe.copy()
    null = np.empty(n_surrogates, dtype=np.float64)
    for i in range(n_surrogates):
        rng.shuffle(shuffled)
        r, _ = spearmanr(theta_power, shuffled)
        null[i] = float(r) if np.isfinite(r) else 0.0
    return float((np.sum(np.abs(null) >= abs(observed_rho)) + 1) / (n_surrogates + 1))


def _analyze_subject(subject: SubjectData, *, rng: np.random.Generator) -> dict[str, Any]:
    raw = subject.raw
    if NEURAL_CHANNEL not in raw.ch_names:
        return {
            "subject": subject.subject_id,
            "status": "SKIPPED",
            "reason": f"{NEURAL_CHANNEL} missing",
        }

    try:
        raw_csd = _apply_csd(raw)
    except Exception as exc:  # noqa: BLE001
        return {"subject": subject.subject_id, "status": "ERROR", "reason": f"CSD: {exc}"}

    events = _parse_feedback_trials(subject)
    if events.n_trials < 20:
        return {
            "subject": subject.subject_id,
            "status": "SKIPPED",
            "reason": f"only {events.n_trials} feedback trials",
        }

    rpe = _q_learning_rpe(events.reward, events.chosen_stim)
    theta_power = _induced_theta_power_per_trial(
        raw_csd, events.fb_onsets_samples, fs=subject.fs
    )

    valid = np.isfinite(theta_power) & np.isfinite(rpe)
    theta_power = theta_power[valid]
    rpe = rpe[valid]
    n_valid = int(len(theta_power))
    if n_valid < 20:
        return {
            "subject": subject.subject_id,
            "status": "SKIPPED",
            "reason": f"only {n_valid} trials after epoch bounds",
        }

    rho, _ = spearmanr(theta_power, rpe)
    rho_f = float(rho) if np.isfinite(rho) else 0.0
    p_surr = _surrogate_p_value(
        theta_power, rpe, rho_f, n_surrogates=N_SURROGATES, rng=rng
    )

    return {
        "subject": subject.subject_id,
        "status": "OK",
        "n_trials_used": n_valid,
        "mean_rpe": float(np.mean(rpe)),
        "mean_abs_rpe": float(np.mean(np.abs(rpe))),
        "mean_theta_power_csd": float(np.mean(theta_power)),
        "spearman_rho": rho_f,
        "surrogate_p": p_surr,
        "significant": bool(p_surr < ALPHA),
    }


def run_csd_analysis(
    data_root: str | Path = "data/ds003458",
    *,
    seed: int = 42,
) -> dict[str, Any]:
    loader = DS003458Loader(data_root)
    all_subjects = loader.list_subjects()
    subjects = [s for s in all_subjects if s not in SKIP_SUBJECTS]
    skipped = [s for s in all_subjects if s in SKIP_SUBJECTS]

    print(
        f"Found {len(all_subjects)} subjects; skipping {len(skipped)} "
        f"({', '.join(skipped)}); running {len(subjects)}"
    )
    print(
        f"Channel: {NEURAL_CHANNEL}  Band: {THETA_BAND} Hz  "
        f"Epoch: {EPOCH_SEC} s  Window: {POWER_WINDOW_MS} ms  "
        f"Surrogates: {N_SURROGATES}"
    )
    print()

    rng = np.random.default_rng(seed)
    per_subject: list[dict[str, Any]] = []

    for sid in subjects:
        print(f"  {sid}...", end=" ", flush=True)
        try:
            subject = loader.load_subject(sid)
            row = _analyze_subject(subject, rng=rng)
        except Exception as exc:  # noqa: BLE001
            row = {"subject": sid, "status": "ERROR", "reason": str(exc)}
        per_subject.append(row)

        if row.get("status") == "OK":
            tag = "SIG" if row["significant"] else "ns "
            print(
                f"rho={row['spearman_rho']:+.4f}  p={row['surrogate_p']:.4f}  "
                f"{tag}  n={row['n_trials_used']}"
            )
        else:
            print(f"{row['status']}: {row.get('reason','')}")

    valid = [r for r in per_subject if r.get("status") == "OK"]
    n_valid = len(valid)
    n_sig = sum(1 for r in valid if r["significant"])

    if n_valid == 0:
        verdict = "REJECTED"
        group_p = 1.0
    else:
        binom = binomtest(n_sig, n=n_valid, p=ALPHA, alternative="greater")
        group_p = float(binom.pvalue)
        frac_sig = n_sig / n_valid
        if frac_sig > 0.5 and group_p < ALPHA:
            verdict = "CONFIRMED"
        elif group_p < ALPHA:
            verdict = "MARGINAL"
        else:
            verdict = "REJECTED"

    return {
        "experiment": "ds003458_csd_analysis",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": "OpenNeuro ds003458 v1.1.0",
        "hypothesis": "Induced FMθ power at FCz (CSD) ~ trial-wise RPE (Q-learning)",
        "method": {
            "channel": NEURAL_CHANNEL,
            "preprocessing": "CSD (standard_1005 montage, sphere=(0,0,0,0.095))",
            "time_frequency": "Morlet wavelet, freqs=arange(4,9), n_cycles=3",
            "epoch_sec": list(EPOCH_SEC),
            "power_window_ms": list(POWER_WINDOW_MS),
            "band_hz": list(THETA_BAND),
            "rpe_model": f"delta-rule, alpha={Q_LEARNING_RATE}, V0={Q_INIT}, per-arm",
            "statistic": "Spearman rho of trial θ-power vs RPE",
            "null": f"{N_SURROGATES}× RPE shuffle (two-sided)",
            "alpha": ALPHA,
            "group_test": "one-sided binomial vs chance rate 0.05",
        },
        "skip_subjects": sorted(skipped),
        "n_subjects_run": len(subjects),
        "n_subjects_valid": n_valid,
        "n_significant": n_sig,
        "group_p": group_p,
        "verdict": verdict,
        "per_subject": per_subject,
    }


def save_results(results: dict[str, Any], output_dir: str | Path = "results") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    path = out_dir / f"ds003458_csd_{date_str}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def main() -> None:
    print("=" * 64)
    print("  NEUROPHASE · ds003458 CSD + Morlet — Cavanagh method")
    print("=" * 64)
    print()

    results = run_csd_analysis()
    path = save_results(results)

    print()
    print(f"Results saved to: {path}")
    print(f"Valid subjects: {results['n_subjects_valid']}")
    print(f"Significant:    {results['n_significant']}")
    print(f"Group p:        {results['group_p']:.4g}")
    print(f"Verdict:        {results['verdict']}")


if __name__ == "__main__":
    main()
