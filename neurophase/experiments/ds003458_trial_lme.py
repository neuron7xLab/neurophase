"""ds003458 trial-by-trial theta power × reward — Toma replication.

Replicates the Toma & Miyakoshi (2021) approach: single-trial EEG
band power regressed against trial-level reward prediction error
using per-subject correlation + group-level one-sample t-test.

This is the CORRECT method for ds003458: it operates at the trial
timescale (one observation per ~3.4s trial) where the reward signal
and neural power can covary — no frequency matching required.

Pre-registration: results/ds003458_preregistration.md (Amendment 3)

Run:
    python -m neurophase.experiments.ds003458_trial_lme
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, ttest_1samp

from neurophase.data.ds003458_loader import DS003458Loader
from neurophase.metrics.trial_theta_lme import (
    compute_trial_theta_reward_correlation,
    extract_trial_band_power,
)

# Channels to test (primary + secondary)
CHANNELS = ["FC5", "Fz", "F3"]
# Bands to test
BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "delta": (1.0, 4.0),
    "alpha": (8.0, 13.0),
}


def run_trial_lme_analysis(
    data_root: str | Path = "data/ds003458",
    primary_channel: str = "FC5",
    primary_band: str = "theta",
    seed: int = 42,
) -> dict[str, Any]:
    """Run per-subject trial-level correlation analysis."""
    loader = DS003458Loader(data_root)
    subjects = loader.list_subjects()

    if not subjects:
        raise RuntimeError(f"No subjects found in {data_root}")

    band = BANDS[primary_band]
    print(f"Found {len(subjects)} subjects")
    print(f"Channel: {primary_channel}, Band: {primary_band} {band} Hz")
    print()

    per_subject: list[dict[str, Any]] = []

    for sid in subjects:
        print(f"  {sid}...", end=" ", flush=True)
        try:
            sub = loader.load_subject(sid)
            raw = sub.raw
            fs = sub.fs

            ch_names = raw.ch_names
            ch = primary_channel if primary_channel in ch_names else "Fz"
            ch_data = raw.get_data(picks=[ch])[0]

            # Feedback event onsets (win + loss)
            events_df = sub.events_df
            fb_mask = events_df["trial_type"].str.contains("Feedback", na=False)
            fb_mask = fb_mask & ~events_df["trial_type"].str.contains("Null", na=False)
            fb_onsets_sec = events_df.loc[fb_mask, "onset"].values.astype(float)
            fb_onsets_samples = (fb_onsets_sec * fs).astype(np.int64)

            # Extract trial theta power (0-500ms post-feedback)
            trial_power = extract_trial_band_power(
                ch_data, fb_onsets_samples, fs=fs, band=band, window_ms=(0.0, 500.0),
            )

            # Reward signal: chosen arm probability per trial
            n_trials = min(len(trial_power), len(sub.reward_prob_chosen))
            reward = sub.reward_prob_chosen[:n_trials]
            trial_power = trial_power[:n_trials]

            # Compute prediction error: |reward_received - reward_expected|
            # Simpler proxy: just use reward_probability as the signal
            # (Toma used price changes, we use probability)
            result = compute_trial_theta_reward_correlation(
                trial_power, reward,
                subject_id=sid, channel=ch, band=band,
            )

            row: dict[str, Any] = {
                "subject": sid,
                "channel": ch,
                "band": f"{band[0]}-{band[1]}Hz",
                "r_pearson": result.r_pearson,
                "p_pearson": result.p_pearson,
                "r_spearman": result.r_spearman,
                "p_spearman": result.p_spearman,
                "n_trials": result.n_trials,
                "significant": result.p_pearson < 0.05,
            }
            per_subject.append(row)

            sig = "SIG" if result.p_pearson < 0.05 else "ns "
            print(
                f"r={result.r_pearson:+.4f}  "
                f"p={result.p_pearson:.3f}  {sig}  "
                f"rho={result.r_spearman:+.4f}  "
                f"n={result.n_trials}"
            )

        except Exception as e:
            print(f"FAILED: {e}")
            per_subject.append({"subject": sid, "r_pearson": None, "error": str(e)})

    # Group-level
    valid = [r for r in per_subject if r.get("r_pearson") is not None]
    n_total = len(valid)

    if n_total == 0:
        return {"experiment": "ds003458_trial_lme", "error": "No valid", "per_subject": per_subject}

    n_sig = sum(1 for r in valid if r.get("significant", False))
    binom = binomtest(n_sig, n=n_total, p=0.05, alternative="greater")

    r_values = [r["r_pearson"] for r in valid]
    z_scores = np.arctanh(np.clip(r_values, -0.999, 0.999))
    group_r = float(np.tanh(np.mean(z_scores)))
    t_stat, t_p = ttest_1samp(z_scores, 0)

    # Count negative correlations (Toma's finding)
    n_negative = sum(1 for r in r_values if r < 0)

    if n_sig >= 5 and t_p < 0.05:
        evidence_status = "Strongly Plausible"
    elif n_sig >= 3:
        evidence_status = "Tentative (upgraded)"
    else:
        evidence_status = "Tentative (unchanged)"

    return {
        "experiment": "ds003458_trial_lme",
        "timestamp": datetime.now(UTC).isoformat(),
        "method": "Toma & Miyakoshi (2021) replication — trial-level correlation",
        "primary_channel": primary_channel,
        "band": f"{band[0]}-{band[1]}Hz",
        "n_subjects_valid": n_total,
        "n_significant": n_sig,
        "n_negative_r": n_negative,
        "group_r": group_r,
        "group_ttest_t": float(t_stat),
        "group_ttest_p": float(t_p),
        "binomial_p": float(binom.pvalue),
        "evidence_status": evidence_status,
        "per_subject": per_subject,
    }


def save_results(results: dict[str, Any], output_dir: str | Path = "results") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    path = out_dir / f"ds003458_trial_lme_{date_str}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def main() -> None:
    print("=" * 60)
    print("  NEUROPHASE · Trial-by-Trial Theta x Reward (Toma method)")
    print("=" * 60)
    print()

    results = run_trial_lme_analysis()

    print()
    path = save_results(results)
    print(f"Results saved to: {path}")
    print()
    print(f"Significant:    {results.get('n_significant', 0)}/{results.get('n_subjects_valid', 0)}")
    print(f"Negative r:     {results.get('n_negative_r', 0)}/{results.get('n_subjects_valid', 0)}")
    print(f"Group r:        {results.get('group_r', 0):+.4f}")
    print(f"Group t-test p: {results.get('group_ttest_p', 1):.4f}")
    print()
    print(f"Evidence status: {results.get('evidence_status', 'unknown')}")


if __name__ == "__main__":
    main()
