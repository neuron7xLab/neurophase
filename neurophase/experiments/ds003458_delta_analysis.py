"""ds003458 delta power × reward probability cross-correlation analysis.

Tests whether frontal delta (1–4 Hz) power envelope correlates with
trial-by-trial reward probability changes. Operates at the correct
timescale (trial-level, ~3 s) unlike the PLV analysis which failed
due to frequency mismatch.

Pre-registration amendment: results/ds003458_preregistration.md

Run:
    python -m neurophase.experiments.ds003458_delta_analysis
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, ttest_1samp

from neurophase.data.ds003458_loader import DS003458Loader, interpolate_to_eeg_rate
from neurophase.metrics.delta_power import extract_delta_power
from neurophase.metrics.delta_price_xcorr import compute_delta_price_xcorr

# Pre-registered channel
PRIMARY_CHANNEL = "FC5"


def _build_reward_returns(
    reward_prob: np.ndarray,
    trial_onsets: np.ndarray,
    fs: float,
    total_dur: float,
    n_eeg: int,
) -> np.ndarray:
    """Build trial-by-trial reward returns, interpolated to EEG rate."""
    # Compute returns (diff of reward probability)
    returns_trial = np.diff(reward_prob, prepend=reward_prob[0])
    # Interpolate to EEG sample rate
    returns_cont = interpolate_to_eeg_rate(returns_trial, trial_onsets, fs, total_dur)
    returns_cont = returns_cont[:n_eeg]
    # Z-score
    m, s = float(np.mean(returns_cont)), float(np.std(returns_cont))
    if s > 0:
        returns_cont = (returns_cont - m) / s
    return returns_cont.astype(np.float64)


def run_delta_analysis(
    data_root: str | Path = "data/ds003458",
    n_surrogates: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run per-subject delta power × reward xcorr analysis."""
    loader = DS003458Loader(data_root)
    subjects = loader.list_subjects()

    if not subjects:
        raise RuntimeError(f"No subjects found in {data_root}")

    print(f"Found {len(subjects)} subjects")
    print()

    per_subject: list[dict[str, Any]] = []

    for sid in subjects:
        print(f"  {sid}...", end=" ", flush=True)
        try:
            sub = loader.load_subject(sid)
            raw = sub.raw
            fs = sub.fs

            # Check channel availability
            ch_names = raw.ch_names
            ch = PRIMARY_CHANNEL if PRIMARY_CHANNEL in ch_names else "Fz"

            # Extract delta power from primary channel
            raw_filt = raw.copy()
            ch_data = raw_filt.get_data(picks=[ch])[0]  # (n_samples,)

            delta = extract_delta_power(
                ch_data,
                fs=fs,
                band=(1.0, 4.0),
                smooth_ms=500.0,
                channel_name=ch,
            )

            # Build reward returns at EEG rate
            total_dur = float(raw.times[-1])
            n_eeg = len(ch_data)
            returns = _build_reward_returns(
                sub.reward_prob_chosen,
                sub.trial_onsets_sec,
                fs,
                total_dur,
                n_eeg,
            )

            # Trim edges (5%)
            n_trim = int(0.05 * min(len(delta.envelope), len(returns)))
            env = delta.envelope[n_trim:-n_trim] if n_trim > 0 else delta.envelope
            ret = returns[n_trim:-n_trim] if n_trim > 0 else returns

            # Held-out: last 30%
            n_ho = int(0.30 * len(env))
            env_ho = env[-n_ho:]
            ret_ho = ret[-n_ho:]

            # Cross-correlation
            xcorr = compute_delta_price_xcorr(
                env_ho,
                ret_ho,
                fs=fs,
                max_lag_ms=2000.0,
                n_surrogates=n_surrogates,
                seed=seed,
            )

            row: dict[str, Any] = {
                "subject": sid,
                "channel": ch,
                "max_xcorr": xcorr.max_xcorr,
                "lag_ms": xcorr.lag_ms,
                "p_value": xcorr.p_value,
                "significant": xcorr.significant,
                "direction": xcorr.direction,
                "n_samples_held_out": len(env_ho),
            }
            per_subject.append(row)

            sig_str = "SIG" if xcorr.significant else "ns "
            print(
                f"xcorr={xcorr.max_xcorr:+.4f}  "
                f"lag={xcorr.lag_ms:+.0f}ms  "
                f"p={xcorr.p_value:.3f}  {sig_str}  "
                f"{xcorr.direction}"
            )

        except Exception as e:
            print(f"FAILED: {e}")
            per_subject.append(
                {
                    "subject": sid,
                    "max_xcorr": None,
                    "error": str(e),
                }
            )

    # --- Group-level ---
    valid = [r for r in per_subject if r.get("max_xcorr") is not None]
    n_total = len(valid)

    if n_total == 0:
        return {
            "experiment": "ds003458_delta",
            "error": "No valid subjects",
            "per_subject": per_subject,
        }

    n_sig = sum(1 for r in valid if r.get("significant", False))
    binom = binomtest(n_sig, n=n_total, p=0.05, alternative="greater")

    xcorr_vals = [r["max_xcorr"] for r in valid]
    z_scores = np.arctanh(np.clip(xcorr_vals, -0.999, 0.999))
    group_xcorr = float(np.tanh(np.mean(z_scores)))
    _, xcorr_p = ttest_1samp(z_scores, 0)

    lag_vals = [r["lag_ms"] for r in valid if r.get("lag_ms") is not None]
    mean_lag = float(np.mean(lag_vals)) if lag_vals else 0.0

    # Interpretation
    if n_sig >= 5 and group_xcorr < -0.10 and mean_lag <= 0:
        evidence_status = "Strongly Plausible"
    elif n_sig >= 2:
        evidence_status = "Tentative (upgraded)"
    else:
        evidence_status = "Tentative (unchanged)"

    return {
        "experiment": "ds003458_delta_power_xcorr",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": "OpenNeuro ds003458 v1.1.0",
        "preregistration": "results/ds003458_preregistration.md (amendment 2026-04-12)",
        "primary_channel": PRIMARY_CHANNEL,
        "n_subjects_total": len(subjects),
        "n_subjects_valid": n_total,
        "n_significant": n_sig,
        "group_xcorr": group_xcorr,
        "group_xcorr_ttest_p": float(xcorr_p),
        "binomial_p": float(binom.pvalue),
        "mean_lag_ms": mean_lag,
        "evidence_status": evidence_status,
        "per_subject": per_subject,
        "analysis_params": {
            "delta_band_hz": [1.0, 4.0],
            "channel": PRIMARY_CHANNEL,
            "smooth_ms": 500.0,
            "max_lag_ms": 2000.0,
            "n_surrogates": n_surrogates,
            "seed": seed,
            "held_out_fraction": 0.30,
            "edge_trim_fraction": 0.05,
        },
    }


def save_results(results: dict[str, Any], output_dir: str | Path = "results") -> Path:
    """Save to timestamped JSON."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    path = out_dir / f"ds003458_delta_xcorr_{date_str}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def main() -> None:
    """Entry point."""
    print("=" * 60)
    print("  NEUROPHASE · ds003458 Delta Power x Reward Xcorr")
    print("=" * 60)
    print()

    results = run_delta_analysis()

    print()
    path = save_results(results)
    print(f"Results saved to: {path}")
    print()
    print(f"Significant:    {results.get('n_significant', 0)}/{results.get('n_subjects_valid', 0)}")
    print(f"Group xcorr:    {results.get('group_xcorr', 0):+.4f}")
    print(f"Binomial p:     {results.get('binomial_p', 1):.4f}")
    print(f"Mean lag:       {results.get('mean_lag_ms', 0):+.1f} ms")
    print()
    print(f"Evidence status: {results.get('evidence_status', 'unknown')}")


if __name__ == "__main__":
    main()
