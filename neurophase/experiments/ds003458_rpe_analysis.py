"""ds003458 FMθ power × |RPE| trial-locked analysis.

New hypothesis (post-failure of cross-frequency PLV):

    FMθ power at Fz correlates with reward-prediction-error magnitude
    on a trial-by-trial basis.

This is a within-band analysis, not cross-frequency coupling. For each
trial we:

    1. Find the feedback-onset sample.
    2. Extract the Fz EEG epoch [-0.2, 0.8] s around feedback onset.
    3. Bandpass 4–8 Hz and compute instantaneous power via Hilbert.
    4. Average power over the epoch as the trial-level FMθ statistic.
    5. Compute RPE = reward_received - reward_expected, where
       reward_received ∈ {0, 1} from the Feedback Win/Loss event and
       reward_expected = reward probability of the chosen arm.
    6. Correlate |RPE| against FMθ power across trials (Spearman ρ).
    7. Surrogate null: shuffle |RPE| 1000× and recompute ρ; p is the
       two-sided tail proportion.

Group level: binomial test of how many subjects show p < 0.05 against
the chance rate of 0.05.

Run:
    python -m neurophase.experiments.ds003458_rpe_analysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import binomtest, spearmanr

from neurophase.data.ds003458_loader import DS003458Loader, SubjectData
from neurophase.metrics.trial_theta_lme import extract_trial_band_power

FloatArray = npt.NDArray[np.float64]

FMTHETA_BAND: tuple[float, float] = (4.0, 8.0)
NEURAL_CHANNEL: str = "Fz"
EPOCH_MS: tuple[float, float] = (-200.0, 800.0)
N_SURROGATES: int = 1000
ALPHA: float = 0.05

SKIP_SUBJECTS: frozenset[str] = frozenset(
    {"sub-006", "sub-017", "sub-020", "sub-021", "sub-022", "sub-023"}
)


@dataclass(frozen=True)
class TrialAlignment:
    fb_onsets_samples: npt.NDArray[np.int64]
    reward_received: FloatArray  # 0 or 1 per trial (NaN if no valid feedback)
    reward_expected: FloatArray  # reward probability of chosen arm
    n_trials: int


def _extract_trial_alignment(subject: SubjectData) -> TrialAlignment:
    """Pair feedback events with chosen-arm expected reward.

    reward_prob_chosen is one value per completed trial (stimulus→
    response→feedback triple). We walk the events in order and keep
    only the trials that produced a Win/Loss feedback event, mapping
    each to the corresponding entry of reward_prob_chosen.
    """
    events = subject.events_df
    fs = subject.fs

    trial_types = events["trial_type"].astype(str)
    fb_onsets: list[float] = []
    rewards: list[float] = []
    trial_indices: list[int] = []  # index into reward_prob_chosen

    trial_counter = -1
    for _idx, row in events.iterrows():
        tt = str(row["trial_type"])
        if tt.startswith("Stimulus"):
            trial_counter += 1
        elif tt.startswith("Feedback") and "Null" not in tt:
            if trial_counter < 0:
                continue
            if "Win" in tt:
                rewards.append(1.0)
            elif "Loss" in tt:
                rewards.append(0.0)
            else:
                continue
            fb_onsets.append(float(row["onset"]))
            trial_indices.append(trial_counter)

    if not fb_onsets:
        return TrialAlignment(
            fb_onsets_samples=np.zeros(0, dtype=np.int64),
            reward_received=np.zeros(0, dtype=np.float64),
            reward_expected=np.zeros(0, dtype=np.float64),
            n_trials=0,
        )

    expected_full = subject.reward_prob_chosen
    expected = np.asarray(
        [expected_full[i] if i < len(expected_full) else np.nan for i in trial_indices],
        dtype=np.float64,
    )

    fb_samples = (np.asarray(fb_onsets, dtype=np.float64) * fs).astype(np.int64)
    return TrialAlignment(
        fb_onsets_samples=fb_samples,
        reward_received=np.asarray(rewards, dtype=np.float64),
        reward_expected=expected,
        n_trials=len(fb_samples),
    )


def _surrogate_p_value(
    fmtheta_power: FloatArray,
    abs_rpe: FloatArray,
    observed_rho: float,
    *,
    n_surrogates: int,
    rng: np.random.Generator,
) -> tuple[float, FloatArray]:
    """Two-sided surrogate p-value from shuffling |RPE| labels."""
    null_rhos = np.empty(n_surrogates, dtype=np.float64)
    shuffled = abs_rpe.copy()
    for i in range(n_surrogates):
        rng.shuffle(shuffled)
        rho_i, _ = spearmanr(fmtheta_power, shuffled)
        null_rhos[i] = float(rho_i) if np.isfinite(rho_i) else 0.0
    # Two-sided tail
    p = float((np.sum(np.abs(null_rhos) >= abs(observed_rho)) + 1) / (n_surrogates + 1))
    return p, null_rhos


def _analyze_subject(
    subject: SubjectData,
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Per-subject RPE × FMθ-power trial analysis."""
    raw = subject.raw
    fs = subject.fs

    if NEURAL_CHANNEL not in raw.ch_names:
        return {
            "subject": subject.subject_id,
            "status": "SKIPPED",
            "reason": f"{NEURAL_CHANNEL} not in channels",
        }

    alignment = _extract_trial_alignment(subject)
    if alignment.n_trials < 20:
        return {
            "subject": subject.subject_id,
            "status": "SKIPPED",
            "reason": f"only {alignment.n_trials} feedback trials",
        }

    rpe = alignment.reward_received - alignment.reward_expected
    abs_rpe = np.abs(rpe)

    ch_data = raw.get_data(picks=[NEURAL_CHANNEL])[0].astype(np.float64)
    fmtheta_power = extract_trial_band_power(
        ch_data,
        alignment.fb_onsets_samples,
        fs=fs,
        band=FMTHETA_BAND,
        window_ms=EPOCH_MS,
    )

    n_common = min(len(fmtheta_power), len(abs_rpe))
    fmtheta_power = fmtheta_power[:n_common]
    abs_rpe = abs_rpe[:n_common]
    rpe_trim = rpe[:n_common]

    valid = np.isfinite(fmtheta_power) & np.isfinite(abs_rpe)
    fmtheta_power = fmtheta_power[valid]
    abs_rpe = abs_rpe[valid]
    rpe_trim = rpe_trim[valid]
    n_valid = int(len(fmtheta_power))

    if n_valid < 20:
        return {
            "subject": subject.subject_id,
            "status": "SKIPPED",
            "reason": f"only {n_valid} valid trials after epoch bounds",
        }

    rho, _ = spearmanr(fmtheta_power, abs_rpe)
    rho_f = float(rho) if np.isfinite(rho) else 0.0
    p_surr, _null = _surrogate_p_value(
        fmtheta_power,
        abs_rpe,
        rho_f,
        n_surrogates=N_SURROGATES,
        rng=rng,
    )

    return {
        "subject": subject.subject_id,
        "status": "OK",
        "n_trials_used": n_valid,
        "n_wins": int(np.sum(alignment.reward_received[:n_common][valid] > 0.5)),
        "n_losses": int(np.sum(alignment.reward_received[:n_common][valid] < 0.5)),
        "mean_abs_rpe": float(np.mean(abs_rpe)),
        "mean_fmtheta_power": float(np.mean(fmtheta_power)),
        "spearman_rho": rho_f,
        "surrogate_p": p_surr,
        "significant": bool(p_surr < ALPHA),
    }


def run_rpe_analysis(
    data_root: str | Path = "data/ds003458",
    *,
    seed: int = 42,
) -> dict[str, Any]:
    loader = DS003458Loader(data_root)
    all_subjects = loader.list_subjects()
    subjects = [s for s in all_subjects if s not in SKIP_SUBJECTS]
    skipped = [s for s in all_subjects if s in SKIP_SUBJECTS]

    print(f"Found {len(all_subjects)} subjects; skipping {len(skipped)} "
          f"({', '.join(skipped)}); running {len(subjects)}")
    print(f"Channel: {NEURAL_CHANNEL}  Band: {FMTHETA_BAND} Hz  "
          f"Epoch: {EPOCH_MS} ms  Surrogates: {N_SURROGATES}")
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
        "experiment": "ds003458_rpe_analysis",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": "OpenNeuro ds003458 v1.1.0",
        "hypothesis": "FMθ_power ~ |RPE| trial-by-trial",
        "method": {
            "channel": NEURAL_CHANNEL,
            "band_hz": list(FMTHETA_BAND),
            "epoch_ms": list(EPOCH_MS),
            "statistic": "Spearman rho of trial FMθ power vs |RPE|",
            "null": f"{N_SURROGATES}× |RPE| shuffle (two-sided)",
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
    path = out_dir / f"ds003458_rpe_{date_str}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def main() -> None:
    print("=" * 64)
    print("  NEUROPHASE · ds003458 FMθ power × |RPE| — trial-locked")
    print("=" * 64)
    print()

    results = run_rpe_analysis()
    path = save_results(results)

    print()
    print(f"Results saved to: {path}")
    print()
    print(f"Valid subjects: {results['n_subjects_valid']}")
    print(f"Significant:    {results['n_significant']}")
    print(f"Group p:        {results['group_p']:.4g}")
    print(f"Verdict:        {results['verdict']}")


if __name__ == "__main__":
    main()
