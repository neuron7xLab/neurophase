"""ds003458 — continuous regression on the adaptation sub-hypothesis.

Motivation
----------

In the quartile-split utility benchmark (``ds003458_delta_q_20260413.json``),
adaptation was the only primary metric whose group mean ΔQ trended
positive (+0.073) with 11/17 subjects positive, though the effect did
not survive Wilcoxon on Δs (p=0.548) or BH-FDR (2/17 significant).
Value_loss and accuracy trended near zero.

The user asked for a continuous follow-up: replace the quartile
contrast with a Spearman rank regression on the same predictive
alignment, test the sub-hypothesis directly, and commit regardless
of outcome.

Hypothesis
----------

    Spearman ρ( theta_power(t), adaptation_speed(t+1) | RPE(t)<0 ) > 0

Predictive alignment, neural feature, RPE model, skip list, seed, and
α are all inherited verbatim from the validated CSD / ΔQ pipelines —
no re-tuning.

Design invariants
-----------------

* **Same-trial circularity avoided.** θ at trial t predicts adaptation
  on trial t+1. The last trial is always dropped; the first trial has
  no RPE(t-1) constraint of its own (RPE is computed over all trials
  and masked afterwards).

* **Continuous test, not quartile.** All trials where (RPE(t) < 0 and
  both RTs are finite) enter the Spearman ρ. No split, no quartile,
  no post-hoc threshold.

* **Primary null = shuffled-theta within subject**, 1000x. Two-sided
  empirical p = (1 + #|ρ_null| >= |ρ_obs|) / (1 + N).

* **Group test = Wilcoxon signed-rank on Fisher-z(ρ)** vs 0.  BH-FDR
  on subject empirical p-values.  Binomial on sign of ρ.

* **Scope:** this is an *exploratory follow-up* on a single dataset,
  motivated by a subthreshold trend in the preceding benchmark. A
  positive result here does NOT constitute independent replication
  and does NOT promote C5 — it only quantifies the sub-hypothesis
  more tightly on ds003458. Replication on an independent dataset
  with a prespecified direction remains required.

Run::

    python -m neurophase.experiments.ds003458_adaptation_regression
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import binomtest, spearmanr, wilcoxon

from neurophase.data.ds003458_loader import DS003458Loader, SubjectData
from neurophase.experiments.ds003458_csd_analysis import (
    NEURAL_CHANNEL,
    POWER_WINDOW_MS,
    THETA_BAND,
    _apply_csd,
    _induced_theta_power_per_trial,
    _q_learning_rpe,
)
from neurophase.experiments.ds003458_delta_q import (
    _compute_behavioural_targets,
    _parse_trials_full,
)

FloatArray = npt.NDArray[np.float64]

SKIP_SUBJECTS: frozenset[str] = frozenset(
    {"sub-006", "sub-017", "sub-020", "sub-021", "sub-022", "sub-023"}
)
N_THETA_SHUFFLES: int = 1000
RANDOM_SEED: int = 42
ALPHA: float = 0.05
MIN_TRIALS_REQUIRED: int = 20


@dataclass(frozen=True)
class SubjectRegression:
    subject: str
    status: str
    reason: str
    n_pairs_after_rpe_filter: int
    rho: float
    p_parametric: float
    p_empirical_shuffled: float
    fisher_z: float


def _bh_fdr(pvals: FloatArray) -> FloatArray:
    """Benjamini-Hochberg q-values; NaNs pass through."""
    p = np.asarray(pvals, dtype=np.float64)
    mask = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if not np.any(mask):
        return q
    valid = p[mask]
    n = valid.size
    order = np.argsort(valid)
    ranked = valid[order]
    adj = ranked * n / (np.arange(n) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n, dtype=np.float64)
    out[order] = adj
    q[mask] = out
    return q


def _analyse_subject(
    subject: SubjectData,
    *,
    rng: np.random.Generator,
) -> dict[str, Any]:
    sid = subject.subject_id
    if NEURAL_CHANNEL not in subject.raw.ch_names:
        return {
            "subject": sid,
            "status": "SKIPPED",
            "reason": f"{NEURAL_CHANNEL} missing",
        }

    try:
        raw_csd = _apply_csd(subject.raw)
    except Exception as exc:
        return {"subject": sid, "status": "ERROR", "reason": f"CSD: {exc}"}

    ev = _parse_trials_full(subject)
    if ev.n_trials < MIN_TRIALS_REQUIRED:
        return {
            "subject": sid,
            "status": "SKIPPED",
            "reason": f"only {ev.n_trials} completed triples",
        }

    targets = _compute_behavioural_targets(ev)
    rpe = _q_learning_rpe(ev.reward, ev.chosen_stim)
    theta = _induced_theta_power_per_trial(raw_csd, ev.fb_onsets_samples, fs=subject.fs)
    if len(theta) != ev.n_trials:
        return {"subject": sid, "status": "ERROR", "reason": "theta length mismatch"}

    n = ev.n_trials
    t_idx = np.arange(n - 1)
    theta_t = theta[t_idx]
    z_rt_t = targets["z_rt"][t_idx]
    z_rt_next = targets["z_rt"][t_idx + 1]
    rpe_t = rpe[t_idx]

    adaptation_next = -(z_rt_next - z_rt_t)
    # Valid adaptation trial pair: negative RPE at t, both RTs finite,
    # theta finite.
    valid = np.isfinite(theta_t) & np.isfinite(z_rt_t) & np.isfinite(z_rt_next) & (rpe_t < 0.0)
    n_valid = int(valid.sum())
    if n_valid < MIN_TRIALS_REQUIRED:
        return {
            "subject": sid,
            "status": "SKIPPED",
            "reason": f"only {n_valid} valid post-loss trial pairs",
        }

    x = theta_t[valid]
    y = adaptation_next[valid]

    rho, p_param = spearmanr(x, y)
    rho = float(rho) if np.isfinite(rho) else 0.0
    p_param = float(p_param) if np.isfinite(p_param) else float("nan")

    # Shuffled-theta empirical null. Permute the predictor only, keep
    # behaviour intact. 1000 iterations. Two-sided tail test.
    shuffled = x.copy()
    null_rhos = np.empty(N_THETA_SHUFFLES, dtype=np.float64)
    for i in range(N_THETA_SHUFFLES):
        rng.shuffle(shuffled)
        r, _ = spearmanr(shuffled, y)
        null_rhos[i] = float(r) if np.isfinite(r) else 0.0
    p_emp = float((np.sum(np.abs(null_rhos) >= abs(rho)) + 1) / (N_THETA_SHUFFLES + 1))

    # Fisher-z transform for group-level inference.
    z = float(np.arctanh(np.clip(rho, -0.999999, 0.999999)))

    return {
        "subject": sid,
        "status": "OK",
        "n_pairs_after_rpe_filter": n_valid,
        "rho": rho,
        "fisher_z": z,
        "p_parametric": p_param,
        "p_empirical_shuffled": p_emp,
        "null_mean": float(null_rhos.mean()),
        "null_std": float(null_rhos.std(ddof=0)),
    }


def run_adaptation_regression(
    data_root: str | Path = "data/ds003458",
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    loader = DS003458Loader(data_root)
    all_subjects = loader.list_subjects()
    subjects = [s for s in all_subjects if s not in SKIP_SUBJECTS]
    skipped = [s for s in all_subjects if s in SKIP_SUBJECTS]

    print(f"Found {len(all_subjects)} subjects; skipping {len(skipped)}; running {len(subjects)}")
    print(
        f"Neural feature: {NEURAL_CHANNEL} CSD + Morlet {THETA_BAND} Hz, "
        f"{POWER_WINDOW_MS} ms window (reused from ds003458_csd_analysis)"
    )
    print(
        "Continuous test: Spearman rho(theta(t), -(zRT(t+1)-zRT(t)) | RPE(t)<0), "
        f"N_THETA_SHUFFLES={N_THETA_SHUFFLES}"
    )
    print()

    rng = np.random.default_rng(seed)
    per_subject: list[dict[str, Any]] = []

    for sid in subjects:
        print(f"  {sid}...", end=" ", flush=True)
        try:
            sub = loader.load_subject(sid)
            row = _analyse_subject(sub, rng=rng)
        except Exception as exc:
            row = {"subject": sid, "status": "ERROR", "reason": str(exc)}
        per_subject.append(row)

        if row.get("status") == "OK":
            sig = "SIG" if row["p_empirical_shuffled"] < ALPHA else "ns "
            print(
                f"rho={row['rho']:+.4f}  p_emp={row['p_empirical_shuffled']:.4f}  "
                f"{sig}  n={row['n_pairs_after_rpe_filter']}"
            )
        else:
            print(f"{row['status']}: {row.get('reason', '')}")

    # --- Group-level inference ----------------------------------------
    valid = [r for r in per_subject if r.get("status") == "OK"]
    n_valid = len(valid)
    if n_valid == 0:
        verdict = "UNDETERMINED"
        wil_result: dict[str, Any] = {"stat": float("nan"), "p": float("nan")}
        sign_binom_p = float("nan")
        n_sig_raw = n_sig_fdr = n_pos = 0
        mean_rho_from_z = float("nan")
        mean_z = float("nan")
        pvals_arr: FloatArray = np.zeros(0, dtype=np.float64)
        qvals_arr: FloatArray = np.zeros(0, dtype=np.float64)
    else:
        zs = np.asarray([r["fisher_z"] for r in valid], dtype=np.float64)
        mean_z = float(zs.mean())
        mean_rho_from_z = float(np.tanh(mean_z))
        try:
            wil_stat, wil_p = wilcoxon(zs, alternative="two-sided", zero_method="zsplit")
            wil_result = {"stat": float(wil_stat), "p": float(wil_p), "n": int(zs.size)}
        except ValueError:
            wil_result = {"stat": float("nan"), "p": float("nan"), "n": int(zs.size)}

        rhos = np.asarray([r["rho"] for r in valid], dtype=np.float64)
        n_pos = int(np.sum(rhos > 0))
        bt = binomtest(n_pos, n=n_valid, p=0.5, alternative="two-sided")
        sign_binom_p = float(bt.pvalue)

        pvals_arr = np.asarray([r["p_empirical_shuffled"] for r in valid], dtype=np.float64)
        qvals_arr = _bh_fdr(pvals_arr)
        n_sig_raw = int(np.sum(pvals_arr < ALPHA))
        n_sig_fdr = int(np.sum(qvals_arr < ALPHA))

        if mean_rho_from_z > 0 and wil_result["p"] < ALPHA and n_sig_fdr >= 2:
            verdict = "POSITIVE (Case A — continuous effect survives Wilcoxon + FDR)"
        elif mean_rho_from_z < 0 and wil_result["p"] < ALPHA:
            verdict = "NEGATIVE (Case C — inverted effect)"
        else:
            verdict = "NULL (Case B — continuous sub-hypothesis not supported)"

    return {
        "metadata": {
            "repo_task": "ds003458 adaptation continuous regression (follow-up to delta_q)",
            "dataset": "OpenNeuro ds003458 v1.1.0",
            "date": datetime.now(UTC).isoformat(),
            "random_seed": seed,
            "skipped_subjects": sorted(skipped),
            "subjects_run": subjects,
        },
        "experiment_spec": {
            "neural_feature": {
                "channel": NEURAL_CHANNEL,
                "preprocessing": "CSD (standard_1005)",
                "time_frequency": "Morlet, freqs=arange(4,9), n_cycles=3",
                "band_hz": list(THETA_BAND),
                "window_ms": list(POWER_WINDOW_MS),
                "source": "imported verbatim from ds003458_csd_analysis",
            },
            "predictive_alignment": (
                "theta_power(t) predicts adaptation_speed(t+1) on trials "
                "where RPE(t) < 0; last trial dropped"
            ),
            "adaptation_metric": (
                "adaptation_speed(t+1) = -(zRT(t+1) - zRT(t)); "
                "z-score computed within subject over all valid RTs; "
                "higher = faster recovery after a loss surprise"
            ),
            "statistic": "Spearman rho per subject",
            "null": (
                f"{N_THETA_SHUFFLES}x within-subject theta permutation, two-sided empirical p"
            ),
            "group_tests": [
                "Wilcoxon signed-rank on Fisher-z(rho) vs 0",
                "BH-FDR on subject empirical p-values",
                "Two-sided binomial on sign of rho",
            ],
            "alpha": ALPHA,
            "min_trials_required": MIN_TRIALS_REQUIRED,
            "scope": (
                "Exploratory follow-up motivated by a subthreshold positive trend "
                "in ds003458_delta_q_20260413.json; not an independent replication. "
                "Any positive result requires validation on a separate dataset."
            ),
        },
        "subject_results": per_subject,
        "group_results": {
            "n_subjects_total": len(per_subject),
            "n_subjects_valid": n_valid,
            "n_subjects_positive_rho": n_pos,
            "n_subjects_significant_raw": n_sig_raw,
            "n_subjects_significant_fdr": n_sig_fdr,
            "mean_fisher_z": mean_z,
            "mean_rho_from_z": mean_rho_from_z,
            "wilcoxon_fisher_z": wil_result,
            "sign_binomial_p": sign_binom_p,
            "subject_p_empirical": [float(p) for p in pvals_arr],
            "subject_q_bh_fdr": [float(q) for q in qvals_arr],
        },
        "interpretation": {
            "verdict": verdict,
            "caveat": (
                "Exploratory follow-up. A positive outcome here does not promote "
                "C5; independent replication on a different dataset is still "
                "required before FMtheta becomes a gating primitive."
            ),
        },
        "limitations": [
            "Single-dataset follow-up; motivated by a subthreshold trend in the "
            "preceding delta_q benchmark, therefore NOT an independent replication.",
            "RT derived from (Response.onset - Stimulus.onset) because the "
            "response_time column in events.tsv is 'n/a'.",
            "Q-learning RPE with alpha=0.1, V0=0.5 (same parameters as all prior "
            "neurophase ds003458 analyses); sensitivity to learning-rate choice "
            "not explored here.",
        ],
    }


def save_results(results: dict[str, Any], output_dir: str | Path = "results") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    path = out_dir / f"ds003458_adaptation_regression_{date_str}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def main() -> None:
    print("=" * 72)
    print("  NEUROPHASE - ds003458 adaptation continuous regression")
    print("=" * 72)
    print()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_adaptation_regression()
    path = save_results(results)

    print()
    print(f"Results saved to: {path}")
    g = results["group_results"]
    print(f"Valid subjects:          {g['n_subjects_valid']}")
    print(f"Positive rho:            {g['n_subjects_positive_rho']}")
    print(f"Mean rho (Fisher-z):     {g['mean_rho_from_z']:+.4f}")
    print(f"Wilcoxon (z vs 0) p:     {g['wilcoxon_fisher_z'].get('p', float('nan')):.4f}")
    print(f"Sign binomial p:         {g['sign_binomial_p']:.4f}")
    print(f"Significant (raw):       {g['n_subjects_significant_raw']}")
    print(f"Significant (BH-FDR):    {g['n_subjects_significant_fdr']}")
    print(f"Verdict:                 {results['interpretation']['verdict']}")


if __name__ == "__main__":
    main()
