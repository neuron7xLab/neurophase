"""ds003458 ΔQ_gate utility benchmark — does neural state help decisions?

Preceding work on this dataset established only the *existence* of a
trial-wise FMθ ↔ RPE relationship (see ds003458_csd_20260413.json).
This module asks the next, harder question:

    Does FMθ at trial t predict *better decisions* on trial t+1,
    relative to behaviour-only and against explicit nulls?

Design invariants
-----------------

1. **No same-trial circularity.** Theta is extracted from the
   feedback-evoked window of trial t; it is only ever used to
   predict behaviour on trial t+1. Target indices are shifted
   forward by one; the last trial is dropped.

2. **Fixed, predeclared neural split.** LOW = bottom quartile of
   theta_power(t), HIGH = top quartile. The middle 50% is excluded
   from the quartile contrast. Thresholds are computed on the
   in-subject distribution of theta_t restricted to trial pairs that
   have a finite (value_loss(t+1), accuracy(t+1)) target.

3. **Three primary metrics:**
     A. value_loss(t+1) = best_arm_value(t+1) − chosen_arm_value(t+1)
        — lower is better. Positive ΔQ means LOW > HIGH (HIGH helps).
     B. accuracy(t+1)   = 1[chosen(t+1) == argmax_arm(t+1)]
        — higher is better. Positive ΔQ means HIGH > LOW.
     C. adaptation_speed(t+1) = −(zRT(t+1) − zRT(t)) | RPE(t) < 0
        — larger is better (faster recovery after a loss surprise).

4. **Three baselines:**
     B1 behavior_only  — unconditional subject mean.
     B2 random_gate    — 1000 random 50/50 splits of the same targets.
     B3 shuffled_theta — 1000 within-subject permutations of theta_t,
        recomputing the quartile-based Δ each time.  Primary null.

5. **Reuse the validated neural pipeline.** Theta power per trial is
   taken from ``ds003458_csd_analysis`` (FCz, CSD standard_1005, Morlet
   freqs=arange(4,9), n_cycles=3, 200–500 ms window). The validated
   helpers are imported, not reimplemented.

6. **No post-hoc tuning.** Constants below are fixed before running.
   Direction conventions are spelled out in code so the sign of every
   reported Δ is unambiguous.

Run::

    python -m neurophase.experiments.ds003458_delta_q
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
from scipy.stats import mannwhitneyu, rankdata, spearmanr, wilcoxon

# Reused, validated neural extraction pipeline.
from neurophase.data.ds003458_loader import (
    DS003458Loader,
    SubjectData,
    _compute_reward_probabilities,  # module-private helper, deliberately reused
)
from neurophase.experiments.ds003458_csd_analysis import (
    EPOCH_SEC,
    NEURAL_CHANNEL,
    POWER_WINDOW_MS,
    THETA_BAND,
    _apply_csd,
    _induced_theta_power_per_trial,
    _q_learning_rpe,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# --- Fixed constants (no post-hoc tuning) -------------------------------

SKIP_SUBJECTS: frozenset[str] = frozenset(
    {"sub-006", "sub-017", "sub-020", "sub-021", "sub-022", "sub-023"}
)
N_RANDOM_SPLITS: int = 1000
N_THETA_SHUFFLES: int = 1000
RANDOM_SEED: int = 42
ALPHA: float = 0.05
MIN_TRIALS_REQUIRED: int = 30
MIN_STRATUM_SIZE: int = 5


# --- Data container ------------------------------------------------------


@dataclass(frozen=True)
class TrialEvents:
    """One row per completed (stim → resp → feedback) triple, in task order."""

    stim_onsets_sec: FloatArray
    resp_onsets_sec: FloatArray
    fb_onsets_sec: FloatArray
    fb_onsets_samples: IntArray
    reward: FloatArray  # 0 or 1
    chosen_stim: IntArray  # 0/1/2 (LO/MID/HI)
    n_trials: int


# --- Pure helpers --------------------------------------------------------


def _cliff_delta(x: FloatArray, y: FloatArray) -> float:
    """Cliff's δ = (P(x>y) − P(x<y)); range [−1, 1].

    Implemented via ranks on the concatenated sample so it runs in
    O((n+m) log(n+m)) rather than O(n·m).
    """
    nx = x.size
    ny = y.size
    if nx == 0 or ny == 0:
        return float("nan")
    combined = np.concatenate([x, y])
    ranks = rankdata(combined)
    rank_sum_x = float(ranks[:nx].sum())
    u = rank_sum_x - nx * (nx + 1) / 2.0
    return float(2.0 * u / (nx * ny) - 1.0)


def _bh_fdr(pvals: FloatArray) -> FloatArray:
    """Benjamini–Hochberg-adjusted q-values. NaNs pass through."""
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


def _parse_trials_full(subject: SubjectData) -> TrialEvents:
    """Walk events, keep clean Stimulus→Response→Feedback triples.

    Tracks stim_onset + resp_onset for RT computation in addition to
    fb_onset + reward + chosen_stim.
    """
    events = subject.events_df
    fs = subject.fs

    code_map = {"X": 0, "Y": 1, "Z": 2}
    pos_map = {"Left": 0, "Right": 1, "Up": 2}

    stim_onsets: list[float] = []
    resp_onsets: list[float] = []
    fb_onsets: list[float] = []
    rewards: list[float] = []
    chosen: list[int] = []

    pending_codes: list[str] | None = None
    pending_pos: int | None = None
    pending_stim: float | None = None
    pending_resp: float | None = None

    def _reset() -> None:
        nonlocal pending_codes, pending_pos, pending_stim, pending_resp
        pending_codes = None
        pending_pos = None
        pending_stim = None
        pending_resp = None

    for _, row in events.iterrows():
        tt = str(row["trial_type"])
        if tt.startswith("Stimulus"):
            try:
                stim_part = tt.split(": ", 1)[1]
                pending_codes = [c.strip() for c in stim_part.split(",")]
                pending_pos = None
                pending_stim = float(row["onset"])
                pending_resp = None
            except IndexError:
                _reset()
        elif tt.startswith("Response") and pending_codes is not None:
            for key, val in pos_map.items():
                if key in tt:
                    pending_pos = val
                    break
            pending_resp = float(row["onset"])
        elif tt.startswith("Feedback") and "Null" not in tt:
            if (
                pending_codes is None
                or pending_pos is None
                or pending_stim is None
                or pending_resp is None
            ):
                _reset()
                continue
            if "Win" in tt:
                r = 1.0
            elif "Loss" in tt:
                r = 0.0
            else:
                _reset()
                continue
            try:
                stim_idx = code_map.get(pending_codes[pending_pos].strip(), 1)
            except (IndexError, TypeError):
                stim_idx = 1
            stim_onsets.append(pending_stim)
            resp_onsets.append(pending_resp)
            fb_onsets.append(float(row["onset"]))
            rewards.append(r)
            chosen.append(stim_idx)
            _reset()

    n = len(fb_onsets)
    fb_arr = np.asarray(fb_onsets, dtype=np.float64)
    return TrialEvents(
        stim_onsets_sec=np.asarray(stim_onsets, dtype=np.float64),
        resp_onsets_sec=np.asarray(resp_onsets, dtype=np.float64),
        fb_onsets_sec=fb_arr,
        fb_onsets_samples=(fb_arr * fs).astype(np.int64),
        reward=np.asarray(rewards, dtype=np.float64),
        chosen_stim=np.asarray(chosen, dtype=np.int64),
        n_trials=n,
    )


def _compute_behavioural_targets(events: TrialEvents) -> dict[str, FloatArray]:
    """Compute per-trial arm values, value_loss, accuracy, z-scored RT."""
    n = events.n_trials
    arm_probs_all = _compute_reward_probabilities(480)  # (3, 480)
    # The i-th completed triple aligns positionally with trial i of the task
    # schedule (oscillating reward probs are slow-varying, so any small
    # misalignment from skipped trials is bounded).
    arm_probs = arm_probs_all[:, :n]  # (3, n)
    chosen_value = arm_probs[events.chosen_stim, np.arange(n)]
    best_value = arm_probs.max(axis=0)
    best_arm_idx = arm_probs.argmax(axis=0)

    value_loss = best_value - chosen_value
    accuracy = (events.chosen_stim == best_arm_idx).astype(np.float64)

    rt = events.resp_onsets_sec - events.stim_onsets_sec
    rt_valid = np.isfinite(rt) & (rt > 0.05) & (rt < 5.0)
    if rt_valid.sum() >= 2:
        mu = float(rt[rt_valid].mean())
        sd = float(rt[rt_valid].std(ddof=0))
    else:
        mu, sd = 0.0, 1.0
    z_rt = np.full(n, np.nan, dtype=np.float64)
    if sd > 0:
        z_rt[rt_valid] = (rt[rt_valid] - mu) / sd

    return {
        "value_loss": value_loss.astype(np.float64),
        "accuracy": accuracy.astype(np.float64),
        "z_rt": z_rt,
        "chosen_value": chosen_value.astype(np.float64),
        "best_value": best_value.astype(np.float64),
    }


def _quartile_masks(
    theta_t: FloatArray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (low_mask, high_mask, q1, q3) over indices where *valid* is True."""
    vals = theta_t[valid]
    if vals.size < 4:
        return (
            np.zeros_like(valid),
            np.zeros_like(valid),
            float("nan"),
            float("nan"),
        )
    q1 = float(np.percentile(vals, 25))
    q3 = float(np.percentile(vals, 75))
    low = valid & (theta_t <= q1)
    high = valid & (theta_t >= q3)
    return low, high, q1, q3


# --- Delta computation ---------------------------------------------------


def _delta_lower_better(
    target: FloatArray, low_mask: np.ndarray, high_mask: np.ndarray
) -> tuple[float, float, float]:
    """Δ for lower-is-better metric. Returns (delta, mean_low, mean_high).

    Sign convention: positive delta means LOW-theta trials have
    *higher* (worse) target than HIGH-theta trials, i.e. HIGH theta
    predicts better outcomes.
    """
    lo = target[low_mask]
    hi = target[high_mask]
    if lo.size < MIN_STRATUM_SIZE or hi.size < MIN_STRATUM_SIZE:
        return float("nan"), float("nan"), float("nan")
    return float(lo.mean() - hi.mean()), float(lo.mean()), float(hi.mean())


def _delta_higher_better(
    target: FloatArray, low_mask: np.ndarray, high_mask: np.ndarray
) -> tuple[float, float, float]:
    """Δ for higher-is-better metric. Returns (delta, mean_low, mean_high).

    Positive delta means HIGH-theta trials have higher target.
    """
    lo = target[low_mask]
    hi = target[high_mask]
    if lo.size < MIN_STRATUM_SIZE or hi.size < MIN_STRATUM_SIZE:
        return float("nan"), float("nan"), float("nan")
    return float(hi.mean() - lo.mean()), float(lo.mean()), float(hi.mean())


# --- Core subject-level benchmark ---------------------------------------


def _analyze_subject(subject: SubjectData, *, rng: np.random.Generator) -> dict[str, Any]:
    sid = subject.subject_id
    if NEURAL_CHANNEL not in subject.raw.ch_names:
        return {"subject": sid, "status": "SKIPPED", "reason": f"{NEURAL_CHANNEL} missing"}

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

    n = ev.n_trials
    if len(theta) != n:  # defensive — extractor aligns to len(fb_samples)
        return {"subject": sid, "status": "ERROR", "reason": "theta length mismatch"}

    # -- Predictive alignment: theta at t → targets at t+1 -------------
    t_idx = np.arange(n - 1)
    theta_t = theta[t_idx]
    value_loss_next = targets["value_loss"][t_idx + 1]
    accuracy_next = targets["accuracy"][t_idx + 1]
    z_rt_t = targets["z_rt"][t_idx]
    z_rt_next = targets["z_rt"][t_idx + 1]
    rpe_t = rpe[t_idx]

    # Adaptation: defined only when trial t had a negative RPE and both
    # RTs are valid. Sign convention: larger = faster (smaller next RT).
    adapt_next = -(z_rt_next - z_rt_t)
    adapt_valid = np.isfinite(adapt_next) & (rpe_t < 0.0)
    adapt_target = np.where(adapt_valid, adapt_next, np.nan)

    # Main-metric validity: theta, value_loss, accuracy all finite.
    valid_main = np.isfinite(theta_t) & np.isfinite(value_loss_next) & np.isfinite(accuracy_next)
    n_main = int(valid_main.sum())
    if n_main < MIN_TRIALS_REQUIRED:
        return {
            "subject": sid,
            "status": "SKIPPED",
            "reason": f"only {n_main} valid main-metric pairs",
        }

    # -- Quartile split on the main-metric-valid subset ---------------
    low_m, high_m, q1, q3 = _quartile_masks(theta_t, valid_main)
    n_low = int(low_m.sum())
    n_high = int(high_m.sum())
    if n_low < MIN_STRATUM_SIZE or n_high < MIN_STRATUM_SIZE:
        return {
            "subject": sid,
            "status": "SKIPPED",
            "reason": f"tiny strata: low={n_low} high={n_high}",
        }

    # Adaptation uses the same θ thresholds but restricted to adapt_valid.
    low_a = low_m & adapt_valid
    high_a = high_m & adapt_valid
    n_low_a = int(low_a.sum())
    n_high_a = int(high_a.sum())

    # -- Observed deltas -----------------------------------------------
    dq_vl, mvl_lo, mvl_hi = _delta_lower_better(value_loss_next, low_m, high_m)
    dq_ac, mac_lo, mac_hi = _delta_higher_better(accuracy_next, low_m, high_m)
    dq_ad, mad_lo, mad_hi = _delta_higher_better(adapt_target, low_a, high_a)

    # -- Mann–Whitney U tests + Cliff's δ -----------------------------
    def _mwu_with_cliff(target: FloatArray, lo_m: np.ndarray, hi_m: np.ndarray) -> dict[str, float]:
        lo = target[lo_m]
        hi = target[hi_m]
        lo = lo[np.isfinite(lo)]
        hi = hi[np.isfinite(hi)]
        if lo.size < MIN_STRATUM_SIZE or hi.size < MIN_STRATUM_SIZE:
            return {"u": float("nan"), "p": float("nan"), "cliff": float("nan")}
        u, p = mannwhitneyu(lo, hi, alternative="two-sided")
        return {"u": float(u), "p": float(p), "cliff": _cliff_delta(lo, hi)}

    mwu_vl = _mwu_with_cliff(value_loss_next, low_m, high_m)
    mwu_ac = _mwu_with_cliff(accuracy_next, low_m, high_m)
    mwu_ad = _mwu_with_cliff(adapt_target, low_a, high_a)

    # -- Continuous (robustness) ---------------------------------------
    def _spearman(x: FloatArray, y: FloatArray) -> dict[str, float]:
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < MIN_TRIALS_REQUIRED:
            return {"rho": float("nan"), "p": float("nan"), "n": int(mask.sum())}
        rho, p = spearmanr(x[mask], y[mask])
        return {
            "rho": float(rho) if np.isfinite(rho) else float("nan"),
            "p": float(p) if np.isfinite(p) else float("nan"),
            "n": int(mask.sum()),
        }

    sp_vl = _spearman(theta_t[valid_main], -value_loss_next[valid_main])
    sp_ac = _spearman(theta_t[valid_main], accuracy_next[valid_main])
    sp_ad = _spearman(theta_t[adapt_valid], adapt_target[adapt_valid])

    # -- Null 1: random 50/50 gate ------------------------------------
    rg_vl = _random_gate_distribution(
        value_loss_next, valid_main, kind="lower_better", observed=dq_vl, rng=rng
    )
    rg_ac = _random_gate_distribution(
        accuracy_next, valid_main, kind="higher_better", observed=dq_ac, rng=rng
    )
    rg_ad = _random_gate_distribution(
        adapt_target, adapt_valid, kind="higher_better", observed=dq_ad, rng=rng
    )

    # -- Null 2: shuffled theta (primary null) ------------------------
    sh_vl = _shuffled_theta_null(
        theta_t,
        value_loss_next,
        valid_main,
        kind="lower_better",
        observed=dq_vl,
        rng=rng,
    )
    sh_ac = _shuffled_theta_null(
        theta_t,
        accuracy_next,
        valid_main,
        kind="higher_better",
        observed=dq_ac,
        rng=rng,
    )
    sh_ad = _shuffled_theta_null(
        theta_t,
        adapt_target,
        adapt_valid,
        kind="higher_better",
        observed=dq_ad,
        rng=rng,
    )

    # -- Behaviour-only reference -------------------------------------
    b1_vl = float(np.nanmean(value_loss_next[valid_main]))
    b1_ac = float(np.nanmean(accuracy_next[valid_main]))
    b1_ad = (
        float(np.nanmean(adapt_target[adapt_valid]))
        if adapt_valid.sum() >= MIN_STRATUM_SIZE
        else float("nan")
    )

    return {
        "subject": sid,
        "status": "OK",
        "n_trials_total": n,
        "n_main_valid_pairs": n_main,
        "n_adapt_valid_pairs": int(adapt_valid.sum()),
        "theta_q1": q1,
        "theta_q3": q3,
        "n_low": n_low,
        "n_high": n_high,
        "n_low_adapt": n_low_a,
        "n_high_adapt": n_high_a,
        "behaviour_only": {"value_loss": b1_vl, "accuracy": b1_ac, "adaptation": b1_ad},
        "value_loss": {
            "mean_low": mvl_lo,
            "mean_high": mvl_hi,
            "delta_q": dq_vl,
            "mwu_u": mwu_vl["u"],
            "mwu_p": mwu_vl["p"],
            "cliff_delta": mwu_vl["cliff"],
            "spearman_rho": sp_vl["rho"],
            "spearman_p": sp_vl["p"],
            "spearman_n": sp_vl["n"],
            "null_shuffled_theta": sh_vl,
            "null_random_gate": rg_vl,
        },
        "accuracy": {
            "mean_low": mac_lo,
            "mean_high": mac_hi,
            "delta_q": dq_ac,
            "mwu_u": mwu_ac["u"],
            "mwu_p": mwu_ac["p"],
            "cliff_delta": mwu_ac["cliff"],
            "spearman_rho": sp_ac["rho"],
            "spearman_p": sp_ac["p"],
            "spearman_n": sp_ac["n"],
            "null_shuffled_theta": sh_ac,
            "null_random_gate": rg_ac,
        },
        "adaptation": {
            "mean_low": mad_lo,
            "mean_high": mad_hi,
            "delta_q": dq_ad,
            "mwu_u": mwu_ad["u"],
            "mwu_p": mwu_ad["p"],
            "cliff_delta": mwu_ad["cliff"],
            "spearman_rho": sp_ad["rho"],
            "spearman_p": sp_ad["p"],
            "spearman_n": sp_ad["n"],
            "null_shuffled_theta": sh_ad,
            "null_random_gate": rg_ad,
        },
    }


# --- Null-distribution builders -----------------------------------------


def _random_gate_distribution(
    target: FloatArray,
    valid: np.ndarray,
    *,
    kind: str,
    observed: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """1000 random 50/50 splits of valid-target values; two-sided empirical p."""
    tgt = target[valid]
    tgt = tgt[np.isfinite(tgt)]
    n_t = tgt.size
    if n_t < 2 * MIN_STRATUM_SIZE or not np.isfinite(observed):
        return {
            "n_iter": N_RANDOM_SPLITS,
            "mean": float("nan"),
            "std": float("nan"),
            "p_empirical": float("nan"),
        }
    half = n_t // 2
    deltas = np.empty(N_RANDOM_SPLITS, dtype=np.float64)
    idx = np.arange(n_t)
    for i in range(N_RANDOM_SPLITS):
        rng.shuffle(idx)
        a = tgt[idx[:half]].mean()
        b = tgt[idx[half : 2 * half]].mean()
        # kind='lower_better' → observed dq = mean_low − mean_high.
        # Map random halves similarly (arbitrary label).
        deltas[i] = a - b if kind == "lower_better" else b - a
    p = float((np.sum(np.abs(deltas) >= abs(observed)) + 1) / (N_RANDOM_SPLITS + 1))
    return {
        "n_iter": N_RANDOM_SPLITS,
        "mean": float(deltas.mean()),
        "std": float(deltas.std(ddof=0)),
        "p_empirical": p,
    }


def _shuffled_theta_null(
    theta_t: FloatArray,
    target: FloatArray,
    valid: np.ndarray,
    *,
    kind: str,
    observed: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Primary null: permute θ across valid trials, recompute quartile Δ."""
    if not np.isfinite(observed):
        return {
            "n_iter": N_THETA_SHUFFLES,
            "mean": float("nan"),
            "std": float("nan"),
            "p_empirical": float("nan"),
            "null_deltas": [],
        }

    idx = np.where(valid)[0]
    tgt_valid = target[idx]
    theta_valid = theta_t[idx].copy()
    n_v = idx.size
    if n_v < 4 * MIN_STRATUM_SIZE:
        return {
            "n_iter": N_THETA_SHUFFLES,
            "mean": float("nan"),
            "std": float("nan"),
            "p_empirical": float("nan"),
            "null_deltas": [],
        }

    deltas = np.empty(N_THETA_SHUFFLES, dtype=np.float64)
    for i in range(N_THETA_SHUFFLES):
        shuffled = theta_valid.copy()
        rng.shuffle(shuffled)
        q1 = float(np.percentile(shuffled, 25))
        q3 = float(np.percentile(shuffled, 75))
        lo = shuffled <= q1
        hi = shuffled >= q3
        if lo.sum() < MIN_STRATUM_SIZE or hi.sum() < MIN_STRATUM_SIZE:
            deltas[i] = np.nan
            continue
        lo_vals = tgt_valid[lo]
        hi_vals = tgt_valid[hi]
        lo_vals = lo_vals[np.isfinite(lo_vals)]
        hi_vals = hi_vals[np.isfinite(hi_vals)]
        if lo_vals.size < MIN_STRATUM_SIZE or hi_vals.size < MIN_STRATUM_SIZE:
            deltas[i] = np.nan
            continue
        if kind == "lower_better":
            deltas[i] = lo_vals.mean() - hi_vals.mean()
        else:
            deltas[i] = hi_vals.mean() - lo_vals.mean()

    finite = deltas[np.isfinite(deltas)]
    if finite.size < N_THETA_SHUFFLES // 10:
        return {
            "n_iter": N_THETA_SHUFFLES,
            "mean": float("nan"),
            "std": float("nan"),
            "p_empirical": float("nan"),
            "null_deltas": [],
        }
    p = float((np.sum(np.abs(finite) >= abs(observed)) + 1) / (finite.size + 1))
    return {
        "n_iter": N_THETA_SHUFFLES,
        "n_finite": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "p_empirical": p,
    }


# --- Group-level aggregation --------------------------------------------


def _aggregate_group(per_subject: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in per_subject if r.get("status") == "OK"]
    n_total = len(per_subject)
    n_valid = len(valid)

    def _gather(metric: str, field: str = "delta_q") -> FloatArray:
        return np.asarray(
            [r[metric][field] for r in valid if np.isfinite(r[metric][field])],
            dtype=np.float64,
        )

    def _collect_pvals(metric: str) -> FloatArray:
        return np.asarray(
            [r[metric]["null_shuffled_theta"]["p_empirical"] for r in valid],
            dtype=np.float64,
        )

    def _wilcoxon(deltas: FloatArray) -> dict[str, float]:
        d = deltas[np.isfinite(deltas)]
        if d.size < 3 or np.all(d == 0.0):
            return {"stat": float("nan"), "p": float("nan"), "n": int(d.size)}
        try:
            s, p = wilcoxon(d, alternative="two-sided", zero_method="zsplit")
        except ValueError:
            return {"stat": float("nan"), "p": float("nan"), "n": int(d.size)}
        return {"stat": float(s), "p": float(p), "n": int(d.size)}

    def _sign_binom(deltas: FloatArray) -> dict[str, float]:
        d = deltas[np.isfinite(deltas)]
        if d.size == 0:
            return {"n_positive": 0, "n_total": 0, "p": float("nan")}
        n_pos = int(np.sum(d > 0))
        n = int(d.size)
        from scipy.stats import binomtest

        bt = binomtest(n_pos, n=n, p=0.5, alternative="two-sided")
        return {"n_positive": n_pos, "n_total": n, "p": float(bt.pvalue)}

    summary: dict[str, Any] = {
        "n_subjects_total": n_total,
        "n_subjects_valid": n_valid,
    }

    for metric in ("value_loss", "accuracy", "adaptation"):
        deltas = _gather(metric)
        pvals = _collect_pvals(metric)
        qvals = _bh_fdr(pvals)
        n_sig = int(np.sum(pvals < ALPHA)) if pvals.size else 0
        n_sig_fdr = int(np.sum(qvals < ALPHA)) if qvals.size else 0
        wil = _wilcoxon(deltas)
        sign = _sign_binom(deltas)
        summary[metric] = {
            "mean_delta_q": float(deltas.mean()) if deltas.size else float("nan"),
            "median_delta_q": float(np.median(deltas)) if deltas.size else float("nan"),
            "std_delta_q": float(deltas.std(ddof=0)) if deltas.size else float("nan"),
            "n_subjects_with_delta": int(deltas.size),
            "n_subjects_positive": int(np.sum(deltas > 0)),
            "n_subjects_negative": int(np.sum(deltas < 0)),
            "sign_binomial": sign,
            "wilcoxon_signed_rank": wil,
            "subject_p_shuffled_theta": [float(p) for p in pvals],
            "subject_q_shuffled_theta_bh": [float(q) for q in qvals],
            "n_subjects_significant_raw": n_sig,
            "n_subjects_significant_fdr": n_sig_fdr,
        }

    return summary


def _classify_interpretation(group: dict[str, Any]) -> dict[str, str]:
    """Apply the strict interpretation rules A/B/C/D to the group summary."""
    verdicts: dict[str, str] = {}
    for metric in ("value_loss", "accuracy", "adaptation"):
        g = group[metric]
        mean_d = g["mean_delta_q"]
        wil_p = g["wilcoxon_signed_rank"]["p"]
        n_sig_fdr = g["n_subjects_significant_fdr"]

        if not np.isfinite(mean_d) or not np.isfinite(wil_p):
            verdicts[metric] = "UNDETERMINED"
            continue
        if mean_d > 0 and wil_p < ALPHA and n_sig_fdr >= 2:
            verdicts[metric] = "USEFUL (Case A — positive utility, survives null + FDR)"
        elif mean_d < 0 and wil_p < ALPHA:
            verdicts[metric] = "MALADAPTIVE (Case C — elevated θ predicts worse outcomes)"
        elif abs(mean_d) < 1e-6 or wil_p >= ALPHA:
            verdicts[metric] = "NULL (Case B — FMθ exists but not decision-relevant here)"
        else:
            verdicts[metric] = "MIXED"

    signs = {m: verdicts[m].split()[0] for m in verdicts}
    agreed = len(set(signs.values())) == 1
    verdicts["_across_metrics"] = (
        "CONSISTENT"
        if agreed
        else "DISAGREEMENT across primary metrics — do not collapse into one sentence"
    )
    return verdicts


# --- Runner --------------------------------------------------------------


def run_delta_q_analysis(
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
        f"Predictive alignment: theta(t) → targets(t+1); quartile split; "
        f"N_RANDOM_SPLITS={N_RANDOM_SPLITS}  N_THETA_SHUFFLES={N_THETA_SHUFFLES}"
    )
    print()

    rng = np.random.default_rng(seed)
    per_subject: list[dict[str, Any]] = []
    for sid in subjects:
        print(f"  {sid}...", end=" ", flush=True)
        try:
            subject = loader.load_subject(sid)
            row = _analyze_subject(subject, rng=rng)
        except Exception as exc:
            row = {"subject": sid, "status": "ERROR", "reason": str(exc)}
        per_subject.append(row)

        if row.get("status") == "OK":
            dqv = row["value_loss"]["delta_q"]
            dqa = row["accuracy"]["delta_q"]
            dqd = row["adaptation"]["delta_q"]
            pv = row["value_loss"]["null_shuffled_theta"]["p_empirical"]
            pa = row["accuracy"]["null_shuffled_theta"]["p_empirical"]
            print(
                f"ΔQ_vl={dqv:+.4f}(p={pv:.3f})  "
                f"ΔQ_ac={dqa:+.4f}(p={pa:.3f})  "
                f"ΔQ_ad={dqd:+.4f}  "
                f"n_low/high={row['n_low']}/{row['n_high']}"
            )
        else:
            print(f"{row['status']}: {row.get('reason', '')}")

    group = _aggregate_group(per_subject)
    interp = _classify_interpretation(group)

    # Baseline summary (B1 is subject-level already; B2/B3 are per-subject nulls).
    baseline_summary = {
        "B1_behavior_only": "per-subject means reported in each subject.behaviour_only",
        "B2_random_gate": (
            f"{N_RANDOM_SPLITS} random 50/50 splits per metric per subject; "
            "two-sided empirical p vs |observed Δ|"
        ),
        "B3_shuffled_theta": (
            f"{N_THETA_SHUFFLES} within-subject θ permutations per metric; "
            "quartile-based Δ recomputed each iteration; primary null"
        ),
    }

    limitations = [
        "Internal replication on a single dataset (ds003458). Not peer-reviewed.",
        "Arm values for value_loss / accuracy are the task's programmed reward "
        "probabilities, not participant-internal Q-estimates. Q-learned arm "
        "values are not used for these metrics to avoid circularity with the "
        "same Q-learning model that generated RPE.",
        "Adaptation metric assumes RT ~ post-feedback motor readiness; RT was "
        "derived from (Response.onset - Stimulus.onset) because response_time "
        "column in events.tsv is 'n/a'.",
        "Trial indexing uses the i-th completed (Stim, Resp, Feedback) triple; "
        "subjects with missed responses may have minor misalignment against "
        "the 480-trial oscillating schedule (typically 0-2 trials).",
        "Group null via pooled subject-level shuffled-theta p-values + BH-FDR. "
        "No full cross-subject permutation was run.",
    ]

    return {
        "metadata": {
            "repo_task": "ds003458 ΔQ_gate utility benchmark",
            "dataset": "OpenNeuro ds003458 v1.1.0 (Cavanagh 2021)",
            "date": datetime.now(UTC).isoformat(),
            "random_seed": seed,
            "skipped_subjects": sorted(skipped),
            "valid_subjects_run": subjects,
        },
        "experiment_spec": {
            "neural_feature_definition": {
                "channel": NEURAL_CHANNEL,
                "preprocessing": "CSD (standard_1005 montage)",
                "time_frequency": "Morlet, freqs=arange(4,9), n_cycles=3",
                "band_hz": list(THETA_BAND),
                "window_ms": list(POWER_WINDOW_MS),
                "epoch_sec": list(EPOCH_SEC),
                "source_of_truth": "imported from ds003458_csd_analysis (pre-validated)",
            },
            "predictive_alignment": "theta_power(t) → decision_quality(t+1)",
            "split": {
                "low": "theta_t <= 25th percentile",
                "high": "theta_t >= 75th percentile",
                "excluded": "middle 50% in quartile contrast",
            },
            "metrics": {
                "value_loss": "best_arm_value(t+1) - chosen_arm_value(t+1), lower better; "
                "delta_q = mean_low - mean_high (positive = HIGH theta helps)",
                "accuracy": "1[chosen(t+1) == argmax_arm(t+1)], higher better; "
                "delta_q = mean_high - mean_low (positive = HIGH theta helps)",
                "adaptation": "-(zRT(t+1) - zRT(t)) | RPE(t)<0, higher=faster; "
                "delta_q = mean_high - mean_low (positive = HIGH theta helps)",
            },
            "statistical_tests": [
                "Mann-Whitney U (two-sided) on LOW vs HIGH per metric",
                "Cliff delta on LOW vs HIGH per metric",
                "Spearman rho (continuous secondary)",
                "Shuffled-theta empirical p (primary null, 1000x)",
                "Random 50/50-gate empirical p (secondary null, 1000x)",
                "Wilcoxon signed-rank on subject deltas vs 0 (group)",
                "Two-sided binomial on sign of subject deltas (group)",
                "Benjamini-Hochberg FDR on shuffled-theta subject p-values",
            ],
            "n_random_splits": N_RANDOM_SPLITS,
            "n_theta_shuffles": N_THETA_SHUFFLES,
            "alpha": ALPHA,
            "min_trials_required": MIN_TRIALS_REQUIRED,
            "min_stratum_size": MIN_STRATUM_SIZE,
        },
        "subject_results": per_subject,
        "group_results": group,
        "baselines": baseline_summary,
        "interpretation": interp,
        "limitations": limitations,
    }


def save_results(results: dict[str, Any], output_dir: str | Path = "results") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    path = out_dir / f"ds003458_delta_q_{date_str}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def main() -> None:
    print("=" * 72)
    print("  NEUROPHASE · ds003458 ΔQ_gate utility benchmark")
    print("=" * 72)
    print()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_delta_q_analysis()
    path = save_results(results)
    print()
    print(f"Results saved to: {path}")
    print()
    print("Group summary:")
    for m in ("value_loss", "accuracy", "adaptation"):
        g = results["group_results"][m]
        print(
            f"  {m:<12}: mean Δ={g['mean_delta_q']:+.4f}  "
            f"median={g['median_delta_q']:+.4f}  "
            f"pos/total={g['n_subjects_positive']}/{g['n_subjects_with_delta']}  "
            f"Wilcoxon p={g['wilcoxon_signed_rank']['p']:.4f}  "
            f"sig(FDR)={g['n_subjects_significant_fdr']}"
        )
    print()
    print("Interpretation:")
    for k, v in results["interpretation"].items():
        print(f"  {k:<18}: {v}")


if __name__ == "__main__":
    main()
