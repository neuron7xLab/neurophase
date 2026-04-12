"""ds003458 real-data PLV/PPC analysis.

Runs the three-gate PLVVerdict on each subject from ds003458
and computes group-level statistics.

Pre-registration: results/ds003458_preregistration.md
Must be committed before this script runs.

Run:
    python -m neurophase.experiments.ds003458_analysis
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from neurophase.data.ds003458_loader import DS003458Loader
from neurophase.data.eeg_preprocessor import extract_phases
from neurophase.metrics.plv_verdict import compute_verdict


def run_analysis(
    data_root: str | Path = "data/ds003458",
    n_surrogates: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run per-subject PLVVerdict and group-level test.

    Returns
    -------
    dict
        JSON-serializable results.
    """
    loader = DS003458Loader(data_root)
    subjects = loader.list_subjects()

    if not subjects:
        raise RuntimeError(f"No subjects found in {data_root}")

    print(f"Found {len(subjects)} subjects")
    print()

    per_subject: list[dict[str, Any]] = []

    for sid in subjects:
        print(f"  Processing {sid}...", end=" ", flush=True)
        try:
            sub_data = loader.load_subject(sid)
            phases = extract_phases(sub_data)

            verdict = compute_verdict(
                phases.phi_neural,
                phases.phi_market,
                coupling_k=None,  # real data — theory gate auto-passes
                n_surrogates=n_surrogates,
                seed=seed,
            )

            row: dict[str, Any] = {
                "subject": sid,
                "ppc": verdict.ppc,
                "rayleigh_R": verdict.rayleigh_r,
                "rayleigh_effect": verdict.rayleigh_effect,
                "p_cyclic": verdict.dual_surrogate.p_cyclic_shift,
                "p_reversal": verdict.dual_surrogate.p_time_reversal,
                "both_significant": verdict.dual_surrogate.both_significant,
                "directional": verdict.dual_surrogate.directional,
                "verdict": verdict.verdict,
                "n_samples": phases.n_samples,
                "n_artifacts": phases.n_artifacts_rejected,
            }
            per_subject.append(row)

            print(f"PPC={verdict.ppc:.4f}  R={verdict.rayleigh_r:.4f}  verdict={verdict.verdict}")

        except Exception as e:
            print(f"FAILED: {e}")
            per_subject.append(
                {
                    "subject": sid,
                    "ppc": None,
                    "verdict": "ERROR",
                    "error": str(e),
                }
            )

    # --- Group-level analysis ---
    valid = [r for r in per_subject if r["verdict"] != "ERROR"]
    n_total = len(valid)

    if n_total == 0:
        return {
            "experiment": "ds003458_analysis",
            "error": "No valid subjects",
            "per_subject": per_subject,
        }

    verdicts = [r["verdict"] for r in valid]
    n_confirmed = verdicts.count("CONFIRMED")
    n_marginal = verdicts.count("MARGINAL")
    n_rejected = verdicts.count("REJECTED")

    # Binomial test: is n_confirmed > chance (5%)?
    binom = binomtest(n_confirmed, n=n_total, p=0.05, alternative="greater")
    group_p = float(binom.pvalue)

    # Group PPC via Fisher z-transform
    ppc_values = [r["ppc"] for r in valid if r["ppc"] is not None]
    if ppc_values:
        # Clamp PPC to (0, 1) for arctanh
        ppc_clamped = np.clip(ppc_values, 1e-10, 1 - 1e-10)
        z_scores = np.arctanh(ppc_clamped)
        group_ppc = float(np.tanh(np.mean(z_scores)))
    else:
        group_ppc = 0.0

    # Interpretation
    confirmed_fraction = n_confirmed / n_total if n_total > 0 else 0.0
    if confirmed_fraction >= 0.60 and group_p < 0.05:
        evidence_status = "Strongly Plausible"
    elif confirmed_fraction >= 0.30:
        evidence_status = "Tentative (upgraded)"
    else:
        evidence_status = "Tentative (unchanged)"

    return {
        "experiment": "ds003458_analysis",
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": "OpenNeuro ds003458 v1.1.0",
        "preregistration": "results/ds003458_preregistration.md",
        "n_subjects_total": len(subjects),
        "n_subjects_valid": n_total,
        "n_confirmed": n_confirmed,
        "n_marginal": n_marginal,
        "n_rejected": n_rejected,
        "confirmed_fraction": confirmed_fraction,
        "group_p_value": group_p,
        "group_ppc": group_ppc,
        "evidence_status": evidence_status,
        "per_subject": per_subject,
        "analysis_params": {
            "neural_band_hz": [4.0, 8.0],
            "neural_channel": "Fz",
            "market_band_hz": [0.005, 0.05],
            "n_surrogates": n_surrogates,
            "seed": seed,
            "edge_trim_fraction": 0.05,
            "artifact_threshold_uv": 150.0,
        },
    }


def save_results(results: dict[str, Any], output_dir: str | Path = "results") -> Path:
    """Save analysis results to timestamped JSON."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    path = out_dir / f"ds003458_verdict_{date_str}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def main() -> None:
    """Entry point."""
    print("=" * 60)
    print("  NEUROPHASE · ds003458 Real Data Analysis")
    print("  Three-gate PLVVerdict on oscillating reward EEG")
    print("=" * 60)
    print()

    results = run_analysis()

    print()
    path = save_results(results)
    print(f"Results saved to: {path}")
    print()
    print(f"CONFIRMED:  {results.get('n_confirmed', 0)}/{results.get('n_subjects_valid', 0)}")
    print(f"MARGINAL:   {results.get('n_marginal', 0)}/{results.get('n_subjects_valid', 0)}")
    print(f"REJECTED:   {results.get('n_rejected', 0)}/{results.get('n_subjects_valid', 0)}")
    print(f"Group PPC:  {results.get('group_ppc', 0):.4f}")
    print(f"Group p:    {results.get('group_p_value', 1):.4f}")
    print()
    print(f"Evidence status: {results.get('evidence_status', 'unknown')}")


if __name__ == "__main__":
    main()
