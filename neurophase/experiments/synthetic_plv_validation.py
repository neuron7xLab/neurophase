"""Synthetic PLV validation experiment.

Sweeps coupling_k ∈ [0.0, 0.5, 1.0, 2.0, 3.0, 5.0] and for each:
    1. Generates synthetic market phase (sinusoidal + noise)
    2. Generates NeuralPhaseTrace with that coupling
    3. Computes iPLV on held-out split (last 30% of samples)
    4. Runs NullModelHarness (1000 cyclic-shift surrogates)
    5. Records: k · PLV · iPLV · p_value · significant

Expected results:
    k=0.0 → PLV ≈ 0.05 ± 0.03, p > 0.05, significant=False
    k=0.5 → PLV > 0.10,         p < 0.05, significant=True
    k=5.0 → PLV > 0.80,         p < 0.001, significant=True

Output: results/synthetic_plv_sweep_YYYYMMDD.json

Run:
    python -m neurophase.experiments.synthetic_plv_validation
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from neurophase.benchmarks.neural_phase_generator import (
    generate_neural_phase_trace,
    generate_synthetic_market_phase,
)
from neurophase.metrics.iplv import iplv
from neurophase.metrics.plv import HeldOutSplit, plv_on_held_out

# Sweep configuration
DEFAULT_COUPLING_VALUES: list[float] = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
DEFAULT_N_SAMPLES: int = 4096
DEFAULT_FS: float = 256.0
DEFAULT_SEED: int = 42
DEFAULT_N_SURROGATES: int = 1000
DEFAULT_HELD_OUT_FRACTION: float = 0.30


def run_sweep(
    coupling_values: list[float] | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    fs: float = DEFAULT_FS,
    seed: int = DEFAULT_SEED,
    n_surrogates: int = DEFAULT_N_SURROGATES,
    held_out_fraction: float = DEFAULT_HELD_OUT_FRACTION,
) -> dict[str, Any]:
    """Run the full coupling sweep and return structured results.

    Returns
    -------
    dict
        JSON-serializable results dictionary.
    """
    if coupling_values is None:
        coupling_values = DEFAULT_COUPLING_VALUES

    # Generate shared synthetic market phase
    phi_market = generate_synthetic_market_phase(
        n_samples=n_samples, fs=fs, seed=seed,
    )

    # Build held-out split: last 30% is test
    n_test = int(n_samples * held_out_fraction)
    n_train = n_samples - n_test
    train_indices = np.arange(n_train, dtype=np.int64)
    test_indices = np.arange(n_train, n_samples, dtype=np.int64)

    split = HeldOutSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        total_length=n_samples,
    )

    results_plv: list[float] = []
    results_iplv: list[float] = []
    results_p_value: list[float] = []
    results_significant: list[bool] = []

    for k in coupling_values:
        # Generate neural trace with this coupling
        trace = generate_neural_phase_trace(
            phi_market,
            n_samples=n_samples,
            fs=fs,
            coupling_k=k,
            seed=seed,
        )

        # Compute PLV on held-out partition
        plv_result = plv_on_held_out(
            trace.phi_neural,
            trace.phi_market,
            split,
            n_surrogates=n_surrogates,
            seed=seed,
        )

        # Compute iPLV on held-out partition
        test_neural = split.test_slice(trace.phi_neural)
        test_market = split.test_slice(trace.phi_market)
        iplv_val = iplv(test_neural, test_market)

        results_plv.append(plv_result.plv)
        results_iplv.append(iplv_val)
        results_p_value.append(plv_result.p_value)
        results_significant.append(plv_result.significant)

        print(
            f"  k={k:4.1f}  PLV={plv_result.plv:.4f}  "
            f"iPLV={iplv_val:.4f}  "
            f"p={plv_result.p_value:.4f}  "
            f"sig={'YES' if plv_result.significant else 'no '}"
        )

    return {
        "experiment": "synthetic_plv_sweep",
        "timestamp": datetime.now(UTC).isoformat(),
        "coupling_k": coupling_values,
        "plv": results_plv,
        "iplv": results_iplv,
        "p_value": results_p_value,
        "significant": results_significant,
        "seed": seed,
        "n_surrogates": n_surrogates,
        "n_samples": n_samples,
        "fs": fs,
        "held_out_fraction": held_out_fraction,
        "evidence_status": "Tentative",
        "note": "Synthetic validation only — not real EEG confirmation",
    }


def save_results(results: dict[str, Any], output_dir: str | Path = "results") -> Path:
    """Save sweep results to a timestamped JSON file."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y%m%d")
    filename = f"synthetic_plv_sweep_{date_str}.json"
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def main() -> None:
    """Entry point for python -m neurophase.experiments.synthetic_plv_validation."""
    print("=" * 60)
    print("  NEUROPHASE · Synthetic PLV Validation Sweep")
    print("=" * 60)
    print()

    results = run_sweep()

    print()
    path = save_results(results)
    print(f"Results saved to: {path}")
    print()

    # Verify basic expectations
    plv_values = results["plv"]
    p_values = results["p_value"]
    sig_values = results["significant"]

    ok = True

    # k=0 should not be significant
    if sig_values[0]:
        print("WARNING: k=0 is significant — false positive!")
        ok = False
    else:
        print("PASS: k=0 is not significant (null confirmed)")

    # k=5 should be significant
    if not sig_values[-1]:
        print("WARNING: k=5 is not significant — pipeline issue!")
        ok = False
    else:
        print(f"PASS: k=5 is significant (PLV={plv_values[-1]:.4f}, p={p_values[-1]:.4f})")

    # PLV at max k should exceed PLV at k=0 (overall trend)
    # Note: Kuramoto coupling has a phase transition — PLV may be
    # non-monotonic at low k where coupling is sub-critical.
    if plv_values[-1] > plv_values[0]:
        print("PASS: PLV at max-k exceeds PLV at null-k")
    else:
        print("WARNING: PLV at max-k does not exceed PLV at null-k")
        ok = False

    print()
    if ok:
        print("ALL CHECKS PASSED — pipeline is calibrated")
    else:
        print("SOME CHECKS FAILED — investigate before proceeding")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
