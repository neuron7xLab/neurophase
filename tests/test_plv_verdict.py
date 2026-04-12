"""Tests for the three-gate PLV/PPC verdict."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.benchmarks.neural_phase_generator import (
    generate_neural_phase_trace,
    generate_synthetic_market_phase,
)
from neurophase.metrics.plv_verdict import compute_verdict

N_SAMPLES = 4096
FS = 256.0
SEED = 42


@pytest.fixture()
def phi_market() -> np.ndarray:
    return generate_synthetic_market_phase(n_samples=N_SAMPLES, fs=FS, seed=SEED)


class TestPLVVerdict:
    def test_null_rejected(self, phi_market: np.ndarray) -> None:
        """k=0 → verdict REJECTED (R < 0.10, surrogates not significant)."""
        trace = generate_neural_phase_trace(
            phi_market,
            n_samples=N_SAMPLES,
            fs=FS,
            coupling_k=0.0,
            seed=SEED,
        )
        verdict = compute_verdict(
            trace.phi_neural,
            trace.phi_market,
            coupling_k=0.0,
            n_surrogates=200,
            seed=SEED,
        )
        assert verdict.verdict == "REJECTED", (
            f"k=0 should be REJECTED, got {verdict.verdict} "
            f"(gates={verdict.gates_passed}, R={verdict.rayleigh_r:.4f}, "
            f"effect={verdict.rayleigh_effect})"
        )

    def test_strong_coupling_confirmed(self, phi_market: np.ndarray) -> None:
        """k=5 → verdict CONFIRMED."""
        trace = generate_neural_phase_trace(
            phi_market,
            n_samples=N_SAMPLES,
            fs=FS,
            coupling_k=5.0,
            seed=SEED,
        )
        verdict = compute_verdict(
            trace.phi_neural,
            trace.phi_market,
            coupling_k=5.0,
            n_surrogates=200,
            seed=SEED,
        )
        assert verdict.verdict == "CONFIRMED", (
            f"k=5 should be CONFIRMED, got {verdict.verdict} "
            f"(gates={verdict.gates_passed}, R={verdict.rayleigh_r:.4f}, "
            f"theory_delta={verdict.theory_delta})"
        )

    def test_all_gates_required(self, phi_market: np.ndarray) -> None:
        """PLV-V1: CONFIRMED requires all three gates."""
        trace = generate_neural_phase_trace(
            phi_market,
            n_samples=N_SAMPLES,
            fs=FS,
            coupling_k=5.0,
            seed=SEED,
        )
        verdict = compute_verdict(
            trace.phi_neural,
            trace.phi_market,
            coupling_k=5.0,
            n_surrogates=200,
            seed=SEED,
        )
        if verdict.verdict == "CONFIRMED":
            assert verdict.gates_passed == 3

    def test_gates_passed_count(self, phi_market: np.ndarray) -> None:
        """gates_passed ∈ {0, 1, 2, 3}."""
        for k in [0.0, 1.0, 5.0]:
            trace = generate_neural_phase_trace(
                phi_market,
                n_samples=N_SAMPLES,
                fs=FS,
                coupling_k=k,
                seed=SEED,
            )
            verdict = compute_verdict(
                trace.phi_neural,
                trace.phi_market,
                coupling_k=k,
                n_surrogates=50,
                seed=SEED,
            )
            assert 0 <= verdict.gates_passed <= 3

    def test_real_data_mode_no_theory(self, phi_market: np.ndarray) -> None:
        """Without coupling_k, theory gate auto-passes, theory_delta is None."""
        trace = generate_neural_phase_trace(
            phi_market,
            n_samples=N_SAMPLES,
            fs=FS,
            coupling_k=5.0,
            seed=SEED,
        )
        verdict = compute_verdict(
            trace.phi_neural,
            trace.phi_market,
            coupling_k=None,
            n_surrogates=200,
            seed=SEED,
        )
        assert verdict.theory_delta is None

    def test_verdict_has_rayleigh_r(self, phi_market: np.ndarray) -> None:
        trace = generate_neural_phase_trace(
            phi_market,
            n_samples=N_SAMPLES,
            fs=FS,
            coupling_k=1.0,
            seed=SEED,
        )
        verdict = compute_verdict(
            trace.phi_neural,
            trace.phi_market,
            n_surrogates=50,
            seed=SEED,
        )
        assert 0.0 <= verdict.rayleigh_r <= 1.0
        assert verdict.rayleigh_effect in {"negligible", "small", "medium", "large"}

    def test_verdict_values(self, phi_market: np.ndarray) -> None:
        """verdict is one of the three allowed strings."""
        for k in [0.0, 1.0, 3.0, 5.0]:
            trace = generate_neural_phase_trace(
                phi_market,
                n_samples=N_SAMPLES,
                fs=FS,
                coupling_k=k,
                seed=SEED,
            )
            verdict = compute_verdict(
                trace.phi_neural,
                trace.phi_market,
                coupling_k=k,
                n_surrogates=50,
                seed=SEED,
            )
            assert verdict.verdict in {"CONFIRMED", "MARGINAL", "REJECTED"}
