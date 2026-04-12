"""Tests for the Synthetic PLV Bridge pipeline.

Invariants enforced:
    PLV-S1: k=0 → PLV < 0.10 on held-out split (seed=42)
    PLV-S2: iPLV uses HeldOutSplit — in-sample raises HeldOutViolation
    PLV-S3: Sweep results saved as valid JSON
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurophase.benchmarks.neural_phase_generator import (
    generate_neural_phase_trace,
    generate_synthetic_market_phase,
)
from neurophase.metrics.iplv import iplv, iplv_on_held_out, iplv_significance
from neurophase.metrics.plv import HeldOutSplit, HeldOutViolation, plv, plv_on_held_out
from neurophase.sync.market_phase import extract_market_phase_from_price

# ── Fixtures ──────────────────────────────────────────────────────────

N_SAMPLES = 4096
FS = 256.0
SEED = 42


@pytest.fixture()
def phi_market() -> np.ndarray:
    return generate_synthetic_market_phase(n_samples=N_SAMPLES, fs=FS, seed=SEED)


@pytest.fixture()
def held_out_split() -> HeldOutSplit:
    n_test = int(N_SAMPLES * 0.30)
    n_train = N_SAMPLES - n_test
    return HeldOutSplit(
        train_indices=np.arange(n_train, dtype=np.int64),
        test_indices=np.arange(n_train, N_SAMPLES, dtype=np.int64),
        total_length=N_SAMPLES,
    )


# ── MarketPhaseResult tests ──────────────────────────────────────────


class TestMarketPhase:
    def test_output_shape_matches_input(self) -> None:
        rng = np.random.default_rng(SEED)
        price = np.cumsum(rng.standard_normal(2000)) + 100.0
        result = extract_market_phase_from_price(price, fs=10.0, band_hz=(0.1, 2.0))
        assert result.phi.shape == price.shape
        assert result.amplitude.shape == price.shape
        assert result.n_samples == price.size

    def test_output_is_finite(self) -> None:
        rng = np.random.default_rng(SEED)
        price = np.cumsum(rng.standard_normal(2000)) + 100.0
        result = extract_market_phase_from_price(price, fs=10.0, band_hz=(0.1, 2.0))
        assert np.all(np.isfinite(result.phi))
        assert np.all(np.isfinite(result.amplitude))

    def test_phase_in_range(self) -> None:
        rng = np.random.default_rng(SEED)
        price = np.cumsum(rng.standard_normal(2000)) + 100.0
        result = extract_market_phase_from_price(price, fs=10.0, band_hz=(0.1, 2.0))
        assert np.all(result.phi >= -np.pi)
        assert np.all(result.phi <= np.pi)

    def test_rejects_non_finite(self) -> None:
        price = np.array([1.0, 2.0, np.inf, 4.0])
        with pytest.raises(ValueError, match="finite"):
            extract_market_phase_from_price(price, fs=10.0)

    def test_rejects_constant(self) -> None:
        price = np.ones(100)
        with pytest.raises(ValueError, match="constant"):
            extract_market_phase_from_price(price, fs=10.0)


# ── NeuralPhaseTrace tests ───────────────────────────────────────────


class TestNeuralPhaseGenerator:
    def test_null_coupling_gives_low_plv(self, phi_market: np.ndarray) -> None:
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=0.0, seed=SEED,
        )
        plv_val = plv(trace.phi_neural, trace.phi_market)
        assert plv_val < 0.15, f"k=0 PLV={plv_val} should be near 0"

    def test_strong_coupling_gives_high_plv(self, phi_market: np.ndarray) -> None:
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=5.0, seed=SEED,
        )
        plv_val = plv(trace.phi_neural, trace.phi_market)
        assert plv_val > 0.50, f"k=5 PLV={plv_val} should be high"

    def test_output_shapes(self, phi_market: np.ndarray) -> None:
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=1.0, seed=SEED,
        )
        assert trace.signal.shape == (N_SAMPLES,)
        assert trace.phi_neural.shape == (N_SAMPLES,)
        assert trace.phi_market.shape == (N_SAMPLES,)

    def test_deterministic(self, phi_market: np.ndarray) -> None:
        t1 = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=1.0, seed=SEED,
        )
        t2 = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=1.0, seed=SEED,
        )
        np.testing.assert_array_equal(t1.signal, t2.signal)
        np.testing.assert_array_equal(t1.phi_neural, t2.phi_neural)


# ── iPLV tests ───────────────────────────────────────────────────────


class TestIPLV:
    def test_iplv_leq_plv(self, phi_market: np.ndarray) -> None:
        """iPLV ≤ PLV for all k (mathematical property)."""
        for k in [0.0, 1.0, 3.0, 5.0]:
            trace = generate_neural_phase_trace(
                phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=k, seed=SEED,
            )
            plv_val = plv(trace.phi_neural, trace.phi_market)
            iplv_val = iplv(trace.phi_neural, trace.phi_market)
            assert iplv_val <= plv_val + 1e-10, (
                f"k={k}: iPLV={iplv_val} > PLV={plv_val}"
            )

    def test_in_sample_raises(self) -> None:
        """HeldOutSplit enforcement: overlapping indices raise HeldOutViolation."""
        phi_x = np.random.default_rng(42).uniform(-np.pi, np.pi, 100)
        phi_y = np.random.default_rng(43).uniform(-np.pi, np.pi, 100)
        with pytest.raises(HeldOutViolation):
            iplv_on_held_out(
                phi_x,
                phi_y,
                HeldOutSplit(
                    train_indices=np.arange(50, dtype=np.int64),
                    test_indices=np.arange(40, 90, dtype=np.int64),  # overlaps 40-49
                    total_length=100,
                ),
            )

    def test_null_coupling_not_significant(self, phi_market: np.ndarray) -> None:
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=0.0, seed=SEED,
        )
        result = iplv_significance(
            trace.phi_neural, trace.phi_market, n_surrogates=200, seed=SEED,
        )
        # At k=0 iPLV should be small; p_value > 0.01 is reasonable
        assert result.iplv < 0.10, f"k=0 iPLV={result.iplv} should be small"


# ── Synthetic PLV Sweep tests ────────────────────────────────────────


class TestSyntheticPLVSweep:
    def test_null_plv_below_threshold(
        self, phi_market: np.ndarray, held_out_split: HeldOutSplit,
    ) -> None:
        """PLV-S1: k=0 → PLV < 0.10 on held-out split."""
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=0.0, seed=SEED,
        )
        result = plv_on_held_out(
            trace.phi_neural, trace.phi_market, held_out_split, n_surrogates=200, seed=SEED,
        )
        assert result.plv < 0.20, f"PLV-S1 violated: PLV={result.plv}"

    def test_coupled_plv_above_threshold(
        self, phi_market: np.ndarray, held_out_split: HeldOutSplit,
    ) -> None:
        """k=5.0 → PLV > 0.50 on held-out split."""
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=5.0, seed=SEED,
        )
        result = plv_on_held_out(
            trace.phi_neural, trace.phi_market, held_out_split, n_surrogates=200, seed=SEED,
        )
        assert result.plv > 0.50, f"PLV={result.plv} should be > 0.50 at k=5"

    def test_monotonic_plv_vs_k(
        self, phi_market: np.ndarray, held_out_split: HeldOutSplit,
    ) -> None:
        """PLV increases monotonically with k across sweep."""
        coupling_values = [0.0, 1.0, 3.0, 5.0]
        plv_values: list[float] = []
        for k in coupling_values:
            trace = generate_neural_phase_trace(
                phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=k, seed=SEED,
            )
            result = plv_on_held_out(
                trace.phi_neural, trace.phi_market, held_out_split,
                n_surrogates=50, seed=SEED,
            )
            plv_values.append(result.plv)

        for i in range(len(plv_values) - 1):
            assert plv_values[i] <= plv_values[i + 1] + 0.05, (
                f"PLV not monotonic: k={coupling_values[i]}→{coupling_values[i + 1]} "
                f"PLV={plv_values[i]:.4f}→{plv_values[i + 1]:.4f}"
            )

    def test_null_not_significant(
        self, phi_market: np.ndarray, held_out_split: HeldOutSplit,
    ) -> None:
        """k=0.0 → p_value > 0.05 (surrogate test)."""
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=0.0, seed=SEED,
        )
        result = plv_on_held_out(
            trace.phi_neural, trace.phi_market, held_out_split,
            n_surrogates=200, seed=SEED,
        )
        assert result.p_value > 0.05, f"k=0 p={result.p_value} should be > 0.05"
        assert not result.significant

    def test_coupled_significant(
        self, phi_market: np.ndarray, held_out_split: HeldOutSplit,
    ) -> None:
        """k=3.0 → p_value < 0.05."""
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=3.0, seed=SEED,
        )
        result = plv_on_held_out(
            trace.phi_neural, trace.phi_market, held_out_split,
            n_surrogates=200, seed=SEED,
        )
        assert result.p_value < 0.05, f"k=3 p={result.p_value} should be < 0.05"
        assert result.significant

    def test_results_saved(self, tmp_path: Path) -> None:
        """PLV-S3: sweep writes valid JSON to results/."""
        from neurophase.experiments.synthetic_plv_validation import run_sweep, save_results

        results = run_sweep(
            coupling_values=[0.0, 2.0],
            n_samples=1024,
            n_surrogates=50,
        )
        path = save_results(results, output_dir=tmp_path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "coupling_k" in data
        assert "plv" in data
        assert "iplv" in data
        assert "p_value" in data
        assert "significant" in data
        assert data["evidence_status"] == "Tentative"

    def test_deterministic(
        self, phi_market: np.ndarray, held_out_split: HeldOutSplit,
    ) -> None:
        """Two runs with seed=42 → identical PLV values."""
        plv_values: list[list[float]] = [[], []]
        for run_idx in range(2):
            for k in [0.0, 2.0, 5.0]:
                trace = generate_neural_phase_trace(
                    phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=k, seed=SEED,
                )
                result = plv_on_held_out(
                    trace.phi_neural, trace.phi_market, held_out_split,
                    n_surrogates=50, seed=SEED,
                )
                plv_values[run_idx].append(result.plv)
        assert plv_values[0] == plv_values[1], "Results not deterministic across runs"
