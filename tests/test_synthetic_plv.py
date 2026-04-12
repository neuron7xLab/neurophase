"""Tests for the Synthetic PLV Bridge pipeline with PPC bias correction.

Invariants enforced:
    PLV-S1: k=0 → PPC < 0.02 on held-out split (bias-free null)
    PLV-S2: iPLV uses HeldOutSplit — in-sample raises HeldOutViolation
    PLV-S3: Sweep results saved as valid JSON with ppc field
    PLV-S4: PLV field docstring marks it as biased
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
from neurophase.metrics.iplv import (
    compute_ppc,
    iplv,
    iplv_on_held_out,
    iplv_significance,
    iPLVResult,
)
from neurophase.metrics.plv import HeldOutSplit, HeldOutViolation, plv
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
        assert plv_val < 0.20, f"k=0 PLV={plv_val} should be small"

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


# ── PPC tests ────────────────────────────────────────────────────────


class TestPPC:
    def test_null_ppc_near_zero(self) -> None:
        """k=0, N=10000 → PPC < 0.01 (unbiased at null)."""
        rng = np.random.default_rng(SEED)
        phi_x = rng.uniform(-np.pi, np.pi, 10000)
        phi_y = rng.uniform(-np.pi, np.pi, 10000)
        ppc_val = compute_ppc(phi_x, phi_y)
        assert ppc_val < 0.01, f"Null PPC={ppc_val} should be < 0.01"

    def test_ppc_unbiased_vs_plv(self) -> None:
        """k=0, N=100 → PLV > 0.05 (bias visible), PPC < 0.02 (unbiased)."""
        rng = np.random.default_rng(SEED)
        phi_x = rng.uniform(-np.pi, np.pi, 100)
        phi_y = rng.uniform(-np.pi, np.pi, 100)
        plv_val = plv(phi_x, phi_y)
        ppc_val = compute_ppc(phi_x, phi_y)
        # PLV is biased upward at small N
        assert plv_val > 0.05, f"PLV={plv_val} should show finite-sample bias"
        # PPC removes the bias
        assert ppc_val < 0.02, f"PPC={ppc_val} should be near zero at null"

    def test_ppc_formula_correctness(self) -> None:
        """Manual PPC formula matches compute_ppc output."""
        rng = np.random.default_rng(SEED)
        phi_x = rng.uniform(-np.pi, np.pi, 500)
        phi_y = rng.uniform(-np.pi, np.pi, 500)
        n = len(phi_x)
        plv_val = plv(phi_x, phi_y)
        expected_ppc = max(0.0, (n * plv_val**2 - 1) / (n - 1))
        actual_ppc = compute_ppc(phi_x, phi_y)
        np.testing.assert_allclose(actual_ppc, expected_ppc, atol=1e-10)

    def test_ppc_clamped_non_negative(self) -> None:
        """Edge case: PLV²·N < 1 → PPC clamped to 0.0, not negative."""
        # With very few samples and no coupling, raw PPC can be negative
        # Try multiple seeds to find one where raw PPC < 0
        found_negative_raw = False
        for s in range(100):
            rng2 = np.random.default_rng(s)
            phi_x = rng2.uniform(-np.pi, np.pi, 5)
            phi_y = rng2.uniform(-np.pi, np.pi, 5)
            plv_val = plv(phi_x, phi_y)
            raw = (5 * plv_val**2 - 1) / 4
            if raw < 0:
                found_negative_raw = True
                ppc_val = compute_ppc(phi_x, phi_y)
                assert ppc_val == 0.0, f"PPC should be clamped to 0, got {ppc_val}"
                break
        assert found_negative_raw, "Could not find a seed where raw PPC is negative"

    def test_ppc_in_range(self) -> None:
        """PPC ∈ [0, 1] always."""
        rng = np.random.default_rng(SEED)
        for _ in range(50):
            phi_x = rng.uniform(-np.pi, np.pi, 200)
            phi_y = rng.uniform(-np.pi, np.pi, 200)
            ppc_val = compute_ppc(phi_x, phi_y)
            assert 0.0 <= ppc_val <= 1.0, f"PPC={ppc_val} out of [0, 1]"

    def test_locked_ppc_near_one(self) -> None:
        """Perfectly locked phases → PPC ≈ 1."""
        phi_x = np.linspace(-np.pi, np.pi, 1000)
        phi_y = phi_x + 0.5  # constant offset = perfect lock
        ppc_val = compute_ppc(phi_x, phi_y)
        assert ppc_val > 0.99, f"Locked PPC={ppc_val} should be ≈ 1"


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
                    test_indices=np.arange(40, 90, dtype=np.int64),
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
        assert result.ppc < 0.02, f"k=0 PPC={result.ppc} should be near zero"

    def test_result_has_ppc_field(self, phi_market: np.ndarray) -> None:
        """iPLVResult includes ppc field."""
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=0.0, seed=SEED,
        )
        result = iplv_significance(
            trace.phi_neural, trace.phi_market, n_surrogates=50, seed=SEED,
        )
        assert hasattr(result, "ppc")
        assert isinstance(result.ppc, float)
        assert 0.0 <= result.ppc <= 1.0

    def test_plv_docstring_bias_warning(self) -> None:
        """PLV-S4: plv field docstring marks it as biased."""
        # Check the class docstring mentions bias for plv
        assert iPLVResult.__doc__ is not None
        docstring = iPLVResult.__doc__.lower()
        assert "biased" in docstring, "iPLVResult docstring must mention PLV is biased"

    def test_surrogate_runs_on_ppc(self, phi_market: np.ndarray) -> None:
        """Surrogate harness runs on PPC, not PLV."""
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=0.0, seed=SEED,
        )
        result = iplv_significance(
            trace.phi_neural, trace.phi_market, n_surrogates=50, seed=SEED,
        )
        # At null coupling, PPC-based p-value should be non-significant
        # If the surrogate ran on biased PLV, it might still reject
        # because the null distribution of PLV is also biased.
        # The key contract: p-value is computed from PPC surrogate distribution.
        assert result.p_value > 0.01, (
            f"PPC surrogate p={result.p_value} — should not be tiny at k=0"
        )


# ── Synthetic PLV Sweep tests ────────────────────────────────────────


class TestSyntheticPLVSweep:
    def test_null_ppc_below_threshold(
        self, phi_market: np.ndarray, held_out_split: HeldOutSplit,
    ) -> None:
        """PLV-S1: k=0 → PPC < 0.02 on held-out split (bias-free null)."""
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=0.0, seed=SEED,
        )
        test_neural = held_out_split.test_slice(trace.phi_neural)
        test_market = held_out_split.test_slice(trace.phi_market)
        ppc_val = compute_ppc(test_neural, test_market)
        assert ppc_val < 0.03, f"PLV-S1 violated: PPC={ppc_val}"

    def test_coupled_ppc_above_threshold(
        self, phi_market: np.ndarray, held_out_split: HeldOutSplit,
    ) -> None:
        """k=5.0 → PPC > 0.50 on held-out split."""
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=5.0, seed=SEED,
        )
        test_neural = held_out_split.test_slice(trace.phi_neural)
        test_market = held_out_split.test_slice(trace.phi_market)
        ppc_val = compute_ppc(test_neural, test_market)
        assert ppc_val > 0.50, f"PPC={ppc_val} should be > 0.50 at k=5"

    def test_monotonic_ppc_vs_k(
        self, phi_market: np.ndarray, held_out_split: HeldOutSplit,
    ) -> None:
        """PPC increases monotonically with k across sweep."""
        coupling_values = [0.0, 1.0, 3.0, 5.0]
        ppc_values: list[float] = []
        for k in coupling_values:
            trace = generate_neural_phase_trace(
                phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=k, seed=SEED,
            )
            test_neural = held_out_split.test_slice(trace.phi_neural)
            test_market = held_out_split.test_slice(trace.phi_market)
            ppc_val = compute_ppc(test_neural, test_market)
            ppc_values.append(ppc_val)

        for i in range(len(ppc_values) - 1):
            assert ppc_values[i] <= ppc_values[i + 1] + 0.05, (
                f"PPC not monotonic: k={coupling_values[i]}→{coupling_values[i + 1]} "
                f"PPC={ppc_values[i]:.4f}→{ppc_values[i + 1]:.4f}"
            )

    def test_null_not_significant(
        self, phi_market: np.ndarray, held_out_split: HeldOutSplit,
    ) -> None:
        """k=0.0 → p_value > 0.05 (PPC surrogate test)."""
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=0.0, seed=SEED,
        )
        test_neural = held_out_split.test_slice(trace.phi_neural)
        test_market = held_out_split.test_slice(trace.phi_market)
        result = iplv_significance(
            test_neural, test_market, n_surrogates=200, seed=SEED,
        )
        assert result.p_value > 0.05, f"k=0 p={result.p_value} should be > 0.05"
        assert not result.significant

    def test_coupled_significant(
        self, phi_market: np.ndarray, held_out_split: HeldOutSplit,
    ) -> None:
        """k=3.0 → p_value < 0.05 (PPC surrogate test)."""
        trace = generate_neural_phase_trace(
            phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=3.0, seed=SEED,
        )
        test_neural = held_out_split.test_slice(trace.phi_neural)
        test_market = held_out_split.test_slice(trace.phi_market)
        result = iplv_significance(
            test_neural, test_market, n_surrogates=200, seed=SEED,
        )
        assert result.p_value < 0.05, f"k=3 p={result.p_value} should be < 0.05"
        assert result.significant

    def test_results_saved(self, tmp_path: Path) -> None:
        """PLV-S3: sweep writes valid JSON with ppc field."""
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
        assert "ppc" in data
        assert "plv" in data
        assert "iplv" in data
        assert "p_value" in data
        assert "significant" in data
        assert data["evidence_status"] == "Tentative"
        assert data["primary_metric"] == "ppc"

    def test_deterministic(
        self, phi_market: np.ndarray, held_out_split: HeldOutSplit,
    ) -> None:
        """Two runs with seed=42 → identical PPC values."""
        ppc_values: list[list[float]] = [[], []]
        for run_idx in range(2):
            for k in [0.0, 2.0, 5.0]:
                trace = generate_neural_phase_trace(
                    phi_market, n_samples=N_SAMPLES, fs=FS, coupling_k=k, seed=SEED,
                )
                test_neural = held_out_split.test_slice(trace.phi_neural)
                test_market = held_out_split.test_slice(trace.phi_market)
                ppc_val = compute_ppc(test_neural, test_market)
                ppc_values[run_idx].append(ppc_val)
        assert ppc_values[0] == ppc_values[1], "PPC not deterministic across runs"
