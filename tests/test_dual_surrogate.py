"""Tests for dual surrogate verification (PATH 2)."""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.benchmarks.neural_phase_generator import (
    generate_neural_phase_trace,
    generate_synthetic_market_phase,
)
from neurophase.metrics.plv_verdict import dual_surrogate_test
from neurophase.validation.surrogates import time_reversal

N_SAMPLES = 4096
FS = 256.0
SEED = 42


class TestTimeReversal:
    def test_preserves_spectrum(self) -> None:
        """PSD before/after reversal identical to 1e-10."""
        rng = np.random.default_rng(SEED)
        y = rng.standard_normal(1024).astype(np.float64)
        y_rev = time_reversal(y, rng=rng)
        psd_orig = np.abs(np.fft.rfft(y)) ** 2
        psd_rev = np.abs(np.fft.rfft(y_rev)) ** 2
        np.testing.assert_allclose(psd_orig, psd_rev, atol=1e-10)

    def test_reversal_is_deterministic(self) -> None:
        """Time reversal does not depend on rng."""
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(999)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r1 = time_reversal(y, rng=rng1)
        r2 = time_reversal(y, rng=rng2)
        np.testing.assert_array_equal(r1, r2)

    def test_reversal_output(self) -> None:
        """y[::-1] is the output."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rng = np.random.default_rng(SEED)
        result = time_reversal(y, rng=rng)
        np.testing.assert_array_equal(result, np.array([5.0, 4.0, 3.0, 2.0, 1.0]))


class TestDualSurrogate:
    @pytest.fixture()
    def phi_market(self) -> np.ndarray:
        return generate_synthetic_market_phase(n_samples=N_SAMPLES, fs=FS, seed=SEED)

    def test_null_both_not_significant(self, phi_market: np.ndarray) -> None:
        """k=0 → both surrogates give p > 0.05."""
        trace = generate_neural_phase_trace(
            phi_market,
            n_samples=N_SAMPLES,
            fs=FS,
            coupling_k=0.0,
            seed=SEED,
        )
        result = dual_surrogate_test(
            trace.phi_neural,
            trace.phi_market,
            n_surrogates=200,
            seed=SEED,
        )
        assert not result.both_significant, (
            f"k=0 should not be significant: "
            f"p_cs={result.p_cyclic_shift}, p_tr={result.p_time_reversal}"
        )

    def test_coupled_both_significant(self, phi_market: np.ndarray) -> None:
        """k=5 → both surrogates give p < 0.05."""
        trace = generate_neural_phase_trace(
            phi_market,
            n_samples=N_SAMPLES,
            fs=FS,
            coupling_k=5.0,
            seed=SEED,
        )
        result = dual_surrogate_test(
            trace.phi_neural,
            trace.phi_market,
            n_surrogates=200,
            seed=SEED,
        )
        assert result.both_significant, (
            f"k=5 should be significant: "
            f"p_cs={result.p_cyclic_shift}, p_tr={result.p_time_reversal}"
        )

    def test_result_has_both_p_values(self, phi_market: np.ndarray) -> None:
        trace = generate_neural_phase_trace(
            phi_market,
            n_samples=N_SAMPLES,
            fs=FS,
            coupling_k=1.0,
            seed=SEED,
        )
        result = dual_surrogate_test(
            trace.phi_neural,
            trace.phi_market,
            n_surrogates=50,
            seed=SEED,
        )
        assert 0.0 <= result.p_cyclic_shift <= 1.0
        assert 0.0 <= result.p_time_reversal <= 1.0
        assert isinstance(result.directional, bool)

    def test_reversal_independent_of_cyclic(self, phi_market: np.ndarray) -> None:
        """Two surrogate distributions should differ."""
        trace = generate_neural_phase_trace(
            phi_market,
            n_samples=N_SAMPLES,
            fs=FS,
            coupling_k=1.0,
            seed=SEED,
        )
        result = dual_surrogate_test(
            trace.phi_neural,
            trace.phi_market,
            n_surrogates=200,
            seed=SEED,
        )
        # p-values from different surrogate strategies should not be identical
        # (they CAN be close, but exact equality is astronomically unlikely)
        assert result.p_cyclic_shift != result.p_time_reversal, (
            "Cyclic-shift and time-reversal gave identical p-values — "
            "likely sharing the same null distribution"
        )
