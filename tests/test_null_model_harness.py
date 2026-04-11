"""Tests for ``neurophase.validation`` (C1 + C2).

Covers:

* each of the three surrogate generators' null-hypothesis contracts
* ``NullModelHarness`` construction, determinism, and p-value math
* the Phipson–Smyth ``+1`` smoothing (no p = 0 for finite samples)
* rejection band behavior under the harness's own alpha
* integration: PLV-like statistic on a perfectly-coupled pair is
  rejected by cyclic-shift surrogates, while an uncoupled pair is not
* integration: phase-shuffle preserves amplitude spectrum
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from neurophase.validation.null_model import (
    DEFAULT_N_SURROGATES,
    NullModelHarness,
    NullModelResult,
)
from neurophase.validation.surrogates import (
    block_bootstrap,
    cyclic_shift,
    phase_shuffle,
)

# ---------------------------------------------------------------------------
# Surrogate generator contracts
# ---------------------------------------------------------------------------


class TestPhaseShuffle:
    def test_preserves_amplitude_spectrum(self) -> None:
        rng = np.random.default_rng(17)
        y = np.sin(np.linspace(0, 10 * np.pi, 512)) + 0.3 * rng.standard_normal(512)
        surrogate = phase_shuffle(y, rng=rng)
        orig_spec = np.abs(np.fft.rfft(y))
        surr_spec = np.abs(np.fft.rfft(surrogate))
        np.testing.assert_allclose(orig_spec, surr_spec, atol=1e-10)

    def test_returns_real_output(self) -> None:
        rng = np.random.default_rng(19)
        y = rng.standard_normal(256)
        s = phase_shuffle(y, rng=rng)
        assert np.all(np.isfinite(s))
        assert s.dtype.kind == "f"

    def test_rejects_non_1d(self) -> None:
        rng = np.random.default_rng(23)
        with pytest.raises(ValueError):
            phase_shuffle(np.zeros((3, 3)), rng=rng)

    def test_rejects_non_finite(self) -> None:
        rng = np.random.default_rng(23)
        with pytest.raises(ValueError):
            phase_shuffle(np.array([1.0, float("nan"), 2.0]), rng=rng)

    def test_different_seeds_give_different_phases(self) -> None:
        y = np.sin(np.linspace(0, 4 * np.pi, 256))
        a = phase_shuffle(y, rng=np.random.default_rng(1))
        b = phase_shuffle(y, rng=np.random.default_rng(2))
        # Same amplitude spectrum, but different realizations.
        assert not np.allclose(a, b)


class TestCyclicShift:
    def test_preserves_value_set(self) -> None:
        """A rotation is a permutation, so sorted values are identical."""
        rng = np.random.default_rng(29)
        y = rng.standard_normal(100)
        s = cyclic_shift(y, rng=rng)
        np.testing.assert_allclose(np.sort(y), np.sort(s))

    def test_preserves_autocorrelation(self) -> None:
        rng = np.random.default_rng(31)
        y = rng.standard_normal(200)
        s = cyclic_shift(y, rng=rng)

        # Normalized autocorrelation at lag 1 must match.
        def autocorr_lag1(v: np.ndarray) -> float:
            v = v - v.mean()
            return float((v[:-1] @ v[1:]) / (v @ v))

        # Not exactly preserved (rotation is not a true time-domain invariant
        # for non-periodic signals), but correlation structure is preserved
        # globally — the sum of all lags is identical up to rotation.
        # We check that a full circular autocorrelation matches.
        def circ_autocorr(v: np.ndarray) -> np.ndarray:
            V = np.fft.fft(v)
            ac = np.real(np.fft.ifft(V * np.conj(V)))
            return ac

        np.testing.assert_allclose(circ_autocorr(y), circ_autocorr(s), atol=1e-10)

    def test_is_nontrivial_rotation(self) -> None:
        """k=0 is excluded, so the surrogate must differ from the input."""
        rng = np.random.default_rng(37)
        y = np.arange(10, dtype=np.float64)
        s = cyclic_shift(y, rng=rng)
        assert not np.array_equal(y, s)

    def test_rejects_short_input(self) -> None:
        rng = np.random.default_rng(41)
        with pytest.raises(ValueError):
            cyclic_shift(np.array([1.0]), rng=rng)


class TestBlockBootstrap:
    def test_preserves_length(self) -> None:
        rng = np.random.default_rng(43)
        y = rng.standard_normal(100)
        s = block_bootstrap(y, rng=rng, block=8)
        assert s.size == y.size

    def test_block_equals_n_is_identity(self) -> None:
        """When block == n there is only one possible starting position, so
        the bootstrap collapses to the identity transformation."""
        rng = np.random.default_rng(47)
        y = np.arange(16, dtype=np.float64)
        s = block_bootstrap(y, rng=rng, block=16)
        np.testing.assert_array_equal(s, y)

    def test_small_block_diverges_from_input(self) -> None:
        rng = np.random.default_rng(53)
        y = np.arange(100, dtype=np.float64)
        s = block_bootstrap(y, rng=rng, block=2)
        # With block=2 and 50 blocks, almost certainly the bootstrap
        # differs from the original.
        assert not np.array_equal(s, y)

    def test_rejects_invalid_block(self) -> None:
        rng = np.random.default_rng(59)
        y = np.arange(10, dtype=np.float64)
        with pytest.raises(ValueError):
            block_bootstrap(y, rng=rng, block=0)
        with pytest.raises(ValueError):
            block_bootstrap(y, rng=rng, block=20)


# ---------------------------------------------------------------------------
# NullModelHarness — construction + validation
# ---------------------------------------------------------------------------


class TestHarnessConstruction:
    def test_defaults(self) -> None:
        h = NullModelHarness()
        assert h.n_surrogates == DEFAULT_N_SURROGATES
        assert h.alpha == 0.05

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_surrogates": 5},
            {"n_surrogates": 0},
            {"n_surrogates": -10},
            {"alpha": 0.0},
            {"alpha": 1.0},
            {"alpha": -0.1},
            {"alpha": 1.1},
        ],
    )
    def test_rejects_bad_config(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            NullModelHarness(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Determinism + p-value math
# ---------------------------------------------------------------------------


def _dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """A toy statistic for harness tests: normalized inner product."""
    return float(x @ y / x.size)


def _seeded_cyclic(
    seed: int,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    return lambda y: cyclic_shift(y, rng=rng)


class TestDeterminism:
    def test_same_seed_same_null_distribution(self) -> None:
        x = np.sin(np.linspace(0, 4 * np.pi, 200))
        y = np.cos(np.linspace(0, 4 * np.pi, 200))
        harness = NullModelHarness(n_surrogates=50)
        r1 = harness.test(
            x,
            y,
            statistic=_dot_product,
            surrogate_fn=_seeded_cyclic(42),
            seed=42,
        )
        r2 = harness.test(
            x,
            y,
            statistic=_dot_product,
            surrogate_fn=_seeded_cyclic(42),
            seed=42,
        )
        np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)
        assert r1.p_value == r2.p_value

    def test_different_seeds_give_different_nulls(self) -> None:
        x = np.sin(np.linspace(0, 4 * np.pi, 200))
        y = np.cos(np.linspace(0, 4 * np.pi, 200))
        harness = NullModelHarness(n_surrogates=50)
        r1 = harness.test(
            x,
            y,
            statistic=_dot_product,
            surrogate_fn=_seeded_cyclic(1),
            seed=1,
        )
        r2 = harness.test(
            x,
            y,
            statistic=_dot_product,
            surrogate_fn=_seeded_cyclic(2),
            seed=2,
        )
        assert not np.array_equal(r1.null_distribution, r2.null_distribution)


class TestPValueMath:
    def test_p_value_is_phipson_smyth_smoothed(self) -> None:
        """When all surrogates beat the observed statistic, p = (n+1)/(n+1) = 1;
        when none do, p = 1/(n+1) — never 0."""
        x = np.ones(10)

        def stat_returns_zero(a: np.ndarray, b: np.ndarray) -> float:
            return 0.0

        def stat_returns_huge(a: np.ndarray, b: np.ndarray) -> float:
            return 1e12

        harness = NullModelHarness(n_surrogates=10)

        # Surrogate that always makes the statistic 10.
        r = harness.test(
            x,
            x,
            statistic=stat_returns_zero,
            surrogate_fn=lambda y: y.copy(),
            seed=0,
        )
        # observed=0, null all 0 → p = (1+10)/(1+10) = 1.0
        assert r.p_value == pytest.approx(1.0)

        r_huge = harness.test(
            x,
            x,
            statistic=stat_returns_huge,
            surrogate_fn=lambda y: y.copy(),
            seed=0,
        )
        # observed=huge, null all huge → still p=1.0 because ≥ includes equality.
        assert r_huge.p_value == pytest.approx(1.0)

        # To get the 1/(n+1) minimum, use a statistic where the null is smaller:
        counter = {"i": 0}

        def decreasing_stat(a: np.ndarray, b: np.ndarray) -> float:
            counter["i"] += 1
            return 100.0 if counter["i"] == 1 else 0.0

        r_min = harness.test(
            x,
            x,
            statistic=decreasing_stat,
            surrogate_fn=lambda y: y.copy(),
            seed=0,
        )
        # observed=100, null all 0 → p = 1/(1+10) ≈ 0.0909
        assert r_min.p_value == pytest.approx(1.0 / 11.0)
        # And the p-value is NEVER zero.
        assert r_min.p_value > 0.0


# ---------------------------------------------------------------------------
# Rejection band
# ---------------------------------------------------------------------------


class TestRejection:
    def test_perfectly_coupled_is_rejected_under_cyclic_shift(self) -> None:
        """A perfectly correlated pair has a normalized inner product ≈ 1.
        Under cyclic shift, the distribution of the inner product sits near 0
        for an un-aligned signal, so the observed value is in the far right
        tail and p-value ≈ 1/(n+1)."""
        n = 256
        x = np.sin(np.linspace(0, 4 * np.pi, n))
        y = x.copy()

        harness = NullModelHarness(n_surrogates=200, alpha=0.05)
        rng = np.random.default_rng(101)
        result = harness.test(
            x,
            y,
            statistic=_dot_product,
            surrogate_fn=lambda arr: cyclic_shift(arr, rng=rng),
            seed=101,
        )
        assert result.rejected
        assert result.p_value < 0.05

    def test_independent_pair_is_not_rejected(self) -> None:
        n = 512
        rng_a = np.random.default_rng(103)
        x = rng_a.standard_normal(n)
        y = rng_a.standard_normal(n)

        harness = NullModelHarness(n_surrogates=200, alpha=0.05)
        rng = np.random.default_rng(107)
        result = harness.test(
            x,
            y,
            statistic=_dot_product,
            surrogate_fn=lambda arr: cyclic_shift(arr, rng=rng),
            seed=107,
        )
        # Should not reject the null more often than the nominal 5% rate.
        # We just assert it didn't reject on this seed.
        assert not result.rejected


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------


class TestResultContract:
    def test_result_is_frozen(self) -> None:
        import dataclasses

        r = NullModelResult(
            observed=1.0,
            null_distribution=np.zeros(10),
            p_value=0.5,
            n_surrogates=10,
            seed=0,
            rejected=False,
            alpha=0.05,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.observed = 2.0  # type: ignore[misc]

    def test_shape_validation_rejects_mismatched_inputs(self) -> None:
        harness = NullModelHarness(n_surrogates=20)
        with pytest.raises(ValueError, match="same length"):
            harness.test(
                np.zeros(10),
                np.zeros(5),
                statistic=_dot_product,
                surrogate_fn=lambda y: y.copy(),
                seed=0,
            )

    def test_rejects_non_1d_input(self) -> None:
        harness = NullModelHarness(n_surrogates=20)
        with pytest.raises(ValueError, match="1-D"):
            harness.test(
                np.zeros((3, 3)),
                np.zeros(3),
                statistic=_dot_product,
                surrogate_fn=lambda y: y.copy(),
                seed=0,
            )


# ---------------------------------------------------------------------------
# Integration: PLV-like statistic with phase-shuffle surrogates
# ---------------------------------------------------------------------------


def _circular_plv(x: np.ndarray, y: np.ndarray) -> float:
    """Phase-locking value on two already-phase signals, in [0, 1]."""
    return float(np.abs(np.mean(np.exp(1j * (x - y)))))


class TestPLVIntegration:
    def test_locked_phases_rejected_by_cyclic_shift(self) -> None:
        n = 256
        t = np.linspace(0, 4 * np.pi, n)
        # Two perfectly locked phase signals with a fixed offset.
        phi_x = t.copy()
        phi_y = t + 0.3
        harness = NullModelHarness(n_surrogates=200, alpha=0.05)
        rng = np.random.default_rng(211)
        result = harness.test(
            phi_x,
            phi_y,
            statistic=_circular_plv,
            surrogate_fn=lambda y: cyclic_shift(y, rng=rng),
            seed=211,
        )
        assert math.isclose(result.observed, 1.0, abs_tol=1e-9)
        assert result.rejected

    def test_random_phases_not_rejected(self) -> None:
        n = 256
        rng_data = np.random.default_rng(223)
        phi_x = rng_data.uniform(-np.pi, np.pi, size=n)
        phi_y = rng_data.uniform(-np.pi, np.pi, size=n)
        harness = NullModelHarness(n_surrogates=200, alpha=0.05)
        rng = np.random.default_rng(227)
        result = harness.test(
            phi_x,
            phi_y,
            statistic=_circular_plv,
            surrogate_fn=lambda y: cyclic_shift(y, rng=rng),
            seed=227,
        )
        assert not result.rejected
