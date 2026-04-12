"""HN_SEED contract tests — NullModelHarness.test_seeded reproducibility.

Six tests lock in the HN_SEED invariant:

1. test_seeded_same_seed_identical_results        — two calls with seed=42
   produce byte-identical null_distribution arrays.
2. test_seeded_different_seed_different_results   — seed=42 vs seed=99
   produce different null distributions (stochastic sanity check).
3. test_seeded_result_records_seed                — result.seed == input seed.
4. test_seeded_cyclic_shift_deterministic         — cyclic_shift generator
   is bit-identical across two seeded calls.
5. test_seeded_phase_shuffle_deterministic        — phase_shuffle generator
   is bit-identical across two seeded calls.
6. test_seeded_block_bootstrap_deterministic      — block_bootstrap generator
   is bit-identical across two seeded calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from neurophase.validation.null_model import NullModelHarness, ReproducibilityWarning
from neurophase.validation.surrogates import block_bootstrap, cyclic_shift, phase_shuffle

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 128  # signal length — long enough for stable stats
RNG_FIXED = np.random.default_rng(0)


def _make_signals(n: int = N) -> tuple[np.ndarray, np.ndarray]:
    """Return two correlated 1-D float64 signals."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(n)
    y = 0.6 * x + 0.4 * rng.standard_normal(n)
    return x.astype(np.float64), y.astype(np.float64)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pure correlation statistic (no external state)."""
    return float(np.corrcoef(x, y)[0, 1])


_HARNESS = NullModelHarness(n_surrogates=100)
_X, _Y = _make_signals()


# ---------------------------------------------------------------------------
# 1. Same seed → identical null distribution
# ---------------------------------------------------------------------------


def test_seeded_same_seed_identical_results() -> None:
    """Two calls with seed=42 must produce byte-identical null_distribution."""
    r1 = _HARNESS.test_seeded(_X, _Y, statistic=_pearson, generator=cyclic_shift, seed=42)
    r2 = _HARNESS.test_seeded(_X, _Y, statistic=_pearson, generator=cyclic_shift, seed=42)
    np.testing.assert_array_equal(
        r1.null_distribution,
        r2.null_distribution,
        err_msg="HN_SEED violated: same seed produced different null distributions",
    )
    assert r1.observed == r2.observed
    assert r1.p_value == r2.p_value


# ---------------------------------------------------------------------------
# 2. Different seeds → different distributions (probabilistic sanity check)
# ---------------------------------------------------------------------------


def test_seeded_different_seed_different_results() -> None:
    """seed=42 and seed=99 must produce different null distributions."""
    r42 = _HARNESS.test_seeded(_X, _Y, statistic=_pearson, generator=cyclic_shift, seed=42)
    r99 = _HARNESS.test_seeded(_X, _Y, statistic=_pearson, generator=cyclic_shift, seed=99)
    assert not np.array_equal(r42.null_distribution, r99.null_distribution), (
        "Unexpected: seed=42 and seed=99 produced identical null distributions"
    )


# ---------------------------------------------------------------------------
# 3. result.seed records the supplied seed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42, 99, 2**31 - 1])
def test_seeded_result_records_seed(seed: int) -> None:
    """NullModelResult.seed must equal the seed passed to test_seeded."""
    result = _HARNESS.test_seeded(_X, _Y, statistic=_pearson, generator=cyclic_shift, seed=seed)
    assert result.seed == seed, f"Expected seed={seed}, got result.seed={result.seed}"


# ---------------------------------------------------------------------------
# 4. cyclic_shift is bit-identical across two seeded calls
# ---------------------------------------------------------------------------


def test_seeded_cyclic_shift_deterministic() -> None:
    """cyclic_shift generator: same seed → byte-identical null distribution."""
    r1 = _HARNESS.test_seeded(_X, _Y, statistic=_pearson, generator=cyclic_shift, seed=42)
    r2 = _HARNESS.test_seeded(_X, _Y, statistic=_pearson, generator=cyclic_shift, seed=42)
    np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)


# ---------------------------------------------------------------------------
# 5. phase_shuffle is bit-identical across two seeded calls
# ---------------------------------------------------------------------------


def test_seeded_phase_shuffle_deterministic() -> None:
    """phase_shuffle generator: same seed → byte-identical null distribution."""
    r1 = _HARNESS.test_seeded(_X, _Y, statistic=_pearson, generator=phase_shuffle, seed=7)
    r2 = _HARNESS.test_seeded(_X, _Y, statistic=_pearson, generator=phase_shuffle, seed=7)
    np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)


# ---------------------------------------------------------------------------
# 6. block_bootstrap is bit-identical across two seeded calls
# ---------------------------------------------------------------------------


def test_seeded_block_bootstrap_deterministic() -> None:
    """block_bootstrap generator: same seed → byte-identical null distribution."""
    import functools

    gen = functools.partial(block_bootstrap, block=8)
    r1 = _HARNESS.test_seeded(_X, _Y, statistic=_pearson, generator=gen, seed=13)
    r2 = _HARNESS.test_seeded(_X, _Y, statistic=_pearson, generator=gen, seed=13)
    np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)


# ---------------------------------------------------------------------------
# Bonus: ReproducibilityWarning is importable and is a UserWarning subclass
# ---------------------------------------------------------------------------


def test_reproducibility_warning_is_user_warning() -> None:
    assert issubclass(ReproducibilityWarning, UserWarning)
