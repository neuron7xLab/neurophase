"""Property-based fuzz on :mod:`neurophase.metrics.transfer_entropy`.

Promotes TE invariants from *"tested on chosen examples"* to *"verified
on every (source, target, k, n_levels) configuration a hypothesis
strategy can generate"*.

Locks in:

* **Non-negativity** — ``transfer_entropy(...) ≥ 0`` on every finite
  input.
* **Self-TE is exactly zero** — ``TE(X → X, k) = 0`` identically, for
  every ``X`` and every ``k``. Conditioning on a variable's own past
  eliminates every remaining bit of information its past could carry.
* **Rank invariance** — TE depends only on the rank ordering of its
  inputs, not their magnitudes, so any strictly monotonic transform
  (scale, shift, exponentiation) leaves the estimate byte-identical.
* **Swap antisymmetry of net flow** — ``TE_net`` flips sign exactly on
  argument swap under a shared seed.
* **Determinism under seed** — same inputs + same seed ⇒ byte-identical
  :class:`TEResult`.
* **Degenerate honest-null** — constant or too-short inputs ⇒ 0.0
  without raising.
* **Finite-sample bound on independent Gaussian** — plug-in TE on
  independent standard-normal pairs stays within a bias envelope.

Each property carries an explicit ``@settings(max_examples=...)`` cap
so the suite stays within CI budget.
"""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from neurophase.metrics.transfer_entropy import (
    TEResult,
    transfer_entropy,
    transfer_entropy_with_significance,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_FINITE: dict[str, float | bool] = {
    "allow_nan": False,
    "allow_infinity": False,
    "min_value": -1e6,
    "max_value": 1e6,
}


def _series(min_size: int = 50, max_size: int = 400) -> st.SearchStrategy[np.ndarray]:
    """Finite float64 series in a reasonable magnitude range."""
    return hnp.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=min_size, max_value=max_size),
        elements=st.floats(**_FINITE),
    )


def _paired_series(
    min_size: int = 50, max_size: int = 400
) -> st.SearchStrategy[tuple[np.ndarray, np.ndarray]]:
    """Two series that share the same length."""
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda n: st.tuples(
            hnp.arrays(np.float64, shape=n, elements=st.floats(**_FINITE)),
            hnp.arrays(np.float64, shape=n, elements=st.floats(**_FINITE)),
        )
    )


# ---------------------------------------------------------------------------
# Non-negativity — TE ≥ 0 on every finite input
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    pair=_paired_series(),
    k=st.integers(min_value=1, max_value=3),
    n_levels=st.integers(min_value=2, max_value=3),
)
def test_te_is_non_negative(pair: tuple[np.ndarray, np.ndarray], k: int, n_levels: int) -> None:
    x, y = pair
    assert transfer_entropy(x, y, k=k, n_levels=n_levels) >= 0.0
    assert transfer_entropy(y, x, k=k, n_levels=n_levels) >= 0.0


# ---------------------------------------------------------------------------
# Self-TE — TE(X → X, k) = 0 identically
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    series=_series(),
    k=st.integers(min_value=1, max_value=3),
    n_levels=st.integers(min_value=2, max_value=3),
)
def test_te_of_self_is_zero(series: np.ndarray, k: int, n_levels: int) -> None:
    """Conditioning a variable on its own past removes every remaining bit."""
    assert transfer_entropy(series, series, k=k, n_levels=n_levels) == 0.0


# ---------------------------------------------------------------------------
# Rank invariance — monotonic transforms leave TE byte-identical
# ---------------------------------------------------------------------------


# Rank invariance holds for any *ideal* monotonic affine map but fails
# under float64 when magnitudes collide with shift precision
# (e.g. ``1.0 * 1e-106 + 1.0 == 1.0``). Using integer-valued floats keeps
# the inputs well-separated at every magnitude the affine map can
# produce, so the property becomes a sharp correctness check on the
# algorithm rather than a float-precision stress test.
_RANK_ELEMENTS = st.integers(min_value=-1000, max_value=1000).map(float)


def _well_conditioned_pair(
    min_size: int = 50, max_size: int = 400
) -> st.SearchStrategy[tuple[np.ndarray, np.ndarray]]:
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda n: st.tuples(
            hnp.arrays(np.float64, shape=n, elements=_RANK_ELEMENTS),
            hnp.arrays(np.float64, shape=n, elements=_RANK_ELEMENTS),
        )
    )


@settings(max_examples=75, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    pair=_well_conditioned_pair(),
    scale=st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
    shift=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
)
def test_te_invariant_under_affine_transform(
    pair: tuple[np.ndarray, np.ndarray], scale: float, shift: float
) -> None:
    """Positive-scale affine on either series is a monotonic rank-preserving
    map, and quantile binning depends only on ranks."""
    x, y = pair
    te_raw = transfer_entropy(x, y)
    te_scaled_source = transfer_entropy(scale * x + shift, y)
    te_scaled_target = transfer_entropy(x, scale * y + shift)
    assert te_raw == te_scaled_source
    assert te_raw == te_scaled_target


# ---------------------------------------------------------------------------
# Determinism under seed
# ---------------------------------------------------------------------------


@settings(max_examples=25, deadline=None)
@given(
    pair=_paired_series(min_size=200, max_size=400),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_significance_is_byte_identical_under_seed(
    pair: tuple[np.ndarray, np.ndarray], seed: int
) -> None:
    x, y = pair
    a = transfer_entropy_with_significance(x, y, n_surrogates=20, seed=seed)
    b = transfer_entropy_with_significance(x, y, n_surrogates=20, seed=seed)
    assert a == b


# ---------------------------------------------------------------------------
# Swap antisymmetry of net flow under a shared seed
# ---------------------------------------------------------------------------


@settings(max_examples=25, deadline=None)
@given(
    pair=_paired_series(min_size=200, max_size=400),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_te_net_flips_on_swap(pair: tuple[np.ndarray, np.ndarray], seed: int) -> None:
    x, y = pair
    forward = transfer_entropy_with_significance(x, y, n_surrogates=20, seed=seed)
    backward = transfer_entropy_with_significance(y, x, n_surrogates=20, seed=seed)
    # Same seed → same surrogate shifts, so raw surrogate means also swap.
    assert forward.te_net == -backward.te_net


# ---------------------------------------------------------------------------
# Honest null — degenerate inputs stay at 0
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    constant=st.floats(**_FINITE),
    length=st.integers(min_value=3, max_value=500),
    k=st.integers(min_value=1, max_value=3),
)
def test_constant_input_has_zero_te(constant: float, length: int, k: int) -> None:
    """A constant series carries no variation — TE in both directions is 0."""
    other = np.arange(length, dtype=np.float64)
    const = np.full(length, constant, dtype=np.float64)
    assert transfer_entropy(const, other, k=k) == 0.0
    assert transfer_entropy(other, const, k=k) == 0.0


@settings(max_examples=25, deadline=None)
@given(length=st.integers(min_value=0, max_value=2))
def test_short_input_returns_zero(length: int) -> None:
    x = np.zeros(length, dtype=np.float64)
    y = np.zeros(length, dtype=np.float64)
    assert transfer_entropy(x, y) == 0.0
    result = transfer_entropy_with_significance(x, y, n_surrogates=5)
    assert isinstance(result, TEResult)
    assert result.te_xy == 0.0
    assert result.te_yx == 0.0
    assert result.p_xy == 1.0
    assert result.p_yx == 1.0


# ---------------------------------------------------------------------------
# Finite-sample bias envelope on independent Gaussian
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_independent_gaussian_bias_is_bounded(seed: int) -> None:
    """Plug-in TE on genuinely independent Gaussians has a small positive
    finite-sample bias; with n = 3000 and k = 1, binary quantisation, the
    bias stays comfortably under 0.05 nats across any seed."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(3000)
    y = rng.standard_normal(3000)
    assert transfer_entropy(x, y) < 0.05
    assert transfer_entropy(y, x) < 0.05


# ---------------------------------------------------------------------------
# Output-type integrity — TEResult fields satisfy declared invariants
# ---------------------------------------------------------------------------


@settings(max_examples=15, deadline=None)
@given(
    pair=_paired_series(min_size=200, max_size=400),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_result_fields_are_well_formed(pair: tuple[np.ndarray, np.ndarray], seed: int) -> None:
    x, y = pair
    r = transfer_entropy_with_significance(x, y, n_surrogates=20, seed=seed)
    assert r.te_xy >= 0.0
    assert r.te_yx >= 0.0
    assert 0.0 < r.p_xy <= 1.0
    assert 0.0 < r.p_yx <= 1.0
    assert r.te_net == r.te_xy - r.te_yx
    assert r.n_surrogates == 20
    assert r.k == 1
    assert r.n_levels == 2
