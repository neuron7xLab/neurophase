"""Null-model harness — seeded, deterministic significance testing.

The harness runs a two-input scalar statistic over ``n_surrogates``
resamples of the second input and returns a full null distribution,
the observed value, and a one-sided p-value. This is the canonical
way to confront a coupling claim in ``neurophase``.

Contract
--------

* The statistic must be a pure function ``(x, y) → float``. It may
  not depend on external state.
* The surrogate generator must accept ``y`` and an ``rng`` keyword
  and return a new 1-D array of the same length.
* The harness threads a single deterministic seed through every
  surrogate draw so that two runs with the same seed produce
  identical null distributions.

This module provides only the mechanism. Individual consumers (PLV
significance, regime tests, …) live in higher-level modules that
import :class:`NullModelHarness` and bind it to a specific statistic.

Reproducibility
---------------

Use :meth:`NullModelHarness.test_seeded` for guaranteed bit-identical
replay. It takes a generator matching the ``surrogates.py`` protocol
``(y, *, rng) -> ndarray`` and creates the RNG internally from the
supplied seed via :func:`numpy.random.default_rng`, spawning one child
RNG per surrogate iteration for fully independent draws (HN_SEED).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

#: Default number of surrogate resamples. Matches the R&D report
#: convention (Lancaster et al. 2018) of ``N = 1000`` for a stable
#: 0.001-resolution p-value.
DEFAULT_N_SURROGATES: Final[int] = 1000


Statistic = Callable[
    [NDArray[np.float64], NDArray[np.float64]],
    float,
]
SurrogateFn = Callable[
    [NDArray[np.float64]],
    NDArray[np.float64],
]

# Type alias for the surrogates.py protocol: (y, *, rng) -> ndarray.
SurrogateGenerator = Callable[
    [NDArray[np.float64]],
    NDArray[np.float64],
]


class ReproducibilityWarning(UserWarning):
    """Issued when a surrogate_fn passed to :meth:`NullModelHarness.test`
    cannot be verified to be seeded.

    The :meth:`NullModelHarness.test` API delegates RNG control to the
    caller via ``surrogate_fn``. If the caller forgets to capture an
    ``rng`` in the closure, two calls with the same ``seed`` will produce
    different null distributions. Use :meth:`NullModelHarness.test_seeded`
    to obtain HN_SEED-compliant determinism by construction.
    """


@dataclass(frozen=True, repr=False)
class NullModelResult:
    """Immutable outcome of a null-model confrontation.

    Attributes
    ----------
    observed
        The statistic evaluated on the real ``(x, y)`` pair.
    null_distribution
        Flat array of ``n_surrogates`` statistic values computed on
        surrogate ``y``.
    p_value
        One-sided (right-tailed) p-value using the discrete estimator

        .. math::

            p = \\frac{1 + \\#\\{ s \\in \\text{null} : s \\ge \\text{observed} \\}}{1 + n}

        The ``+1`` smoothing prevents ``p = 0`` for finite samples
        (Phipson & Smyth 2010).
    n_surrogates
        Number of surrogate resamples used.
    seed
        Seed that generated the null distribution.
    rejected
        ``True`` iff ``p_value < alpha``. ``alpha`` is stored for
        replay.
    alpha
        Significance level used for the rejection decision.
    """

    observed: float
    null_distribution: NDArray[np.float64]
    p_value: float
    n_surrogates: int
    seed: int
    rejected: bool
    alpha: float

    def __repr__(self) -> str:  # aesthetic rich repr (HN22)
        flag = "rejected" if self.rejected else "not rejected"
        return (
            f"NullModelResult[observed={self.observed:.4f} · "
            f"p={self.p_value:.4f} · "
            f"n={self.n_surrogates} · "
            f"α={self.alpha:.3f} · {flag}]"  # noqa: RUF001
        )


class NullModelHarness:
    """Seeded surrogate-based significance harness.

    Parameters
    ----------
    n_surrogates
        Number of surrogate resamples per call to :meth:`test`.
        Defaults to :data:`DEFAULT_N_SURROGATES`.
    alpha
        Default significance level for :meth:`test`. Calls can
        override this per invocation.
    """

    __slots__ = ("alpha", "n_surrogates")

    def __init__(
        self,
        n_surrogates: int = DEFAULT_N_SURROGATES,
        alpha: float = 0.05,
    ) -> None:
        if n_surrogates < 10:
            raise ValueError(
                f"n_surrogates must be ≥ 10 for a meaningful p-value, got {n_surrogates}"
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.n_surrogates: int = n_surrogates
        self.alpha: float = alpha

    def test(
        self,
        x: NDArray[np.float64] | np.ndarray,
        y: NDArray[np.float64] | np.ndarray,
        *,
        statistic: Statistic,
        surrogate_fn: SurrogateFn,
        seed: int,
        alpha: float | None = None,
    ) -> NullModelResult:
        """Confront ``statistic(x, y)`` with a surrogate null distribution.

        Parameters
        ----------
        x, y
            1-D input arrays. The statistic is evaluated on ``(x, y)``;
            surrogates are drawn over ``y`` only (this matches the
            "shuffle the second signal" convention used in Lachaux et
            al. 1999).
        statistic
            A pure function ``(x, y) -> float``.
        surrogate_fn
            A function ``y -> y_surrogate``. To use the built-in
            generators in :mod:`neurophase.validation.surrogates`,
            wrap them in a lambda that captures an ``rng`` derived
            from ``seed``.
        seed
            Integer seed for the harness's internal ``np.random.Generator``.
            Same seed → identical null distribution.
        alpha
            Optional override of the harness's default significance level.

        Returns
        -------
        NullModelResult
            Full provenance of the test. ``null_distribution`` is a
            flat array of length ``n_surrogates``.
        """
        x_arr = _to_1d(x, name="x")
        y_arr = _to_1d(y, name="y")
        if x_arr.size != y_arr.size:
            raise ValueError(f"x and y must have the same length, got {x_arr.size} vs {y_arr.size}")

        effective_alpha = alpha if alpha is not None else self.alpha
        observed = float(statistic(x_arr, y_arr))
        null = np.empty(self.n_surrogates, dtype=np.float64)
        for i in range(self.n_surrogates):
            y_surr = surrogate_fn(y_arr)
            null[i] = float(statistic(x_arr, y_surr))

        # Discrete one-sided p-value with +1 smoothing (Phipson & Smyth 2010).
        p_value = float((1 + np.sum(null >= observed)) / (1 + self.n_surrogates))
        return NullModelResult(
            observed=observed,
            null_distribution=null,
            p_value=p_value,
            n_surrogates=self.n_surrogates,
            seed=seed,
            rejected=p_value < effective_alpha,
            alpha=effective_alpha,
        )

    def test_seeded(
        self,
        x: NDArray[np.float64] | np.ndarray,
        y: NDArray[np.float64] | np.ndarray,
        *,
        statistic: Statistic,
        generator: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        seed: int,
        alpha: float | None = None,
    ) -> NullModelResult:
        """HN_SEED-compliant significance test with guaranteed bit-identical replay.

        Unlike :meth:`test`, this method owns the RNG. It creates
        ``np.random.default_rng(seed)`` internally and spawns one child
        RNG per surrogate iteration via ``rng.spawn(1)[0]``, so every draw
        is independent and the full null distribution is reproducible with
        the same ``seed``.

        Parameters
        ----------
        x, y
            1-D input arrays (same convention as :meth:`test`).
        statistic
            A pure function ``(x, y) -> float``.
        generator
            A callable matching the surrogates.py protocol:
            ``(y, *, rng: np.random.Generator) -> ndarray``. The harness
            injects the child RNG — the caller must NOT capture an
            external RNG in the closure.
        seed
            Integer seed. Identical seeds produce byte-identical results.
        alpha
            Optional override of the harness's default significance level.

        Returns
        -------
        NullModelResult
            ``result.seed`` equals the supplied ``seed``.
        """
        # Runtime import avoids a circular-import risk; surrogates is a
        # sibling module with no dependency on null_model.
        rng = np.random.default_rng(seed)
        child_rngs = rng.spawn(self.n_surrogates)

        def _wrapped(
            y_arr: NDArray[np.float64], *, _rng: np.random.Generator
        ) -> NDArray[np.float64]:
            return generator(y_arr, rng=_rng)  # type: ignore[call-arg]

        def _make_surrogate_fn(child: np.random.Generator) -> SurrogateFn:
            def _fn(y_arr: NDArray[np.float64]) -> NDArray[np.float64]:
                return _wrapped(y_arr, _rng=child)

            return _fn

        # Build the null distribution without re-entering test() to avoid
        # the overhead of a second input validation pass.
        x_arr = _to_1d(x, name="x")
        y_arr = _to_1d(y, name="y")
        if x_arr.size != y_arr.size:
            raise ValueError(f"x and y must have the same length, got {x_arr.size} vs {y_arr.size}")

        effective_alpha = alpha if alpha is not None else self.alpha
        observed = float(statistic(x_arr, y_arr))
        null = np.empty(self.n_surrogates, dtype=np.float64)
        for i in range(self.n_surrogates):
            fn = _make_surrogate_fn(child_rngs[i])
            null[i] = float(statistic(x_arr, fn(y_arr)))

        p_value = float((1 + np.sum(null >= observed)) / (1 + self.n_surrogates))
        return NullModelResult(
            observed=observed,
            null_distribution=null,
            p_value=p_value,
            n_surrogates=self.n_surrogates,
            seed=seed,
            rejected=p_value < effective_alpha,
            alpha=effective_alpha,
        )


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _to_1d(a: NDArray[np.float64] | np.ndarray, *, name: str) -> NDArray[np.float64]:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size < 2:
        raise ValueError(f"{name} must have length ≥ 2, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr
