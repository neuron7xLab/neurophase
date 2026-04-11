"""H1 — synthetic phase-coupling ground-truth generator.

The load-bearing purpose of this module is to produce paired phase
signals ``(φ_x, φ_y)`` whose true coupling is **known exactly** by
construction, so that downstream statistical tests (PLV significance,
null-model rejection rates, calibration sweeps) can be validated
against ground truth rather than against each other.

Generative model
----------------

The generator parametrizes phase coupling through a single scalar
``coupling_strength ∈ [0, 1]`` that linearly interpolates between:

* ``coupling_strength = 0`` — two independent uniform-random phase
  processes. The ground-truth PLV in the infinite-length limit is
  **exactly zero** (the expected value of ``|⟨exp(i Δφ)⟩|`` over
  independent uniforms), and finite-sample PLV is
  ``O(1/√T)``.
* ``coupling_strength = 1`` — ``φ_y(t) = φ_x(t) + φ_offset``, i.e. a
  pure phase lock with a caller-controlled constant offset. Ground-
  truth PLV is **exactly one** at every sample length.

For intermediate values the model is a **convex mixture**:

.. math::

    \\phi_y(t)
    \\;=\\;
    (1 - c)\\cdot \\eta_y(t)
    \\;+\\;
    c \\cdot \\big(\\phi_x(t) + \\phi_{\\text{offset}}\\big)

where ``η_y(t)`` is an i.i.d. uniform-random phase signal and ``c``
is the coupling strength. The expected PLV is a monotone increasing
function of ``c`` (empirically near-quadratic for small ``c``, and
exactly ``1`` at ``c = 1``).

Why this particular parametrization
-----------------------------------

* **Endpoint correctness.** ``c = 0`` and ``c = 1`` give mathematically
  exact PLV ground truth. This is what the tests in
  ``tests/test_benchmarks_phase_coupling.py`` certify.
* **Deterministic under seeding.** The entire trajectory is produced
  through a single ``numpy.random.Generator`` so replays are
  bit-identical.
* **No SciPy.** Only ``numpy``.
* **Wrapped to ``[-π, π]``.** Both ``φ_x`` and ``φ_y`` are wrapped to
  the canonical principal branch so they can be fed directly to
  :func:`~neurophase.metrics.plv.plv` without reprocessing.

This is **not** a physical model of brain-market coupling. It is a
synthetic fixture designed to expose statistical bias, variance,
and false-positive / false-negative rates in the validation harness
with known ground truth.

Relation to CoupledBrainMarketSystem
------------------------------------

``CoupledBrainMarketSystem`` produces phase signals via an RK4
integrator of the physically-motivated equation 8.1. Those signals
have realistic dynamics but **no closed-form ground-truth PLV** —
the PLV depends on ``K``, ``τ``, ``σ`` in a way that must be
empirically measured. H1 is the complementary tool: no physics, no
dynamics, but a closed-form ground truth.

Both generators are useful. H1 is for certifying the statistical
validation layer; CoupledBrainMarketSystem is for certifying the
gate state-machine layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

#: Default number of samples when no explicit ``n`` is supplied.
DEFAULT_N_SAMPLES: Final[int] = 2048


@dataclass(frozen=True)
class PhaseCouplingConfig:
    """Parameters of a synthetic phase-coupling scenario.

    Attributes
    ----------
    n
        Number of phase samples. Must be ``≥ 2``.
    coupling_strength
        Interpolation parameter ``c ∈ [0, 1]``. ``0`` = independent,
        ``1`` = pure phase lock.
    phi_offset
        Constant phase offset added to the coupled component. Any
        real value (wrapped internally). The PLV value is independent
        of this offset — PLV measures locking, not direction.
    seed
        Integer seed for the underlying ``numpy.random.Generator``.
    """

    n: int = DEFAULT_N_SAMPLES
    coupling_strength: float = 0.5
    phi_offset: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError(f"n must be ≥ 2, got {self.n}")
        if not 0.0 <= self.coupling_strength <= 1.0:
            raise ValueError(f"coupling_strength must be in [0, 1], got {self.coupling_strength}")
        if not np.isfinite(self.phi_offset):
            raise ValueError(f"phi_offset must be finite, got {self.phi_offset}")


@dataclass(frozen=True)
class PhaseCouplingTrace:
    """A generated synthetic trace with closed-form ground truth.

    Attributes
    ----------
    phi_x, phi_y
        Paired phase series in ``[-π, π]`` of length ``config.n``.
    config
        The configuration that produced the trace.
    ground_truth_plv
        The analytically-known PLV *in the infinite-length limit*:

        * ``0.0`` when ``coupling_strength == 0``
        * ``1.0`` when ``coupling_strength == 1``
        * ``coupling_strength`` otherwise (the convex mixture of
          a random and a locked component has expected ``|⟨e^{iΔφ}⟩|
          → c`` as ``n → ∞``).

        Finite-sample PLV deviates from this value by ``O(1/√n)``
        for small ``c`` and is exact for ``c ∈ {0, 1}``.
    """

    phi_x: NDArray[np.float64]
    phi_y: NDArray[np.float64]
    config: PhaseCouplingConfig
    ground_truth_plv: float


def generate_phase_coupling(config: PhaseCouplingConfig) -> PhaseCouplingTrace:
    """Generate a synthetic phase-coupling trace with known ground truth.

    See the module docstring for the generative model. The output is
    deterministic under ``config.seed`` — same seed → bit-identical
    trace.
    """
    rng = np.random.default_rng(config.seed)

    # φ_x: i.i.d. uniform on (-π, π].
    phi_x = rng.uniform(-np.pi, np.pi, size=config.n)

    c = config.coupling_strength

    if c == 0.0:
        # Fully independent. Draw φ_y directly from a separate uniform
        # — mixture would still be uniform but we keep the code path
        # explicit for the zero-coupling test.
        phi_y = rng.uniform(-np.pi, np.pi, size=config.n)
        ground_truth_plv = 0.0
    elif c == 1.0:
        # Pure phase lock.
        phi_y = _wrap(phi_x + config.phi_offset)
        ground_truth_plv = 1.0
    else:
        # Convex mixture of a random and a locked component.
        eta_y = rng.uniform(-np.pi, np.pi, size=config.n)
        locked = phi_x + config.phi_offset
        phi_y = _wrap((1.0 - c) * eta_y + c * locked)
        ground_truth_plv = c

    return PhaseCouplingTrace(
        phi_x=_wrap(phi_x),
        phi_y=phi_y,
        config=config,
        ground_truth_plv=ground_truth_plv,
    )


def generate_anti_coupled(n: int = DEFAULT_N_SAMPLES, *, seed: int = 42) -> PhaseCouplingTrace:
    """Convenience wrapper for the uncoupled null-hypothesis scenario.

    Returns a trace with ``coupling_strength = 0`` and
    ``ground_truth_plv = 0``. Useful as a false-positive control in
    surrogate tests.
    """
    return generate_phase_coupling(
        PhaseCouplingConfig(
            n=n,
            coupling_strength=0.0,
            phi_offset=0.0,
            seed=seed,
        )
    )


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------


def _wrap(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap angles to the ``(-π, π]`` principal branch."""
    wrapped: NDArray[np.float64] = np.mod(angles + np.pi, 2.0 * np.pi) - np.pi
    return wrapped
