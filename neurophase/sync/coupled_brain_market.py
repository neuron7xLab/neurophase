"""Coupled brain × market Kuramoto system — equation 8.1.

Implements the shared-`R(t)` coupled-Kuramoto model from the R&D report
(section 8.1). The key property: brain and market oscillators live in one
network and are both driven by the **same** order parameter ``R(t)``.

Governing equations (for each oscillator ``k``)::

    dθ_k/dt = ω_k + K · R(t) · sin(Ψ_other(t − τ) − θ_k(t)) + σ · ξ_k(t)

where

* ``ω_k`` — natural frequency of oscillator ``k``
* ``R(t) exp(i Ψ(t)) = (1/N) Σ_j exp(i θ_j(t))`` is the **joint** order
  parameter over the full brain ∪ market population
* ``Ψ_brain`` / ``Ψ_market`` are the mean phases of the brain-side and
  market-side sub-populations (the "other" population drives each oscillator)
* ``τ`` is an optional integer-step propagation delay
* ``σ ξ_k`` is i.i.d. Gaussian noise with variance ``σ²``

This mirrors the Fioriti & Chinnici (2012) formulation in which two
sub-populations are mutually driven by one shared synchronization signal
and diverge only through their natural-frequency distributions.

Design notes
------------

* **RK4 integration.** The deterministic drift is integrated with a
  classical fourth-order Runge–Kutta scheme. Noise is added with the
  Euler–Maruyama ``√dt`` scaling *after* the deterministic step — this is
  the standard semi-implicit SDE convention used in the wider Kuramoto
  literature and keeps the drift fourth-order accurate.
* **Delay.** ``tau`` is expressed in **seconds** and is rounded to the
  nearest integer number of steps. A ring buffer of past mean phases is
  maintained so that the driving signal seen at time ``t`` is
  ``Ψ_other(t − τ_steps · dt)``. For ``tau = 0`` the delay is disabled.
* **Joint R(t).** The published order parameter is computed over the
  union of brain and market oscillators. Individual sub-population mean
  phases ``Ψ_brain`` / ``Ψ_market`` are returned alongside for diagnostics
  and for downstream prediction-error monitors.
* **Deterministic seeding.** All sources of randomness (ω draws, noise)
  go through a single ``np.random.Generator`` seeded by ``seed``.

Sources
-------
Fioriti & Chinnici (2012); R&D report section 8.1; Kuramoto (1984);
Acebrón et al., Rev. Mod. Phys. 77 (2005).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from neurophase.gate.execution_gate import DEFAULT_THRESHOLD

# Natural-frequency dispersion for the brain and market populations.
# Chosen to mirror the R&D report — brain oscillations concentrate around
# ω ≈ 2π·1 Hz (slow cognitive tempo), market around ω ≈ 2π·0.5 Hz.
_BRAIN_OMEGA_MEAN: Final[float] = 2.0 * np.pi * 1.0
_MARKET_OMEGA_MEAN: Final[float] = 2.0 * np.pi * 0.5
_OMEGA_SPREAD: Final[float] = 0.15


@dataclass(frozen=True)
class CoupledStep:
    """One integration step output.

    Attributes
    ----------
    t
        Simulation time in seconds.
    R
        Joint order parameter over brain ∪ market oscillators.
    psi_brain
        Mean phase of the brain sub-population, wrapped to ``(-π, π]``.
    psi_market
        Mean phase of the market sub-population, wrapped to ``(-π, π]``.
    execution_allowed
        Gate decision based on ``R >= threshold``. This mirrors
        ``ExecutionGate`` invariant I₁ for downstream consumers that
        only need the scalar flag.
    """

    t: float
    R: float
    psi_brain: float
    psi_market: float
    execution_allowed: bool


class CoupledBrainMarketSystem:
    """Shared-`R(t)` coupled Kuramoto system for brain × market oscillators.

    Parameters
    ----------
    n_brain
        Number of brain-side oscillators. Defaults to 3 (EEG α, EEG β, HRV).
    n_market
        Number of market-side oscillators. Defaults to 3 (price, vol, spread).
    K
        Coupling strength. Higher ``K`` → stronger tendency to synchronize.
    tau
        Communication delay in seconds. ``0.0`` disables the delay.
    sigma
        Gaussian noise amplitude (standard deviation of ``dθ`` per √s).
    dt
        Integrator step size in seconds.
    seed
        RNG seed for reproducibility.
    threshold
        Gate threshold on ``R(t)``. Defaults to the package default.

    Raises
    ------
    ValueError
        If any parameter is outside its physically meaningful range.
    """

    def __init__(
        self,
        n_brain: int = 3,
        n_market: int = 3,
        K: float = 1.0,
        tau: float = 0.0,
        sigma: float = 0.01,
        dt: float = 0.01,
        seed: int = 42,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if n_brain < 1:
            raise ValueError(f"n_brain must be ≥ 1, got {n_brain}")
        if n_market < 1:
            raise ValueError(f"n_market must be ≥ 1, got {n_market}")
        if K < 0:
            raise ValueError(f"K must be ≥ 0, got {K}")
        if tau < 0:
            raise ValueError(f"tau must be ≥ 0, got {tau}")
        if sigma < 0:
            raise ValueError(f"sigma must be ≥ 0, got {sigma}")
        if dt <= 0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}")

        self.n_brain: int = n_brain
        self.n_market: int = n_market
        self.N: int = n_brain + n_market
        self.K: float = K
        self.tau: float = tau
        self.sigma: float = sigma
        self.dt: float = dt
        self.threshold: float = threshold

        self._rng: np.random.Generator = np.random.default_rng(seed)

        # Natural frequencies: brain centered around ~1 Hz, market ~0.5 Hz.
        brain_omegas = self._rng.normal(_BRAIN_OMEGA_MEAN, _OMEGA_SPREAD, size=n_brain)
        market_omegas = self._rng.normal(_MARKET_OMEGA_MEAN, _OMEGA_SPREAD, size=n_market)
        self.omega: NDArray[np.float64] = np.concatenate([brain_omegas, market_omegas]).astype(
            np.float64
        )

        # Initial phases: uniformly distributed on (-π, π].
        self.theta: NDArray[np.float64] = self._rng.uniform(-np.pi, np.pi, size=self.N).astype(
            np.float64
        )

        # Delay in integer steps. ``max(0, …)`` guards the tau=0 case.
        self._tau_steps: int = round(tau / dt) if tau > 0 else 0

        # Ring buffer of past (psi_brain, psi_market). We seed it with the
        # current means so that delayed lookups before warm-up simply return
        # the initial state — no NaN / None to propagate.
        psi_b0, psi_m0 = self._subpop_means()
        buffer_len = max(1, self._tau_steps + 1)
        self._psi_brain_hist: list[float] = [psi_b0] * buffer_len
        self._psi_market_hist: list[float] = [psi_m0] * buffer_len

        self._t: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> tuple[float, float, float]:
        """Advance the system by one RK4 step.

        Returns
        -------
        tuple[float, float, float]
            ``(R(t), ψ_brain, ψ_market)`` **after** the step, with phases
            wrapped to ``(-π, π]``.
        """
        # Classical RK4 on the deterministic drift.
        #
        # For tau = 0 the driving mean phases (ψ_brain, ψ_market) are the
        # **current** sub-step means: the canonical Kuramoto RK4 recomputes
        # them at every evaluation of f(θ). For tau > 0 the drivers are
        # frozen to the buffer read at ``t − τ`` (a classic delay-DDE step).
        if self._tau_steps == 0:
            psi_override: tuple[float, float] | None = None
        else:
            psi_override = self._delayed_drivers()

        theta = self.theta
        k1 = self._drift(theta, psi_override)
        k2 = self._drift(theta + 0.5 * self.dt * k1, psi_override)
        k3 = self._drift(theta + 0.5 * self.dt * k2, psi_override)
        k4 = self._drift(theta + self.dt * k3, psi_override)
        theta_next = theta + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # 3. Euler–Maruyama noise kick (σ · √dt · ξ with ξ ~ N(0, 1)).
        if self.sigma > 0:
            noise = self._rng.standard_normal(self.N) * (self.sigma * np.sqrt(self.dt))
            theta_next = theta_next + noise

        # 4. Wrap + commit + advance time.
        self.theta = _wrap(theta_next)
        self._t += self.dt

        # 5. Update history buffers and emit diagnostics.
        psi_b_now, psi_m_now = self._subpop_means()
        self._psi_brain_hist.append(psi_b_now)
        self._psi_market_hist.append(psi_m_now)
        # Keep the ring buffer bounded.
        max_len = max(1, self._tau_steps + 1)
        if len(self._psi_brain_hist) > max_len:
            self._psi_brain_hist.pop(0)
            self._psi_market_hist.pop(0)

        R_now = self._joint_R()
        return R_now, psi_b_now, psi_m_now

    def run(self, n_steps: int) -> pd.DataFrame:
        """Integrate ``n_steps`` forward and return the full trajectory.

        Parameters
        ----------
        n_steps
            Number of RK4 steps to advance.

        Returns
        -------
        pandas.DataFrame
            Columns: ``t``, ``R``, ``psi_brain``, ``psi_market``,
            ``execution_allowed``.
        """
        if n_steps < 0:
            raise ValueError(f"n_steps must be ≥ 0, got {n_steps}")
        rows: list[CoupledStep] = []
        for _ in range(n_steps):
            R_now, psi_b, psi_m = self.step()
            rows.append(
                CoupledStep(
                    t=self._t,
                    R=R_now,
                    psi_brain=psi_b,
                    psi_market=psi_m,
                    execution_allowed=bool(R_now >= self.threshold),
                )
            )
        df = pd.DataFrame(
            [
                {
                    "t": r.t,
                    "R": r.R,
                    "psi_brain": r.psi_brain,
                    "psi_market": r.psi_market,
                    "execution_allowed": r.execution_allowed,
                }
                for r in rows
            ]
        )
        return df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def t(self) -> float:
        """Current simulation time in seconds."""
        return self._t

    @property
    def R(self) -> float:
        """Current joint order parameter."""
        return self._joint_R()

    def psi_brain(self) -> float:
        """Current brain-subpopulation mean phase."""
        return self._subpop_means()[0]

    def psi_market(self) -> float:
        """Current market-subpopulation mean phase."""
        return self._subpop_means()[1]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _drift(
        self,
        theta: NDArray[np.float64],
        psi_override: tuple[float, float] | None,
    ) -> NDArray[np.float64]:
        """Deterministic RHS of equation 8.1.

        Brain oscillators are driven by ``Ψ_market``; market oscillators
        are driven by ``Ψ_brain``. When ``psi_override`` is ``None`` the
        driving mean phases are recomputed from ``theta`` (canonical
        Kuramoto RK4 at ``τ = 0``). When provided, they are frozen to the
        delayed buffer values (DDE step at ``τ > 0``). The coupling gain
        ``K · R(t)`` is always evaluated at the *current* phases.
        """
        R_now = _order_parameter_from_phases(theta)
        if psi_override is None:
            psi_b = _mean_phase(theta[: self.n_brain])
            psi_m = _mean_phase(theta[self.n_brain :])
        else:
            psi_b, psi_m = psi_override

        drift = np.empty_like(theta)
        # Brain block — driven by ψ_market.
        drift[: self.n_brain] = self.omega[: self.n_brain] + self.K * R_now * np.sin(
            psi_m - theta[: self.n_brain]
        )
        # Market block — driven by ψ_brain.
        drift[self.n_brain :] = self.omega[self.n_brain :] + self.K * R_now * np.sin(
            psi_b - theta[self.n_brain :]
        )
        return drift

    def _joint_R(self) -> float:
        """Joint order parameter over the full brain ∪ market population."""
        return _order_parameter_from_phases(self.theta)

    def _subpop_means(self) -> tuple[float, float]:
        """Return the (brain, market) mean phases."""
        psi_b = _mean_phase(self.theta[: self.n_brain])
        psi_m = _mean_phase(self.theta[self.n_brain :])
        return psi_b, psi_m

    def _delayed_drivers(self) -> tuple[float, float]:
        """Return ``(Ψ_brain(t − τ), Ψ_market(t − τ))``.

        When ``tau == 0`` the history buffer has length 1 and the method
        returns the current means — there is no delay.
        """
        if self._tau_steps == 0:
            psi_b_lag, psi_m_lag = self._subpop_means()
            return psi_b_lag, psi_m_lag
        # index 0 is the oldest entry → time t − tau_steps·dt.
        return self._psi_brain_hist[0], self._psi_market_hist[0]


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


def _wrap(angles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap angles to the ``(-π, π]`` principal branch."""
    wrapped: NDArray[np.float64] = np.mod(angles + np.pi, 2.0 * np.pi) - np.pi
    return wrapped


def _order_parameter_from_phases(theta: NDArray[np.float64]) -> float:
    """Kuramoto order parameter magnitude."""
    return float(np.abs(np.mean(np.exp(1j * theta))))


def _mean_phase(theta: NDArray[np.float64]) -> float:
    """Circular mean (phase of the mean complex exponent)."""
    if theta.size == 0:
        return 0.0
    return float(np.angle(np.mean(np.exp(1j * theta))))
