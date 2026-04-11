"""Kuramoto network with optional delays, noise, and liquidity-modulated coupling.

Standard Kuramoto model for N phase oscillators θ_i(t):

    dθ_i/dt = ω_i + (K(t) / N) · Σ_{j=1..N} sin(θ_j(t - τ_ij) - θ_i(t)) + ξ_i(t)

where:
    ω_i                — intrinsic natural frequency of oscillator i
    K(t) = K_0 · L(t)  — coupling strength scaled by an optional liquidity factor
    τ_ij               — pairwise information delay (in simulation steps)
    ξ_i(t) ~ N(0, σ²)  — additive Gaussian phase noise

This module implements a vectorised 4-th order Runge–Kutta integrator for
the non-delayed case (τ = 0) and a fixed-history buffer for the delayed case.
Delays are specified as integer step counts; the network keeps the last
``max_delay`` phase snapshots and indexes into them per pair.

Design notes:
- The integrator is deliberately explicit and dependency-free (NumPy only).
- For very large N, the O(N²) coupling matrix dominates — acceptable for
  research-grade networks up to a few thousand oscillators.
- All arrays are float64; no in-place mutation of caller inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class KuramotoParams:
    """Static parameters for a Kuramoto network.

    Attributes
    ----------
    omega : FloatArray, shape (N,)
        Natural frequencies of the N oscillators (radians / time unit).
    coupling : float
        Base coupling strength K_0. Must be non-negative.
    dt : float
        Integration step size. Must be positive.
    noise_sigma : float
        Standard deviation of additive phase noise per step. Zero disables noise.
    delays : IntArray | None, shape (N, N)
        Integer delay matrix (in simulation steps). Use None for no delay.
        Element ``delays[i, j]`` is the lag at which oscillator j influences i.
        Diagonal must be zero.
    liquidity_fn : object | None
        Callable ``(step: int, state: FloatArray) -> float`` returning a
        non-negative scalar multiplier of the coupling. None means constant
        coupling. Stored as Any to keep the dataclass frozen + picklable.
    """

    omega: FloatArray
    coupling: float = 1.0
    dt: float = 0.05
    noise_sigma: float = 0.0
    delays: IntArray | None = None
    liquidity_fn: object | None = field(default=None)

    def __post_init__(self) -> None:
        if self.omega.ndim != 1:
            raise ValueError(f"omega must be 1-D, got shape {self.omega.shape}")
        if self.omega.size < 2:
            raise ValueError(f"need at least 2 oscillators, got {self.omega.size}")
        if self.coupling < 0:
            raise ValueError(f"coupling must be non-negative, got {self.coupling}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.noise_sigma < 0:
            raise ValueError(f"noise_sigma must be non-negative, got {self.noise_sigma}")
        if self.delays is not None:
            expected = (self.omega.size, self.omega.size)
            if self.delays.shape != expected:
                raise ValueError(f"delays must have shape {expected}, got {self.delays.shape}")
            if np.any(self.delays < 0):
                raise ValueError("delays must be non-negative")
            if np.any(np.diag(self.delays) != 0):
                raise ValueError("delays diagonal must be zero")


class KuramotoNetwork:
    """Kuramoto oscillator network with RK4 integration.

    Parameters
    ----------
    omega : array_like, shape (N,)
        Natural frequencies.
    coupling : float
        Base coupling K_0.
    dt : float
        Integration step.
    noise_sigma : float
        Phase noise standard deviation.
    delays : array_like | None, shape (N, N)
        Integer delays in simulation steps. Diagonal must be zero.
    seed : int | None
        PRNG seed for reproducibility.
    """

    def __init__(
        self,
        omega: npt.ArrayLike,
        coupling: float = 1.0,
        dt: float = 0.05,
        noise_sigma: float = 0.0,
        delays: npt.ArrayLike | None = None,
        seed: int | None = None,
    ) -> None:
        omega_arr = np.asarray(omega, dtype=np.float64)
        delays_arr: IntArray | None = None if delays is None else np.asarray(delays, dtype=np.int64)
        self.params = KuramotoParams(
            omega=omega_arr,
            coupling=float(coupling),
            dt=float(dt),
            noise_sigma=float(noise_sigma),
            delays=delays_arr,
        )
        self._rng = np.random.default_rng(seed)
        self._N = omega_arr.size
        # Initial phases: uniform in [-π, π).
        self._theta: FloatArray = self._rng.uniform(-np.pi, np.pi, size=self._N).astype(np.float64)

    @property
    def N(self) -> int:
        """Number of oscillators."""
        return self._N

    @property
    def theta(self) -> FloatArray:
        """Current phase vector (copy)."""
        return self._theta.copy()

    def set_initial_phases(self, theta: npt.ArrayLike) -> None:
        """Override the current phases. Shape must match N."""
        arr = np.asarray(theta, dtype=np.float64)
        if arr.shape != (self._N,):
            raise ValueError(f"expected shape ({self._N},), got {arr.shape}")
        self._theta = arr.copy()

    def _coupling_force(self, theta: FloatArray, history: FloatArray | None) -> FloatArray:
        """Compute the mean-field coupling term for a given phase vector.

        When ``history`` is None (no delays), uses ``theta`` directly:
            F_i = (1/N) Σ_j sin(θ_j − θ_i)

        With delays, uses the history buffer:
            F_i = (1/N) Σ_j sin(θ_j(t − τ_ij) − θ_i(t))
        """
        params = self.params
        if params.delays is None or history is None:
            diff = theta[np.newaxis, :] - theta[:, np.newaxis]
            out_now: FloatArray = np.mean(np.sin(diff), axis=1).astype(np.float64)
            return out_now
        # history[k, :] is theta at history buffer index k (0 = oldest, -1 = newest).
        # delays[i, j] gives the lag in steps for j -> i.
        H = history.shape[0]
        delays = params.delays
        # Indices into history: newest is H-1. lag d -> index H-1-d, clipped.
        idx = np.clip(H - 1 - delays, 0, H - 1)
        # Build delayed_theta_j[i, j] = history[idx[i, j], j]
        j_idx = np.arange(self._N)[np.newaxis, :].repeat(self._N, axis=0)
        delayed = history[idx, j_idx]  # shape (N, N)
        diff = delayed - theta[:, np.newaxis]
        out_delayed: FloatArray = np.mean(np.sin(diff), axis=1).astype(np.float64)
        return out_delayed

    def _derivative(
        self,
        theta: FloatArray,
        history: FloatArray | None,
        step: int,
    ) -> FloatArray:
        """dθ/dt — deterministic part only (noise added in the Euler–Maruyama tail)."""
        params = self.params
        K = params.coupling
        if params.liquidity_fn is not None:
            fn = params.liquidity_fn
            K = K * float(fn(step, theta))  # type: ignore[operator]
            if K < 0:
                raise ValueError("liquidity_fn must return a non-negative multiplier")
        coupling_force = self._coupling_force(theta, history)
        return params.omega + K * coupling_force

    def step(self, history: FloatArray | None = None, step_idx: int = 0) -> FloatArray:
        """Advance one RK4 step (with optional Euler–Maruyama noise tail).

        Returns the new phase vector and updates the internal state.
        """
        params = self.params
        dt = params.dt
        theta = self._theta
        k1 = self._derivative(theta, history, step_idx)
        k2 = self._derivative(theta + 0.5 * dt * k1, history, step_idx)
        k3 = self._derivative(theta + 0.5 * dt * k2, history, step_idx)
        k4 = self._derivative(theta + dt * k3, history, step_idx)
        new_theta = theta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if params.noise_sigma > 0:
            noise = self._rng.normal(0.0, params.noise_sigma * np.sqrt(dt), size=self._N)
            new_theta = new_theta + noise
        # Wrap to (-π, π] for numerical hygiene.
        new_theta = ((new_theta + np.pi) % (2 * np.pi)) - np.pi
        self._theta = new_theta.astype(np.float64)
        return self._theta.copy()

    def run(self, n_steps: int) -> FloatArray:
        """Integrate the network for ``n_steps`` and return the full trajectory.

        Parameters
        ----------
        n_steps : int
            Number of integration steps.

        Returns
        -------
        FloatArray, shape (n_steps, N)
            Phase trajectory. Row t is the state after integration step t.
        """
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        params = self.params
        if params.delays is None:
            trajectory = np.empty((n_steps, self._N), dtype=np.float64)
            for t in range(n_steps):
                trajectory[t] = self.step(history=None, step_idx=t)
            return trajectory

        max_delay = int(np.max(params.delays)) + 1
        history = np.tile(self._theta, (max_delay, 1)).astype(np.float64)
        trajectory = np.empty((n_steps, self._N), dtype=np.float64)
        for t in range(n_steps):
            new_theta = self.step(history=history, step_idx=t)
            history = np.roll(history, -1, axis=0)
            history[-1] = new_theta
            trajectory[t] = new_theta
        return trajectory
