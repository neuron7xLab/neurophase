"""Synthetic neural phase generator with controlled Kuramoto coupling.

Generates a synthetic neural signal whose phase is coupled to an
external driving phase (market or synthetic proxy) via the Kuramoto
equation:

    dφ/dt = ω_neu + k·sin(φ_mkt(t) − φ(t))

The coupling strength k controls the PLV:
    k=0   → PLV ≈ O(1/√T)  (null)
    k>0   → PLV > 0          (coupled)
    k→∞   → PLV → 1          (locked)

**Important:** the returned ``phi_neural`` is the raw Kuramoto-
integrated phase, NOT re-extracted via Hilbert transform. This is
deliberate: the synthetic bridge tests the *PLV estimator*, not the
*phase extraction* pipeline. Hilbert extraction is tested separately
in :mod:`neurophase.sync.market_phase`.

This generator differs from :mod:`neurophase.benchmarks.phase_coupling`
(H1) which uses a convex-mixture model with closed-form ground truth.
This generator uses a *dynamical* Kuramoto coupling — no closed-form
PLV, but the dynamics are physically motivated and match the
neurophase hypothesis equation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class NeuralPhaseTrace:
    """Synthetic neural signal with controlled coupling to market phase.

    Attributes
    ----------
    signal : FloatArray
        Raw synthetic EEG-like signal x(t) = A·sin(φ(t)) + noise(t).
    phi_neural : FloatArray
        Raw Kuramoto-integrated phase (not Hilbert-extracted).
    phi_market : FloatArray
        The driving market phase.
    coupling_k : float
        Kuramoto coupling strength used.
    plv_ground_truth : float | None
        Known only at k=0 (≈0). None for k>0 (no closed form).
    """

    signal: FloatArray
    phi_neural: FloatArray
    phi_market: FloatArray
    coupling_k: float
    plv_ground_truth: float | None


def _sim_powerlaw_seeded(
    n_samples: int,
    fs: float,
    exponent: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Generate 1/f^α noise with a seeded RNG (deterministic).

    Uses spectral synthesis: generate white noise in freq domain,
    shape amplitude by f^(exponent/2), inverse FFT.
    """
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    # Avoid division by zero at DC
    freqs[0] = 1.0
    # Shape amplitude: |H(f)| = f^(exponent/2)
    amplitude = np.power(freqs, exponent / 2.0)
    amplitude[0] = 0.0  # zero DC component
    # Random phase
    phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
    phases[0] = 0.0
    if n_samples % 2 == 0:
        phases[-1] = 0.0
    # Construct spectrum and inverse FFT
    spectrum = amplitude * np.exp(1j * phases)
    signal = np.fft.irfft(spectrum, n=n_samples)
    return signal.astype(np.float64)


def generate_neural_phase_trace(
    phi_market: npt.ArrayLike,
    *,
    n_samples: int | None = None,
    fs: float = 256.0,
    f_neural: float = 1.0,
    coupling_k: float = 0.0,
    noise_exponent: float = 1.0,
    noise_amplitude: float = 0.3,
    signal_amplitude: float = 1.0,
    seed: int = 42,
) -> NeuralPhaseTrace:
    """Generate a synthetic neural signal coupled to an external phase.

    Parameters
    ----------
    phi_market : array_like, shape (T,)
        External driving phase in radians.
    n_samples : int | None
        If None, uses len(phi_market).
    fs : float
        Sampling rate in Hz.
    f_neural : float
        Neural oscillation frequency in Hz. Default 1.0 Hz to match
        typical market phase frequencies for coupling detection.
    coupling_k : float
        Kuramoto coupling strength. Must be ≥ 0.
    noise_exponent : float
        Exponent for 1/f noise (1.0 = pink, 2.0 = brown).
    noise_amplitude : float
        Scale of additive aperiodic noise.
    signal_amplitude : float
        Amplitude of the sinusoidal component.
    seed : int
        For deterministic generation.

    Returns
    -------
    NeuralPhaseTrace
    """
    phi_mkt = np.asarray(phi_market, dtype=np.float64)
    if phi_mkt.ndim != 1:
        raise ValueError(f"phi_market must be 1-D, got shape {phi_mkt.shape}")
    if not np.all(np.isfinite(phi_mkt)):
        raise ValueError("phi_market must contain only finite values")
    if coupling_k < 0:
        raise ValueError(f"coupling_k must be ≥ 0, got {coupling_k}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    T = n_samples if n_samples is not None else phi_mkt.size
    if T < 4:
        raise ValueError(f"n_samples must be ≥ 4, got {T}")
    if phi_mkt.size < T:
        raise ValueError(f"n_samples={T} exceeds phi_market length={phi_mkt.size}")
    phi_mkt = phi_mkt[:T]

    rng = np.random.default_rng(seed)
    dt = 1.0 / fs

    # Step 1: Generate deterministic 1/f noise
    noise = _sim_powerlaw_seeded(T, fs, -noise_exponent, rng)
    noise_std = float(np.std(noise))
    if noise_std > 0:
        noise = noise * (noise_amplitude / noise_std)

    # Step 2: Integrate Kuramoto ODE via RK4
    # dφ/dt = 2π·f_neural + k·sin(φ_mkt(t) − φ(t))
    phi = np.empty(T, dtype=np.float64)
    phi[0] = rng.uniform(-np.pi, np.pi)

    omega = 2.0 * np.pi * f_neural

    for t in range(T - 1):
        p = phi[t]
        m = phi_mkt[t]
        # RK4 with fixed phi_mkt at this step
        k1 = omega + coupling_k * np.sin(m - p)
        k2 = omega + coupling_k * np.sin(m - (p + 0.5 * dt * k1))
        k3 = omega + coupling_k * np.sin(m - (p + 0.5 * dt * k2))
        k4 = omega + coupling_k * np.sin(m - (p + dt * k3))
        phi[t + 1] = p + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Wrap to (-π, π]
    phi = ((phi + np.pi) % (2.0 * np.pi)) - np.pi

    # Step 3: Construct signal x(t) = A·sin(φ(t)) + noise(t)
    signal = signal_amplitude * np.sin(phi) + noise

    # Ground truth: only known at k=0
    gt: float | None = 0.0 if coupling_k == 0.0 else None

    return NeuralPhaseTrace(
        signal=signal.astype(np.float64),
        phi_neural=phi,
        phi_market=phi_mkt,
        coupling_k=coupling_k,
        plv_ground_truth=gt,
    )


def generate_synthetic_market_phase(
    n_samples: int = 4096,
    fs: float = 256.0,
    f_market: float = 0.5,
    noise_amplitude: float = 0.2,
    seed: int = 42,
) -> FloatArray:
    """Generate a synthetic market-like phase signal.

    Produces a slowly oscillating phase signal with additive noise,
    mimicking intraday price cycles. This is a synthetic proxy —
    for real market phase extraction, use
    :func:`~neurophase.sync.market_phase.extract_market_phase_from_price`.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    fs : float
        Sampling rate in Hz.
    f_market : float
        Market oscillation frequency in Hz.
    noise_amplitude : float
        Amplitude of phase noise perturbation.
    seed : int
        For determinism.

    Returns
    -------
    FloatArray, shape (n_samples,)
        Synthetic market phase in [−π, π].
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    phi_base = 2.0 * np.pi * f_market * t
    phi_noise = noise_amplitude * np.cumsum(rng.standard_normal(n_samples)) / np.sqrt(fs)
    phi = phi_base + phi_noise
    # Wrap to (-π, π]
    wrapped: FloatArray = ((phi + np.pi) % (2.0 * np.pi)) - np.pi
    return wrapped.astype(np.float64)
