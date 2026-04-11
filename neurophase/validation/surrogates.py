"""Surrogate generators — destroy coupling while preserving marginals.

Three generators are provided. Each one preserves a specific set of
distributional properties of its input and destroys exactly the
relationship under test.

================ ======================================================
Generator        Null hypothesis (what is destroyed)
================ ======================================================
phase_shuffle    Destroys phase coupling between two signals by
                 randomly permuting the phase of the Fourier-transformed
                 second signal. Preserves amplitude spectrum and
                 therefore autocorrelation of ``y``.
cyclic_shift     Destroys cross-signal phase locking by applying a
                 random integer rotation to ``y``. Exactly preserves
                 the autocorrelation function of ``y`` (not just the
                 spectrum). The recommended default for PLV surrogates.
block_bootstrap  Destroys long-range coherence by resampling ``y`` in
                 contiguous blocks of length ``block`` with
                 replacement. Preserves short-range autocorrelation
                 up to lag ``block``.
================ ======================================================

All three functions share the same signature:

.. code-block:: python

    def generator(y, *, rng) -> numpy.ndarray:
        ...

``y`` is a 1-D float array; ``rng`` is a ``numpy.random.Generator``.
The return value is a new 1-D array of the same length — the input is
never modified in place.

No SciPy. No stochastic state outside ``rng``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_1d(y: NDArray[np.float64] | np.ndarray, *, name: str = "y") -> NDArray[np.float64]:
    """Coerce to 1-D float64 and validate finiteness."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size < 2:
        raise ValueError(f"{name} must have length ≥ 2, got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def phase_shuffle(
    y: NDArray[np.float64] | np.ndarray,
    *,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Phase-randomize ``y`` in the Fourier domain (destroys phase coupling).

    The algorithm is the standard Theiler et al. (1992) IAAFT-light:

    1. Compute the FFT of ``y``.
    2. Replace the phase of every non-DC / non-Nyquist bin with a
       uniformly random value in :math:`[0, 2\\pi)`.
    3. Preserve conjugate symmetry so the inverse transform is real.
    4. Inverse-transform.

    Null hypothesis: the amplitude spectrum (and therefore the
    autocorrelation) of ``y`` is preserved; the phase coupling to any
    external signal is destroyed.
    """
    arr = _validate_1d(y, name="y")
    n = arr.size
    Y = np.fft.rfft(arr)

    # Randomize phases of the non-DC / non-Nyquist bins only.
    random_phases = rng.uniform(0.0, 2.0 * np.pi, size=Y.size)
    # DC stays real; Nyquist (if present) also stays real.
    random_phases[0] = 0.0
    if n % 2 == 0:
        random_phases[-1] = 0.0

    magnitude = np.abs(Y)
    Y_shuffled = magnitude * np.exp(1j * random_phases)
    out: NDArray[np.float64] = np.fft.irfft(Y_shuffled, n=n)
    return out


def cyclic_shift(
    y: NDArray[np.float64] | np.ndarray,
    *,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Apply a random integer cyclic rotation (destroys cross-signal coupling).

    For any integer shift ``k``, ``y_rot[i] = y[(i + k) mod n]``. A
    rotation exactly preserves the autocorrelation function of ``y``
    (not just its spectrum — the circular-mean equivalence holds) and
    therefore destroys only the cross-signal phase relationship with
    an external reference. This is the recommended default for PLV
    surrogate tests (Lachaux et al., 1999).

    ``k`` is drawn uniformly from ``[1, n-1]`` so the identity rotation
    (``k = 0``) is excluded — the surrogate must differ from ``y``.
    """
    arr = _validate_1d(y, name="y")
    n = arr.size
    k = int(rng.integers(1, n))
    return np.roll(arr, k)


def block_bootstrap(
    y: NDArray[np.float64] | np.ndarray,
    *,
    rng: np.random.Generator,
    block: int = 16,
) -> NDArray[np.float64]:
    """Block-bootstrap resample ``y`` (destroys long-range coherence).

    The signal is cut into contiguous blocks of length ``block`` and
    resampled with replacement. The output has the same length as the
    input. Null hypothesis: short-range autocorrelation (up to lag
    ``block``) is preserved; long-range coherence is destroyed.

    ``block`` must be in ``[1, n]``; smaller blocks destroy more
    coherence.
    """
    arr = _validate_1d(y, name="y")
    n = arr.size
    if block < 1 or block > n:
        raise ValueError(f"block must be in [1, {n}], got {block}")

    # Non-overlapping starting indices for the source blocks.
    starts = np.arange(0, n - block + 1)
    if starts.size == 0:
        # Edge case: block == n → bootstrap is the identity.
        return arr.copy()

    # Sample ceil(n / block) starting indices and concatenate blocks.
    n_blocks = int(np.ceil(n / block))
    chosen = rng.choice(starts, size=n_blocks, replace=True)
    pieces = [arr[s : s + block] for s in chosen]
    out_full = np.concatenate(pieces)
    result: NDArray[np.float64] = out_full[:n].copy()
    return result
