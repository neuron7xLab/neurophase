"""Shannon, Tsallis, and Rényi entropies of a time series.

Probabilities are estimated via a histogram with adaptive binning
(Freedman–Diaconis rule). All three functional forms are regularised to
avoid log(0) / 0^q singularities.

Entropy as a phase-transition diagnostic:
    H(t) rising  → disorder growing → possible regime shift
    H(t) falling → coherence growing → possible emergent phase

Normalised change:

    ΔH(t) = H(t) − H(t − τ)

serves as an input to the 4-condition emergent phase criterion from the
π-system reference.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import iqr

FloatArray = npt.NDArray[np.float64]

_EPSILON = 1e-12


def freedman_diaconis_bins(series: npt.ArrayLike, min_bins: int = 2) -> int:
    """Freedman–Diaconis bin count for a series.

    n = ceil((max − min) / (2 · IQR · n^(-1/3)))

    Falls back to ``min_bins`` when IQR is zero (degenerate / constant input).
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 2:
        return min_bins
    q = float(iqr(arr))
    if q == 0:
        return min_bins
    bin_width = 2.0 * q * arr.size ** (-1.0 / 3.0)
    if bin_width <= 0:
        return min_bins
    span = float(np.max(arr) - np.min(arr))
    if span <= 0:
        return min_bins
    return max(min_bins, int(np.ceil(span / bin_width)))


def _histogram_probs(series: FloatArray, bins: int) -> FloatArray:
    """Histogram probabilities with additive regularisation."""
    hist, _ = np.histogram(series, bins=bins, density=False)
    hist_f = hist.astype(np.float64)
    total = float(np.sum(hist_f))
    if total == 0.0:
        return np.full(bins, 1.0 / bins, dtype=np.float64)
    probs = hist_f / total
    # Regularise: p' = (p + ε) / (1 + n·ε)
    n_bins = probs.size
    probs = (probs + _EPSILON) / (1.0 + n_bins * _EPSILON)
    return probs.astype(np.float64)


def shannon_entropy(series: npt.ArrayLike, bins: int | None = None) -> float:
    """Shannon entropy H_S = -Σ p log p (natural log, nats).

    Parameters
    ----------
    series : array_like
        Samples from which to estimate probabilities.
    bins : int | None
        Histogram bin count. If None, uses ``freedman_diaconis_bins``.
    """
    arr = np.asarray(series, dtype=np.float64)
    n_bins = bins if bins is not None else freedman_diaconis_bins(arr)
    probs = _histogram_probs(arr, n_bins)
    return float(-np.sum(probs * np.log(probs)))


def tsallis_entropy(series: npt.ArrayLike, q: float = 1.5, bins: int | None = None) -> float:
    """Tsallis entropy H_T = (1 − Σ p^q) / (q − 1).

    The q → 1 limit recovers Shannon. For q > 1 the measure penalises
    heavy tails less aggressively, which matches crypto return distributions.

    Parameters
    ----------
    series : array_like
    q : float
        Tsallis parameter; must be ≠ 1.
    bins : int | None
    """
    if q == 1.0:
        raise ValueError("q must be != 1 for Tsallis entropy")
    arr = np.asarray(series, dtype=np.float64)
    n_bins = bins if bins is not None else freedman_diaconis_bins(arr)
    probs = _histogram_probs(arr, n_bins)
    return float((1.0 - np.sum(probs**q)) / (q - 1.0))


def renyi_entropy(series: npt.ArrayLike, alpha: float = 2.0, bins: int | None = None) -> float:
    """Rényi entropy H_α = (1 / (1 − α)) · log Σ p^α.

    The α → 1 limit recovers Shannon; α = 2 gives collision entropy.

    Parameters
    ----------
    series : array_like
    alpha : float
        Rényi parameter; must be > 0 and ≠ 1.
    bins : int | None
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if alpha == 1.0:
        raise ValueError("alpha must be != 1 for Rényi entropy")
    arr = np.asarray(series, dtype=np.float64)
    n_bins = bins if bins is not None else freedman_diaconis_bins(arr)
    probs = _histogram_probs(arr, n_bins)
    return float(np.log(np.sum(probs**alpha)) / (1.0 - alpha))


def delta_entropy(
    series: npt.ArrayLike,
    window: int,
    bins: int | None = None,
) -> float:
    """Shannon entropy change between the two most recent non-overlapping windows.

        ΔH = H(series[-window:]) − H(series[-2·window:-window])

    Negative ΔH signals rising coherence (entropy dropping) — a candidate
    emergent-phase indicator in the π-system criterion.

    Parameters
    ----------
    series : array_like
    window : int
        Trailing window size; needs at least ``2 · window`` samples.
    bins : int | None

    Raises
    ------
    ValueError
        If ``window`` is not positive or the series is too short.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 2 * window:
        raise ValueError(f"need at least 2·window={2 * window} samples, got {arr.size}")
    current = shannon_entropy(arr[-window:], bins=bins)
    previous = shannon_entropy(arr[-2 * window : -window], bins=bins)
    return float(current - previous)
