"""Branching ratio σ — proximity of a driven system to its critical point.

Originally defined for neural avalanches as

    σ = ⟨A_{t+1}⟩ / ⟨A_t⟩

the branching ratio reports, in a single scale-free scalar, whether a
perturbation tends to decay, perpetuate, or amplify in an
activity-driven system:

    σ < 1 → sub-critical   (perturbations decay, coherence is lost)
    σ ≈ 1 → critical       (scale-free cascades, maximal information flow)
    σ > 1 → super-critical (perturbations amplify, runaway synchrony)

In ``neurophase`` the same estimator applies to any non-negative
activity-like series — ``|dR/dt|``, envelope of an order-parameter
excursion, count of gate-admits per window. Read against the critical
band ``(0.95, 1.05)`` it augments the symmetric synchrony metrics with
a **dynamical-regime** coordinate that the PLV/iPLV family cannot see.

Elegance
--------
* Two objects: :func:`branching_ratio` for one-shot estimates, and
  :class:`BranchingRatioEMA` for streaming use inside the gate.
* No hidden smoothing, no default thresholds buried in the estimator.
  Phase classification is a small free function,
  :func:`critical_phase`, trivially testable against its band.
* Honest-null semantics: zero activity returns ``σ = 1.0`` rather than
  a numerical artefact like ``A_{t+1} / ε``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt

_EPSILON = 1e-12
_DEFAULT_SUBCRITICAL_MAX = 0.95
_DEFAULT_SUPERCRITICAL_MIN = 1.05


class CriticalPhase(Enum):
    """Position of a driven system relative to its critical point."""

    SUBCRITICAL = "subcritical"
    CRITICAL = "critical"
    SUPERCRITICAL = "supercritical"

    def __str__(self) -> str:
        return self.value


def branching_ratio(
    activity: npt.ArrayLike,
    *,
    eps: float = _EPSILON,
) -> float:
    """Mean-ratio estimator ``σ = ⟨A_{t+1}⟩ / ⟨A_t⟩``.

    Parameters
    ----------
    activity : array_like
        Non-negative activity-like time series. Length ≥ 2.
    eps : float
        Guard on the denominator mean. When ⟨A_t⟩ ≤ eps the series
        carries no activity information and the estimator returns the
        honest-null value 1.0 (critical, i.e. "no evidence either way").

    Returns
    -------
    float
        Estimated σ. Returns ``1.0`` for degenerate input (fewer than
        two samples, or activity indistinguishable from zero).

    Raises
    ------
    ValueError
        If any activity sample is negative (the branching-ratio
        abstraction is only defined on non-negative activities).
    """
    arr = np.asarray(activity, dtype=np.float64).ravel()
    if arr.size < 2:
        return 1.0
    if np.any(arr < 0.0):
        raise ValueError("activity must be non-negative")
    denom = float(np.mean(arr[:-1]))
    if denom <= eps:
        return 1.0
    numer = float(np.mean(arr[1:]))
    return numer / denom


def critical_phase(
    sigma: float,
    *,
    subcritical_max: float = _DEFAULT_SUBCRITICAL_MAX,
    supercritical_min: float = _DEFAULT_SUPERCRITICAL_MIN,
) -> CriticalPhase:
    """Classify σ against a critical band.

    Defaults ``(0.95, 1.05)`` follow the neural-avalanche literature.
    The band is a *closed* critical interval: σ equal to either bound
    is labelled :class:`CriticalPhase.CRITICAL`.

    Raises
    ------
    ValueError
        If the band is degenerate or non-positive.
    """
    if not (0.0 < subcritical_max < supercritical_min):
        raise ValueError(
            "require 0 < subcritical_max < supercritical_min; "
            f"got {subcritical_max=!r}, {supercritical_min=!r}"
        )
    if sigma < subcritical_max:
        return CriticalPhase.SUBCRITICAL
    if sigma > supercritical_min:
        return CriticalPhase.SUPERCRITICAL
    return CriticalPhase.CRITICAL


@dataclass
class BranchingRatioEMA:
    """Streaming branching-ratio estimator with exponential smoothing.

        σ_{t+1} = (1 − α) · σ_t + α · (A_{t+1} + ε) / (A_t + ε)

    The *symmetric* ε is a deliberate departure from the classical
    one-sided guard: when both activity samples are zero the ratio
    collapses to 1, keeping σ anchored at the critical prior rather
    than snapping to 0 or ∞. This is the correct honest-null for "no
    activity means nothing can be said about branching".

    Parameters
    ----------
    alpha : float
        EMA coefficient in ``(0, 1]``. Larger α → more responsive,
        noisier. Default 0.05 matches the slow-tracking regime in the
        avalanche literature.
    initial_sigma : float
        Prior σ before any observation. Default 1.0 (critical).
    eps : float
        Symmetric guard added to both numerator and denominator.
    """

    alpha: float = 0.05
    initial_sigma: float = 1.0
    eps: float = _EPSILON
    _sigma: float = field(init=False)
    _observations: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]; got {self.alpha!r}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive; got {self.eps!r}")
        self._sigma = float(self.initial_sigma)

    @property
    def sigma(self) -> float:
        """Current smoothed σ estimate."""
        return self._sigma

    @property
    def observations(self) -> int:
        """Number of (A_t, A_{t+1}) pairs observed since the last reset."""
        return self._observations

    @property
    def phase(self) -> CriticalPhase:
        """Critical-band classification of the current σ."""
        return critical_phase(self._sigma)

    def update(self, a_t: float, a_t1: float) -> float:
        """Observe ``(A_t, A_{t+1})`` and return the updated σ."""
        a_t_f = float(a_t)
        a_t1_f = float(a_t1)
        if a_t_f < 0.0 or a_t1_f < 0.0:
            raise ValueError("activity counts must be non-negative")
        instant = (a_t1_f + self.eps) / (a_t_f + self.eps)
        self._sigma = (1.0 - self.alpha) * self._sigma + self.alpha * instant
        self._observations += 1
        return self._sigma

    def reset(self) -> None:
        """Reset σ to ``initial_sigma`` and clear the observation counter."""
        self._sigma = float(self.initial_sigma)
        self._observations = 0


__all__ = [
    "BranchingRatioEMA",
    "CriticalPhase",
    "branching_ratio",
    "critical_phase",
]
