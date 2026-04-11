"""Emergent phase detector — the 4-condition criterion.

A market is in an *emergent phase* (metastable, high-information, coherent)
when all four conditions hold simultaneously:

    R(t)       > R_min               (Kuramoto synchronization)
    ΔH_S(t)    < dH_max   (< 0)      (entropy collapsing)
    κ̄(t)       < kappa_max (< 0)     (graph unstable, ready to reconfigure)
    ISM(t)     ∈ [ism_low, ism_high] (balanced regime)

Default thresholds from the π-system reference (IEEE Technical Documentation
v2.0, section 5.1):

    R_min      = 0.75
    dH_max     = -0.05
    kappa_max  = -0.10
    ism_low    = 0.80
    ism_high   = 1.20

This is the upstream trigger: the execution gate (invariant I1) still
blocks when R < θ even if the emergent criterion says otherwise. The two
layers compose — gate first, direction-giving phase detector second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class EmergentPhaseCriteria:
    """Thresholds for the 4-condition detector.

    Defaults follow the π-system reference. All comparisons are strict.
    """

    R_min: float = 0.75
    dH_max: float = -0.05
    kappa_max: float = -0.10
    ism_low: float = 0.80
    ism_high: float = 1.20

    def __post_init__(self) -> None:
        if not 0.0 < self.R_min < 1.0:
            raise ValueError(f"R_min must be in (0, 1), got {self.R_min}")
        if self.ism_low >= self.ism_high:
            raise ValueError(f"ism_low ({self.ism_low}) must be < ism_high ({self.ism_high})")


@dataclass(frozen=True)
class EmergentPhaseDecision:
    """Per-component evaluation of the 4-condition criterion.

    ``is_emergent`` is True iff all four checks pass.
    """

    is_emergent: bool
    R_ok: bool
    dH_ok: bool
    kappa_ok: bool
    ism_ok: bool
    R: float
    dH: float
    kappa: float
    ism: float

    def reasons(self) -> list[str]:
        """Human-readable list of failed checks (empty if emergent)."""
        out: list[str] = []
        if not self.R_ok:
            out.append(f"R={self.R:.3f} below threshold")
        if not self.dH_ok:
            out.append(f"ΔH={self.dH:.3f} above threshold (entropy not collapsing)")
        if not self.kappa_ok:
            out.append(f"κ̄={self.kappa:.3f} above threshold (graph still stable)")
        if not self.ism_ok:
            out.append(f"ISM={self.ism:.3f} outside balanced band")
        return out


DEFAULT_CRITERIA: Final[EmergentPhaseCriteria] = EmergentPhaseCriteria()


def detect_emergent_phase(
    R: float,
    dH: float,
    kappa: float,
    ism: float,
    criteria: EmergentPhaseCriteria = DEFAULT_CRITERIA,
) -> EmergentPhaseDecision:
    """Evaluate the 4-condition emergent-phase criterion.

    Parameters
    ----------
    R : float
        Current Kuramoto order parameter in [0, 1].
    dH : float
        Short-horizon Shannon entropy change ΔH_S(t).
    kappa : float
        Mean Ricci curvature κ̄(t) of the market graph.
    ism : float
        Current Information-Structural Metric value.
    criteria : EmergentPhaseCriteria
        Thresholds. Defaults to the π-system reference.

    Returns
    -------
    EmergentPhaseDecision
    """
    R_ok = R > criteria.R_min  # noqa: SIM300 — physical semantics: R compared to floor
    dH_ok = dH < criteria.dH_max
    kappa_ok = kappa < criteria.kappa_max
    ism_ok = criteria.ism_low <= ism <= criteria.ism_high
    return EmergentPhaseDecision(
        is_emergent=R_ok and dH_ok and kappa_ok and ism_ok,
        R_ok=R_ok,
        dH_ok=dH_ok,
        kappa_ok=kappa_ok,
        ism_ok=ism_ok,
        R=R,
        dH=dH,
        kappa=kappa,
        ism=ism,
    )
