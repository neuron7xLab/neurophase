"""Tail-risk and multifractal risk modules.

Production-grade risk post-gate: once the physics layers allow execution,
this package decides how much to expose.

Three concentric rings:
    1. evt      — EVT / Peaks-Over-Threshold VaR and CVaR
    2. mfdfa    — multifractal detrended fluctuation analysis + instability
    3. sizer    — composite position sizing from the above + gate state
"""

from __future__ import annotations

from neurophase.risk.evt import (
    EVTFit,
    compute_cvar,
    compute_var,
    fit_gpd_pot,
)
from neurophase.risk.mfdfa import MFDFAResult, mfdfa, multifractal_instability
from neurophase.risk.sizer import PositionSize, RiskProfile, size_position

__all__ = [
    "EVTFit",
    "MFDFAResult",
    "PositionSize",
    "RiskProfile",
    "compute_cvar",
    "compute_var",
    "fit_gpd_pot",
    "mfdfa",
    "multifractal_instability",
    "size_position",
]
