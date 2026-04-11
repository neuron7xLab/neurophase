"""neurophase — phase synchronization as execution gate.

Kuramoto model of market × trader nervous system.

Three invariants that cannot be overridden:
    I1: R(t) < θ  ⇒  execution_allowed = False
    I2: PLV computed on held-out data only
    I3: Bio-sensor absent  ⇒  SENSOR_ABSENT (no synthetic fallback)
"""

from __future__ import annotations

__version__ = "0.2.0"

from neurophase.core.kuramoto import KuramotoNetwork, KuramotoParams
from neurophase.core.order_parameter import OrderParameterResult, order_parameter
from neurophase.core.phase import (
    adaptive_threshold,
    compute_phase,
    preprocess_signal,
)
from neurophase.gate.direction_index import (
    Direction,
    DirectionDecision,
    DirectionIndexWeights,
    direction_index,
)
from neurophase.gate.emergent_phase import (
    EmergentPhaseCriteria,
    EmergentPhaseDecision,
    detect_emergent_phase,
)
from neurophase.gate.execution_gate import (
    DEFAULT_THRESHOLD,
    ExecutionGate,
    GateDecision,
    GateState,
)
from neurophase.indicators.fmn import compute_fmn
from neurophase.indicators.qilm import compute_qilm
from neurophase.metrics.asymmetry import kurtosis, skewness, topological_asymmetry
from neurophase.metrics.entropy import (
    delta_entropy,
    freedman_diaconis_bins,
    renyi_entropy,
    shannon_entropy,
    tsallis_entropy,
)
from neurophase.metrics.hurst import hurst_dfa, hurst_rs
from neurophase.metrics.ism import compute_ism, compute_topological_energy, ism_derivative
from neurophase.metrics.plv import PLVResult, plv, plv_significance, rolling_plv
from neurophase.metrics.ricci import forman_ricci, mean_ricci, ollivier_ricci
from neurophase.risk.evt import EVTFit, compute_cvar, compute_var, fit_gpd_pot
from neurophase.risk.mfdfa import MFDFAResult, mfdfa, multifractal_instability
from neurophase.risk.sizer import PositionSize, RiskProfile, size_position

__all__ = [
    "DEFAULT_THRESHOLD",
    "Direction",
    "DirectionDecision",
    "DirectionIndexWeights",
    "EVTFit",
    "EmergentPhaseCriteria",
    "EmergentPhaseDecision",
    "ExecutionGate",
    "GateDecision",
    "GateState",
    "KuramotoNetwork",
    "KuramotoParams",
    "MFDFAResult",
    "OrderParameterResult",
    "PLVResult",
    "PositionSize",
    "RiskProfile",
    "__version__",
    "adaptive_threshold",
    "compute_cvar",
    "compute_fmn",
    "compute_ism",
    "compute_phase",
    "compute_qilm",
    "compute_topological_energy",
    "compute_var",
    "delta_entropy",
    "detect_emergent_phase",
    "direction_index",
    "fit_gpd_pot",
    "forman_ricci",
    "freedman_diaconis_bins",
    "hurst_dfa",
    "hurst_rs",
    "ism_derivative",
    "kurtosis",
    "mean_ricci",
    "mfdfa",
    "multifractal_instability",
    "ollivier_ricci",
    "order_parameter",
    "plv",
    "plv_significance",
    "preprocess_signal",
    "renyi_entropy",
    "rolling_plv",
    "shannon_entropy",
    "size_position",
    "skewness",
    "topological_asymmetry",
    "tsallis_entropy",
]
