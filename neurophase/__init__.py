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
from neurophase.gate.execution_gate import (
    DEFAULT_THRESHOLD,
    ExecutionGate,
    GateDecision,
    GateState,
)
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

__all__ = [
    "DEFAULT_THRESHOLD",
    "ExecutionGate",
    "GateDecision",
    "GateState",
    "KuramotoNetwork",
    "KuramotoParams",
    "OrderParameterResult",
    "PLVResult",
    "__version__",
    "adaptive_threshold",
    "compute_ism",
    "compute_phase",
    "compute_topological_energy",
    "delta_entropy",
    "forman_ricci",
    "freedman_diaconis_bins",
    "hurst_dfa",
    "hurst_rs",
    "ism_derivative",
    "mean_ricci",
    "ollivier_ricci",
    "order_parameter",
    "plv",
    "plv_significance",
    "preprocess_signal",
    "renyi_entropy",
    "rolling_plv",
    "shannon_entropy",
    "tsallis_entropy",
]
