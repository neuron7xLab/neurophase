"""neurophase — phase synchronization as execution gate.

Kuramoto model of market × trader nervous system.

Four invariants that cannot be overridden:

    I₁: R(t) < θ              ⇒ execution_allowed = False
    I₂: bio-sensor absent     ⇒ execution_allowed = False
    I₃: R(t) invalid / OOR    ⇒ execution_allowed = False
    I₄: stillness             ⇒ execution_allowed = False  (action_unnecessary)

See ``docs/theory/scientific_basis.md`` and
``docs/theory/stillness_invariant.md`` for the full derivation.
"""

from __future__ import annotations

__version__ = "0.3.0"

from neurophase.agents.pi_agent import (
    AgentEfficiency,
    MarketContext,
    PiAgent,
    PiRule,
    SemanticMemory,
)
from neurophase.analysis.prediction_error import (
    CognitiveState,
    PredictionErrorMonitor,
    PredictionErrorSample,
)
from neurophase.core.kuramoto import KuramotoNetwork, KuramotoParams
from neurophase.core.order_parameter import OrderParameterResult, order_parameter
from neurophase.core.phase import (
    adaptive_threshold,
    compute_phase,
    preprocess_signal,
)
from neurophase.data.temporal_validator import (
    TemporalError,
    TemporalQualityDecision,
    TemporalValidator,
    TimeQuality,
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
from neurophase.gate.stillness_detector import (
    StillnessDecision,
    StillnessDetector,
    StillnessState,
    free_energy_proxy,
)
from neurophase.indicators.fmn import compute_fmn
from neurophase.indicators.qilm import compute_qilm
from neurophase.intel.btc_field_order import (
    BTCFieldOrderRequest,
    DerivativesBlock,
    OnchainBlock,
    OrderBookBlock,
    Scenario,
    SpotBlock,
    WhaleEvent,
    build_signal_scan_payload,
    validate_request,
)
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
from neurophase.oscillators.market import MarketOscillators, extract_market_phase
from neurophase.oscillators.neural_protocol import (
    NeuralFrame,
    NeuralPhaseExtractor,
    NullNeuralExtractor,
    SensorStatus,
)
from neurophase.risk.evt import EVTFit, compute_cvar, compute_var, fit_gpd_pot
from neurophase.risk.mfdfa import MFDFAResult, mfdfa, multifractal_instability
from neurophase.risk.sizer import PositionSize, RiskProfile, size_position
from neurophase.state.executive_monitor import (
    ExecutiveMonitor,
    ExecutiveMonitorConfig,
    ExecutiveSample,
    OverloadIndex,
    PacingDirective,
    VerificationStep,
)
from neurophase.sync.coupled_brain_market import (
    CoupledBrainMarketSystem,
    CoupledStep,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "AgentEfficiency",
    "BTCFieldOrderRequest",
    "CognitiveState",
    "CoupledBrainMarketSystem",
    "CoupledStep",
    "DerivativesBlock",
    "Direction",
    "DirectionDecision",
    "DirectionIndexWeights",
    "EVTFit",
    "EmergentPhaseCriteria",
    "EmergentPhaseDecision",
    "ExecutionGate",
    "ExecutiveMonitor",
    "ExecutiveMonitorConfig",
    "ExecutiveSample",
    "GateDecision",
    "GateState",
    "KuramotoNetwork",
    "KuramotoParams",
    "MFDFAResult",
    "MarketContext",
    "MarketOscillators",
    "NeuralFrame",
    "NeuralPhaseExtractor",
    "NullNeuralExtractor",
    "OnchainBlock",
    "OrderBookBlock",
    "OrderParameterResult",
    "OverloadIndex",
    "PLVResult",
    "PacingDirective",
    "PiAgent",
    "PiRule",
    "PositionSize",
    "PredictionErrorMonitor",
    "PredictionErrorSample",
    "RiskProfile",
    "Scenario",
    "SemanticMemory",
    "SensorStatus",
    "SpotBlock",
    "StillnessDecision",
    "StillnessDetector",
    "StillnessState",
    "TemporalError",
    "TemporalQualityDecision",
    "TemporalValidator",
    "TimeQuality",
    "VerificationStep",
    "WhaleEvent",
    "__version__",
    "adaptive_threshold",
    "build_signal_scan_payload",
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
    "extract_market_phase",
    "fit_gpd_pot",
    "forman_ricci",
    "free_energy_proxy",
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
    "validate_request",
]
