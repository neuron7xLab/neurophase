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

__version__ = "0.4.0"

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
from neurophase.audit.decision_ledger import (
    GENESIS_HASH,
    DecisionTraceLedger,
    DecisionTraceRecord,
    LedgerError,
    LedgerVerification,
    fingerprint_parameters,
    verify_ledger,
)
from neurophase.audit.replay import ReplayInput, ReplayResult, replay_ledger
from neurophase.benchmarks.neural_phase_generator import (
    NeuralPhaseTrace,
    generate_neural_phase_trace,
    generate_synthetic_market_phase,
)
from neurophase.benchmarks.phase_coupling import (
    PhaseCouplingConfig,
    PhaseCouplingTrace,
    generate_anti_coupled,
    generate_phase_coupling,
)
from neurophase.benchmarks.ppc_analytical import (
    calibrated_ppc,
    ott_antonsen_order_parameter,
    ott_antonsen_ppc,
    theoretical_plv,
    theoretical_ppc,
)
from neurophase.calibration.stillness import (
    DEFAULT_DELTA_MIN_GRID,
    DEFAULT_EPS_F_GRID,
    DEFAULT_EPS_R_GRID,
    DEFAULT_WINDOW_GRID,
    StillnessCalibrationReport,
    StillnessCellEvaluation,
    StillnessGrid,
    calibrate_stillness_parameters,
)
from neurophase.calibration.threshold import (
    DEFAULT_THRESHOLD_GRID,
    ThresholdCalibrationReport,
    ThresholdEvaluation,
    ThresholdGrid,
    calibrate_gate_threshold,
)
from neurophase.core.kuramoto import KuramotoNetwork, KuramotoParams
from neurophase.core.order_parameter import OrderParameterResult, order_parameter
from neurophase.core.phase import (
    adaptive_threshold,
    compute_phase,
    preprocess_signal,
)
from neurophase.data.stream_detector import (
    StreamQualityDecision,
    StreamQualityStats,
    StreamRegime,
    TemporalStreamDetector,
)
from neurophase.data.temporal_validator import (
    TemporalError,
    TemporalQualityDecision,
    TemporalValidator,
    TimeQuality,
)
from neurophase.experiments.ds003458_analysis import (
    run_analysis as run_ds003458_analysis,
)
from neurophase.experiments.ds003458_delta_analysis import (
    run_delta_analysis as run_ds003458_delta_analysis,
)
from neurophase.experiments.ds003458_scp_analysis import (
    run_scp_analysis as run_ds003458_scp_analysis,
)
from neurophase.experiments.synthetic_plv_validation import (
    run_sweep as run_synthetic_plv_sweep,
)
from neurophase.explain import (
    Contract,
    DecisionExplanation,
    ExplanationStep,
    Verdict,
    explain_decision,
    explain_gate,
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
from neurophase.metrics.delta_power import DeltaPowerTrace, extract_delta_power
from neurophase.metrics.delta_price_xcorr import (
    DeltaPriceXCorrResult,
    compute_delta_price_xcorr,
)
from neurophase.metrics.effect_size import (
    EffectSizeReport,
    cohens_d,
    cohens_d_one_sample,
    confidence_interval_d,
    effect_size_report,
    hedges_g,
    statistical_power,
)
from neurophase.metrics.entropy import (
    delta_entropy,
    freedman_diaconis_bins,
    renyi_entropy,
    shannon_entropy,
    tsallis_entropy,
)
from neurophase.metrics.hurst import hurst_dfa, hurst_rs
from neurophase.metrics.iplv import (
    compute_ppc,
    iplv,
    iplv_on_held_out,
    iplv_significance,
    iPLVResult,
)
from neurophase.metrics.ism import compute_ism, compute_topological_energy, ism_derivative
from neurophase.metrics.plv import (
    DEFAULT_PLV_N_SURROGATES,
    HeldOutSplit,
    HeldOutViolation,
    PLVResult,
    plv,
    plv_on_held_out,
    plv_significance,
    rolling_plv,
)
from neurophase.metrics.plv_verdict import (
    DualSurrogateResult,
    PLVVerdict,
    compute_verdict,
    dual_surrogate_test,
)
from neurophase.metrics.rayleigh import RayleighResult, rayleigh_test
from neurophase.metrics.ricci import forman_ricci, mean_ricci, ollivier_ricci
from neurophase.metrics.scp import SCPTrace, extract_scp
from neurophase.oscillators.market import MarketOscillators, extract_market_phase
from neurophase.oscillators.neural_protocol import (
    NeuralFrame,
    NeuralPhaseExtractor,
    NullNeuralExtractor,
    SensorStatus,
)
from neurophase.reset import (
    Curriculum,
    KetamineLikeResetController,
    KLRConfig,
    KLRPipeline,
    ResetReport,
    ResetState,
    SystemMetrics,
    SystemState,
)
from neurophase.risk.evt import EVTFit, compute_cvar, compute_var, fit_gpd_pot
from neurophase.risk.mfdfa import MFDFAResult, mfdfa, multifractal_instability
from neurophase.risk.sizer import PositionSize, RiskProfile, size_position
from neurophase.runtime.pipeline import (
    DecisionFrame,
    PipelineConfig,
    StreamingPipeline,
)
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
from neurophase.sync.market_phase import (
    MarketPhaseResult,
    extract_market_phase_from_price,
)
from neurophase.validation.null_model import (
    DEFAULT_N_SURROGATES,
    NullModelHarness,
    NullModelResult,
)
from neurophase.validation.surrogates import (
    block_bootstrap,
    cyclic_shift,
    phase_shuffle,
    time_reversal,
)

__all__ = [
    "DEFAULT_DELTA_MIN_GRID",
    "DEFAULT_EPS_F_GRID",
    "DEFAULT_EPS_R_GRID",
    "DEFAULT_N_SURROGATES",
    "DEFAULT_PLV_N_SURROGATES",
    "DEFAULT_THRESHOLD",
    "DEFAULT_THRESHOLD_GRID",
    "DEFAULT_WINDOW_GRID",
    "GENESIS_HASH",
    "AgentEfficiency",
    "BTCFieldOrderRequest",
    "CognitiveState",
    "Contract",
    "CoupledBrainMarketSystem",
    "CoupledStep",
    "Curriculum",
    "DecisionExplanation",
    "DecisionFrame",
    "DecisionTraceLedger",
    "DecisionTraceRecord",
    "DeltaPowerTrace",
    "DeltaPriceXCorrResult",
    "DerivativesBlock",
    "Direction",
    "DirectionDecision",
    "DirectionIndexWeights",
    "DualSurrogateResult",
    "EVTFit",
    "EffectSizeReport",
    "EmergentPhaseCriteria",
    "EmergentPhaseDecision",
    "ExecutionGate",
    "ExecutiveMonitor",
    "ExecutiveMonitorConfig",
    "ExecutiveSample",
    "ExplanationStep",
    "GateDecision",
    "GateState",
    "HeldOutSplit",
    "HeldOutViolation",
    "KLRConfig",
    "KLRPipeline",
    "KetamineLikeResetController",
    "KuramotoNetwork",
    "KuramotoParams",
    "LedgerError",
    "LedgerVerification",
    "MFDFAResult",
    "MarketContext",
    "MarketOscillators",
    "MarketPhaseResult",
    "NeuralFrame",
    "NeuralPhaseExtractor",
    "NeuralPhaseTrace",
    "NullModelHarness",
    "NullModelResult",
    "NullNeuralExtractor",
    "OnchainBlock",
    "OrderBookBlock",
    "OrderParameterResult",
    "OverloadIndex",
    "PLVResult",
    "PLVVerdict",
    "PacingDirective",
    "PhaseCouplingConfig",
    "PhaseCouplingTrace",
    "PiAgent",
    "PiRule",
    "PipelineConfig",
    "PositionSize",
    "PredictionErrorMonitor",
    "PredictionErrorSample",
    "RayleighResult",
    "ReplayInput",
    "ReplayResult",
    "ResetReport",
    "ResetState",
    "RiskProfile",
    "SCPTrace",
    "Scenario",
    "SemanticMemory",
    "SensorStatus",
    "SpotBlock",
    "StillnessCalibrationReport",
    "StillnessCellEvaluation",
    "StillnessDecision",
    "StillnessDetector",
    "StillnessGrid",
    "StillnessState",
    "StreamQualityDecision",
    "StreamQualityStats",
    "StreamRegime",
    "StreamingPipeline",
    "SystemMetrics",
    "SystemState",
    "TemporalError",
    "TemporalQualityDecision",
    "TemporalStreamDetector",
    "TemporalValidator",
    "ThresholdCalibrationReport",
    "ThresholdEvaluation",
    "ThresholdGrid",
    "TimeQuality",
    "Verdict",
    "VerificationStep",
    "WhaleEvent",
    "__version__",
    "adaptive_threshold",
    "block_bootstrap",
    "build_signal_scan_payload",
    "calibrate_gate_threshold",
    "calibrate_stillness_parameters",
    "calibrated_ppc",
    "cohens_d",
    "cohens_d_one_sample",
    "compute_cvar",
    "compute_delta_price_xcorr",
    "compute_fmn",
    "compute_ism",
    "compute_phase",
    "compute_ppc",
    "compute_qilm",
    "compute_topological_energy",
    "compute_var",
    "compute_verdict",
    "confidence_interval_d",
    "cyclic_shift",
    "delta_entropy",
    "detect_emergent_phase",
    "direction_index",
    "dual_surrogate_test",
    "effect_size_report",
    "explain_decision",
    "explain_gate",
    "extract_delta_power",
    "extract_market_phase",
    "extract_market_phase_from_price",
    "extract_scp",
    "fingerprint_parameters",
    "fit_gpd_pot",
    "forman_ricci",
    "free_energy_proxy",
    "freedman_diaconis_bins",
    "generate_anti_coupled",
    "generate_neural_phase_trace",
    "generate_phase_coupling",
    "generate_synthetic_market_phase",
    "hedges_g",
    "hurst_dfa",
    "hurst_rs",
    "iPLVResult",
    "iplv",
    "iplv_on_held_out",
    "iplv_significance",
    "ism_derivative",
    "kurtosis",
    "mean_ricci",
    "mfdfa",
    "multifractal_instability",
    "ollivier_ricci",
    "order_parameter",
    "ott_antonsen_order_parameter",
    "ott_antonsen_ppc",
    "phase_shuffle",
    "plv",
    "plv_on_held_out",
    "plv_significance",
    "preprocess_signal",
    "rayleigh_test",
    "renyi_entropy",
    "replay_ledger",
    "rolling_plv",
    "run_ds003458_analysis",
    "run_ds003458_delta_analysis",
    "run_ds003458_scp_analysis",
    "run_synthetic_plv_sweep",
    "shannon_entropy",
    "size_position",
    "skewness",
    "statistical_power",
    "theoretical_plv",
    "theoretical_ppc",
    "time_reversal",
    "topological_asymmetry",
    "tsallis_entropy",
    "validate_request",
    "verify_ledger",
]
