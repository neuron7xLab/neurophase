"""Synthetic world & benchmark lab — Program H of the Evolution Board.

Ground-truth generators and benchmark scenarios that back every
claim Program C (falsification) and Program D (calibration) make.
Without H1 all scientific validation is empirical; with H1 the
ground truth is known and the tests become exact.

Public API:

* :class:`PhaseCouplingConfig` — parameters of a synthetic phase-coupling scenario.
* :class:`PhaseCouplingTrace` — generated phase signals with ground-truth PLV.
* :func:`generate_phase_coupling` — seeded ground-truth generator.
* :func:`generate_anti_coupled` — two signals that are guaranteed to be uncoupled.
"""

from __future__ import annotations

from neurophase.benchmarks.neural_phase_generator import (
    NeuralPhaseTrace,
    generate_neural_phase_trace,
    generate_synthetic_market_phase,
)
from neurophase.benchmarks.parameter_sweep import (
    SweepCellResult,
    SweepError,
    SweepGrid,
    SweepReport,
    sweep_parameters,
)
from neurophase.benchmarks.phase_coupling import (
    PhaseCouplingConfig,
    PhaseCouplingTrace,
    generate_anti_coupled,
    generate_phase_coupling,
)

__all__ = [
    "NeuralPhaseTrace",
    "PhaseCouplingConfig",
    "PhaseCouplingTrace",
    "SweepCellResult",
    "SweepError",
    "SweepGrid",
    "SweepReport",
    "generate_anti_coupled",
    "generate_neural_phase_trace",
    "generate_phase_coupling",
    "generate_synthetic_market_phase",
    "sweep_parameters",
]
