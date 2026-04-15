"""Coupled synchronization dynamics — brain × market Kuramoto systems.

This subpackage hosts full coupled-oscillator models where brain-side
oscillators (EEG α / β, HRV, pupil) and market-side oscillators (price,
volume, spread) share a **single order parameter** ``R(t)``.

The flagship model is ``CoupledBrainMarketSystem`` (equation 8.1 from the
R&D report; cf. Fioriti & Chinnici, 2012).

The :mod:`coupling_direction` layer answers the *who-leads-whom*
question that the symmetric ``R(t)`` cannot — see
:func:`analyse_coupling` and :class:`CouplingDirection`.
"""

from __future__ import annotations

from neurophase.sync.coupled_brain_market import (
    CoupledBrainMarketSystem,
    CoupledStep,
)
from neurophase.sync.coupling_direction import (
    CouplingDirection,
    analyse_coupling,
)
from neurophase.sync.market_phase import (
    MarketPhaseResult,
    extract_market_phase_from_price,
)

__all__ = [
    "CoupledBrainMarketSystem",
    "CoupledStep",
    "CouplingDirection",
    "MarketPhaseResult",
    "analyse_coupling",
    "extract_market_phase_from_price",
]
