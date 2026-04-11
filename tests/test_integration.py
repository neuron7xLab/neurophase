"""End-to-end integration test: physics → gate → direction → sizer.

Exercises the full pipeline on synthetic data to prove the layers
compose coherently. This is deliberately a *single* test — smaller
module-level tests cover the details.
"""

from __future__ import annotations

import numpy as np

from neurophase.core.kuramoto import KuramotoNetwork
from neurophase.core.order_parameter import order_parameter
from neurophase.gate.direction_index import Direction, direction_index
from neurophase.gate.emergent_phase import detect_emergent_phase
from neurophase.gate.execution_gate import ExecutionGate, GateState
from neurophase.metrics.asymmetry import skewness
from neurophase.risk.sizer import RiskProfile, size_position


def test_full_pipeline_on_synchronised_market() -> None:
    """When Kuramoto is synchronised and inputs are positively skewed,
    the pipeline should end in READY + LONG with a non-zero position."""
    # Step 1: physics — strongly coupled Kuramoto network → high R(t).
    omega = np.linspace(-0.3, 0.3, 20)
    net = KuramotoNetwork(omega, coupling=5.0, dt=0.05, seed=0)
    trajectory = net.run(n_steps=600)
    R = order_parameter(trajectory[-1]).R
    assert isinstance(R, float)
    assert R > 0.85

    # Step 2: execution gate — physical permission.
    gate = ExecutionGate(threshold=0.65)
    decision = gate.evaluate(R=R, sensor_present=True)
    assert decision.state is GateState.READY
    assert decision.execution_allowed

    # Step 3: emergent phase — 4-condition trigger (hand-fed numbers that
    # would come from entropy / ricci / ism elsewhere in the pipeline).
    emergent = detect_emergent_phase(R=R, dH=-0.08, kappa=-0.15, ism=1.0)
    assert emergent.is_emergent

    # Step 4: direction index with positively-skewed returns.
    rng = np.random.default_rng(42)
    returns = rng.exponential(1.0, 500) - 0.9
    sk = skewness(returns)
    di = direction_index(skew=sk, curv=0.2, bias=0.1)
    assert di.direction is Direction.LONG

    # Step 5: position sizing against a plausible CVaR.
    size = size_position(
        R=R,
        threshold=0.65,
        cvar=0.05,
        multifractal_instability_value=0.1,
        profile=RiskProfile(multifractal_penalty=1.5, max_leverage=3.0),
    )
    assert size.fraction > 0.0
    assert size.fraction <= 3.0
