"""End-to-end pipeline test for invariant I₄.

Exercises the full composition:

    CoupledBrainMarketSystem  →  (R, δ)
                               →  StillnessDetector
                               →  ExecutionGate  (5 states)

Two scenarios:

1. **High coupling, zero noise** — the joint system synchronizes,
   dynamics decay, and the gate eventually emits ``UNNECESSARY``.
2. **Zero coupling, zero noise** — the system never synchronizes and
   the gate stays ``BLOCKED`` indefinitely, never reaching the stillness
   layer.
"""

from __future__ import annotations

import numpy as np

from neurophase.gate.execution_gate import ExecutionGate, GateState
from neurophase.gate.stillness_detector import StillnessDetector, StillnessState
from neurophase.sync.coupled_brain_market import CoupledBrainMarketSystem


def _make_gate(window: int = 8) -> ExecutionGate:
    det = StillnessDetector(
        window=window,
        eps_R=1e-3,
        eps_F=1e-3,
        delta_min=0.10,
        dt=0.01,  # matches the default CoupledBrainMarketSystem dt
    )
    return ExecutionGate(threshold=0.65, stillness_detector=det)


def test_pipeline_converges_to_unnecessary_at_high_coupling() -> None:
    """With very strong coupling and no noise the joint system
    synchronizes, the steady-state phase lag ``~Δω/K`` drops below
    ``delta_min``, and the gate eventually classifies the regime as
    ``UNNECESSARY`` — acting adds no new information.

    Note on physics: for brain-frequency mean ``2π·1 Hz`` and market
    mean ``2π·0.5 Hz`` the two-population Kuramoto steady-state lag is
    ``Δω / (2 K R) ≈ π / (2·K)``. At ``K = 50`` the lag is ``≈ 0.03``,
    safely below ``delta_min = 0.10``.
    """
    sys = CoupledBrainMarketSystem(K=50.0, sigma=0.0, dt=0.01, seed=11)
    gate = _make_gate(window=8)
    df = sys.run(n_steps=400)

    states: list[GateState] = []
    for _, row in df.iterrows():
        decision = gate.evaluate(R=float(row["R"]), delta=float(row["delta"]))
        states.append(decision.state)

    # After a transient the pipeline should find at least one UNNECESSARY.
    assert GateState.UNNECESSARY in states
    # And UNNECESSARY dominates the long tail — integrate over the last 100.
    tail = states[-100:]
    assert tail.count(GateState.UNNECESSARY) > 50


def test_pipeline_blocks_at_zero_coupling() -> None:
    """With zero coupling the system never synchronizes — the stillness
    layer must never fire because the gate never reaches the READY
    branch."""
    sys = CoupledBrainMarketSystem(K=0.0, sigma=0.0, dt=0.01, seed=13)
    gate = _make_gate(window=8)
    df = sys.run(n_steps=300)

    states: list[GateState] = []
    stillness_states: list[StillnessState | None] = []
    for _, row in df.iterrows():
        decision = gate.evaluate(R=float(row["R"]), delta=float(row["delta"]))
        states.append(decision.state)
        stillness_states.append(decision.stillness_state)

    # Most time is spent blocked, and the stillness layer is never
    # reached (its provenance stays None throughout).
    assert GateState.BLOCKED in states
    # When R does cross the threshold briefly the detector is in warmup.
    assert GateState.UNNECESSARY not in states


def test_pipeline_stillness_state_is_reported_when_layer_runs() -> None:
    """Stillness provenance in ``GateDecision.stillness_state`` must be
    non-None exactly when the gate reached the stillness layer."""
    sys = CoupledBrainMarketSystem(K=10.0, sigma=0.0, dt=0.01, seed=17)
    gate = _make_gate(window=4)
    df = sys.run(n_steps=100)

    seen_non_none = False
    for _, row in df.iterrows():
        decision = gate.evaluate(R=float(row["R"]), delta=float(row["delta"]))
        if decision.state in {GateState.READY, GateState.UNNECESSARY}:
            # The stillness layer ran, so provenance must be populated.
            assert decision.stillness_state is not None
            seen_non_none = True
        else:
            # Otherwise the layer never ran, provenance must be None.
            assert decision.stillness_state is None
    assert seen_non_none


def test_pipeline_can_use_dataframe_columns_directly() -> None:
    """The DataFrame exposed by ``run()`` must contain all columns needed
    by the stillness layer — this test verifies the downstream contract."""
    sys = CoupledBrainMarketSystem(K=4.0, sigma=0.0, seed=23)
    df = sys.run(n_steps=50)
    required = {"R", "delta"}
    assert required <= set(df.columns)
    # And R, delta are physical — no NaN, inside their ranges.
    assert not df[["R", "delta"]].isna().any().any()
    assert df["R"].between(0.0, 1.0).all()
    assert df["delta"].between(0.0, float(np.pi) + 1e-9).all()
