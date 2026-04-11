"""Runnable demo: CoupledBrainMarketSystem → StillnessDetector → ExecutionGate.

Runs the full ``I₁``–``I₄`` pipeline on 500 RK4 steps of a coupled
brain–market Kuramoto system with strong coupling and no noise,
printing a histogram of the five gate states and the first sample at
which each state is reached.

Usage::

    python examples/stillness_demo.py

Expected output (seed=11, K=50):

    Gate state histogram over 500 steps:
      BLOCKED        : NN
      READY          : NN
      UNNECESSARY    : NNN
      ...

and a time-to-first for each reached state.
"""

from __future__ import annotations

from collections import Counter

from neurophase.gate.execution_gate import ExecutionGate, GateState
from neurophase.gate.stillness_detector import StillnessDetector
from neurophase.sync.coupled_brain_market import CoupledBrainMarketSystem


def main() -> None:
    # Build the full pipeline.
    system = CoupledBrainMarketSystem(K=50.0, sigma=0.0, dt=0.01, seed=11)
    stillness = StillnessDetector(
        window=8,
        eps_R=1e-3,
        eps_F=1e-3,
        delta_min=0.10,
        dt=0.01,
    )
    gate = ExecutionGate(threshold=0.65, stillness_detector=stillness)

    # Integrate and classify.
    df = system.run(n_steps=500)
    counts: Counter[GateState] = Counter()
    first_seen: dict[GateState, int] = {}

    for step_idx, (_, row) in enumerate(df.iterrows()):
        decision = gate.evaluate(R=float(row["R"]), delta=float(row["delta"]))
        counts[decision.state] += 1
        if decision.state not in first_seen:
            first_seen[decision.state] = step_idx

    print("Gate state histogram over 500 steps:")
    for state in GateState:
        print(f"  {state.name:<14}: {counts.get(state, 0):>4}")
    print()
    print("First step at which each state is reached:")
    for state, idx in sorted(first_seen.items(), key=lambda kv: kv[1]):
        print(f"  {state.name:<14}: step {idx}")


if __name__ == "__main__":
    main()
