"""F3 — same-input → same-decision certification suite.

This is the **reproducibility certification** for the `neurophase`
pipeline. It is not a unit test of any single module — it is a
cross-module contract that the composition of every core layer is
bit-deterministic under same-seed replay.

Doctrine #5 (``docs/EVOLUTION_BOARD.md``):

    An unreplayable result is not yet a system result.

What this suite proves
----------------------

1. **Same seed → bit-identical RK4 trajectory.** Two
   ``CoupledBrainMarketSystem`` instances with identical
   ``(n_brain, n_market, K, tau, sigma, dt, seed)`` produce
   numerically identical ``(R, δ, ψ_brain, ψ_market)`` sequences
   via ``pandas.testing.assert_frame_equal``.

2. **Same trajectory → same gate decisions.** The full
   ``ExecutionGate`` (with a ``StillnessDetector`` + a
   ``TemporalValidator``) produces identical state sequences over
   two replays.

3. **Same decisions → bit-identical ledger tip hash.** Two
   ``DecisionTraceLedger`` instances fed the pipeline output
   produce the same ``last_hash`` to the last hex digit. This is
   the hash-chain-level guarantee of reproducibility.

4. **Null-model harness determinism.** Two invocations of
   ``NullModelHarness.test`` with the same seed on the same inputs
   produce identical ``null_distribution`` arrays.

5. **Ledger reopen is lossless.** A ledger that is closed and
   re-opened continues the chain at the same ``last_hash`` with
   the same ``n_appended`` offset.

6. **Cross-module end-to-end replay.** A synthetic trace spanning
   every layer (coupled system → temporal validator → gate →
   ledger) replays identically on disk, i.e. the JSONL file
   content is byte-identical across two runs.

If any of these assertions fails, the system is not yet replayable
and no downstream claim (calibration, significance, regime
classification) can be trusted.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from neurophase.audit.decision_ledger import (
    DecisionTraceLedger,
    fingerprint_parameters,
    verify_ledger,
)
from neurophase.data.temporal_validator import TemporalValidator
from neurophase.gate.execution_gate import ExecutionGate
from neurophase.gate.stillness_detector import StillnessDetector
from neurophase.sync.coupled_brain_market import CoupledBrainMarketSystem
from neurophase.validation.null_model import NullModelHarness
from neurophase.validation.surrogates import cyclic_shift

# ---------------------------------------------------------------------------
# 1. Coupled system trajectory determinism
# ---------------------------------------------------------------------------


class TestCoupledSystemDeterminism:
    def test_same_seed_same_trajectory(self) -> None:
        a = CoupledBrainMarketSystem(K=5.0, sigma=0.01, dt=0.01, seed=101).run(200)
        b = CoupledBrainMarketSystem(K=5.0, sigma=0.01, dt=0.01, seed=101).run(200)
        pd.testing.assert_frame_equal(a, b)

    def test_same_seed_same_noisy_trajectory(self) -> None:
        """Even with σ > 0, the RNG is fully seeded, so replays are identical."""
        a = CoupledBrainMarketSystem(K=3.0, sigma=0.1, dt=0.005, seed=103).run(300)
        b = CoupledBrainMarketSystem(K=3.0, sigma=0.1, dt=0.005, seed=103).run(300)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_different_trajectory(self) -> None:
        a = CoupledBrainMarketSystem(K=3.0, sigma=0.1, seed=1).run(100)
        b = CoupledBrainMarketSystem(K=3.0, sigma=0.1, seed=2).run(100)
        assert not a.equals(b)


# ---------------------------------------------------------------------------
# 2. Gate decision sequence determinism
# ---------------------------------------------------------------------------


def _make_gate() -> ExecutionGate:
    return ExecutionGate(
        threshold=0.65,
        stillness_detector=StillnessDetector(
            window=8, eps_R=1e-3, eps_F=1e-3, delta_min=0.10, dt=0.01
        ),
    )


def _drive_gate(
    gate: ExecutionGate, df: pd.DataFrame, validator: TemporalValidator | None = None
) -> list[str]:
    """Iterate the gate over a trajectory DataFrame and return state names."""
    out: list[str] = []
    for i, (_, row) in enumerate(df.iterrows()):
        if validator is not None:
            tq = validator.validate(float(i) * 0.01)
            decision = gate.evaluate(
                R=float(row["R"]),
                delta=float(row["delta"]),
                time_quality=tq,
            )
        else:
            decision = gate.evaluate(
                R=float(row["R"]),
                delta=float(row["delta"]),
            )
        out.append(decision.state.name)
    return out


class TestGateDecisionDeterminism:
    def test_same_trajectory_same_gate_states(self) -> None:
        df = CoupledBrainMarketSystem(K=5.0, sigma=0.0, dt=0.01, seed=211).run(300)
        states_a = _drive_gate(_make_gate(), df)
        states_b = _drive_gate(_make_gate(), df)
        assert states_a == states_b

    def test_full_pipeline_with_temporal_validator(self) -> None:
        df = CoupledBrainMarketSystem(K=5.0, sigma=0.0, dt=0.01, seed=217).run(200)
        states_a = _drive_gate(_make_gate(), df, TemporalValidator(max_gap_seconds=1.0))
        states_b = _drive_gate(_make_gate(), df, TemporalValidator(max_gap_seconds=1.0))
        assert states_a == states_b
        # And the expected state set is non-trivial — we exercised more
        # than just one regime.
        assert len(set(states_a)) >= 2

    def test_distinct_seeds_produce_distinct_decisions(self) -> None:
        df1 = CoupledBrainMarketSystem(K=5.0, sigma=0.1, dt=0.01, seed=1).run(200)
        df2 = CoupledBrainMarketSystem(K=5.0, sigma=0.1, dt=0.01, seed=2).run(200)
        s1 = _drive_gate(_make_gate(), df1)
        s2 = _drive_gate(_make_gate(), df2)
        assert s1 != s2


# ---------------------------------------------------------------------------
# 3. Ledger tip hash determinism
# ---------------------------------------------------------------------------


def _build_pipeline_ledger(tmp_path: Path, name: str, seed: int) -> str:
    system = CoupledBrainMarketSystem(K=5.0, sigma=0.0, dt=0.01, seed=seed)
    gate = _make_gate()
    fp = fingerprint_parameters(
        {
            "threshold": gate.threshold,
            "K": system.K,
            "dt": system.dt,
            "seed": seed,
        }
    )
    ledger = DecisionTraceLedger(tmp_path / f"{name}.jsonl", fp)
    df = system.run(150)
    for i, (_, row) in enumerate(df.iterrows()):
        decision = gate.evaluate(R=float(row["R"]), delta=float(row["delta"]))
        ledger.append(
            timestamp=float(i) * system.dt,
            gate_state=decision.state.name,
            execution_allowed=decision.execution_allowed,
            R=decision.R,
            threshold=decision.threshold,
            reason=decision.reason,
        )
    return ledger.last_hash


class TestLedgerTipDeterminism:
    def test_same_pipeline_same_tip(self, tmp_path: Path) -> None:
        h1 = _build_pipeline_ledger(tmp_path, "a", seed=311)
        h2 = _build_pipeline_ledger(tmp_path, "b", seed=311)
        assert h1 == h2

    def test_different_seed_different_tip(self, tmp_path: Path) -> None:
        h1 = _build_pipeline_ledger(tmp_path, "c", seed=321)
        h2 = _build_pipeline_ledger(tmp_path, "d", seed=322)
        assert h1 != h2

    def test_ledger_file_is_byte_identical_on_replay(self, tmp_path: Path) -> None:
        """The JSONL file content must be byte-identical across two runs
        with the same seed — this is the strongest form of
        reproducibility and the foundation of postmortem replay."""
        _build_pipeline_ledger(tmp_path, "first", seed=401)
        _build_pipeline_ledger(tmp_path, "second", seed=401)
        bytes_a = (tmp_path / "first.jsonl").read_bytes()
        bytes_b = (tmp_path / "second.jsonl").read_bytes()
        assert bytes_a == bytes_b


# ---------------------------------------------------------------------------
# 4. Null-model harness determinism
# ---------------------------------------------------------------------------


def _dot(x: np.ndarray, y: np.ndarray) -> float:
    return float(x @ y / x.size)


class TestHarnessDeterminism:
    def test_same_seed_same_null(self) -> None:
        n = 128
        t = np.linspace(0, 4 * np.pi, n)
        x = np.sin(t)
        y = np.cos(t)
        harness = NullModelHarness(n_surrogates=100)

        def run() -> np.ndarray:
            rng = np.random.default_rng(42)
            return harness.test(
                x,
                y,
                statistic=_dot,
                surrogate_fn=lambda a: cyclic_shift(a, rng=rng),
                seed=42,
            ).null_distribution

        np.testing.assert_array_equal(run(), run())


# ---------------------------------------------------------------------------
# 5. Ledger reopen-and-continue correctness
# ---------------------------------------------------------------------------


class TestLedgerReopenCorrectness:
    def test_reopen_continues_chain_identically(self, tmp_path: Path) -> None:
        fp = fingerprint_parameters({"threshold": 0.65})
        path = tmp_path / "l.jsonl"

        # Run A: append all 20 records in one session.
        ledger_all = DecisionTraceLedger(path, fp)
        for i in range(20):
            ledger_all.append(
                timestamp=float(i),
                gate_state="READY",
                execution_allowed=True,
                R=0.9,
                threshold=0.65,
                reason=f"r{i}",
            )
        tip_all = ledger_all.last_hash

        # Run B: write 10 records, close, re-open, write the remaining 10.
        path2 = tmp_path / "l2.jsonl"
        ledger_b = DecisionTraceLedger(path2, fp)
        for i in range(10):
            ledger_b.append(
                timestamp=float(i),
                gate_state="READY",
                execution_allowed=True,
                R=0.9,
                threshold=0.65,
                reason=f"r{i}",
            )
        # "Close" by dropping the reference.
        del ledger_b
        ledger_b_reopened = DecisionTraceLedger(path2, fp)
        for i in range(10, 20):
            ledger_b_reopened.append(
                timestamp=float(i),
                gate_state="READY",
                execution_allowed=True,
                R=0.9,
                threshold=0.65,
                reason=f"r{i}",
            )
        tip_reopen = ledger_b_reopened.last_hash

        assert tip_all == tip_reopen
        # And the raw file bytes are identical.
        assert path.read_bytes() == path2.read_bytes()


# ---------------------------------------------------------------------------
# 6. Full end-to-end reproducibility
# ---------------------------------------------------------------------------


class TestEndToEndReplay:
    def test_full_stack_byte_identical_replay(self, tmp_path: Path) -> None:
        """Full stack: CoupledBrainMarketSystem → TemporalValidator →
        ExecutionGate(stillness) → DecisionTraceLedger must produce
        byte-identical ledger files across two runs with the same seed."""

        def run(name: str) -> Path:
            system = CoupledBrainMarketSystem(K=6.0, sigma=0.0, dt=0.01, seed=513)
            validator = TemporalValidator(max_gap_seconds=1.0, warmup_samples=2)
            gate = _make_gate()
            fp = fingerprint_parameters(
                {
                    "threshold": gate.threshold,
                    "K": system.K,
                    "dt": system.dt,
                    "seed": 513,
                    "max_gap_seconds": validator.max_gap_seconds,
                }
            )
            path = tmp_path / f"{name}.jsonl"
            ledger = DecisionTraceLedger(path, fp)
            df = system.run(120)
            for i, (_, row) in enumerate(df.iterrows()):
                tq = validator.validate(float(i) * system.dt)
                decision = gate.evaluate(
                    R=float(row["R"]),
                    delta=float(row["delta"]),
                    time_quality=tq,
                )
                ledger.append(
                    timestamp=float(i) * system.dt,
                    gate_state=decision.state.name,
                    execution_allowed=decision.execution_allowed,
                    R=decision.R,
                    threshold=decision.threshold,
                    reason=decision.reason,
                    extras={"delta": float(row["delta"])},
                )
            return path

        p1 = run("run_alpha")
        p2 = run("run_beta")

        # Byte-identical on disk.
        assert p1.read_bytes() == p2.read_bytes()

        # Both ledgers verify cleanly.
        v1 = verify_ledger(p1)
        v2 = verify_ledger(p2)
        assert v1.ok is True
        assert v2.ok is True
        assert v1.n_records == v2.n_records == 120


# ---------------------------------------------------------------------------
# Anti-regression: executive monitor determinism
# ---------------------------------------------------------------------------


class TestStillnessDetectorDeterminism:
    def test_same_input_same_decision_stream(self) -> None:
        def run() -> list[tuple[str, float]]:
            det = StillnessDetector(window=4, eps_R=1e-3, eps_F=1e-3, delta_min=0.05)
            out: list[tuple[str, float]] = []
            for i in range(20):
                R = 0.9 + 0.01 * math.sin(i * 0.1)
                delta = 0.01
                d = det.update(R=R, delta=delta)
                out.append((d.state.name, d.R))
            return out

        assert run() == run()
