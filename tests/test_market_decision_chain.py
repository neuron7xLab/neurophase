"""End-to-end composition test for the market-side decision chain.

The market-side decision primitives have lived as four STANDALONE
modules without a test that composes them in the order a real caller
would assemble them:

    ExecutionGate.evaluate(R, delta)                 # kernel admission
      -> direction_index(skew, curv, bias)           # LONG / SHORT / FLAT
      -> size_position(R, θ, cvar, m)                # capital fraction
      -> DecisionTraceLedger.append(...)             # audit record

This file is that test. It verifies:

1. a READY gate + LONG direction + non-zero sizer fraction produces
   a well-formed ledger record with all the expected fields;
2. a BLOCKED gate short-circuits the chain to a zero-size record
   with ``execution_allowed=False``, and the ledger still records
   the rejection (audit trail covers negative paths);
3. every record is SHA256-chained to its predecessor (the ledger
   contract is end-to-end on the composed chain, not only on
   hand-fed frames);
4. the ledger survives a full round-trip via :func:`verify_ledger`
   after the composed chain has written N records.

This closes the v1.2-rc1 audit gap: market-side ledger
(``neurophase.audit.decision_ledger.DecisionTraceLedger``) has been
available and wired into ``runtime/pipeline.py`` since v1.0, but no
public test walked the whole DI + Sizer + Gate + Ledger chain in
one go. The assumption "the ledger also records market decisions"
now has a mechanical guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurophase.audit.decision_ledger import (
    DecisionTraceLedger,
    fingerprint_parameters,
    verify_ledger,
)
from neurophase.gate.direction_index import (
    DEFAULT_WEIGHTS,
    Direction,
    direction_index,
)
from neurophase.gate.execution_gate import ExecutionGate, GateState
from neurophase.risk.sizer import RiskProfile, size_position


def _fingerprint(threshold: float) -> str:
    """Deterministic fingerprint for a single-threshold gate configuration."""
    return fingerprint_parameters({"gate.threshold": threshold, "version": "v1.2-chain-test"})


class TestMarketDecisionChainHappyPath:
    """R above threshold, well-formed DI inputs, positive CVaR."""

    def test_composed_chain_produces_ledger_record(self, tmp_path: Path) -> None:
        threshold = 0.65
        gate = ExecutionGate(threshold=threshold)
        profile = RiskProfile(risk_per_trade=0.01, confidence=0.99, max_leverage=3.0)
        ledger = DecisionTraceLedger(
            tmp_path / "chain.jsonl",
            _fingerprint(threshold),
        )

        # 1. Kernel admission.
        R = 0.82
        decision = gate.evaluate(R=R)
        assert decision.state is GateState.READY
        assert decision.execution_allowed is True

        # 2. Direction.
        di = direction_index(skew=0.18, curv=0.05, bias=0.02, weights=DEFAULT_WEIGHTS)
        assert di.direction is Direction.LONG
        assert di.value > 0

        # 3. Sizing.
        size = size_position(
            R=R,
            threshold=threshold,
            cvar=0.03,
            multifractal_instability_value=0.0,
            profile=profile,
        )
        assert size.fraction > 0.0

        # 4. Audit record.
        record = ledger.append(
            timestamp=1.0,
            gate_state=decision.state.name,
            execution_allowed=decision.execution_allowed,
            R=R,
            threshold=threshold,
            reason=decision.reason,
            extras={
                "direction": di.direction.name,
                "di_value": di.value,
                "size_fraction": size.fraction,
                "size_reason": size.reason,
            },
        )

        assert record.execution_allowed is True
        assert record.gate_state == "READY"
        assert pytest.approx(R) == record.R
        assert pytest.approx(threshold) == record.threshold
        assert record.extras["direction"] == "LONG"
        assert record.extras["size_fraction"] > 0

        # 5. Ledger integrity across the chain.
        verification = verify_ledger(tmp_path / "chain.jsonl")
        assert verification.ok is True
        assert verification.n_records == 1


class TestMarketDecisionChainBlockedPath:
    """R below threshold: negative path must still leave an audit trail."""

    def test_blocked_gate_records_zero_size_with_reason(self, tmp_path: Path) -> None:
        threshold = 0.65
        gate = ExecutionGate(threshold=threshold)
        profile = RiskProfile()
        ledger = DecisionTraceLedger(tmp_path / "blocked.jsonl", _fingerprint(threshold))

        R = 0.30  # below gate threshold
        decision = gate.evaluate(R=R)
        assert decision.state is GateState.BLOCKED
        assert decision.execution_allowed is False

        # Direction is computed regardless (signal quality is independent
        # of the kernel gate); sizer short-circuits to zero due to R < θ.
        di = direction_index(skew=0.2, curv=-0.1, bias=0.05)
        size = size_position(R=R, threshold=threshold, cvar=0.03, profile=profile)
        assert size.fraction == pytest.approx(0.0)
        assert "gate blocked" in size.reason

        record = ledger.append(
            timestamp=2.0,
            gate_state=decision.state.name,
            execution_allowed=decision.execution_allowed,
            R=R,
            threshold=threshold,
            reason=decision.reason,
            extras={
                "direction": di.direction.name,
                "di_value": di.value,
                "size_fraction": size.fraction,
                "size_reason": size.reason,
            },
        )
        assert record.execution_allowed is False
        assert record.gate_state == "BLOCKED"
        assert record.extras["size_fraction"] == pytest.approx(0.0)

        # Ledger still intact.
        assert verify_ledger(tmp_path / "blocked.jsonl").ok is True


class TestMarketDecisionChainSHA256Chain:
    """Multi-record chain: verify SHA256 linkage across N composed decisions."""

    def test_chain_of_ten_records_verifies(self, tmp_path: Path) -> None:
        threshold = 0.5
        gate = ExecutionGate(threshold=threshold)
        profile = RiskProfile()
        ledger = DecisionTraceLedger(tmp_path / "ten.jsonl", _fingerprint(threshold))

        # Deterministic R sweep: 5 above threshold, 5 below.
        for i, R in enumerate([0.80, 0.30, 0.72, 0.20, 0.95, 0.10, 0.55, 0.42, 0.99, 0.05]):
            decision = gate.evaluate(R=R)
            di = direction_index(
                skew=0.1 * (i % 3 - 1),
                curv=0.05 * (i % 2),
                bias=0.02,
            )
            size = size_position(R=R, threshold=threshold, cvar=0.03, profile=profile)
            ledger.append(
                timestamp=float(i),
                gate_state=decision.state.name,
                execution_allowed=decision.execution_allowed,
                R=R,
                threshold=threshold,
                reason=decision.reason,
                extras={
                    "tick": i,
                    "direction": di.direction.name,
                    "size_fraction": size.fraction,
                },
            )

        verification = verify_ledger(tmp_path / "ten.jsonl")
        assert verification.ok is True
        assert verification.n_records == 10

    def test_tampered_mid_chain_record_breaks_verification(self, tmp_path: Path) -> None:
        """Silent-drift guard: if any record is mutated post-append,
        verify_ledger must refuse the whole chain. This proves the
        audit trail is tamper-evident across the composed chain."""
        threshold = 0.5
        gate = ExecutionGate(threshold=threshold)
        ledger = DecisionTraceLedger(tmp_path / "tamper.jsonl", _fingerprint(threshold))
        for i, R in enumerate([0.8, 0.7, 0.6]):
            decision = gate.evaluate(R=R)
            ledger.append(
                timestamp=float(i),
                gate_state=decision.state.name,
                execution_allowed=decision.execution_allowed,
                R=R,
                threshold=threshold,
                reason=decision.reason,
                extras={"tick": i},
            )
        # Tamper with the middle record's R value without recomputing
        # the hash. verify_ledger must catch it.
        import json

        path = tmp_path / "tamper.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        mid = json.loads(lines[1])
        mid["R"] = 0.01  # drastic change; hash would no longer match
        lines[1] = json.dumps(mid)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        verification = verify_ledger(path)
        assert verification.ok is False
        assert verification.first_broken_index == 1


class TestDirectionIndexFlatPath:
    """Sanity: a DI whose magnitude is below flat_tolerance resolves to FLAT."""

    def test_flat_tolerance_honoured(self, tmp_path: Path) -> None:
        # Weights sum to 1.0; symmetric inputs -> DI == 0 exactly.
        di = direction_index(skew=0.0, curv=0.0, bias=0.0)
        assert di.direction is Direction.FLAT
        assert di.value == pytest.approx(0.0)
