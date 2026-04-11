"""Tests for ``neurophase.audit.decision_ledger`` (F1).

Covers:

* append/read round-trip
* SHA256 hash chain linkage
* tamper detection (every mutation type must be caught at the first
  broken record and the index reported)
* same-input → same-hash determinism
* parameter-fingerprint uniqueness
* append continuation across two ledger instances on the same file
* schema-missing-field detection
* genesis-hash correctness
* end-to-end gate → ledger → replay-verify test
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from neurophase.audit.decision_ledger import (
    GENESIS_HASH,
    DecisionTraceLedger,
    DecisionTraceRecord,
    LedgerError,
    LedgerVerification,
    fingerprint_parameters,
    verify_ledger,
)
from neurophase.gate.execution_gate import ExecutionGate

# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_deterministic(self) -> None:
        p = {"threshold": 0.65, "dt": 0.01, "seed": 42}
        assert fingerprint_parameters(p) == fingerprint_parameters(p)

    def test_key_order_invariant(self) -> None:
        p1 = {"threshold": 0.65, "dt": 0.01}
        p2 = {"dt": 0.01, "threshold": 0.65}
        assert fingerprint_parameters(p1) == fingerprint_parameters(p2)

    def test_distinct_params_distinct_fingerprints(self) -> None:
        a = fingerprint_parameters({"threshold": 0.65})
        b = fingerprint_parameters({"threshold": 0.70})
        assert a != b


# ---------------------------------------------------------------------------
# Basic append + verify
# ---------------------------------------------------------------------------


def _sample_fingerprint() -> str:
    return fingerprint_parameters({"threshold": 0.65, "dt": 0.01})


class TestAppendAndVerify:
    def test_single_append_and_verify(self, tmp_path: Path) -> None:
        ledger = DecisionTraceLedger(tmp_path / "decisions.jsonl", _sample_fingerprint())
        record = ledger.append(
            timestamp=1.0,
            gate_state="READY",
            execution_allowed=True,
            R=0.9,
            threshold=0.65,
            reason="test",
        )
        assert record.index == 0
        assert record.prev_hash == GENESIS_HASH
        assert record.record_hash != GENESIS_HASH
        assert ledger.n_appended == 1
        assert ledger.last_hash == record.record_hash

        verification = verify_ledger(ledger.path)
        assert verification.ok
        assert verification.n_records == 1
        assert verification.first_broken_index is None

    def test_multi_append_and_verify(self, tmp_path: Path) -> None:
        ledger = DecisionTraceLedger(tmp_path / "decisions.jsonl", _sample_fingerprint())
        for i in range(10):
            ledger.append(
                timestamp=float(i),
                gate_state="READY" if i % 2 == 0 else "BLOCKED",
                execution_allowed=(i % 2 == 0),
                R=0.8 + 0.01 * i,
                threshold=0.65,
                reason=f"tick {i}",
            )
        verification = verify_ledger(ledger.path)
        assert verification.ok
        assert verification.n_records == 10

    def test_chain_links_correctly(self, tmp_path: Path) -> None:
        """Each record's prev_hash must equal the previous record's record_hash."""
        ledger = DecisionTraceLedger(tmp_path / "decisions.jsonl", _sample_fingerprint())
        records: list[DecisionTraceRecord] = []
        for i in range(5):
            records.append(
                ledger.append(
                    timestamp=float(i),
                    gate_state="READY",
                    execution_allowed=True,
                    R=0.9,
                    threshold=0.65,
                    reason="test",
                )
            )
        assert records[0].prev_hash == GENESIS_HASH
        for prev, curr in itertools.pairwise(records):
            assert curr.prev_hash == prev.record_hash


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------


def _write_and_tamper(
    tmp_path: Path,
    *,
    mutator: Callable[[list[str]], list[str]],
    expected_broken_index: int,
) -> LedgerVerification:
    """Helper: write a 5-record ledger then apply ``mutator`` to a line."""
    ledger = DecisionTraceLedger(tmp_path / "decisions.jsonl", _sample_fingerprint())
    for i in range(5):
        ledger.append(
            timestamp=float(i),
            gate_state="READY",
            execution_allowed=True,
            R=0.9 + 0.01 * i,
            threshold=0.65,
            reason=f"r{i}",
        )

    lines = ledger.path.read_text(encoding="utf-8").splitlines()
    lines = mutator(lines)
    ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    v = verify_ledger(ledger.path)
    assert v.ok is False
    assert v.first_broken_index == expected_broken_index
    return v


class TestTamperDetection:
    def test_mutated_reason_detected(self, tmp_path: Path) -> None:
        def mutate(lines: list[str]) -> list[str]:
            payload = json.loads(lines[2])
            payload["reason"] = "tampered"
            lines[2] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            return lines

        _write_and_tamper(tmp_path, mutator=mutate, expected_broken_index=2)

    def test_mutated_R_detected(self, tmp_path: Path) -> None:
        def mutate(lines: list[str]) -> list[str]:
            payload = json.loads(lines[1])
            payload["R"] = 0.0001
            lines[1] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            return lines

        _write_and_tamper(tmp_path, mutator=mutate, expected_broken_index=1)

    def test_swapped_records_detected(self, tmp_path: Path) -> None:
        def mutate(lines: list[str]) -> list[str]:
            lines[1], lines[2] = lines[2], lines[1]
            return lines

        _write_and_tamper(tmp_path, mutator=mutate, expected_broken_index=1)

    def test_deleted_record_detected(self, tmp_path: Path) -> None:
        def mutate(lines: list[str]) -> list[str]:
            return lines[:2] + lines[3:]

        # Deleting line 2 means line 3's prev_hash still points at line 1's
        # record_hash but its index is now out of order.
        _write_and_tamper(tmp_path, mutator=mutate, expected_broken_index=2)

    def test_bogus_hash_detected(self, tmp_path: Path) -> None:
        def mutate(lines: list[str]) -> list[str]:
            payload = json.loads(lines[3])
            payload["record_hash"] = "f" * 64
            lines[3] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            return lines

        _write_and_tamper(tmp_path, mutator=mutate, expected_broken_index=3)

    def test_truncated_line_detected(self, tmp_path: Path) -> None:
        def mutate(lines: list[str]) -> list[str]:
            lines[2] = lines[2][: len(lines[2]) // 2]
            return lines

        _write_and_tamper(tmp_path, mutator=mutate, expected_broken_index=2)


# ---------------------------------------------------------------------------
# Missing / malformed file
# ---------------------------------------------------------------------------


class TestFileIO:
    def test_missing_file_verification(self, tmp_path: Path) -> None:
        v = verify_ledger(tmp_path / "nope.jsonl")
        assert v.ok is False
        assert v.n_records == 0
        assert v.reason.startswith("ledger file not found")

    def test_missing_field_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text(
            json.dumps({"index": 0, "timestamp": 0.0}) + "\n",
            encoding="utf-8",
        )
        v = verify_ledger(path)
        assert v.ok is False
        assert v.first_broken_index == 0
        assert "missing field" in v.reason

    def test_junk_json_detected(self, tmp_path: Path) -> None:
        path = tmp_path / "junk.jsonl"
        path.write_text("{not json\n", encoding="utf-8")
        v = verify_ledger(path)
        assert v.ok is False
        assert "not valid JSON" in v.reason


# ---------------------------------------------------------------------------
# Append continuation across instances
# ---------------------------------------------------------------------------


class TestAppendContinuation:
    def test_second_instance_continues_chain(self, tmp_path: Path) -> None:
        path = tmp_path / "decisions.jsonl"
        fp = _sample_fingerprint()

        ledger_a = DecisionTraceLedger(path, fp)
        ledger_a.append(
            timestamp=1.0,
            gate_state="READY",
            execution_allowed=True,
            R=0.9,
            threshold=0.65,
            reason="a",
        )

        ledger_b = DecisionTraceLedger(path, fp)
        assert ledger_b.n_appended == 1
        assert ledger_b.last_hash == ledger_a.last_hash

        ledger_b.append(
            timestamp=2.0,
            gate_state="READY",
            execution_allowed=True,
            R=0.92,
            threshold=0.65,
            reason="b",
        )
        v = verify_ledger(path)
        assert v.ok
        assert v.n_records == 2

    def test_reopening_broken_ledger_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "decisions.jsonl"
        fp = _sample_fingerprint()
        ledger = DecisionTraceLedger(path, fp)
        ledger.append(
            timestamp=1.0,
            gate_state="READY",
            execution_allowed=True,
            R=0.9,
            threshold=0.65,
            reason="first",
        )
        # Corrupt.
        raw = path.read_text(encoding="utf-8")
        path.write_text(raw.replace("first", "tampered"), encoding="utf-8")

        with pytest.raises(LedgerError, match="broken"):
            DecisionTraceLedger(path, fp)


# ---------------------------------------------------------------------------
# Determinism: same input → same hash
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_ledgers_produce_identical_hash_chain(self, tmp_path: Path) -> None:
        fp = _sample_fingerprint()

        def build(p: Path) -> str:
            ledger = DecisionTraceLedger(p, fp)
            for i in range(5):
                ledger.append(
                    timestamp=float(i),
                    gate_state="READY",
                    execution_allowed=True,
                    R=0.9,
                    threshold=0.65,
                    reason=f"tick {i}",
                )
            return ledger.last_hash

        h_a = build(tmp_path / "a.jsonl")
        h_b = build(tmp_path / "b.jsonl")
        assert h_a == h_b

    def test_distinct_parameters_yield_distinct_hashes(self, tmp_path: Path) -> None:
        fp_a = fingerprint_parameters({"threshold": 0.65})
        fp_b = fingerprint_parameters({"threshold": 0.70})

        ledger_a = DecisionTraceLedger(tmp_path / "a.jsonl", fp_a)
        ledger_b = DecisionTraceLedger(tmp_path / "b.jsonl", fp_b)
        ledger_a.append(
            timestamp=1.0,
            gate_state="READY",
            execution_allowed=True,
            R=0.9,
            threshold=0.65,
            reason="test",
        )
        ledger_b.append(
            timestamp=1.0,
            gate_state="READY",
            execution_allowed=True,
            R=0.9,
            threshold=0.65,
            reason="test",
        )
        assert ledger_a.last_hash != ledger_b.last_hash


# ---------------------------------------------------------------------------
# Frozen record
# ---------------------------------------------------------------------------


class TestFrozenRecord:
    def test_record_is_frozen(self) -> None:
        r = DecisionTraceRecord(
            index=0,
            timestamp=0.0,
            gate_state="READY",
            execution_allowed=True,
            R=0.9,
            threshold=0.65,
            reason="test",
            parameter_fingerprint="abc",
            extras={},
            prev_hash=GENESIS_HASH,
            record_hash="deadbeef",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.index = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# End-to-end: gate → ledger → verify
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_gate_decisions_written_and_verified(self, tmp_path: Path) -> None:
        gate = ExecutionGate(threshold=0.65)
        fp = fingerprint_parameters({"threshold": gate.threshold, "has_stillness": False})
        ledger = DecisionTraceLedger(tmp_path / "decisions.jsonl", fp)

        for i, R in enumerate([0.30, 0.50, 0.70, 0.85, 0.99]):
            decision = gate.evaluate(R=R)
            ledger.append(
                timestamp=float(i),
                gate_state=decision.state.name,
                execution_allowed=decision.execution_allowed,
                R=decision.R,
                threshold=decision.threshold,
                reason=decision.reason,
            )

        v = verify_ledger(ledger.path)
        assert v.ok
        assert v.n_records == 5

        # Parse the ledger back and confirm the gate states.
        states = []
        for line in ledger.path.read_text(encoding="utf-8").splitlines():
            states.append(json.loads(line)["gate_state"])
        assert states == ["BLOCKED", "BLOCKED", "READY", "READY", "READY"]
