from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neurophase.reset import (
    Curriculum,
    FrozenNodeAnalyzer,
    IntegrityOracle,
    LedgerEntry,
    MarketCouplingValidator,
    MarketPhase,
    PlasticityMonitor,
    RollbackLedger,
    SystemMetrics,
    SystemState,
)
from neurophase.reset.controller import ResetReport


def _state(n: int = 4) -> SystemState:
    return SystemState(
        weights=np.full((n, n), 1 / n),
        confidence=np.full(n, 0.6),
        usage=np.full(n, 0.5),
        utility=np.full(n, 0.8),
        inhibition=np.full(n, 0.7),
        topology=np.ones((n, n)),
    )


def test_ledger_append_query_export_stats(tmp_path: Path) -> None:
    ledger = RollbackLedger(capacity=2)
    state = _state()
    metrics = SystemMetrics(0.9, 0.9, 0.1, 0.1, 0.0, 0.0)
    report = ResetReport("ROLLBACK", "x", 1.0, 0.0, 0)
    ledger.record_attempt(state, metrics, report)
    assert len(ledger.query_by_reason("x")) == 1
    out = tmp_path / "ledger.json"
    ledger.export_json(out)
    payload = json.loads(out.read_text())
    assert len(payload) == 1
    stats = ledger.statistics()
    assert stats["total"] == 1


def test_frozen_analyzer_and_premature() -> None:
    pre = _state()
    post = _state()
    assert post.frozen is not None
    post.frozen[:] = True
    post.utility[:] = 0.7
    post.weights[0, 0] = 0.9
    ok, warns = FrozenNodeAnalyzer().verify_consolidation(
        pre, post, Curriculum(np.zeros(4), np.zeros(4), np.zeros(4))
    )
    assert not ok
    assert warns
    history = [pre, post]
    flagged = FrozenNodeAnalyzer().detect_premature_freeze(history)
    assert flagged


def test_integrity_oracle() -> None:
    s = _state()
    oracle = IntegrityOracle()
    h1 = oracle.checksum_state(s)
    s.weights[0, 0] = 0.9
    h2 = oracle.checksum_state(s)
    oracle.log_mutation(h1, h2, "test")
    oracle.assert_mutation_logged(h1, h2)
    assert oracle.audit_trail()


def test_market_coupling_and_monitor() -> None:
    v = MarketCouplingValidator()
    aligned, gamma = v.verify_regime_alignment(np.array([1.0, 0.5, 0.2]), np.array([1.0, 0.4, 0.3]))
    assert isinstance(aligned, bool)
    assert gamma >= 0.0
    c = v.curriculum_from_market_regime(MarketPhase.BREAKDOWN, 4)
    assert np.allclose(c.corrective_signal, 0.0)

    monitor = PlasticityMonitor()
    rep = monitor.compute(
        usage=np.array([0.8, 0.4, 0.2, 0.1], dtype=float),
        frozen=np.array([False, False, True, False]),
        ntk_rank=0.2,
        plasticity_floor=0.3,
        injection_triggered=True,
    )
    assert rep.injection_triggered
    assert rep.ntk_rank_vs_floor < 0.0


def test_ledger_entry_type_access() -> None:
    e = LedgerEntry(0.0, "h", {"error": 0.1}, "SKIPPED", 0.0, 0.0, "ok", [])
    assert e.decision == "SKIPPED"
