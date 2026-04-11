from __future__ import annotations

from neurophase.reset.adaptive_threshold import AdaptiveThreshold
from neurophase.reset.ledger import LedgerEntry


def _entry(decision: str, score: float, ts: float) -> LedgerEntry:
    return LedgerEntry(
        timestamp=ts,
        state_hash="h",
        metrics_snapshot={"lockin_score": score},
        decision=decision,
        relapse_ratio=0.0,
        improvement_ratio=1.0,
        reason="r",
        new_frozen_nodes=[],
    )


def test_fallback_and_serialize_roundtrip() -> None:
    t = AdaptiveThreshold()
    assert t.update([]) == 0.72
    blob = t.serialize()
    rt = AdaptiveThreshold.deserialize(blob)
    assert rt.serialize() == blob


def test_threshold_moves_with_outcomes_and_freeze() -> None:
    t = AdaptiveThreshold(update_interval=50)
    hist = [_entry("ROLLBACK", 0.60, i) for i in range(50)]
    assert t.update(hist) <= 0.72
    prev = t.current()
    t.freeze()
    _ = t.update([*hist, _entry("ROLLBACK", 0.55, 100.0)])
    assert t.current() == prev
    t.unfreeze()
    hist2 = hist + [_entry("SUCCESS", 0.85, i + 200.0) for i in range(60)]
    assert t.update(hist2) >= prev
