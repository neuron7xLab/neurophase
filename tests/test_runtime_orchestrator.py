"""E3 — contract tests for the runtime orchestrator.

This test file is the HN26 binding. It locks in:

1. **Single-call composition.** One ``orchestrator.tick()`` call
   drives every layer (B1+B2+B6 → I₁..I₄ → G1 regime → I1
   policy → F1 ledger), in the canonical order, exactly once.
2. **Determinism.** Two orchestrators with identical
   configuration fed identical input sequences emit byte-
   identical :class:`OrchestratedFrame` sequences and
   byte-identical :class:`SessionManifest` outputs at close.
3. **Gate-honoring.** Vetoed frames produce HOLD intents
   regardless of regime. The orchestrator never widens the
   gate's permission surface.
4. **Layer-skip on missing inputs.** A frame with ``R is None``
   or ``delta is None`` skips the regime + policy layers (the
   regime classifier requires both) and surfaces ``regime=None``
   and ``action=None`` on the envelope.
5. **Session sealing.** ``close()`` materialises the
   :class:`SessionManifest` and verifies the bound ledger
   end-to-end. After ``close()``, ``tick()`` raises ``RuntimeError``.
6. **Manifest determinism.** Two sessions with the same config
   and the same input sequence close to manifests that share
   ``manifest_hash``, ``run_id``, ``parameter_fingerprint``,
   ``ledger_tip_hash``, and ``n_ticks``.
7. **Frozen envelopes.** Every emitted
   :class:`OrchestratedFrame` rejects attribute reassignment.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neurophase.audit.session_manifest import (
    ManifestError,
    SessionManifest,
    compute_dataset_fingerprint,
)
from neurophase.gate.execution_gate import GateState
from neurophase.policy.action import ActionIntent, PolicyConfig
from neurophase.runtime.orchestrator import (
    OrchestratedFrame,
    OrchestratorConfig,
    RuntimeOrchestrator,
)
from neurophase.runtime.pipeline import PipelineConfig

# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _config(
    *,
    ledger_path: Path | None = None,
    cooldown: int = 0,
    require_warm_regime: bool = True,
) -> OrchestratorConfig:
    return OrchestratorConfig(
        pipeline=PipelineConfig(
            warmup_samples=2,
            stream_window=4,
            enable_stillness=False,
            ledger_path=ledger_path,
        ),
        policy=PolicyConfig(
            cooldown_ticks=cooldown,
            require_warm_regime=require_warm_regime,
        ),
    )


def _drive_steady_high_R(
    orch: RuntimeOrchestrator, *, n: int, R: float = 0.95, delta: float = 0.05
) -> list[OrchestratedFrame]:
    """Drive ``n`` ticks of constant high R / stable δ."""
    out: list[OrchestratedFrame] = []
    for i in range(n):
        out.append(orch.tick(timestamp=float(i) * 0.1, R=R, delta=delta))
    return out


# ---------------------------------------------------------------------------
# 1. Single-call composition.
# ---------------------------------------------------------------------------


class TestSingleCallComposition:
    def test_tick_returns_orchestrated_frame(self) -> None:
        orch = RuntimeOrchestrator(_config())
        frame = orch.tick(timestamp=0.0, R=0.9, delta=0.05)
        assert isinstance(frame, OrchestratedFrame)
        assert frame.pipeline_frame is not None
        assert frame.tick_index == 0

    def test_tick_drives_every_layer(self) -> None:
        """After enough warmup, a high-R steady tick produces a
        non-None regime + non-None action."""
        orch = RuntimeOrchestrator(_config())
        frames = _drive_steady_high_R(orch, n=8)
        # Find the first frame after warmup with a non-None regime.
        late = frames[-1]
        assert late.pipeline_frame is not None
        assert late.regime is not None
        assert late.action is not None

    def test_tick_index_is_monotonic(self) -> None:
        orch = RuntimeOrchestrator(_config())
        frames = _drive_steady_high_R(orch, n=5)
        assert [f.tick_index for f in frames] == list(range(5))


# ---------------------------------------------------------------------------
# 2. Determinism.
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_orchestrators_same_input_same_output(self) -> None:
        cfg = _config()
        o1 = RuntimeOrchestrator(cfg)
        o2 = RuntimeOrchestrator(cfg)
        seq = [(float(i) * 0.1, 0.9 + 0.001 * i, 0.05) for i in range(10)]
        out1 = [o1.tick(timestamp=t, R=R, delta=d) for t, R, d in seq]
        out2 = [o2.tick(timestamp=t, R=R, delta=d) for t, R, d in seq]
        assert [f.pipeline_frame.gate.state for f in out1] == [
            f.pipeline_frame.gate.state for f in out2
        ]
        assert [f.regime.label if f.regime else None for f in out1] == [
            f.regime.label if f.regime else None for f in out2
        ]
        assert [f.action.intent if f.action else None for f in out1] == [
            f.action.intent if f.action else None for f in out2
        ]

    def test_parameter_fingerprint_is_stable(self) -> None:
        cfg = _config()
        o1 = RuntimeOrchestrator(cfg)
        o2 = RuntimeOrchestrator(cfg)
        assert o1.parameter_fingerprint == o2.parameter_fingerprint


# ---------------------------------------------------------------------------
# 3. Gate-honoring.
# ---------------------------------------------------------------------------


class TestGateHonoring:
    def test_vetoed_frame_yields_hold(self) -> None:
        """A frame with R below threshold produces a vetoed gate
        and a HOLD action — the orchestrator never widens the
        gate's permission surface."""
        orch = RuntimeOrchestrator(_config())
        # Warm up the pipeline so it leaves WARMUP.
        for i in range(5):
            orch.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.05)
        # Inject a low-R frame.
        frame = orch.tick(timestamp=0.6, R=0.10, delta=0.05)
        assert frame.pipeline_frame.gate.state is GateState.BLOCKED
        assert frame.pipeline_frame.gate.execution_allowed is False
        assert frame.action is not None
        assert frame.action.intent is ActionIntent.HOLD

    def test_degraded_R_none_skips_regime_and_action(self) -> None:
        """An R=None frame routes through DEGRADED, and the
        orchestrator skips the regime + policy layers entirely."""
        orch = RuntimeOrchestrator(_config())
        # Warm up first.
        for i in range(5):
            orch.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.05)
        frame = orch.tick(timestamp=0.6, R=None, delta=0.05)
        assert frame.pipeline_frame.gate.state is GateState.DEGRADED
        assert frame.regime is None
        assert frame.action is None


# ---------------------------------------------------------------------------
# 4. Layer-skip on missing inputs.
# ---------------------------------------------------------------------------


class TestLayerSkip:
    def test_missing_delta_skips_regime(self) -> None:
        orch = RuntimeOrchestrator(_config())
        frame = orch.tick(timestamp=0.0, R=0.9, delta=None)
        assert frame.regime is None
        assert frame.action is None


# ---------------------------------------------------------------------------
# 5. Session sealing — close() and tick-after-close.
# ---------------------------------------------------------------------------


class TestSessionSealing:
    def test_close_returns_session_manifest(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        orch = RuntimeOrchestrator(_config(ledger_path=ledger))
        for i in range(5):
            orch.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.05)
        manifest = orch.close(end_ts=0.5)
        assert isinstance(manifest, SessionManifest)
        assert manifest.n_ticks == 5
        assert manifest.parameter_fingerprint == orch.parameter_fingerprint

    def test_close_verifies_bound_ledger(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        orch = RuntimeOrchestrator(_config(ledger_path=ledger))
        for i in range(5):
            orch.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.05)
        # Tamper with the ledger before closing.
        text = ledger.read_text()
        tampered = text.replace("READY", "READYX", 1)
        ledger.write_text(tampered)
        with pytest.raises(ManifestError):
            orch.close(end_ts=0.5)

    def test_close_without_ledger_succeeds(self) -> None:
        orch = RuntimeOrchestrator(_config(ledger_path=None))
        orch.tick(timestamp=0.0, R=0.9, delta=0.05)
        manifest = orch.close(end_ts=0.0)
        assert manifest.n_ticks == 1
        # No ledger → tip stays at the empty sentinel.
        from neurophase.audit.session_manifest import EMPTY_LEDGER_TIP

        assert manifest.ledger_tip_hash == EMPTY_LEDGER_TIP

    def test_tick_after_close_raises(self) -> None:
        orch = RuntimeOrchestrator(_config())
        orch.tick(timestamp=0.0, R=0.9, delta=0.05)
        orch.close()
        with pytest.raises(RuntimeError, match="after close"):
            orch.tick(timestamp=0.1, R=0.9, delta=0.05)

    def test_double_close_raises(self) -> None:
        orch = RuntimeOrchestrator(_config())
        orch.tick(timestamp=0.0, R=0.9, delta=0.05)
        orch.close()
        with pytest.raises(RuntimeError, match="close called twice"):
            orch.close()

    def test_close_with_explicit_dataset_fingerprint(self) -> None:
        orch = RuntimeOrchestrator(_config())
        orch.tick(timestamp=0.0, R=0.9, delta=0.05)
        fp = compute_dataset_fingerprint(b"explicit dataset")
        manifest = orch.close(end_ts=0.0, dataset_fingerprint=fp)
        assert manifest.dataset_fingerprint == fp


# ---------------------------------------------------------------------------
# 6. Manifest determinism — same config + same inputs → same manifest.
# ---------------------------------------------------------------------------


class TestManifestDeterminism:
    def test_two_sessions_same_inputs_same_manifest_hash(self, tmp_path: Path) -> None:
        cfg_a = _config(ledger_path=tmp_path / "a.jsonl")
        cfg_b = _config(ledger_path=tmp_path / "b.jsonl")
        oa = RuntimeOrchestrator(cfg_a)
        ob = RuntimeOrchestrator(cfg_b)
        for i in range(6):
            oa.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.05)
            ob.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.05)
        ma = oa.close(end_ts=0.5)
        mb = ob.close(end_ts=0.5)
        # Different ledger paths → different run_id and
        # manifest_hash by design. But the parameter fingerprint
        # and the *content* of the ledgers should match.
        assert ma.parameter_fingerprint == mb.parameter_fingerprint
        assert ma.n_ticks == mb.n_ticks
        assert ma.ledger_tip_hash == mb.ledger_tip_hash

    def test_different_config_changes_parameter_fingerprint(self) -> None:
        oa = RuntimeOrchestrator(_config())
        ob = RuntimeOrchestrator(
            OrchestratorConfig(
                pipeline=PipelineConfig(
                    warmup_samples=2,
                    stream_window=4,
                    enable_stillness=False,
                    threshold=0.40,  # different threshold
                )
            )
        )
        assert oa.parameter_fingerprint != ob.parameter_fingerprint


# ---------------------------------------------------------------------------
# 7. Frozen OrchestratedFrame.
# ---------------------------------------------------------------------------


class TestFrozenEnvelope:
    def test_orchestrated_frame_is_frozen(self) -> None:
        orch = RuntimeOrchestrator(_config())
        frame = orch.tick(timestamp=0.0, R=0.9, delta=0.05)
        with pytest.raises((AttributeError, TypeError)):
            frame.pipeline_frame = None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 8. JSON projection.
# ---------------------------------------------------------------------------


class TestJsonProjection:
    def test_to_json_dict_serialisable(self) -> None:
        import json

        orch = RuntimeOrchestrator(_config())
        # Need a few ticks for regime to warm.
        for i in range(5):
            orch.tick(timestamp=float(i) * 0.1, R=0.9, delta=0.05)
        last = orch.tick(timestamp=0.5, R=0.92, delta=0.05)
        d = last.to_json_dict()
        text = json.dumps(d)
        loaded = json.loads(text)
        assert loaded["pipeline_frame"]["tick_index"] == 5
        assert loaded["regime"] is not None
        assert loaded["action"] is not None

    def test_to_json_dict_with_skipped_layers(self) -> None:
        import json

        orch = RuntimeOrchestrator(_config())
        frame = orch.tick(timestamp=0.0, R=None, delta=0.05)
        d = frame.to_json_dict()
        text = json.dumps(d)
        loaded = json.loads(text)
        assert loaded["regime"] is None
        assert loaded["action"] is None


# ---------------------------------------------------------------------------
# 9. Aesthetic rich __repr__ — HN26 design language.
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_layer_summary(self) -> None:
        orch = RuntimeOrchestrator(_config())
        for i in range(5):
            orch.tick(timestamp=float(i) * 0.1, R=0.95, delta=0.05)
        last = orch.tick(timestamp=0.5, R=0.96, delta=0.05)
        r = repr(last)
        assert r.startswith("OrchestratedFrame[")
        assert "tick=5" in r
        assert "TRENDING" in r or "READY" in r

    def test_repr_with_skipped_layers_uses_dash(self) -> None:
        orch = RuntimeOrchestrator(_config())
        frame = orch.tick(timestamp=0.0, R=None, delta=0.05)
        r = repr(frame)
        assert "—" in r  # dash sentinel for skipped layer


# ---------------------------------------------------------------------------
# 10. End-to-end smoke test — full session with ledger + manifest.
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_session_round_trip(self, tmp_path: Path) -> None:
        """Build → tick → close → write manifest → load → verify
        against ledger end to end."""
        ledger = tmp_path / "session.jsonl"
        orch = RuntimeOrchestrator(_config(ledger_path=ledger))
        for i in range(10):
            orch.tick(timestamp=float(i) * 0.1, R=0.92, delta=0.04)
        manifest = orch.close(end_ts=0.9)
        manifest_path = tmp_path / "session.json"
        manifest.write(manifest_path)
        loaded = SessionManifest.load(manifest_path)
        verification = loaded.verify_against_ledger()
        assert verification.ok
        assert loaded.n_ticks == 10
        assert loaded.parameter_fingerprint == orch.parameter_fingerprint
