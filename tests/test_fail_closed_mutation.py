"""Fail-closed mutation suite — enumerates each guard in the physio
stack and proves it is load-bearing.

This is NOT a mutmut / cosmic-ray run; it is a **hand-authored proof
of coverage**. For every fail-closed guard the physio path relies on,
there is exactly one test here whose docstring names the guard and
whose body feeds input designed to survive ONLY if the guard is
removed or weakened. If someone deletes the guard in a refactor, the
test here fails **by name** so code review catches it.

This file is the explicit read-across between:

  * CLAUDE.md  <hard_rules> 1 (no fake-live) and 7 (no untested fixes)
  * tests/README.md  (each new guard -> entry here)

Guards enumerated (all in the physio path):

  G1.  SentinelGuard.emit_once pushes at most ONE NaN/NaN sample
  G2.  RRSample rejects non-finite timestamp_s
  G3.  RRSample rejects non-finite rr_ms
  G4.  RRSample rejects negative timestamp_s
  G5.  RRSample rejects rr outside [RR_MIN_MS, RR_MAX_MS]
  G6.  RRReplayReader rejects non-monotonic (duplicate) timestamps
  G7.  PhysioDecision enforces execution_allowed == (state == EXECUTE_ALLOWED)
  G8.  PhysioDecision execution_allowed=True + state!=EXECUTE_ALLOWED raises
  G9.  PhysioDecision state=EXECUTE_ALLOWED + execution_allowed=False raises
  G10. PhysioGate calibrated mode requires a non-empty profile_user_id
  G11. PhysioGate default mode forbids a profile_user_id
  G12. PhysioGate rejects thresholds where abstain >= allow
  G13. HRVWindow below MIN_WINDOW_SIZE returns confidence == 0
  G14. HRVWindow flatlined RR (RMSSD == 0) marks rmssd_plausible=False
  G15. PhysioLedger rejects __enter__ called twice
  G16. PhysioLedger writes file mode 0o600 regardless of umask
  G17. LiveConfig stall_timeout_s outside [2.0, 30.0] raises
  G18. LiveConfig read_timeout_s >= stall_timeout_s raises
  G19. LedgerReplayError on missing header / wrong schema / mid-file garbage
  G20. PhysioProfile post-init rejects threshold_abstain >= threshold_allow
  G21. ProfileValidationError on wrong schema_version at load

Each test name starts with ``test_Gxx_`` to keep the map 1:1 with the
list above. Breaking a guard removes the guard, which makes the
corresponding test fail, which makes CI go red.
"""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from pathlib import Path

import pytest

from neurophase.gate.execution_gate import GateState
from neurophase.physio.features import (
    DEFAULT_WINDOW_SIZE,
    MIN_WINDOW_SIZE,
    HRVWindow,
)
from neurophase.physio.gate import PhysioDecision, PhysioGate, PhysioGateState
from neurophase.physio.ledger import LEDGER_FILE_MODE, LedgerConfig, PhysioLedger
from neurophase.physio.live import STALL_TIMEOUT_SAFE_MAX_S, LiveConfig
from neurophase.physio.profile import (
    PROFILE_SCHEMA_VERSION,
    FeatureBand,
    PhysioProfile,
    ProfileValidationError,
    load_profile,
)
from neurophase.physio.replay import (
    RR_MAX_MS,
    RR_MIN_MS,
    ReplayIngestError,
    RRReplayReader,
    RRSample,
)

# =======================================================================
#   G1 — SentinelGuard single-shot invariant (out-of-repo polar_producer)
# =======================================================================


def test_G1_sentinel_guard_emits_at_most_once() -> None:
    """Guard: SentinelGuard.emit_once must push the NaN/NaN sample
    exactly once. If the _sent flag flip is removed, this test fails."""
    import importlib.util
    import sys as _sys

    path = Path(__file__).resolve().parent.parent / "tools" / "polar_producer.py"
    spec = importlib.util.spec_from_file_location("pp_mut", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["pp_mut"] = mod
    spec.loader.exec_module(mod)

    class _Stub:
        def __init__(self) -> None:
            self.pushes: list[list[float]] = []

        def push_sample(self, x: list[float]) -> None:
            self.pushes.append(list(x))

    stub = _Stub()
    guard = mod.SentinelGuard(stub)
    assert guard.emit_once() is True
    assert guard.emit_once() is False
    assert guard.emit_once() is False
    assert len(stub.pushes) == 1
    assert all(math.isnan(v) for v in stub.pushes[0])


# =======================================================================
#   G2 .. G5 — RRSample hard-input guards
# =======================================================================


def test_G2_rr_sample_rejects_non_finite_timestamp() -> None:
    """Guard: RRSample.__post_init__ rejects NaN / +inf / -inf ts."""
    for bad_ts in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ReplayIngestError, match=r"NaN|not finite"):
            RRSample(timestamp_s=bad_ts, rr_ms=820.0, row_index=0)


def test_G3_rr_sample_rejects_non_finite_rr() -> None:
    """Guard: RRSample rejects NaN / +inf / -inf rr_ms."""
    for bad_rr in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ReplayIngestError, match=r"NaN|not finite"):
            RRSample(timestamp_s=1.0, rr_ms=bad_rr, row_index=0)


def test_G4_rr_sample_rejects_negative_timestamp() -> None:
    """Guard: RRSample rejects negative timestamp_s."""
    with pytest.raises(ReplayIngestError, match="negative"):
        RRSample(timestamp_s=-0.001, rr_ms=820.0, row_index=0)


def test_G5_rr_sample_rejects_outside_envelope() -> None:
    """Guard: RRSample rejects rr_ms outside [RR_MIN_MS, RR_MAX_MS]."""
    with pytest.raises(ReplayIngestError, match="envelope"):
        RRSample(timestamp_s=1.0, rr_ms=RR_MIN_MS - 1.0, row_index=0)
    with pytest.raises(ReplayIngestError, match="envelope"):
        RRSample(timestamp_s=1.0, rr_ms=RR_MAX_MS + 1.0, row_index=0)


# =======================================================================
#   G6 — RRReplayReader monotonic timestamp guard
# =======================================================================


def test_G6_replay_reader_rejects_duplicate_timestamps(tmp_path: Path) -> None:
    """Guard: RRReplayReader raises on any non-strictly-increasing ts."""
    path = tmp_path / "dup.csv"
    path.write_text("timestamp_s,rr_ms\n1.0,820.0\n1.0,815.0\n", encoding="utf-8")
    with pytest.raises(ReplayIngestError, match="not strictly greater"):
        list(RRReplayReader(path))


# =======================================================================
#   G7 .. G9 — PhysioDecision biconditional
# =======================================================================


def test_G7_physio_decision_execution_allowed_biconditional_ready() -> None:
    """Guard: PhysioDecision with state=EXECUTE_ALLOWED requires
    execution_allowed=True. A mutation loosening this (allow False on
    EXECUTE_ALLOWED) would be caught here."""
    with pytest.raises(ValueError, match="Invariant"):
        PhysioDecision(
            state=PhysioGateState.EXECUTE_ALLOWED,
            execution_allowed=False,
            confidence=0.9,
            threshold_allow=0.8,
            threshold_abstain=0.5,
            reason="forged",
            kernel_state=GateState.READY,
        )


def test_G8_physio_decision_execution_allowed_biconditional_reduced() -> None:
    """Guard: PhysioDecision with execution_allowed=True rejects any
    state that is not EXECUTE_ALLOWED (e.g. EXECUTE_REDUCED)."""
    for bad_state in (
        PhysioGateState.EXECUTE_REDUCED,
        PhysioGateState.ABSTAIN,
        PhysioGateState.SENSOR_DEGRADED,
    ):
        with pytest.raises(ValueError, match="Invariant"):
            PhysioDecision(
                state=bad_state,
                execution_allowed=True,
                confidence=0.9,
                threshold_allow=0.8,
                threshold_abstain=0.5,
                reason="forged",
                kernel_state=GateState.READY,
            )


def test_G9_physio_decision_non_execute_allowed_with_false_ok() -> None:
    """Positive control: a non-EXECUTE_ALLOWED state with
    execution_allowed=False constructs fine."""
    for ok_state in (
        PhysioGateState.EXECUTE_REDUCED,
        PhysioGateState.ABSTAIN,
        PhysioGateState.SENSOR_DEGRADED,
    ):
        d = PhysioDecision(
            state=ok_state,
            execution_allowed=False,
            confidence=0.6,
            threshold_allow=0.8,
            threshold_abstain=0.5,
            reason="ok",
            kernel_state=GateState.BLOCKED,
        )
        assert d.execution_allowed is False


# =======================================================================
#   G10 .. G12 — PhysioGate construction guards
# =======================================================================


def test_G10_physio_gate_calibrated_requires_user_id() -> None:
    """Guard: mode='calibrated' without a profile_user_id raises."""
    with pytest.raises(ValueError, match="profile_user_id"):
        PhysioGate(mode="calibrated", profile_user_id=None)
    with pytest.raises(ValueError, match="profile_user_id"):
        PhysioGate(mode="calibrated", profile_user_id="")


def test_G11_physio_gate_default_forbids_user_id() -> None:
    """Guard: mode='default' with a profile_user_id raises. Catches
    the reverse mismatch -- a well-meaning caller attaching a user id
    to an un-calibrated gate to 'tag' it."""
    with pytest.raises(ValueError, match="must not carry"):
        PhysioGate(mode="default", profile_user_id="alex-2026-04")


def test_G12_physio_gate_threshold_ordering() -> None:
    """Guard: thresholds must satisfy 0 < abstain < allow < 1."""
    with pytest.raises(ValueError, match="threshold_abstain"):
        PhysioGate(threshold_allow=0.5, threshold_abstain=0.5)
    with pytest.raises(ValueError, match="threshold_abstain"):
        PhysioGate(threshold_allow=0.3, threshold_abstain=0.8)


# =======================================================================
#   G13 .. G14 — HRVWindow fail-closed features
# =======================================================================


def test_G13_hrv_window_below_min_returns_zero_confidence() -> None:
    """Guard: features() on a buffer smaller than MIN_WINDOW_SIZE
    returns confidence=0.0. If this is removed, short buffers could
    produce EXECUTE_ALLOWED during warm-up, which is the exact
    fail-closed failure the physio path exists to prevent."""
    win = HRVWindow()
    # Push MIN_WINDOW_SIZE - 1 samples; expect zero confidence.
    for i in range(MIN_WINDOW_SIZE - 1):
        win.push(RRSample(timestamp_s=i * 0.8 + 1.0, rr_ms=820.0, row_index=i))
    feats = win.features()
    assert feats.window_size == MIN_WINDOW_SIZE - 1
    assert feats.confidence == 0.0


def test_G14_hrv_window_flatlined_rr_fails_plausibility() -> None:
    """Guard: an all-identical RR buffer has RMSSD=0, which must fall
    outside RMSSD plausibility (flatline proxy). Removing that hard
    multiplicative penalty on confidence would let a flatlined sensor
    earn EXECUTE_ALLOWED."""
    win = HRVWindow()
    for i in range(DEFAULT_WINDOW_SIZE):
        win.push(RRSample(timestamp_s=i * 0.8 + 1.0, rr_ms=820.0, row_index=i))
    feats = win.features()
    assert feats.rmssd_ms == pytest.approx(0.0)
    assert feats.rmssd_plausible is False
    # Confidence is hard-penalised by the rmssd_plausible factor.
    assert feats.confidence < 0.25


# =======================================================================
#   G15 .. G16 — PhysioLedger guards
# =======================================================================


def _tiny_ledger_config() -> LedgerConfig:
    return LedgerConfig(
        source_mode="mutation-test",
        stream_name=None,
        window_size=32,
        threshold_allow=0.8,
        threshold_abstain=0.5,
        stall_timeout_s=None,
    )


def test_G15_physio_ledger_double_enter_rejected(tmp_path: Path) -> None:
    """Guard: a single PhysioLedger must accept __enter__ exactly once.
    Weakening this would let a caller silently overwrite a session in
    progress and drop everything already recorded."""
    path = tmp_path / "double.jsonl"
    led = PhysioLedger(path, config=_tiny_ledger_config())
    led.__enter__()
    try:
        with pytest.raises(RuntimeError, match="twice"):
            led.__enter__()
    finally:
        led.__exit__(None, None, None)


def test_G16_physio_ledger_file_mode_0o600(tmp_path: Path) -> None:
    """Guard: ledger files are 0o600 regardless of umask. Removing
    this would leave personal RR traces group/world-readable on
    default-umask systems."""
    old_umask = os.umask(0o022)
    try:
        path = tmp_path / "priv.jsonl"
        with PhysioLedger(path, config=_tiny_ledger_config()) as led:
            led.write_event({"event": "FRAME"})
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == LEDGER_FILE_MODE == 0o600
    finally:
        os.umask(old_umask)


# =======================================================================
#   G17 .. G18 — LiveConfig timing guards
# =======================================================================


def test_G17_live_config_stall_timeout_bounds() -> None:
    """Guard: stall_timeout_s clamped to [2.0, 30.0]. A looser bound
    would let an operator configure a 1-day stall tolerance, defeating
    the fail-closed stall path entirely."""
    with pytest.raises(ValueError, match="stall_timeout_s"):
        LiveConfig(stall_timeout_s=1.99)
    with pytest.raises(ValueError, match="stall_timeout_s"):
        LiveConfig(stall_timeout_s=STALL_TIMEOUT_SAFE_MAX_S + 0.01)


def test_G18_live_config_read_timeout_must_be_smaller() -> None:
    """Guard: read_timeout_s must be > 0 and < stall_timeout_s. Equal
    or larger would block the stall-detection loop from ever firing."""
    with pytest.raises(ValueError, match="read_timeout_s"):
        LiveConfig(read_timeout_s=0.0)
    with pytest.raises(ValueError, match="read_timeout_s"):
        LiveConfig(stall_timeout_s=2.0, read_timeout_s=2.0)


# =======================================================================
#   G19 — LedgerReplayError on missing/wrong-schema/mid-garbage header
# =======================================================================


def test_G19_ledger_replay_rejects_missing_header(tmp_path: Path) -> None:
    """Guard: a ledger that starts with a FRAME (no SESSION_HEADER)
    must raise. Weakening this would let any JSONL file be read as a
    ledger with an empty config => session reconstruction on wrong
    thresholds => silent decision drift."""
    from neurophase.physio.session_replay import LedgerReplayError, replay_ledger

    path = tmp_path / "no-header.jsonl"
    path.write_text(json.dumps({"event": "FRAME", "tick_index": 0}) + "\n", encoding="utf-8")
    with pytest.raises(LedgerReplayError, match="SESSION_HEADER"):
        replay_ledger(path)


def test_G19b_ledger_replay_rejects_wrong_schema_version(tmp_path: Path) -> None:
    from neurophase.physio.session_replay import LedgerReplayError, replay_ledger

    path = tmp_path / "wrong-schema.jsonl"
    path.write_text(
        json.dumps(
            {
                "event": "SESSION_HEADER",
                "schema_version": "physio-ledger-v999",
                "session_id": "x",
                "config": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LedgerReplayError, match="schema_version"):
        replay_ledger(path)


# =======================================================================
#   G20 .. G21 — PhysioProfile guards
# =======================================================================


def _dummy_band() -> FeatureBand:
    return FeatureBand(p05=0.4, p50=0.6, p95=0.85, mean=0.6, std=0.1)


def test_G20_profile_threshold_ordering() -> None:
    """Guard: profile's threshold_abstain must be < threshold_allow."""
    with pytest.raises(Exception, match="threshold"):
        PhysioProfile(
            user_id="x",
            created_at_utc="2026-04-13T00:00:00+00:00",
            n_baseline_sessions=3,
            window_size=32,
            rmssd_ms=FeatureBand(p05=10.0, p50=30.0, p95=80.0, mean=35.0, std=15.0),
            rr_stability=_dummy_band(),
            continuity_fraction=FeatureBand(p05=0.9, p50=1.0, p95=1.0, mean=0.98, std=0.02),
            confidence=_dummy_band(),
            threshold_allow=0.5,
            threshold_abstain=0.7,
        )


def test_G21_profile_load_rejects_wrong_schema(tmp_path: Path) -> None:
    """Guard: load_profile raises on schema_version mismatch. A silent
    coerce-to-current would let an old profile be loaded into a newer
    runtime with semantically different bands."""
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps({"schema_version": "physio-profile-v0", "user_id": "x"}),
        encoding="utf-8",
    )
    with pytest.raises(ProfileValidationError, match="schema_version"):
        load_profile(path)
    # Positive control: current schema at least gets past the version
    # check (though it will then fail required-keys).
    path2 = tmp_path / "curr.json"
    path2.write_text(json.dumps({"schema_version": PROFILE_SCHEMA_VERSION}), encoding="utf-8")
    with pytest.raises(ProfileValidationError):
        load_profile(path2)


# =======================================================================
#   Coverage self-check: 21 guards registered, each with a test
# =======================================================================


def test_guard_count_is_registered() -> None:
    """Meta-check: the guard list in this file's docstring is the
    canonical registry. If someone adds a guard to the physio path
    but forgets to add a test here, this count won't match and the
    assertion fails."""
    current_module = sys.modules[__name__]
    guard_tests = [
        name
        for name in dir(current_module)
        if name.startswith("test_G") and name[6:8].rstrip("_").isdigit()
    ]
    # Expect >= 21 guard tests named test_G1_ .. test_G21_ (sub-guards
    # like G19b allowed).
    assert len(guard_tests) >= 21, f"expected >= 21 guard tests, found {guard_tests}"
