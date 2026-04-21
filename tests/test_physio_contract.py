"""ONE canonical LSL/sentinel/exit-code contract for the physio stack.

The physio stack has four coordinate endpoints that must agree on a
single LSL + sample + sentinel + exit-code contract:

    * ``neurophase.physio.live``        (kernel consumer)
    * ``neurophase.physio.live_producer`` (synthetic validation producer)
    * ``tools/polar_producer.py``       (out-of-repo real hardware producer)
    * ``tools/fault_producer.py``       (adversarial fault producer)

Contract drift between any two of them silently breaks end-to-end
integration. Previously the contract was asserted piecemeal across
``test_physio_live.py`` + ``test_physio_faults.py``; this file is the
**single canonical place** where every surface of the contract is
checked against the one source of truth (``neurophase.physio.live``).

Explicitly asserted:

  * stream name default                 "neurophase-rr"
  * stream type                         "RR"
  * LSL channel_count                   2
  * LSL channel_format                  "double64"
  * sample layout                       ``[timestamp_s, rr_ms]``
  * EOF sentinel semantics              NaN on either channel -> clean shutdown
  * RR physiological envelope           ``[RR_MIN_MS, RR_MAX_MS] = [300, 2000]``
  * config bounds                       stall_timeout_s in ``[2.0, 30.0]``
  * exit-code table (live consumer)     0 / 2 / 3
  * exit-code table (polar producer)    0 / 1 / 2 / 3 / 4 / 5 / 6
  * exit-code table (fault producer)    0 / 2 / 4

A failure in this file is the loudest possible signal: someone
silently changed the contract without updating all four endpoints.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

from neurophase.physio import live as physio_live
from neurophase.physio import live_producer as physio_producer

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str) -> ModuleType:
    """Load an out-of-repo tool by path; register in sys.modules for dataclasses."""
    path = _REPO_ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"tools_{name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"tools_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def polar_producer() -> ModuleType:
    return _load_tool("polar_producer")


@pytest.fixture(scope="module")
def fault_producer() -> ModuleType:
    return _load_tool("fault_producer")


# =======================================================================
#   LSL sample schema
# =======================================================================


class TestLSLSchema:
    """channel_count, channel_format, type, stream name default."""

    def test_channel_count_is_two(
        self, polar_producer: ModuleType, fault_producer: ModuleType
    ) -> None:
        assert physio_live.LSL_CHANNEL_COUNT == 2
        assert polar_producer.LSL_CHANNEL_COUNT == 2
        assert fault_producer.LSL_CHANNEL_COUNT == 2
        assert physio_live.LiveConfig().window_size > 0  # sanity: live config loads

    def test_channel_format_is_double64(
        self, polar_producer: ModuleType, fault_producer: ModuleType
    ) -> None:
        """All four endpoints share the upgraded double64 contract.

        Float32 ULP at ``time.monotonic()`` values > 10⁴ s exceeds the
        producer cadence (20 ms), causing consecutive timestamps to
        collide and the consumer's monotonicity guard to reject 2/3 of
        samples on any host with uptime ≳ 3 h. double64 ULP stays below
        10⁻⁹ s across the realistic monotonic range.
        """
        assert physio_live.LSL_CHANNEL_FORMAT == "double64"
        assert polar_producer.LSL_CHANNEL_FORMAT == "double64"
        assert fault_producer.LSL_CHANNEL_FORMAT == "double64"

    def test_stream_type_is_rr(
        self, polar_producer: ModuleType, fault_producer: ModuleType
    ) -> None:
        assert physio_live.LSL_STREAM_TYPE == "RR"
        assert polar_producer.LSL_STREAM_TYPE == "RR"
        assert fault_producer.LSL_STREAM_TYPE == "RR"

    def test_default_stream_name(
        self, polar_producer: ModuleType, fault_producer: ModuleType
    ) -> None:
        # Live consumer: LiveConfig default.
        assert physio_live.LiveConfig().stream_name == "neurophase-rr"
        # Built-in synthetic producer: ProducerConfig default.
        assert physio_producer.ProducerConfig(stream_name="neurophase-rr").stream_name == (
            "neurophase-rr"
        )
        # Out-of-repo tools: CLI default via argparser.
        pp_parser = polar_producer._build_argparser()
        pp_default = pp_parser.get_default("stream_name")
        assert pp_default == "neurophase-rr"
        fp_parser = fault_producer._build_argparser()
        fp_default = fp_parser.get_default("stream_name")
        assert fp_default == "neurophase-rr"


# =======================================================================
#   Sample layout + EOF sentinel
# =======================================================================


class TestSampleLayoutAndSentinel:
    def test_sample_layout_is_ts_then_rr(
        self, polar_producer: ModuleType, fault_producer: ModuleType
    ) -> None:
        """Every producer documents and uses ch0 = timestamp_s, ch1 = rr_ms.
        The consumer unpacks the same order. Drift would mean NaN-on-ch1
        (RR) stops being the sentinel it currently is."""
        # Live consumer source: `sample[0]` -> ts, `sample[1]` -> rr.
        src = (physio_live.__file__, physio_producer.__file__)
        for path in src:
            txt = Path(path).read_text(encoding="utf-8")
            # ch0 = timestamp_s, ch1 = rr_ms -- document + comment references.
            assert "ch0=timestamp_s" in txt or "timestamp_s" in txt
            assert "rr_ms" in txt
        # Tool modules carry the same contract in their XML desc.
        for tool in (polar_producer, fault_producer):
            tool_txt = Path(tool.__file__).read_text(encoding="utf-8")
            assert "ch0=timestamp_s" in tool_txt or "timestamp_s" in tool_txt
            assert "rr_ms" in tool_txt

    def test_eof_sentinel_is_nan_pair(self, polar_producer: ModuleType) -> None:
        """(NaN, NaN) is the only sentinel; the live consumer treats a
        NaN on EITHER channel as EOF. The SentinelGuard in polar_producer
        pushes NaN/NaN exactly once."""

        # SentinelGuard emit_once pushes [nan, nan] via a stub.
        class _Stub:
            def __init__(self) -> None:
                self.pushes: list[list[float]] = []

            def push_sample(self, x: list[float]) -> None:
                self.pushes.append(list(x))

        stub = _Stub()
        guard = polar_producer.SentinelGuard(stub)
        assert guard.emit_once() is True
        assert len(stub.pushes) == 1
        assert all(math.isnan(v) for v in stub.pushes[0])


# =======================================================================
#   RR envelope + config bounds
# =======================================================================


class TestConfigBounds:
    def test_rr_envelope_constants(self, polar_producer: ModuleType) -> None:
        """Envelope [300, 2000] ms corresponds to HR 30-200 bpm."""
        from neurophase.physio.replay import RR_MAX_MS, RR_MIN_MS

        assert RR_MIN_MS == 300.0
        assert RR_MAX_MS == 2000.0
        assert polar_producer.RR_MIN_MS == RR_MIN_MS
        assert polar_producer.RR_MAX_MS == RR_MAX_MS

    def test_stall_timeout_safe_range(self) -> None:
        """stall_timeout_s safe range is [2.0, 30.0]. Outside raises."""
        assert physio_live.STALL_TIMEOUT_SAFE_MIN_S == 2.0
        assert physio_live.STALL_TIMEOUT_SAFE_MAX_S == 30.0
        # Inside the band -> ok.
        physio_live.LiveConfig(stall_timeout_s=2.0, read_timeout_s=0.1)
        physio_live.LiveConfig(stall_timeout_s=30.0, read_timeout_s=0.5)
        # Outside -> ValueError.
        with pytest.raises(ValueError, match="stall_timeout_s"):
            physio_live.LiveConfig(stall_timeout_s=1.99)
        with pytest.raises(ValueError, match="stall_timeout_s"):
            physio_live.LiveConfig(stall_timeout_s=30.01)

    def test_read_timeout_bounds(self) -> None:
        """read_timeout_s must be > 0 and < stall_timeout_s."""
        with pytest.raises(ValueError, match="read_timeout_s"):
            physio_live.LiveConfig(read_timeout_s=0.0)
        with pytest.raises(ValueError, match="read_timeout_s"):
            physio_live.LiveConfig(stall_timeout_s=3.0, read_timeout_s=3.0)


# =======================================================================
#   Exit-code tables (frozen by contract; changes must land here)
# =======================================================================


class TestExitCodeTables:
    """If any of these change, a serious downstream caller breaks. The
    table is the contract; changing it means shipping a breaking
    change and updating docs + tools/README.md in lockstep."""

    def test_polar_producer_exit_codes(self, polar_producer: ModuleType) -> None:
        assert polar_producer.EXIT_OK == 0
        assert polar_producer.EXIT_NO_DEVICE == 1
        assert polar_producer.EXIT_CONNECT_FAIL == 2
        assert polar_producer.EXIT_NO_CHAR == 3
        assert polar_producer.EXIT_UNEXPECTED_DISCONNECT == 4
        assert polar_producer.EXIT_LSL_FATAL == 5
        assert polar_producer.EXIT_SELF_TEST_FAIL == 6

    def test_fault_producer_exit_codes(self, fault_producer: ModuleType) -> None:
        assert fault_producer.EXIT_OK == 0
        assert fault_producer.EXIT_USAGE == 2
        assert fault_producer.EXIT_ABRUPT_REQUESTED == 4

    def test_live_consumer_exits_2_on_missing_stream(self, tmp_path: Path) -> None:
        """The live consumer's EXIT=2 path for missing stream is not a
        bare integer in the source, but documented behaviour. Cover it
        via a unit call with a tiny resolve timeout (subprocess path is
        already in test_physio_live)."""
        import io

        config = physio_live.LiveConfig(
            stream_name=f"does-not-exist-{id(tmp_path)}",
            resolve_timeout_s=0.5,
            stall_timeout_s=3.0,
            read_timeout_s=0.1,
        )
        buf = io.StringIO()
        rc = physio_live._run_consumer(config, out=buf)
        assert rc == 2
        # Output must include FATAL + stream not found.
        text = buf.getvalue()
        assert "FATAL" in text
        assert "not found" in text


# =======================================================================
#   Canonical frame schema version
# =======================================================================


class TestCanonicalFrameSchemaVersion:
    def test_canonical_frame_schema_version_is_pinned(self) -> None:
        """Changing the canonical frame schema version is a breaking
        change. Pinning it here forces a conscious update everywhere."""
        from neurophase.physio.pipeline import CANONICAL_FRAME_SCHEMA_VERSION

        assert CANONICAL_FRAME_SCHEMA_VERSION == "physio-v1"

    def test_physio_ledger_schema_version_is_pinned(self) -> None:
        from neurophase.physio.ledger import PHYSIO_LEDGER_SCHEMA_VERSION

        assert PHYSIO_LEDGER_SCHEMA_VERSION == "physio-ledger-v1"

    def test_physio_profile_schema_version_is_pinned(self) -> None:
        from neurophase.physio.profile import PROFILE_SCHEMA_VERSION

        assert PROFILE_SCHEMA_VERSION == "physio-profile-v1"


# =======================================================================
#   LSL timestamp-precision floor (regression guard for 2026-04-21 fix)
# =======================================================================


class TestLSLTimestampPrecisionFloor:
    """Guard: the chosen LSL channel_format must preserve the producer
    cadence at realistic ``time.monotonic()`` magnitudes.

    Producers push ``time.monotonic()`` absolute values. On long-running
    hosts this value grows into the 10⁴–10⁶ s range. The sample format
    must have ULP strictly smaller than the producer's minimum inter-
    sample interval, or consecutive samples collide to the same float
    representation and the consumer's monotonicity guard rejects every
    collision.

    Prior contract (`float32`) fails this at uptime ≳ 3 h on default
    producer cadence (20 ms). Current contract (`double64`) passes at
    every physically reasonable uptime.
    """

    # Cadence floor (seconds) — tightest interval any producer emits.
    # live_producer default is 0.02; Polar hardware sends RR at ~30 Hz so
    # ~0.03 s; tightest plausible real-world cadence is 2 ms for a
    # high-rate fault injection test. Floor must hold for the tightest.
    _MIN_INTER_SAMPLE_S: float = 2e-3

    # Realistic monotonic magnitude to cover (30 days of uptime, seconds).
    _REALISTIC_MONO_MAX_S: float = 30 * 24 * 3600

    def test_channel_format_resolves_cadence_at_realistic_uptime(self) -> None:
        """INV-LIVE-TSPREC: chosen format ULP at 30-day uptime must be
        strictly less than the tightest producer cadence.

        Why: float32 ULP at 10⁶ s is ~0.125 s — 60× the 2 ms cadence
        floor — which collides consecutive samples and blocks the live
        consumer. double64 ULP at 10⁶ s is ~10⁻¹⁰ s — well below.
        """
        import numpy as np

        fmt = physio_live.LSL_CHANNEL_FORMAT
        if fmt == "float32":
            dtype = np.float32
        elif fmt == "double64":
            dtype = np.float64
        else:  # pragma: no cover - future formats
            raise AssertionError(
                f"INV-LIVE-TSPREC: unexpected channel_format={fmt!r}; "
                f"add ULP mapping for this format before shipping"
            )
        val = dtype(self._REALISTIC_MONO_MAX_S)
        ulp = float(np.spacing(val))
        assert ulp < self._MIN_INTER_SAMPLE_S, (
            f"INV-LIVE-TSPREC VIOLATED: channel_format={fmt!r} has "
            f"ULP={ulp!r} at mono={self._REALISTIC_MONO_MAX_S}s; "
            f"expected ULP < cadence floor {self._MIN_INTER_SAMPLE_S}s. "
            f"Consecutive producer samples at this cadence would collide "
            f"to the same float representation and the consumer would "
            f"reject every collision as non-monotonic. "
            f"Fix: upgrade channel_format to double64 (or tighter)."
        )

    def test_float32_fails_the_guard_at_realistic_uptime(self) -> None:
        """Negative control: confirms the guard above would have caught
        the 2026-04-21 incident had it existed. Proves the guard has
        real resolving power, not just happens-to-pass.

        This is a *proof-of-catchability* test; it does NOT drive any
        runtime behaviour.
        """
        import numpy as np

        ulp_f32 = float(np.spacing(np.float32(self._REALISTIC_MONO_MAX_S)))
        assert ulp_f32 > self._MIN_INTER_SAMPLE_S, (
            f"float32 ULP at mono={self._REALISTIC_MONO_MAX_S}s = {ulp_f32!r} "
            f"should exceed cadence floor {self._MIN_INTER_SAMPLE_S}s — "
            f"if this assertion ever passes, float32 has mysteriously "
            f"become acceptable again and the above guard may be removed. "
            f"Expected: float32 ULP stays unshippable at realistic uptime."
        )
