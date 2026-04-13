"""Tests for the calibration stack (profile + calibrator + gate mode).

Three invariants:

1. Profile schema is strict: missing fields / bad types / wrong
   version / out-of-order thresholds all raise cleanly.
2. Calibrator is conservative: too few sessions, too few healthy
   frames, or no healthy frames anywhere all raise CalibrationError.
   The floor for calibrated thresholds is the repo defaults (so a
   calibrated user is never more permissive than a default user).
3. ``PhysioGate.from_profile`` produces a gate whose ``mode`` is
   ``"calibrated"`` with the profile's user_id attached. A gate in
   ``calibrated`` mode without a user_id cannot be constructed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurophase.physio.calibration import (
    MIN_BASELINE_SESSIONS,
    MIN_HEALTHY_FRAMES_PER_SESSION,
    CalibrationError,
    SessionSource,
    calibrate_profile,
)
from neurophase.physio.features import DEFAULT_WINDOW_SIZE
from neurophase.physio.gate import (
    DEFAULT_THRESHOLD_ABSTAIN,
    DEFAULT_THRESHOLD_ALLOW,
    PhysioGate,
)
from neurophase.physio.profile import (
    PROFILE_SCHEMA_VERSION,
    FeatureBand,
    PhysioProfile,
    ProfileValidationError,
    load_profile,
    save_profile,
)


def _write_stable_csv(path: Path, *, n: int) -> None:
    """Write a stable baseline CSV that produces plenty of healthy frames."""
    rows = ["timestamp_s,rr_ms"]
    t = 0.0
    for i in range(n):
        rr = 820.0 + (8 if i % 2 == 0 else -8)
        t += rr / 1000.0
        rows.append(f"{t:.3f},{rr:.2f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _dummy_band() -> FeatureBand:
    return FeatureBand(p05=0.4, p50=0.6, p95=0.85, mean=0.6, std=0.1)


def _dummy_profile(**overrides: object) -> PhysioProfile:
    base: dict[str, object] = {
        "user_id": "test-user",
        "created_at_utc": "2026-04-13T00:00:00+00:00",
        "n_baseline_sessions": 3,
        "window_size": 32,
        "rmssd_ms": FeatureBand(p05=10.0, p50=30.0, p95=80.0, mean=35.0, std=15.0),
        "rr_stability": _dummy_band(),
        "continuity_fraction": FeatureBand(p05=0.9, p50=1.0, p95=1.0, mean=0.98, std=0.02),
        "confidence": _dummy_band(),
        "threshold_allow": 0.8,
        "threshold_abstain": 0.5,
    }
    base.update(overrides)
    return PhysioProfile(**base)  # type: ignore[arg-type]


# ------------------- Profile schema ------------------------------------


class TestProfileSchema:
    def test_valid_profile_round_trip(self, tmp_path: Path) -> None:
        profile = _dummy_profile()
        path = save_profile(profile, tmp_path / "p.json")
        loaded = load_profile(path)
        assert loaded == profile

    def test_missing_profile_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ProfileValidationError, match="not found"):
            load_profile(tmp_path / "missing.json")

    def test_wrong_schema_version_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema_version": "physio-profile-v99", "user_id": "x"}))
        with pytest.raises(ProfileValidationError, match="schema_version"):
            load_profile(path)

    def test_missing_required_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({"schema_version": PROFILE_SCHEMA_VERSION, "user_id": "x"}))
        with pytest.raises(ProfileValidationError, match="missing keys"):
            load_profile(path)

    def test_threshold_ordering_is_enforced(self) -> None:
        with pytest.raises(ProfileValidationError, match="threshold_abstain"):
            _dummy_profile(threshold_allow=0.5, threshold_abstain=0.5)

    def test_empty_user_id_rejected(self) -> None:
        with pytest.raises(ProfileValidationError, match="user_id"):
            _dummy_profile(user_id="")

    def test_feature_band_percentiles_must_be_ordered(self) -> None:
        with pytest.raises(ProfileValidationError, match="percentiles"):
            FeatureBand(p05=0.6, p50=0.5, p95=0.8, mean=0.5, std=0.1)

    def test_feature_band_requires_nonneg_std(self) -> None:
        with pytest.raises(ProfileValidationError, match="std"):
            FeatureBand(p05=0.1, p50=0.2, p95=0.3, mean=0.2, std=-0.1)


# ------------------- Calibrator conservatism ---------------------------


class TestCalibratorIsConservative:
    def test_too_few_sessions_raises(self, tmp_path: Path) -> None:
        one_csv = tmp_path / "s1.csv"
        _write_stable_csv(one_csv, n=200)
        with pytest.raises(CalibrationError, match="baseline sessions"):
            calibrate_profile(
                [SessionSource(csv_path=one_csv)],
                user_id="alex",
            )

    def test_too_few_healthy_frames_in_session_raises(self, tmp_path: Path) -> None:
        tiny = tmp_path / "tiny.csv"
        _write_stable_csv(tiny, n=32)  # barely warms the window; few healthy frames
        others = []
        for i in range(MIN_BASELINE_SESSIONS - 1):
            p = tmp_path / f"ok_{i}.csv"
            _write_stable_csv(p, n=200)
            others.append(SessionSource(csv_path=p))
        with pytest.raises(CalibrationError, match="healthy frames"):
            calibrate_profile(
                [SessionSource(csv_path=tiny), *others],
                user_id="alex",
            )

    def test_successful_calibration_floors_at_defaults(self, tmp_path: Path) -> None:
        """Even on beautifully stable baselines, the calibrated thresholds
        must never fall below the repo defaults. Calibration only tightens."""
        sources = []
        for i in range(MIN_BASELINE_SESSIONS):
            p = tmp_path / f"baseline_{i}.csv"
            _write_stable_csv(p, n=200)
            sources.append(SessionSource(csv_path=p))
        profile = calibrate_profile(sources, user_id="alex-2026-04")
        assert profile.user_id == "alex-2026-04"
        assert profile.schema_version == PROFILE_SCHEMA_VERSION
        assert profile.window_size == DEFAULT_WINDOW_SIZE
        assert profile.threshold_abstain >= DEFAULT_THRESHOLD_ABSTAIN
        assert profile.threshold_allow >= DEFAULT_THRESHOLD_ALLOW
        assert profile.threshold_abstain < profile.threshold_allow < 1.0
        assert profile.n_baseline_sessions == MIN_BASELINE_SESSIONS
        # Plausibility: confidence band should be well inside [0, 1].
        assert profile.confidence.p05 >= 0.0
        assert profile.confidence.p95 <= 1.0

    def test_min_healthy_frames_per_session_constant_is_enforced(self, tmp_path: Path) -> None:
        # Sanity check on the constant so it is never silently relaxed.
        assert MIN_HEALTHY_FRAMES_PER_SESSION >= 32


# ------------------- Gate modes ----------------------------------------


class TestGateModes:
    def test_default_mode_has_no_profile_user_id(self) -> None:
        g = PhysioGate()
        assert g.mode == "default"
        assert g.profile_user_id is None

    def test_calibrated_mode_requires_profile_user_id(self) -> None:
        with pytest.raises(ValueError, match="profile_user_id"):
            PhysioGate(
                threshold_allow=0.8,
                threshold_abstain=0.5,
                mode="calibrated",
                profile_user_id=None,
            )

    def test_bad_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            PhysioGate(mode="aggressive")  # type: ignore[arg-type]

    def test_from_profile_builds_calibrated_gate(self) -> None:
        profile = _dummy_profile(
            user_id="alex-2026-04",
            threshold_allow=0.85,
            threshold_abstain=0.6,
        )
        gate = PhysioGate.from_profile(profile)
        assert gate.mode == "calibrated"
        assert gate.profile_user_id == "alex-2026-04"
        assert gate.threshold_allow == pytest.approx(0.85)
        assert gate.threshold_abstain == pytest.approx(0.6)

    def test_from_profile_rejects_non_profile_argument(self) -> None:
        with pytest.raises(TypeError, match="PhysioProfile"):
            PhysioGate.from_profile({"not": "a profile"})  # type: ignore[arg-type]
