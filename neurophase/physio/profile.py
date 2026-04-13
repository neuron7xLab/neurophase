"""Personal calibration profile for the physio gate.

A profile captures a user's in-distribution statistics for the
signal-quality features used by :class:`PhysioGate`. With a calibrated
profile, the gate uses user-specific ``threshold_allow`` and
``threshold_abstain`` values derived from their own baselines rather
than the repo's illustrative defaults.

Design invariants:

* One profile corresponds to one user. ``user_id`` is a mandatory
  opaque identifier (e.g. ``alex-2026-04``); no anonymous profiles.
* All thresholds that the gate consumes come from **one** profile.
  Partial overrides at the CLI level are not permitted -- either the
  gate runs in ``default`` mode on the repo's illustrative constants,
  or it runs in ``calibrated`` mode on a profile.
* ``calibrated`` mode without a valid profile is fail-closed: the
  loader raises :class:`ProfileValidationError` and the gate never
  gets constructed.
* The on-disk format is JSON with a mandatory ``schema_version`` so a
  mismatched version is refused at load time, not silently coerced.

The numeric derivation logic (how a profile is computed from baseline
sessions) lives in :mod:`neurophase.physio.calibration`. This module
is the *shape* and the *loader*; it has no numerical policy of its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROFILE_SCHEMA_VERSION: str = "physio-profile-v1"


class ProfileValidationError(ValueError):
    """Raised when a profile file is missing, malformed, or schema-mismatched."""


@dataclass(frozen=True)
class FeatureBand:
    """Empirical lower/upper summary of a single feature distribution."""

    p05: float
    p50: float
    p95: float
    mean: float
    std: float

    def __post_init__(self) -> None:
        if self.p05 > self.p50 or self.p50 > self.p95:
            raise ProfileValidationError(
                f"FeatureBand percentiles must satisfy p05 <= p50 <= p95; "
                f"got p05={self.p05!r} p50={self.p50!r} p95={self.p95!r}"
            )
        if self.std < 0:
            raise ProfileValidationError(f"FeatureBand.std must be >= 0; got {self.std!r}")

    def to_json_dict(self) -> dict[str, float]:
        return {
            "p05": self.p05,
            "p50": self.p50,
            "p95": self.p95,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> FeatureBand:
        required = {"p05", "p50", "p95", "mean", "std"}
        missing = required - d.keys()
        if missing:
            raise ProfileValidationError(f"FeatureBand missing keys: {sorted(missing)}")
        return cls(
            p05=float(d["p05"]),
            p50=float(d["p50"]),
            p95=float(d["p95"]),
            mean=float(d["mean"]),
            std=float(d["std"]),
        )


@dataclass(frozen=True)
class PhysioProfile:
    """Full per-user calibration profile.

    Attributes
    ----------
    schema_version
        Must equal :data:`PROFILE_SCHEMA_VERSION`. Mismatched versions
        are refused at load time.
    user_id
        Opaque, human-readable identifier (e.g. ``alex-2026-04``).
    created_at_utc
        ISO-8601 UTC timestamp the profile was minted.
    n_baseline_sessions
        Number of sessions that fed into the empirical bands.
    window_size
        The rolling-window depth under which these bands were measured.
        A profile is tied to a specific window size; using it at a
        different window is a contract error.
    rmssd_ms
        Empirical band over RMSSD (ms).
    rr_stability
        Empirical band over the 1-CoV stability score in [0, 1].
    continuity_fraction
        Empirical band over the buffer-fill fraction in [0, 1].
    confidence
        Empirical band over the composite confidence score in [0, 1].
        This is the scalar the gate thresholds consume.
    threshold_allow
        User-specific admission threshold (confidence >= -> EXECUTE_ALLOWED).
    threshold_abstain
        User-specific abstain threshold (confidence < -> ABSTAIN).
    notes
        Free-form operator notes (session dates, mood, environment).
    """

    user_id: str
    created_at_utc: str
    n_baseline_sessions: int
    window_size: int
    rmssd_ms: FeatureBand
    rr_stability: FeatureBand
    continuity_fraction: FeatureBand
    confidence: FeatureBand
    threshold_allow: float
    threshold_abstain: float
    schema_version: str = PROFILE_SCHEMA_VERSION
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.user_id:
            raise ProfileValidationError("user_id must be non-empty")
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ProfileValidationError(
                f"schema_version mismatch: got {self.schema_version!r}, "
                f"expected {PROFILE_SCHEMA_VERSION!r}"
            )
        if self.n_baseline_sessions < 1:
            raise ProfileValidationError(
                f"n_baseline_sessions must be >= 1; got {self.n_baseline_sessions}"
            )
        if self.window_size < 1:
            raise ProfileValidationError(f"window_size must be >= 1; got {self.window_size}")
        if not (0.0 < self.threshold_abstain < self.threshold_allow < 1.0):
            raise ProfileValidationError(
                "need 0 < threshold_abstain < threshold_allow < 1; got "
                f"threshold_abstain={self.threshold_abstain!r}, "
                f"threshold_allow={self.threshold_allow!r}"
            )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "created_at_utc": self.created_at_utc,
            "n_baseline_sessions": self.n_baseline_sessions,
            "window_size": self.window_size,
            "rmssd_ms": self.rmssd_ms.to_json_dict(),
            "rr_stability": self.rr_stability.to_json_dict(),
            "continuity_fraction": self.continuity_fraction.to_json_dict(),
            "confidence": self.confidence.to_json_dict(),
            "threshold_allow": self.threshold_allow,
            "threshold_abstain": self.threshold_abstain,
            "notes": list(self.notes),
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> PhysioProfile:
        schema = str(d.get("schema_version", ""))
        if schema != PROFILE_SCHEMA_VERSION:
            raise ProfileValidationError(
                f"schema_version mismatch: got {schema!r}, expected {PROFILE_SCHEMA_VERSION!r}"
            )
        required = {
            "user_id",
            "created_at_utc",
            "n_baseline_sessions",
            "window_size",
            "rmssd_ms",
            "rr_stability",
            "continuity_fraction",
            "confidence",
            "threshold_allow",
            "threshold_abstain",
        }
        missing = required - d.keys()
        if missing:
            raise ProfileValidationError(f"profile missing keys: {sorted(missing)}")
        return cls(
            user_id=str(d["user_id"]),
            created_at_utc=str(d["created_at_utc"]),
            n_baseline_sessions=int(d["n_baseline_sessions"]),
            window_size=int(d["window_size"]),
            rmssd_ms=FeatureBand.from_json_dict(d["rmssd_ms"]),
            rr_stability=FeatureBand.from_json_dict(d["rr_stability"]),
            continuity_fraction=FeatureBand.from_json_dict(d["continuity_fraction"]),
            confidence=FeatureBand.from_json_dict(d["confidence"]),
            threshold_allow=float(d["threshold_allow"]),
            threshold_abstain=float(d["threshold_abstain"]),
            notes=tuple(str(n) for n in d.get("notes", ())),
        )


def load_profile(path: str | Path) -> PhysioProfile:
    """Load and fully validate a :class:`PhysioProfile` from JSON.

    Raises :class:`ProfileValidationError` on missing file, malformed
    JSON, wrong schema, or any post-init invariant failure. Fail-closed.
    """
    p = Path(path)
    if not p.exists():
        raise ProfileValidationError(f"profile not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProfileValidationError(f"{p}: malformed JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileValidationError(f"{p}: top-level JSON must be an object")
    return PhysioProfile.from_json_dict(raw)


def save_profile(profile: PhysioProfile, path: str | Path) -> Path:
    """Write a :class:`PhysioProfile` to JSON. Creates parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(profile.to_json_dict(), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return p


def current_utc_iso() -> str:
    """Return a fresh ISO-8601 UTC timestamp for profile provenance."""
    return datetime.now(UTC).isoformat()


__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "FeatureBand",
    "PhysioProfile",
    "ProfileValidationError",
    "current_utc_iso",
    "load_profile",
    "save_profile",
]
