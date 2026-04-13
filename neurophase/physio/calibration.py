"""Numerical derivation of a :class:`PhysioProfile` from baseline sessions.

A baseline session is any source of RR samples for which the user was
in a **nominal** state: morning resting, post-focus low-stim, post-load
recovery, etc. The calibrator:

1. consumes N baseline sessions (CSV replays or JSONL ledgers),
2. runs each one through a fresh :class:`PhysioSession` with the same
   window configuration,
3. collects per-frame ``HRVFeatures`` samples *only* from frames that
   would have counted as healthy under the default gate (kernel_state
   == READY; the gate already rejects flatline / low-fill windows),
4. summarises each feature distribution with p05 / p50 / p95 / mean /
   std (:class:`FeatureBand`),
5. derives user-specific admission thresholds from the confidence
   distribution.

Threshold derivation rule (explicit, no hidden policy):

* ``threshold_abstain = max(default_abstain, p05(confidence))``
* ``threshold_allow   = max(default_allow,   p50(confidence))``
* clamped so ``threshold_abstain < threshold_allow < 1``.

Intuition: the abstain floor is the user's 5th percentile on healthy
frames (so we never abstain more often than 5% of healthy frames
would), and the allow ceiling is the user's median (so at least half
of their healthy-frame distribution clears it). Both floors are the
repo defaults so a calibrated user never ends up **more permissive**
than a default user.

The calibrator is intentionally conservative. If the baseline sessions
are too few, too noisy, or dominated by degraded frames, calibration
raises :class:`CalibrationError` rather than emitting a profile that
is effectively a coin flip.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neurophase.physio.features import DEFAULT_WINDOW_SIZE, HRVFeatures
from neurophase.physio.gate import (
    DEFAULT_THRESHOLD_ABSTAIN,
    DEFAULT_THRESHOLD_ALLOW,
)
from neurophase.physio.pipeline import PhysioSession
from neurophase.physio.profile import (
    FeatureBand,
    PhysioProfile,
    current_utc_iso,
)
from neurophase.physio.replay import RRReplayReader, RRSample

MIN_HEALTHY_FRAMES_PER_SESSION: int = 32
MIN_TOTAL_HEALTHY_FRAMES: int = 128
MIN_BASELINE_SESSIONS: int = 3


class CalibrationError(ValueError):
    """Raised when calibration cannot produce a defensible profile."""


@dataclass(frozen=True)
class SessionSource:
    """One input to the calibrator.

    Exactly one of ``csv_path`` / ``ledger_path`` must be set. The
    calibrator iterates the RR sequence, feeds it through a fresh
    :class:`PhysioSession`, and keeps the healthy frames.
    """

    csv_path: Path | None = None
    ledger_path: Path | None = None
    note: str = ""

    def __post_init__(self) -> None:
        n_set = sum(1 for x in (self.csv_path, self.ledger_path) if x is not None)
        if n_set != 1:
            raise CalibrationError("SessionSource requires exactly one of csv_path / ledger_path")


def _iter_samples_from_csv(path: Path) -> list[RRSample]:
    return list(RRReplayReader(path))


def _iter_samples_from_ledger(path: Path) -> list[RRSample]:
    """Reconstruct an RRSample sequence from a ledger's FRAME events."""
    samples: list[RRSample] = []
    raw = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            # Tolerate partial last line.
            if lineno == len(raw.splitlines()):
                break
            raise CalibrationError(f"{path}: malformed JSON at line {lineno}") from None
        if evt.get("event") != "FRAME":
            continue
        samples.append(
            RRSample(
                timestamp_s=float(evt["timestamp_s"]),
                rr_ms=float(evt["rr_ms"]),
                row_index=int(evt.get("tick_index", len(samples))),
            )
        )
    return samples


def _collect_healthy_features(
    samples: list[RRSample],
    *,
    window_size: int,
) -> list[HRVFeatures]:
    """Feed samples through a fresh PhysioSession; keep frames whose
    kernel decision is READY (so we calibrate on the user's *healthy*
    distribution, not on their buffer-fill period)."""
    session = PhysioSession(window_size=window_size)
    healthy: list[HRVFeatures] = []
    from neurophase.physio.gate import PhysioGateState  # local import; small module

    for i, s in enumerate(samples):
        frame = session.step(s, tick_index=i)
        # Only count frames that were usable: EXECUTE_ALLOWED or
        # EXECUTE_REDUCED under default thresholds. This aligns with
        # the "healthy signal" definition we calibrate against.
        if frame.decision.state in (
            PhysioGateState.EXECUTE_ALLOWED,
            PhysioGateState.EXECUTE_REDUCED,
        ):
            healthy.append(frame.features)
    return healthy


def _band_from_values(values: list[float]) -> FeatureBand:
    arr = np.asarray(values, dtype=np.float64)
    return FeatureBand(
        p05=float(np.percentile(arr, 5)),
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=0)),
    )


def calibrate_profile(
    sources: list[SessionSource],
    *,
    user_id: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    notes: tuple[str, ...] = (),
) -> PhysioProfile:
    """Run the calibration pipeline on *sources* and return a profile.

    Raises :class:`CalibrationError` on insufficient data.
    """
    if not user_id:
        raise CalibrationError("user_id is required")
    if len(sources) < MIN_BASELINE_SESSIONS:
        raise CalibrationError(
            f"need at least {MIN_BASELINE_SESSIONS} baseline sessions, got {len(sources)}"
        )

    all_healthy: list[HRVFeatures] = []
    n_sessions_with_healthy = 0
    for src in sources:
        if src.csv_path is not None:
            samples = _iter_samples_from_csv(src.csv_path)
            origin = f"csv:{src.csv_path}"
        else:
            assert src.ledger_path is not None
            samples = _iter_samples_from_ledger(src.ledger_path)
            origin = f"ledger:{src.ledger_path}"
        healthy = _collect_healthy_features(samples, window_size=window_size)
        if len(healthy) < MIN_HEALTHY_FRAMES_PER_SESSION:
            raise CalibrationError(
                f"{origin}: only {len(healthy)} healthy frames "
                f"(need >= {MIN_HEALTHY_FRAMES_PER_SESSION})"
            )
        all_healthy.extend(healthy)
        n_sessions_with_healthy += 1

    if len(all_healthy) < MIN_TOTAL_HEALTHY_FRAMES:
        raise CalibrationError(
            f"total healthy frames {len(all_healthy)} < "
            f"MIN_TOTAL_HEALTHY_FRAMES={MIN_TOTAL_HEALTHY_FRAMES}"
        )

    rmssd_band = _band_from_values([f.rmssd_ms for f in all_healthy])
    stability_band = _band_from_values([f.stability for f in all_healthy])
    continuity_band = _band_from_values([f.continuity_fraction for f in all_healthy])
    confidence_band = _band_from_values([f.confidence for f in all_healthy])

    # Threshold derivation rule (explicit, documented above).
    threshold_abstain = max(DEFAULT_THRESHOLD_ABSTAIN, confidence_band.p05)
    threshold_allow = max(DEFAULT_THRESHOLD_ALLOW, confidence_band.p50)
    # Keep a minimum headroom so the bands stay strictly ordered.
    if threshold_abstain >= threshold_allow:
        threshold_allow = min(0.99, threshold_abstain + 0.05)

    return PhysioProfile(
        user_id=user_id,
        created_at_utc=current_utc_iso(),
        n_baseline_sessions=n_sessions_with_healthy,
        window_size=window_size,
        rmssd_ms=rmssd_band,
        rr_stability=stability_band,
        continuity_fraction=continuity_band,
        confidence=confidence_band,
        threshold_allow=threshold_allow,
        threshold_abstain=threshold_abstain,
        notes=notes,
    )


def calibration_report(profile: PhysioProfile) -> dict[str, Any]:
    """Structured, human-readable summary of a profile."""
    return {
        "user_id": profile.user_id,
        "created_at_utc": profile.created_at_utc,
        "n_baseline_sessions": profile.n_baseline_sessions,
        "window_size": profile.window_size,
        "threshold_allow": profile.threshold_allow,
        "threshold_abstain": profile.threshold_abstain,
        "confidence_band": profile.confidence.to_json_dict(),
        "rmssd_band": profile.rmssd_ms.to_json_dict(),
        "rr_stability_band": profile.rr_stability.to_json_dict(),
        "continuity_band": profile.continuity_fraction.to_json_dict(),
    }


__all__ = [
    "MIN_BASELINE_SESSIONS",
    "MIN_HEALTHY_FRAMES_PER_SESSION",
    "MIN_TOTAL_HEALTHY_FRAMES",
    "CalibrationError",
    "SessionSource",
    "calibrate_profile",
    "calibration_report",
]
