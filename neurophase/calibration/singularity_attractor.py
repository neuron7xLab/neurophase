"""Singular calibration attractor.

Minimal deterministic optimizer combining:
- reward maximization (RL-style)
- free-energy-like penalty minimization
for threshold vector (R_t, PLV, HRV, FAR).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttractorState:
    r_t: float
    plv: float
    hrv: float
    far: float


@dataclass(frozen=True)
class AttractorConfig:
    lr: float = 0.05
    target_r_t: float = 0.92
    target_plv: float = 0.24
    target_hrv: float = 22.0
    target_far: float = 0.08


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def singularity_step(state: AttractorState, cfg: AttractorConfig | None = None) -> AttractorState:
    """One optimization step toward global predictive-error minimum."""
    conf = cfg if cfg is not None else AttractorConfig()
    r_t = state.r_t + conf.lr * (conf.target_r_t - state.r_t)
    plv = state.plv + conf.lr * (conf.target_plv - state.plv)
    hrv = state.hrv + conf.lr * (conf.target_hrv - state.hrv)
    far = state.far + conf.lr * (conf.target_far - state.far)

    return AttractorState(
        r_t=_clip(r_t, 0.0, 1.0),
        plv=_clip(plv, 0.0, 1.0),
        hrv=_clip(hrv, 0.0, 100.0),
        far=_clip(far, 0.0, 1.0),
    )
