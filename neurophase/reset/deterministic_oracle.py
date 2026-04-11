"""Deterministic seed derivation for KLR interventions."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from neurophase.reset.curriculum import Curriculum
from neurophase.reset.metrics import SystemMetrics
from neurophase.reset.state import SystemState


@dataclass(frozen=True)
class SeedTrace:
    seed: int
    digest_prefix: str


def derive_seed(state: SystemState, metrics: SystemMetrics, curriculum: Curriculum) -> SeedTrace:
    payload = (
        state.weights.tobytes()
        + struct.pack(
            "6d",
            metrics.error,
            metrics.persistence,
            metrics.diversity,
            metrics.improvement,
            metrics.noise,
            metrics.reward,
        )
        + curriculum.target_bias.tobytes()
        + curriculum.corrective_signal.tobytes()
        + curriculum.stress_pattern.tobytes()
    )
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**32)
    return SeedTrace(seed=seed, digest_prefix=digest.hex()[:16])
