"""Mutable system tensors used by reset interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SystemState:
    """State bundle operated by the KLR controller."""

    weights: NDArray[np.float64]
    confidence: NDArray[np.float64]
    usage: NDArray[np.float64]
    utility: NDArray[np.float64]
    inhibition: NDArray[np.float64]
    topology: NDArray[np.float64]
    frozen: NDArray[np.bool_] | None = None
    gamma: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.weights.shape[0]
        if self.weights.ndim != 2 or self.weights.shape != (n, n):
            raise ValueError("weights must be square [n, n]")
        if not np.isfinite(self.weights).all():
            raise ValueError("weights must be finite")

        for name, arr in (
            ("confidence", self.confidence),
            ("usage", self.usage),
            ("utility", self.utility),
            ("inhibition", self.inhibition),
        ):
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must be finite")
            if np.any(arr < 0.0) or np.any(arr > 1.0):
                raise ValueError(f"{name} must be in [0, 1]")

        if self.topology.shape != (n, n):
            raise ValueError(f"topology must have shape ({n}, {n}), got {self.topology.shape}")
        if not np.isfinite(self.topology).all():
            raise ValueError("topology must be finite")

        if self.frozen is None:
            self.frozen = np.zeros(n, dtype=bool)
        elif self.frozen.shape != (n,):
            raise ValueError(f"frozen must have shape ({n},), got {self.frozen.shape}")
        if not np.isfinite(float(self.gamma)):
            raise ValueError("gamma must be finite")
        if not 0.0 <= float(self.gamma) <= 1.0:
            raise ValueError("gamma must be in [0, 1]")

        # Canonical semantics: row-stochastic weights.
        row_sums = self.weights.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("weights rows must have positive sum for row-stochastic semantics")
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("weights must be row-stochastic (each row sum ≈ 1.0)")
