"""Curriculum signals used during the plasticity window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Curriculum:
    """Task-aligned corrective targets."""

    target_bias: NDArray[np.float64]
    corrective_signal: NDArray[np.float64]
    stress_pattern: NDArray[np.float64]

    def __post_init__(self) -> None:
        if (
            self.target_bias.ndim != 1
            or self.corrective_signal.ndim != 1
            or self.stress_pattern.ndim != 1
        ):
            raise ValueError("curriculum vectors must be 1D")
        n = self.target_bias.shape[0]
        if self.corrective_signal.shape != (n,) or self.stress_pattern.shape != (n,):
            raise ValueError("curriculum vectors must share identical shape (n,)")
        if not np.isfinite(self.target_bias).all():
            raise ValueError("target_bias must be finite")
        if not np.isfinite(self.corrective_signal).all():
            raise ValueError("corrective_signal must be finite")
        if not np.isfinite(self.stress_pattern).all():
            raise ValueError("stress_pattern must be finite")
