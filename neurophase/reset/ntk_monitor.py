"""NTK-rank proxy monitor used as a falsifiable plasticity predicate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NTKSnapshot:
    ntk_rank_pre: float
    ntk_rank_post: float

    @property
    def rank_delta(self) -> float:
        return float(self.ntk_rank_post - self.ntk_rank_pre)


class NTKMonitor:
    """Computes normalized matrix-rank proxy from state weights."""

    def __init__(self, tol: float = 1e-4) -> None:
        self.tol = tol

    def rank_proxy(self, weights: NDArray[np.float64]) -> float:
        gram = weights @ weights.T
        rank = np.linalg.matrix_rank(gram, tol=self.tol)
        return float(rank / weights.shape[0])

    def compare(
        self,
        before: NDArray[np.float64],
        after: NDArray[np.float64],
    ) -> NTKSnapshot:
        return NTKSnapshot(
            ntk_rank_pre=self.rank_proxy(before),
            ntk_rank_post=self.rank_proxy(after),
        )
