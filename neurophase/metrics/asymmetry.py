"""Asymmetry metrics — skewness, kurtosis, topological curvature asymmetry.

The Direction Index (DI) combines statistical asymmetry of returns with
topological curvature asymmetry of the order-book graph to pick a side
*after* the emergent-phase criterion has fired:

    DI(t) = w_s · Skew(X_t) + w_c · Δ_curv(G_t) + w_b · Bias(agent)

These metrics are scale-invariant by construction and robust to small
perturbations — they survive the honest-null contract.
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
import numpy.typing as npt

from neurophase.metrics.ricci import ollivier_ricci

# Relative tolerance for "effectively constant" detection.
# Float64 summation of N copies of value v accumulates error ~ N·ε·|v|;
# use a comfortable multiple of that floor.
_CONSTANT_REL_TOL = 1e-12


def _is_effectively_constant(arr: npt.NDArray[np.float64], std: float) -> bool:
    """True when the standard deviation is at or below float-summation noise."""
    scale = float(np.max(np.abs(arr))) if arr.size else 0.0
    floor = _CONSTANT_REL_TOL * max(scale, 1.0)
    return std <= floor


def skewness(series: npt.ArrayLike) -> float:
    """Sample skewness ⟨((x − μ) / σ)³⟩.

    Returns 0.0 for degenerate (effectively constant) input — honest null.
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if _is_effectively_constant(arr, std):
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def kurtosis(series: npt.ArrayLike) -> float:
    """Excess kurtosis ⟨((x − μ) / σ)⁴⟩ − 3.

    Returns 0.0 for degenerate (effectively constant) input.
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < 2:
        return 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if _is_effectively_constant(arr, std):
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 4) - 3.0)


def topological_asymmetry(
    G: nx.Graph,
    bullish_edges: Iterable[tuple[int, int]],
    bearish_edges: Iterable[tuple[int, int]],
) -> float:
    """Curvature asymmetry across bullish and bearish subgraphs.

        Δ_curv = ⟨κ_O⟩_{M⁺} − ⟨κ_O⟩_{M⁻}

    Positive values indicate the bullish subgraph is structurally more
    stable than the bearish one (price path less likely to invert on the
    up-side) — a topological directionality signal.

    Parameters
    ----------
    G : networkx.Graph
    bullish_edges, bearish_edges : iterable of (int, int)
        Edge subsets interpreted as up-moves vs down-moves.

    Returns
    -------
    float
        Difference in mean Ollivier-Ricci curvature. Returns 0.0 when
        either subset is empty.
    """
    bull_list = list(bullish_edges)
    bear_list = list(bearish_edges)
    if not bull_list or not bear_list:
        return 0.0
    bull_kappa = np.array([ollivier_ricci(G, x, y) for x, y in bull_list], dtype=np.float64)
    bear_kappa = np.array([ollivier_ricci(G, x, y) for x, y in bear_list], dtype=np.float64)
    return float(np.mean(bull_kappa) - np.mean(bear_kappa))
