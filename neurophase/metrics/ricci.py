"""Ollivier–Ricci and Forman–Ricci curvature for weighted graphs.

Given an undirected weighted graph G = (V, E) — representing a market as
price levels (vertices) connected by transition weights (edges) — the
Ricci curvature of an edge quantifies how strongly local random walks
around its endpoints overlap:

    Ollivier-Ricci:  κ_O(x, y) = 1 - W₁(μ_x, μ_y) / d(x, y)
    Forman-Ricci:    κ_F(x, y) = w_xy · ( w_x/d_x + w_y/d_y
                                          - Σ_{z ~ x, z ~ y}
                                             (w_xz + w_yz) / w_xy )

where W₁ is the Wasserstein-1 distance between the transition
distributions μ_x, μ_y at x and y.

Mean curvature over the edge set is a phase-transition diagnostic:

    κ̄(t) < 0  →  structurally unstable (possible regime shift)
    κ̄(t) > 0  →  stable neighbourhood

Implementation supports both curvatures and a weighted combination used
by the 4-condition emergent-phase criterion (π-system reference).
"""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.stats import wasserstein_distance


def _local_distribution(
    G: nx.Graph, node: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return (support, probabilities) of the one-step random walk from ``node``.

    For an isolated node the walk stays in place with mass 1.
    """
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return np.array([node], dtype=np.float64), np.array([1.0], dtype=np.float64)
    weights = np.array(
        [float(G[node][n].get("weight", 1.0)) for n in neighbors],
        dtype=np.float64,
    )
    total = float(np.sum(weights))
    if total <= 0:
        probs: npt.NDArray[np.float64] = np.full(
            len(neighbors), 1.0 / len(neighbors), dtype=np.float64
        )
    else:
        probs = np.asarray(weights / total, dtype=np.float64)
    support = np.array(neighbors, dtype=np.float64)
    return support, probs


def ollivier_ricci(G: nx.Graph, x: int, y: int) -> float:
    """Ollivier–Ricci curvature of edge (x, y).

    Uses the Wasserstein-1 distance between the one-step transition
    distributions at x and y; divides by the shortest-path distance d(x, y).

    Parameters
    ----------
    G : networkx.Graph
        Weighted undirected graph.
    x, y : int
        Endpoints of the edge.

    Returns
    -------
    float
        Curvature value in (-∞, 1]. Returns 0.0 if x and y are in different
        components (no path) — an honest null, not synthetic.
    """
    support_x, probs_x = _local_distribution(G, x)
    support_y, probs_y = _local_distribution(G, y)
    try:
        d = float(nx.shortest_path_length(G, source=x, target=y, weight="weight"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return 0.0
    if d <= 0:
        return 0.0
    w1 = float(wasserstein_distance(support_x, support_y, probs_x, probs_y))
    return 1.0 - w1 / d


def forman_ricci(G: nx.Graph, x: int, y: int) -> float:
    """Forman–Ricci curvature of edge (x, y).

    Uses the discrete Forman formula over common neighbours, with each
    degree computed as the sum of incident edge weights. No normalisation
    by ``w_xy`` at the end — the raw Forman form preserves scale.

    Parameters
    ----------
    G : networkx.Graph
    x, y : int
    """
    if not G.has_edge(x, y):
        return 0.0
    w_xy = float(G[x][y].get("weight", 1.0))
    if w_xy <= 0:
        return 0.0
    d_x = float(G.degree(x, weight="weight"))
    d_y = float(G.degree(y, weight="weight"))
    common = set(G.neighbors(x)) & set(G.neighbors(y)) - {x, y}
    cross_sum = 0.0
    for z in common:
        w_xz = float(G[x][z].get("weight", 1.0))
        w_yz = float(G[y][z].get("weight", 1.0))
        cross_sum += (w_xz + w_yz) / w_xy
    return w_xy * (d_x / w_xy + d_y / w_xy - cross_sum)


def mean_ricci(
    G: nx.Graph,
    lambda_o: float = 0.5,
    edges: Iterable[tuple[int, int]] | None = None,
) -> float:
    """Weighted mean of Ollivier- and Forman-Ricci curvatures.

        κ̄ = λ_o · mean(κ_O) + (1 − λ_o) · mean(κ_F)

    Parameters
    ----------
    G : networkx.Graph
    lambda_o : float
        Weight on the Ollivier component; must be in [0, 1].
    edges : iterable of (int, int), optional
        Restrict the average to this subset. Defaults to all edges.

    Returns
    -------
    float
        Mean curvature. Returns 0.0 for an empty edge set.
    """
    if not 0.0 <= lambda_o <= 1.0:
        raise ValueError(f"lambda_o must be in [0, 1], got {lambda_o}")
    edge_list = list(edges) if edges is not None else list(G.edges())
    if not edge_list:
        return 0.0
    kappa_o = np.array([ollivier_ricci(G, x, y) for x, y in edge_list], dtype=np.float64)
    kappa_f = np.array([forman_ricci(G, x, y) for x, y in edge_list], dtype=np.float64)
    return float(lambda_o * np.mean(kappa_o) + (1.0 - lambda_o) * np.mean(kappa_f))
