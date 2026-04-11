"""Tests for neurophase.metrics.ricci."""

from __future__ import annotations

import networkx as nx
import pytest

from neurophase.metrics.ricci import forman_ricci, mean_ricci, ollivier_ricci


def _path_graph(n: int) -> nx.Graph:
    G = nx.path_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


def _complete_graph(n: int) -> nx.Graph:
    G = nx.complete_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


def test_ollivier_on_path_is_nonpositive() -> None:
    """A path graph is structurally fragile — κ_O ≤ 0 on interior edges."""
    G = _path_graph(5)
    kappa = ollivier_ricci(G, 1, 2)
    assert kappa <= 0.0 + 1e-9


def test_ollivier_on_complete_graph_is_higher() -> None:
    """A complete graph is well-connected — mean κ_O larger than on a path."""
    path = _path_graph(6)
    complete = _complete_graph(6)
    assert mean_ricci(complete, lambda_o=1.0) >= mean_ricci(path, lambda_o=1.0)


def test_forman_symmetric() -> None:
    G = _complete_graph(5)
    assert forman_ricci(G, 0, 1) == pytest.approx(forman_ricci(G, 1, 0))


def test_forman_returns_zero_for_nonedge() -> None:
    G = nx.Graph()
    G.add_nodes_from([0, 1])
    assert forman_ricci(G, 0, 1) == 0.0


def test_mean_ricci_empty_graph() -> None:
    G = nx.Graph()
    assert mean_ricci(G) == 0.0


def test_mean_ricci_rejects_bad_lambda() -> None:
    G = _path_graph(3)
    with pytest.raises(ValueError, match="lambda_o must be in"):
        mean_ricci(G, lambda_o=1.5)


def test_ollivier_disconnected_returns_zero() -> None:
    """Honest null: no path between components → curvature = 0, not synthetic."""
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    assert ollivier_ricci(G, 0, 1) == 0.0


def test_mean_ricci_on_subset_of_edges() -> None:
    G = _complete_graph(4)
    subset = [(0, 1), (1, 2)]
    value = mean_ricci(G, lambda_o=0.5, edges=subset)
    assert isinstance(value, float)
