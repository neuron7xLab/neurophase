"""Tests for neurophase.metrics.asymmetry."""

from __future__ import annotations

import networkx as nx
import numpy as np

from neurophase.metrics.asymmetry import kurtosis, skewness, topological_asymmetry


def test_skewness_symmetric_is_zero() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000)
    assert abs(skewness(x)) < 0.2


def test_skewness_right_tailed_positive() -> None:
    rng = np.random.default_rng(1)
    x = rng.exponential(1.0, 5000)
    assert skewness(x) > 0.5


def test_skewness_left_tailed_negative() -> None:
    rng = np.random.default_rng(2)
    x = -rng.exponential(1.0, 5000)
    assert skewness(x) < -0.5


def test_skewness_constant_is_zero() -> None:
    assert skewness(np.full(100, 3.14)) == 0.0


def test_kurtosis_normal_near_zero() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(10000)
    # Excess kurtosis of a standard normal is 0.
    assert abs(kurtosis(x)) < 0.3


def test_kurtosis_heavy_tails_positive() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_t(df=3.0, size=5000)
    assert kurtosis(x) > 0.5


def test_kurtosis_constant_is_zero() -> None:
    assert kurtosis(np.full(100, 7.0)) == 0.0


def _bidirectional_graph() -> nx.Graph:
    g = nx.Graph()
    # Bullish subgraph: a triangle (high curvature neighbourhood).
    for u, v in [(0, 1), (1, 2), (0, 2)]:
        g.add_edge(u, v, weight=1.0)
    # Bearish subgraph: a path (fragile).
    for u, v in [(3, 4), (4, 5)]:
        g.add_edge(u, v, weight=1.0)
    return g


def test_topological_asymmetry_favours_triangular_subgraph() -> None:
    g = _bidirectional_graph()
    bull = [(0, 1), (1, 2), (0, 2)]
    bear = [(3, 4), (4, 5)]
    delta = topological_asymmetry(g, bull, bear)
    # Triangles have higher Ollivier-Ricci than a path.
    assert delta > 0.0


def test_topological_asymmetry_empty_subset_returns_zero() -> None:
    g = _bidirectional_graph()
    assert topological_asymmetry(g, [], [(3, 4)]) == 0.0
    assert topological_asymmetry(g, [(0, 1)], []) == 0.0
