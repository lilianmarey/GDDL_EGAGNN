"""Unit tests for `compute_edge_girth` against graphs with known closed-form
(g_e, lambda_e) values: cycles, complete graphs, the Petersen graph, trees
(all-bridge), and K3,3.
"""

import math

import networkx as nx
import pytest

from edgegirth.features.edge_girth import compute_edge_girth


def test_cycle():
    for n in [3, 4, 5, 6, 10]:
        G = nx.cycle_graph(n)
        vals = compute_edge_girth(G)
        assert len(vals) == n
        for g_e, lam_e in vals.values():
            assert g_e == n, f"C_{n}: expected g_e={n}, got {g_e}"
            assert lam_e == 1, f"C_{n}: expected lambda_e=1, got {lam_e}"


def test_complete_graph():
    for n in [3, 4, 5, 6, 8]:
        G = nx.complete_graph(n)
        vals = compute_edge_girth(G)
        assert len(vals) == n * (n - 1) // 2
        for g_e, lam_e in vals.values():
            assert g_e == 3, f"K_{n}: expected g_e=3, got {g_e}"
            assert lam_e == n - 2, f"K_{n}: expected lambda_e={n - 2}, got {lam_e}"


def test_petersen():
    G = nx.petersen_graph()
    vals = compute_edge_girth(G)
    assert len(vals) == 15  # 15 edges
    for g_e, lam_e in vals.values():
        assert g_e == 5, f"Petersen: expected g_e=5, got {g_e}"
        assert lam_e == 4, f"Petersen: expected lambda_e=4, got {lam_e}"


def test_tree():
    trees = [
        nx.path_graph(6),
        nx.star_graph(5),
        nx.random_labeled_tree(20, seed=13),
    ]
    for G in trees:
        vals = compute_edge_girth(G)
        assert len(vals) == G.number_of_edges()
        for g_e, lam_e in vals.values():
            assert g_e == math.inf, f"tree: expected g_e=inf, got {g_e}"
            assert lam_e == 0


def test_complete_bipartite_k33():
    G = nx.complete_bipartite_graph(3, 3)
    vals = compute_edge_girth(G)
    assert len(vals) == 9
    for g_e, lam_e in vals.values():
        assert g_e == 4, f"K_3,3: expected g_e=4, got {g_e}"
        # For e = {a1, b1}, shortest paths avoiding e are a1-bx-ay-b1 with
        # bx in B\{b1} (2 choices) and ay in A\{a1} (2 choices): 2*2 = 4.
        assert lam_e == 4, f"K_3,3: expected lambda_e=4, got {lam_e}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
