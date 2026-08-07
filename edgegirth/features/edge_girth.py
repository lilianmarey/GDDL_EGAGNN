"""Edge-girth and edge-girth-multiplicity computation.

For an edge e = {u, v}, a BFS from u in G \\ {e} gives:
  - d(u, v): the distance from u to v avoiding e
  - the number of shortest u-v paths (via the standard shortest-path
    counting recursion, accumulated during the BFS)

Then:
  - g_e = d(u, v) + 1        (inf if v unreachable, i.e. e is a bridge)
  - lambda_e = # shortest u-v paths in G \\ {e}   (exact cycle count through e
    at length g_e)

This is O(|V| + |E|) per edge, hence O(|E| (|V| + |E|)) for the whole graph.
No path enumeration, no Yen's algorithm: shortest-path counting is exact and
linear-time, unlike counting simple paths of a prescribed length (#W[1]-hard).
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass

import networkx as nx
import numpy as np
from joblib import Parallel, delayed

from edgegirth.paths import CACHE_DIR


@dataclass
class EdgeGirthResult:
    # maps frozenset({u, v}) -> (g_e, lambda_e); g_e is float('inf') for bridges
    values: dict
    elapsed_seconds: float


def _shortest_path_count_from(G: nx.Graph, source):
    """BFS-based shortest path counting from `source` to every reachable node.

    Returns (dist, count) dicts. Standard recursion: count[v] = sum of
    count[u] over predecessors u of v on a shortest path.
    """
    dist = {source: 0}
    count = {source: 1}
    q = deque([source])
    while q:
        u = q.popleft()
        du = dist[u]
        for w in G.neighbors(u):
            if w not in dist:
                dist[w] = du + 1
                count[w] = count[u]
                q.append(w)
            elif dist[w] == du + 1:
                count[w] += count[u]
    return dist, count


def edge_girth_for_edge(G: nx.Graph, u, v):
    """Compute (g_e, lambda_e) for edge e = {u, v} in G."""
    # G \ {e}: remove the edge, run BFS shortest-path count from u, restore.
    G.remove_edge(u, v)
    try:
        dist, count = _shortest_path_count_from(G, u)
    finally:
        G.add_edge(u, v)

    if v not in dist:
        return float("inf"), 0
    g_e = dist[v] + 1
    lambda_e = count[v]
    return g_e, lambda_e


def compute_edge_girth(G: nx.Graph) -> dict:
    """Compute (g_e, lambda_e) for every edge of G.

    Returns dict: frozenset({u, v}) -> (g_e, lambda_e).
    """
    result = {}
    for u, v in G.edges():
        result[frozenset((u, v))] = edge_girth_for_edge(G, u, v)
    return result


def _graph_hash(G: nx.Graph) -> str:
    """Cheap, deterministic hash of a graph's structure (node ids + edge set)."""
    nodes = sorted(G.nodes())
    edges = sorted(tuple(sorted(e)) for e in G.edges())
    h = hashlib.sha256()
    h.update(repr(nodes).encode())
    h.update(repr(edges).encode())
    return h.hexdigest()


def compute_edge_girth_cached(G: nx.Graph, cache_dir: str = CACHE_DIR) -> EdgeGirthResult:
    os.makedirs(cache_dir, exist_ok=True)
    key = _graph_hash(G)
    path = os.path.join(cache_dir, f"{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.perf_counter()
    values = compute_edge_girth(G)
    elapsed = time.perf_counter() - t0
    result = EdgeGirthResult(values=values, elapsed_seconds=elapsed)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def compute_edge_girth_batch(graphs: list[nx.Graph], n_jobs: int = -1, cache_dir: str = CACHE_DIR):
    """Parallel, cached edge-girth computation over a list of graphs.

    One graph per task (joblib.Parallel). Returns a list of EdgeGirthResult,
    one per input graph, in the same order.
    """
    os.makedirs(cache_dir, exist_ok=True)

    def _job(G):
        return compute_edge_girth_cached(G, cache_dir=cache_dir)

    return Parallel(n_jobs=n_jobs)(delayed(_job)(G) for G in graphs)


def is_edge_girth_regular(G: nx.Graph, girth_dict: dict | None = None):
    """egr(n, k, g, lambda) test (Jajcay, Kiss & Miklavic 2018): G is
    edge-girth-regular iff it is degree-regular AND (g_e, lambda_e) is the
    same for every edge. Returns (is_egr, params) where params is
    (k, g, lambda) when is_egr is True, else None.

    A graph with any bridge (g_e = inf) cannot be edge-girth-regular unless
    *every* edge is a bridge (a regular forest, degree 0 or a disjoint union
    of single edges) -- handled correctly by the "identical g_e" check since
    inf == inf is well-defined and compares equal across bridges.
    """
    if girth_dict is None:
        girth_dict = compute_edge_girth(G)
    degrees = {d for _, d in G.degree()}
    if len(degrees) != 1:
        return False, None
    k = degrees.pop()
    values = list(girth_dict.values())
    if not values:
        return False, None
    g0, lam0 = values[0]
    for g_e, lam_e in values[1:]:
        same_g = (g_e == g0) or (math.isinf(g_e) and math.isinf(g0))
        if not (same_g and lam_e == lam0):
            return False, None
    return True, (k, g0, lam0)


def is_strongly_regular(G: nx.Graph):
    """srg(n, k, lambda, mu) test: G is k-regular, every pair of adjacent
    vertices shares exactly lambda common neighbours, every pair of
    non-adjacent vertices shares exactly mu common neighbours. Returns
    (is_srg, params) where params is (n, k, lambda, mu) when is_srg is True
    (mu is None if the graph is complete -- no non-adjacent pairs to define
    it on -- and lambda is None if the graph is empty), else (False, None).

    Distinct from edge-girth-regularity: strongly regular graphs happen to be
    edge-girth-regular whenever lambda > 0 (girth 3, lambda_e = lambda for
    every edge -- see Corollary 2 of the paper), but "strongly regular" is a
    property of 2-step neighbourhoods, unrelated in general to (g_e, lambda_e)
    or to 2-FWL/3-WL indistinguishability, which is a separate, coincidental
    fact about the same graphs, not a consequence of Proposition 1.
    """
    n = G.number_of_nodes()
    if n == 0:
        return False, None
    degrees = {d for _, d in G.degree()}
    if len(degrees) != 1:
        return False, None
    k = degrees.pop()

    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((n, n), dtype=np.int64)
    for u, v in G.edges():
        A[idx[u], idx[v]] = 1
        A[idx[v], idx[u]] = 1
    A2 = A @ A  # A2[i, j] = number of common neighbours of i, j

    adj_mask = A == 1
    nonadj_mask = (A == 0) & ~np.eye(n, dtype=bool)

    lam = None
    if adj_mask.any():
        adj_vals = A2[adj_mask]
        if not np.all(adj_vals == adj_vals[0]):
            return False, None
        lam = int(adj_vals[0])

    mu = None
    if nonadj_mask.any():
        nonadj_vals = A2[nonadj_mask]
        if not np.all(nonadj_vals == nonadj_vals[0]):
            return False, None
        mu = int(nonadj_vals[0])

    return True, (n, k, lam, mu)
