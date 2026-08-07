"""Baseline models: GCN, GIN, GatedGCN, GSN, PPGN, Folklore-2WL (exact,
non-learned), triangle-count-as-edge-feature, and edge-girth-sequence-alone
(exact, non-learned).
"""

from __future__ import annotations

import math
from collections import Counter

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GINEConv, MessagePassing, ResGatedGraphConv, global_add_pool

from .gnn import mlp


# ---------------------------------------------------------------------------
# Generic MPNN backbones: GCN, GIN, GatedGCN
# ---------------------------------------------------------------------------
class _GenericMPNN(nn.Module):
    conv_cls = None

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        out_dim: int = 16,
        edge_dim: int | None = None,
    ):
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.uses_edge_attr = edge_dim is not None and self.conv_cls is ResGatedGraphConv
        if self.uses_edge_attr:
            self.edge_embed = nn.Linear(edge_dim, hidden_dim)
        for _ in range(num_layers):
            self.convs.append(self._make_conv(hidden_dim))
        self.out = nn.Linear(hidden_dim, out_dim)

    def _make_conv(self, hidden_dim):
        if self.conv_cls is GINConv:
            return GINConv(mlp(hidden_dim, hidden_dim))
        if self.conv_cls is ResGatedGraphConv:
            return ResGatedGraphConv(hidden_dim, hidden_dim)
        return GCNConv(hidden_dim, hidden_dim)

    def forward(self, data):
        h = self.node_embed(data.x)
        for conv in self.convs:
            h = torch.relu(conv(h, data.edge_index))
        g = global_add_pool(h, data.batch)
        return self.out(g)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class GCNModel(_GenericMPNN):
    conv_cls = GCNConv


class GINModel(_GenericMPNN):
    conv_cls = GINConv


class GatedGCNModel(_GenericMPNN):
    """ResGatedGraphConv == the gated-edge MPNN of Bresson & Laurent 2017."""

    conv_cls = ResGatedGraphConv


# ---------------------------------------------------------------------------
# GatedGCN with EGAGNN-matched nonlinear depth.
# ResGatedGraphConv's key/query/value/skip transforms are each a single
# Linear layer (0 hidden layers); EGAGNN's phi_e/phi_n/psi are each a 2-layer
# MLP (1 hidden layer, one internal ReLU). This class replaces each of
# ResGatedGraphConv's four Linear transforms with the same mlp() used inside
# EGAGNN, holding the gating mechanism itself (sigmoid(k_i+q_j)*v_j, additive
# skip connection) otherwise identical, so it isolates nonlinear depth from
# every other architectural difference.
# ---------------------------------------------------------------------------
class MLPGatedConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__(aggr="add")
        self.lin_key = mlp(in_dim, out_dim)
        self.lin_query = mlp(in_dim, out_dim)
        self.lin_value = mlp(in_dim, out_dim)
        self.lin_skip = mlp(in_dim, out_dim)

    def forward(self, x, edge_index):
        k = self.lin_key(x)
        q = self.lin_query(x)
        v = self.lin_value(x)
        out = self.propagate(edge_index, k=k, q=q, v=v)
        return out + self.lin_skip(x)

    def message(self, k_i, q_j, v_j):
        return torch.sigmoid(k_i + q_j) * v_j


class GatedGCNMLPModel(nn.Module):
    """Same gating mechanism as GatedGCNModel, but with EGAGNN-matched 2-layer
    MLPs in place of ResGatedGraphConv's single-Linear key/query/value/skip."""

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        out_dim: int = 16,
        edge_dim: int | None = None,
    ):
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.convs = nn.ModuleList([MLPGatedConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        h = self.node_embed(data.x)
        for conv in self.convs:
            h = torch.relu(conv(h, data.edge_index))
        g = global_add_pool(h, data.batch)
        return self.out(g)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def make_regressor(embedder_cls, node_in_dim, hidden_dim=64, num_layers=4, **kwargs):
    class Regressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = embedder_cls(
                node_in_dim, hidden_dim=hidden_dim, num_layers=num_layers, out_dim=hidden_dim, **kwargs
            )
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

        def forward(self, data):
            return self.head(self.backbone(data)).squeeze(-1)

        def num_parameters(self):
            return sum(p.numel() for p in self.parameters())

    return Regressor()


# ---------------------------------------------------------------------------
# GSN: GIN backbone + local substructure counts (triangles, 4-cycles) up to
# k=4, injected as extra node/edge features. Exact counts via networkx
# brute-force enumeration over each node/edge's local neighbourhood --
# tractable at BREC/CSL/ZINC scale (<= ~40 nodes).
# ---------------------------------------------------------------------------
def gsn_node_features(G: nx.Graph) -> dict:
    """count of triangles and 4-cycles through each node, k<=4."""
    tri = nx.triangles(G)
    four_cycles = {n: 0 for n in G.nodes()}
    for n in G.nodes():
        neigh = list(G.neighbors(n))
        for i in range(len(neigh)):
            for j in range(i + 1, len(neigh)):
                a, b = neigh[i], neigh[j]
                if a == b or G.has_edge(a, b):
                    continue
                common = set(G.neighbors(a)) & set(G.neighbors(b))
                common.discard(n)
                four_cycles[n] += len(common)
    return {n: (tri[n], four_cycles[n] // 1) for n in G.nodes()}


class GSNModel(nn.Module):
    """GIN backbone with GSN-style structural node/edge counts concatenated
    to the raw node/edge inputs (cycles up to k=4, the BREC configuration).

    `edge_dim`: optional bond-type/edge-attribute
    dimension. When given, GINConv is replaced by GINEConv, which projects
    `data.edge_attr` into the message alongside the node features -- the
    standard, minimal way to give a GIN-style backbone access to edge
    features without altering anything else about the architecture. Defaults
    to None (plain GINConv, exactly the previous behaviour) so every existing
    caller (BREC, CSL, the cycle-scale probe) is completely unaffected."""

    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        out_dim: int = 16,
        struct_dim: int = 2,
        edge_dim: int | None = None,
    ):
        super().__init__()
        self.node_embed = nn.Linear(node_in_dim + struct_dim, hidden_dim)
        self.uses_edge_attr = edge_dim is not None
        if self.uses_edge_attr:
            self.convs = nn.ModuleList(
                [GINEConv(mlp(hidden_dim, hidden_dim), edge_dim=edge_dim) for _ in range(num_layers)]
            )
        else:
            self.convs = nn.ModuleList([GINConv(mlp(hidden_dim, hidden_dim)) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = torch.cat([data.x, data.gsn_node_feat], dim=1)
        h = self.node_embed(x)
        for conv in self.convs:
            if self.uses_edge_attr:
                h = torch.relu(conv(h, data.edge_index, edge_attr=data.edge_attr))
            else:
                h = torch.relu(conv(h, data.edge_index))
        g = global_add_pool(h, data.batch)
        return self.out(g)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def attach_gsn_features(data, G: nx.Graph):
    node_feat = gsn_node_features(G)
    n = data.num_nodes
    feat = torch.zeros(n, 2)
    for node, (tri, fourc) in node_feat.items():
        feat[node, 0] = tri
        feat[node, 1] = fourc
    data = data.clone()
    data.gsn_node_feat = feat
    return data


# ---------------------------------------------------------------------------
# Generalised GSN(k): exact counts of cycles of every length 3..k through each
# node, via nx.simple_cycles(G, length_bound=k) (networkx >= 3.2). This is the
# motif-dictionary-grows-with-k mechanism the cycle-scale probe
# is built to demonstrate: GSN(k=4) has a 2-dim motif dictionary (lengths 3,4),
# GSN(k=8) has a 6-dim one (lengths 3..8) -- k is a budget fixed in advance,
# unlike edge-girth's unbounded g_e. Kept separate from the k=4-only functions
# above so nothing used by BREC/CSL/ZINC is touched.
# ---------------------------------------------------------------------------
def gsn_node_features_k(G: nx.Graph, k: int = 4) -> dict:
    """count of cycles of each length 3..k through each node. Returns
    node -> tuple of length (k-2), one count per cycle length."""
    dim = k - 2
    counts = {n: [0] * dim for n in G.nodes()}
    for cycle in nx.simple_cycles(G, length_bound=k):
        length = len(cycle)
        if length < 3 or length > k:
            continue
        idx = length - 3
        for n in cycle:
            counts[n][idx] += 1
    return {n: tuple(v) for n, v in counts.items()}


def attach_gsn_features_k(data, G: nx.Graph, k: int = 4):
    dim = k - 2
    node_feat = gsn_node_features_k(G, k)
    n = data.num_nodes
    feat = torch.zeros(n, dim)
    for node, counts in node_feat.items():
        feat[node] = torch.tensor(counts, dtype=torch.float32)
    data = data.clone()
    data.gsn_node_feat = feat
    return data


# ---------------------------------------------------------------------------
# PPGN (Maron et al. 2019): order-2 tensor network, matrix-mult layers.
# Implemented densely, one graph at a time (small graphs at this scale).
# ---------------------------------------------------------------------------
class PPGNLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mlp1 = nn.Conv2d(in_ch, out_ch, 1)
        self.mlp2 = nn.Conv2d(in_ch, out_ch, 1)
        self.mlp_out = nn.Conv2d(in_ch + out_ch, out_ch, 1)

    def forward(self, X):
        # X: (1, C, N, N)
        m1 = torch.relu(self.mlp1(X))
        m2 = torch.relu(self.mlp2(X))
        mult = torch.matmul(m1, m2)  # batched matmul over last two dims
        out = self.mlp_out(torch.cat([X, mult], dim=1))
        return out


class PPGNModel(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        out_dim: int = 16,
        edge_dim: int | None = None,
    ):
        super().__init__()
        in_ch = node_in_dim + 1 + (edge_dim or 0)  # node feat on diag, adjacency channel, edge feat off-diag
        self.in_ch = in_ch
        self.edge_dim = edge_dim
        self.layers = nn.ModuleList()
        c = in_ch
        for _ in range(num_layers):
            self.layers.append(PPGNLayer(c, hidden_dim))
            c = hidden_dim
        self.readout = nn.Linear(2 * hidden_dim, out_dim)

    def _build_tensor(self, data):
        n = data.num_nodes
        C = self.in_ch
        X = torch.zeros(C, n, n)
        adj = torch.zeros(n, n)
        ei = data.edge_index
        adj[ei[0], ei[1]] = 1.0
        X[0] = adj
        for i in range(data.x.size(1)):
            X[1 + i].diagonal().copy_(data.x[:, i])
        if self.edge_dim:
            off = data.x.size(1) + 1
            for k in range(ei.size(1)):
                u, v = int(ei[0, k]), int(ei[1, k])
                X[off : off + self.edge_dim, u, v] = data.edge_attr[k, : self.edge_dim]
        return X.unsqueeze(0)

    def forward(self, data):
        # process each graph in the batch separately (dense per-graph tensors)
        graphs = data.to_data_list() if hasattr(data, "to_data_list") else [data]
        outs = []
        for g in graphs:
            X = self._build_tensor(g)
            for layer in self.layers:
                X = torch.relu(layer(X))
            diag = torch.diagonal(X, dim1=2, dim2=3).sum(-1)  # (1,C)
            offdiag = X.sum(dim=(2, 3)) - torch.diagonal(X, dim1=2, dim2=3).sum(-1)
            outs.append(torch.cat([diag, offdiag], dim=1))
        pooled = torch.cat(outs, dim=0)
        return self.readout(pooled)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Folklore-2WL: exact, non-learned pair-colour refinement. Deterministic ->
# used directly as the "embedding" fed to the RPC T^2 test, no training.
# ---------------------------------------------------------------------------
def _base_pair_color(n: int, adj: np.ndarray) -> np.ndarray:
    diag = np.eye(n, dtype=bool)
    # content-defined base colour: 0 = non-edge/off-diag, 1 = edge, 2 = diagonal.
    # These values mean the same thing regardless of which graph they came from,
    # which is what makes colours comparable *across* the two graphs of a pair.
    return np.where(diag, 2, adj.astype(np.int64))


def folklore_2wl_stable_coloring_pair(G_A: nx.Graph, G_B: nx.Graph, max_rounds: int | None = None):
    """Exact, deterministic 2-FWL colour refinement, run to convergence, with
    colours assigned *jointly* across G_A and G_B so that identical structural
    signatures get identical colour ids in both graphs (a prerequisite for
    directly comparing the two colourings -- np.unique's per-call indexing is
    otherwise graph-specific and not comparable across two separate calls).

    BREC only ever compares graphs of equal order within a pair, which is what
    makes a shared (n, n) working shape possible; this is asserted, not assumed.

    max_rounds defaults to n^2 -- not a heuristic truncation, but a proven upper
    bound: pair-colour refinement over n^2 cells is monotonically non-decreasing
    in the number of distinct colours (each round only ever refines, never
    merges), and the number of distinct colours is bounded by 2*n^2 (the total
    cell count across both graphs), so the partition must stabilise within that
    many rounds. In practice convergence is reached in a handful of rounds; the
    loop breaks on the first round with zero colour changes in either graph.
    """
    assert G_A.number_of_nodes() == G_B.number_of_nodes(), (
        "2-FWL pair comparison requires equal-order graphs (true of every BREC pair)"
    )
    n = G_A.number_of_nodes()

    def _adj(G):
        nodes = list(G.nodes())
        idx = {v: i for i, v in enumerate(nodes)}
        A = np.zeros((n, n), dtype=bool)
        for u, v in G.edges():
            A[idx[u], idx[v]] = True
            A[idx[v], idx[u]] = True
        return A

    color_A = _base_pair_color(n, _adj(G_A))
    color_B = _base_pair_color(n, _adj(G_B))

    max_rounds = max_rounds or (2 * n * n)  # provable termination bound, not a truncation
    rounds_used = 0
    for _ in range(max_rounds):
        rounds_used += 1
        modulus = int(max(color_A.max(), color_B.max())) + 1

        def _sorted_code(color):
            code = color[:, :, None].astype(np.int64) * modulus + color[None, :, :].astype(np.int64)
            return np.sort(code, axis=1)  # canonical multiset over k, per (i, ., j)

        feat_A = np.concatenate(
            [color_A.reshape(n, n, 1), _sorted_code(color_A).transpose(0, 2, 1)], axis=2
        ).reshape(n * n, -1)
        feat_B = np.concatenate(
            [color_B.reshape(n, n, 1), _sorted_code(color_B).transpose(0, 2, 1)], axis=2
        ).reshape(n * n, -1)

        # jointly canonicalised: identical rows (identical content) get identical
        # new ids regardless of whether they came from feat_A or feat_B.
        feat_joint = np.concatenate([feat_A, feat_B], axis=0)
        _, inv = np.unique(feat_joint, axis=0, return_inverse=True)
        new_color_A = inv[: n * n].reshape(n, n)
        new_color_B = inv[n * n :].reshape(n, n)

        stable = np.array_equal(new_color_A, color_A) and np.array_equal(new_color_B, color_B)
        color_A, color_B = new_color_A, new_color_B
        if stable:
            break
    else:
        # only reachable if the provable bound above was actually hit -- would
        # indicate a bug in the refinement, not legitimate slow convergence.
        import warnings

        warnings.warn(
            f"2-FWL refinement did not stabilise within the proven bound "
            f"({max_rounds} rounds, n={n}) -- investigate."
        )

    return color_A, color_B, rounds_used


def folklore_2wl_distinguishable(G_A: nx.Graph, G_B: nx.Graph) -> tuple[bool, int]:
    """True iff exact 2-FWL distinguishes G_A and G_B: their stable pair-colour
    multisets differ. No hash projection, no T^2 test, no training -- direct
    multiset equality on jointly-canonicalised colours. Returns
    (distinguishable, rounds_to_convergence)."""
    color_A, color_B, rounds_used = folklore_2wl_stable_coloring_pair(G_A, G_B)
    counter_A = Counter(color_A.reshape(-1).tolist())
    counter_B = Counter(color_B.reshape(-1).tolist())
    return counter_A != counter_B, rounds_used


def folklore_2wl_single_coloring(G: nx.Graph, max_rounds: int | None = None) -> np.ndarray:
    """Single-graph 2-FWL stable colouring, run to convergence (same provable
    n^2-round bound as the pairwise version). Colour ids here are only
    meaningful *within* this one call (np.unique's per-call indexing), which
    is fine for a canonical per-graph feature (e.g. a classification input)
    but NOT for testing whether two graphs are distinguishable -- use
    folklore_2wl_distinguishable for that (BREC)."""
    n = G.number_of_nodes()
    adj = np.zeros((n, n), dtype=bool)
    idx = {v: i for i, v in enumerate(G.nodes())}
    for u, v in G.edges():
        adj[idx[u], idx[v]] = True
        adj[idx[v], idx[u]] = True
    color = _base_pair_color(n, adj)
    max_rounds = max_rounds or (n * n)
    for _ in range(max_rounds):
        modulus = int(color.max()) + 1
        code = color[:, :, None].astype(np.int64) * modulus + color[None, :, :].astype(np.int64)
        sorted_code = np.sort(code, axis=1)
        feat = np.concatenate([color.reshape(n, n, 1), sorted_code.transpose(0, 2, 1)], axis=2).reshape(
            n * n, -1
        )
        _, new_color_flat = np.unique(feat, axis=0, return_inverse=True)
        new_color = new_color_flat.reshape(n, n)
        if np.array_equal(new_color, color):
            break
        color = new_color
    return color


def folklore_2wl_classification_embedding(G: nx.Graph, out_dim: int = 16) -> np.ndarray:
    """Canonical (deterministic, permutation-invariant) per-graph feature for
    classification (used for CSL only -- not part of the Task-1 BREC fix,
    which needs the exact pairwise test above instead): the sorted histogram
    of stable-colour class sizes, padded/truncated to out_dim."""
    color = folklore_2wl_single_coloring(G)
    _, counts = np.unique(color, return_counts=True)
    counts = np.sort(counts)[::-1].astype(np.float64)
    if len(counts) >= out_dim:
        return counts[:out_dim]
    return np.pad(counts, (0, out_dim - len(counts)))


# ---------------------------------------------------------------------------
# Edge-girth-sequence-only: non-learned, deterministic histogram embedding of
# the multiset {(g_e, lambda_e)}. Used to show that the *sequence alone* (no
# diffusion through message passing) is a much weaker signal (Section 2.2).
# ---------------------------------------------------------------------------
def edge_girth_sequence_embedding(girth_dict: dict, out_dim: int = 16) -> np.ndarray:
    vec = np.zeros(out_dim)
    rng_cache = {}
    for g_e, lam_e in girth_dict.values():
        key = (g_e if math.isfinite(g_e) else "inf", lam_e)
        if key not in rng_cache:
            rs = np.random.RandomState(abs(hash(key)) % (2**32))
            rng_cache[key] = rs.randn(out_dim)
        vec += rng_cache[key]
    return vec


class EdgeGirthSequenceOnlyModel:
    """Non-learned baseline: graph signature = hashed histogram of sigma(G)."""

    def __init__(self, girth_dicts_by_key: dict, out_dim: int = 16):
        self.girth_dicts_by_key = girth_dicts_by_key
        self.out_dim = out_dim
        self.training = False

    def eval(self):
        return self

    def train(self, *a):
        return self

    def parameters(self):
        return iter([])

    def to(self, device):
        return self

    def __call__(self, data):
        graphs = data.to_data_list() if hasattr(data, "to_data_list") else [data]
        embs = []
        for g in graphs:
            key = int(g.brec_key)
            girth_dict = self.girth_dicts_by_key[key]
            embs.append(edge_girth_sequence_embedding(girth_dict, self.out_dim))
        return torch.tensor(np.stack(embs), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Triangle-count-only edge feature (1-dim edge attribute), same EGAGNN-style
# gated architecture as the main model, to isolate "any single scalar edge
# invariant" from "edge-girth specifically".
# ---------------------------------------------------------------------------
def triangle_count_edge_attr(G: nx.Graph, edge_index) -> torch.Tensor:
    counts = {}
    for u, v in G.edges():
        counts[frozenset((u, v))] = len(set(G.neighbors(u)) & set(G.neighbors(v)))
    n_edges = edge_index.size(1)
    out = torch.zeros(n_edges, 1)
    for k in range(n_edges):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        out[k, 0] = counts.get(frozenset((u, v)), 0)
    return out


# ---------------------------------------------------------------------------
# Bounded cycle counts, PER EDGE, injected into EGAGNNRegressor's e0 slot
# . This is deliberately NOT called "GSN": GSN uses
# per-node orbit counts with its own GIN-style backbone (see GSNModel above);
# this is a bounded-k cycle-count-per-edge descriptor, plugged into the exact
# same architecture as edge-girth, so that architecture and access to bond
# types are held fixed and only the structural descriptor varies. GSN is the
# inspiration for the counting mechanism (nx.simple_cycles with a length
# bound), not the method being reproduced.
# ---------------------------------------------------------------------------
def bounded_cycle_count_edge_attr(G: nx.Graph, edge_index, k: int) -> torch.Tensor:
    """For each edge (u, v) and each cycle length j in 3..k, the number of
    simple cycles of length j that pass through that edge. Returns a tensor
    of shape (n_edges, k-2), column order 3, 4, ..., k."""
    dim = k - 2
    counts: dict[frozenset, list[int]] = {frozenset(e): [0] * dim for e in G.edges()}
    for cycle in nx.simple_cycles(G, length_bound=k):
        length = len(cycle)
        if length < 3 or length > k:
            continue
        idx = length - 3
        for i in range(length):
            u, v = cycle[i], cycle[(i + 1) % length]
            key = frozenset((u, v))
            if key in counts:
                counts[key][idx] += 1

    n_edges = edge_index.size(1)
    out = torch.zeros(n_edges, dim)
    for i in range(n_edges):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        vals = counts.get(frozenset((u, v)))
        if vals is not None:
            out[i] = torch.tensor(vals, dtype=torch.float32)
    return out
