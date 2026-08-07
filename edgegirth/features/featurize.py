"""Turns the raw (g_e, lambda_e) dict into the normalised edge feature e^(0)
of Equation (1): [g_tilde, lambda_tilde, 1{g_e = inf}] concatenated with any
dataset edge attributes.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

SENTINEL_G = 0.0
SENTINEL_LAMBDA = 0.0


def data_to_simple_undirected_nx(data: Data) -> nx.Graph:
    """Convert a PyG `Data` object to a simple (no parallel edges, no self-loops) undirected graph."""
    G = to_networkx(data, to_undirected=True)
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


class EdgeGirthNormalizer:
    """Mean/std normalisation for (g_e, lambda_e), fit on a set of graphs' edge-girth dicts."""

    def __init__(self) -> None:
        self.g_mean = 0.0
        self.g_std = 1.0
        self.lambda_mean = 0.0
        self.lambda_std = 1.0
        self.fitted = False

    def fit(self, girth_dicts: list[dict]) -> "EdgeGirthNormalizer":
        """Compute mean/std of g_e and lambda_e over every finite edge in `girth_dicts` (bridges excluded)."""
        finite_g, finite_lambda = [], []
        for d in girth_dicts:
            for g_e, lam_e in d.values():
                if math.isfinite(g_e):
                    finite_g.append(g_e)
                    finite_lambda.append(lam_e)
        if not finite_g:
            finite_g, finite_lambda = [0.0], [0.0]
        self.g_mean = float(np.mean(finite_g))
        self.g_std = float(np.std(finite_g)) or 1.0
        self.lambda_mean = float(np.mean(finite_lambda))
        self.lambda_std = float(np.std(finite_lambda)) or 1.0
        self.fitted = True
        return self

    def transform(self, g_e: float, lam_e: float) -> tuple[float, float, float]:
        """Normalise one (g_e, lambda_e) pair; returns (g_norm, lambda_norm, bridge_indicator)."""
        if not math.isfinite(g_e):
            return SENTINEL_G, SENTINEL_LAMBDA, 1.0
        g_norm = (g_e - self.g_mean) / self.g_std
        lam_norm = (lam_e - self.lambda_mean) / self.lambda_std
        return g_norm, lam_norm, 0.0


def build_edge_attr(
    data: Data,
    girth_dict: dict,
    normalizer: EdgeGirthNormalizer,
    feature_mode: str = "full",
    noise_seed: int | None = None,
) -> torch.Tensor:
    """Builds e^(0) for every directed edge in data.edge_index, in order.

    feature_mode selects the ablation variant:
      "full"      -> [g, lambda, bridge_indicator]
      "g_only"    -> [g, 0, bridge_indicator]
      "lambda_only" -> [0, lambda, bridge_indicator]
      "constant"  -> constant vector (same dim, all zeros)
      "noise"     -> iid Gaussian noise, same dim
    """
    ei = data.edge_index
    n_edges = ei.size(1)
    rows = []
    if feature_mode == "noise":
        g = torch.Generator().manual_seed(noise_seed if noise_seed is not None else 0)
        noise = torch.randn(n_edges, 3, generator=g)
        structural = noise
    else:
        for k in range(n_edges):
            u, v = int(ei[0, k]), int(ei[1, k])
            key = frozenset((u, v))
            g_e, lam_e = girth_dict.get(key, (math.inf, 0))
            g_norm, lam_norm, bridge = normalizer.transform(g_e, lam_e)
            if feature_mode == "g_only":
                lam_norm = 0.0
            elif feature_mode == "lambda_only":
                g_norm = 0.0
            elif feature_mode == "constant":
                g_norm, lam_norm, bridge = 0.0, 0.0, 0.0
            rows.append([g_norm, lam_norm, bridge])
        structural = torch.tensor(rows, dtype=torch.float32)

    return structural


def attach_edge_girth(
    data: Data,
    girth_dict: dict,
    normalizer: EdgeGirthNormalizer,
    feature_mode: str = "full",
    noise_seed: int | None = None,
    keep_dataset_attr: bool = True,
) -> Data:
    structural = build_edge_attr(data, girth_dict, normalizer, feature_mode, noise_seed)
    if keep_dataset_attr and getattr(data, "edge_attr", None) is not None:
        base_attr = data.edge_attr
        if base_attr.dim() == 1:
            base_attr = base_attr.view(-1, 1).float()
        else:
            base_attr = base_attr.float()
        e0 = torch.cat([structural, base_attr], dim=1)
    else:
        e0 = structural
    out = data.clone()
    out.e0 = e0  # raw e^(0), re-injected at every EGAGNN layer
    out.edge_attr = e0
    return out
