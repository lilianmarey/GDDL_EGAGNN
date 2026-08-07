"""EGAGNN: gated message passing on edge-girth features (Equations 1-4).

Task 4 adds four optional, independently-togglable architectural changes on
top of the original Eqs 1-4 (see the project's development history for the per-change ablation):
  4a. edge_readout            -- pool edge states into the graph embedding too
  4b. edge_conditioned_message -- message depends on the edge, not just phi_n(h_v)
  4c. use_batchnorm           -- BatchNorm1d + ReLU on node/edge updates
  4d. use_gin_eps             -- GIN-style (1+eps)*h + agg instead of h + agg
All default to False, reproducing the original architecture exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import scatter as pyg_scatter


def scatter_add(src, index, dim=0, dim_size=None):
    return pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")


def mlp(in_dim, out_dim, hidden=None):
    hidden = hidden or out_dim
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class EGAGNNLayer(nn.Module):
    """One layer of Equations (2)-(4), plus the optional Task-4 changes.

    Original:
      m_uv = sigmoid(phi_e(e_uv)) * phi_n(h_v)     -- message v -> u
      h_u  = h_u + AGG_v({m_uv})                   -- residual aggregation
      e_uv = psi(e0_uv || e_uv || h_u || h_v)       -- e0 re-injected every layer

    4b (edge_conditioned_message): phi_n(h_v) -> phi_m([h_v || e_uv]) -- the
    message depends on which edge it travels along, not just its source node.
    The sigmoid gate from phi_e(e_uv) is kept unchanged (it's the paper's
    architectural contribution and Proposition 1 relies on it).

    4c (use_batchnorm): BatchNorm1d + ReLU on both updates.

    4d (use_gin_eps): h_new = mlp((1+eps)*h + agg) instead of a plain residual.
    """

    def __init__(
        self,
        hidden_dim: int,
        e0_dim: int,
        edge_conditioned_message: bool = False,
        use_batchnorm: bool = False,
        use_gin_eps: bool = False,
    ):
        super().__init__()
        self.edge_conditioned_message = edge_conditioned_message
        self.use_batchnorm = use_batchnorm
        self.use_gin_eps = use_gin_eps

        self.phi_e = mlp(hidden_dim, hidden_dim)
        if edge_conditioned_message:
            self.phi_m = mlp(2 * hidden_dim, hidden_dim)
        else:
            self.phi_n = mlp(hidden_dim, hidden_dim)
        self.psi = mlp(e0_dim + hidden_dim + 2 * hidden_dim, hidden_dim)

        if use_gin_eps:
            self.eps = nn.Parameter(torch.zeros(1))
            self.mlp_upd = mlp(hidden_dim, hidden_dim)

        if use_batchnorm:
            self.bn_h = nn.BatchNorm1d(hidden_dim)
            self.bn_e = nn.BatchNorm1d(hidden_dim)

    def forward(self, h, e, e0, edge_index):
        src, dst = edge_index[0], edge_index[1]  # message sent from v=src to u=dst
        gate = torch.sigmoid(self.phi_e(e))
        if self.edge_conditioned_message:
            msg = gate * self.phi_m(torch.cat([h[src], e], dim=1))
        else:
            msg = gate * self.phi_n(h[src])
        agg = scatter_add(msg, dst, dim=0, dim_size=h.size(0))

        if self.use_gin_eps:
            h_new = self.mlp_upd((1 + self.eps) * h + agg)
        else:
            h_new = h + agg
        if self.use_batchnorm:
            h_new = torch.relu(self.bn_h(h_new))

        e_in = torch.cat([e0, e, h_new[dst], h_new[src]], dim=1)
        e_new = self.psi(e_in)
        if self.use_batchnorm:
            e_new = torch.relu(self.bn_e(e_new))
        return h_new, e_new


class EGAGNN(nn.Module):
    def __init__(
        self,
        node_in_dim: int,
        e0_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        out_dim: int = 16,
        readout: str = "add",
        edge_readout: bool = False,
        edge_conditioned_message: bool = False,
        use_batchnorm: bool = False,
        use_gin_eps: bool = False,
    ):
        super().__init__()
        self.edge_readout = edge_readout
        self.node_embed = nn.Linear(node_in_dim, hidden_dim)
        self.edge_embed = nn.Linear(e0_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                EGAGNNLayer(
                    hidden_dim,
                    e0_dim,
                    edge_conditioned_message=edge_conditioned_message,
                    use_batchnorm=use_batchnorm,
                    use_gin_eps=use_gin_eps,
                )
                for _ in range(num_layers)
            ]
        )
        if readout == "mean":
            self.readout_fn = global_mean_pool
        elif readout == "add":
            self.readout_fn = global_add_pool
        else:
            raise ValueError(f"unknown readout: {readout!r}")
        readout_in_dim = 2 * hidden_dim if edge_readout else hidden_dim
        self.out = nn.Linear(readout_in_dim, out_dim)

    def forward(self, data):
        x, edge_index, e0, batch = data.x, data.edge_index, data.e0, data.batch
        h = self.node_embed(x)
        e = self.edge_embed(e0)
        for layer in self.layers:
            h, e = layer(h, e, e0, edge_index)
        g_nodes = self.readout_fn(h, batch)
        if self.edge_readout:
            # edge_index holds both directions of every undirected edge, so this
            # pools each edge twice -- a constant factor of 2, harmless under a
            # sum/mean readout, documented rather than de-duplicated for simplicity.
            edge_batch = batch[edge_index[0]]
            if self.readout_fn is global_mean_pool:
                g_edges = global_mean_pool(e, edge_batch, size=g_nodes.size(0))
            else:
                g_edges = global_add_pool(e, edge_batch, size=g_nodes.size(0))
            g = torch.cat([g_nodes, g_edges], dim=1)
        else:
            g = g_nodes
        return self.out(g)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


class EGAGNNRegressor(nn.Module):
    """EGAGNN backbone + scalar regression head, for ZINC."""

    def __init__(
        self, node_in_dim: int, e0_dim: int, hidden_dim: int = 64, num_layers: int = 4, **edgegirth_kwargs
    ):
        super().__init__()
        self.backbone = EGAGNN(
            node_in_dim, e0_dim, hidden_dim, num_layers, out_dim=hidden_dim, **edgegirth_kwargs
        )
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, data):
        g = self.backbone(data)
        return self.head(g).squeeze(-1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
