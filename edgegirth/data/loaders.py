"""Dataset loaders for BREC, CSL, ZINC, plus smoke-mode subsampling.

BREC: loaded from the official brec_v3.npy (graph6-encoded, downloaded from
the official BREC dropbox link referenced in the `brec` PyPI package). 400
pairs across 6 categories, each pair relabeled 32x, plus a matched
"reliability" control group of isomorphic pairs offset by SAMPLE_NUM=400
pairs -- exactly the official BREC v3 layout.

CSL: torch_geometric.datasets.GNNBenchmarkDataset(name="CSL"). Note: this is
150 circulant-skip-link graphs over 10 skip-class labels (15 graphs/class),
not 15 classes.

ZINC: torch_geometric.datasets.ZINC(subset=True), the standard 12k subset
with train/val/test splits already defined by Dwivedi et al.
"""

from __future__ import annotations

import os
import random

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset
from torch_geometric.utils.convert import from_networkx

from edgegirth.paths import BREC_NPY, BREC_PYG_CACHE, DATA_ROOT

BREC_PART_DICT = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}
BREC_NUM_RELABEL = 32
BREC_SAMPLE_NUM = 400  # number of pairs in the main comparison group


def _graph6_to_pyg(g6_bytes) -> Data:
    G = nx.from_graph6_bytes(g6_bytes)
    data = from_networkx(G)
    data.x = torch.ones([data.num_nodes, 1])
    return data


def load_brec_raw() -> list[Data]:
    """Load all 51200 BREC graph entries (400 pairs x 32 relabel x 2 groups x 2 graphs).

    Cached to disk after the first graph6->PyG conversion (a ~60s one-off cost).
    """
    if os.path.exists(BREC_PYG_CACHE):
        return torch.load(BREC_PYG_CACHE, weights_only=False)
    if not os.path.exists(BREC_NPY):
        raise FileNotFoundError(
            f"BREC raw data not found at {BREC_NPY}. Run scripts/download_data.sh "
            "first, or set EDGEGIRTH_DATA_DIR to point at an existing download."
        )
    raw = np.load(BREC_NPY, allow_pickle=True)
    graphs = [_graph6_to_pyg(g6) for g6 in raw]
    os.makedirs(os.path.dirname(BREC_PYG_CACHE), exist_ok=True)
    torch.save(graphs, BREC_PYG_CACHE)
    return graphs


def select_smoke_pair_ids(n_per_category: int = 40, seed: int = 13) -> dict[str, list[int]]:
    """Sample up to n_per_category pair ids per BREC category (seed-controlled)."""
    rng = random.Random(seed)
    selected = {}
    for cat, (lo, hi) in BREC_PART_DICT.items():
        ids = list(range(lo, hi))
        k = min(n_per_category, len(ids))
        selected[cat] = sorted(rng.sample(ids, k))
    return selected


def load_csl(root: str | None = None) -> GNNBenchmarkDataset:
    """Load the 150-graph CSL dataset (downloaded/cached under `root`)."""
    root = root or os.path.join(DATA_ROOT, "CSL")
    return GNNBenchmarkDataset(root=root, name="CSL")


def load_zinc(subset: bool = True, root: str | None = None) -> tuple[ZINC, ZINC, ZINC]:
    """Load the official ZINC train/val/test splits (downloaded/cached under `root`)."""
    root = root or os.path.join(DATA_ROOT, "ZINC")
    train = ZINC(root=root, subset=subset, split="train")
    val = ZINC(root=root, subset=subset, split="val")
    test = ZINC(root=root, subset=subset, split="test")
    return train, val, test


def subsample_zinc_train(train_dataset: ZINC, n: int, seed: int = 13) -> ZINC:
    """Return a fixed-seed, without-replacement subsample of `n` training graphs."""
    rng = np.random.RandomState(seed)
    n = min(n, len(train_dataset))
    idx = rng.choice(len(train_dataset), size=n, replace=False)
    return train_dataset[torch.tensor(sorted(idx))]
