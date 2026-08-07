"""ZINC-12k graph regression experiment (test MAE).

Usage:
    python -m edgegirth.experiments.run_zinc --mode smoke
    python -m edgegirth.experiments.run_zinc --mode full
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from torch_geometric.loader import DataLoader

from edgegirth.data.loaders import load_zinc, subsample_zinc_train
from edgegirth.experiments.progress import announce_block, checkpoint_save, finish_block
from edgegirth.features.edge_girth import compute_edge_girth
from edgegirth.features.featurize import EdgeGirthNormalizer, data_to_simple_undirected_nx
from edgegirth.models.baselines import (
    GatedGCNMLPModel,
    GatedGCNModel,
    GCNModel,
    GINModel,
    GSNModel,
    attach_gsn_features_k,
    make_regressor,
    triangle_count_edge_attr,
)
from edgegirth.models.gnn import EGAGNNRegressor
from edgegirth.paths import RESULTS_DIR

ATOM_TYPES = 28  # generous upper bound for the ZINC-12k subset vocabulary
BOND_TYPES = 4
HIDDEN_DIM = 64
NUM_LAYERS = 4


def onehot_featurize(data, normalizer=None, girth_dict=None, structural=True):
    d = data.clone()
    x_ids = d.x.view(-1).long().clamp(max=ATOM_TYPES - 1)
    d.x = torch.nn.functional.one_hot(x_ids, ATOM_TYPES).float()
    bond_ids = d.edge_attr.view(-1).long().clamp(max=BOND_TYPES - 1)
    bond_onehot = torch.nn.functional.one_hot(bond_ids, BOND_TYPES).float()

    if structural:
        ei = d.edge_index
        rows = []
        for k in range(ei.size(1)):
            u, v = int(ei[0, k]), int(ei[1, k])
            g_e, lam_e = girth_dict.get(frozenset((u, v)), (float("inf"), 0))
            g_norm, lam_norm, bridge = normalizer.transform(g_e, lam_e)
            rows.append([g_norm, lam_norm, bridge])
        struct = torch.tensor(rows, dtype=torch.float32)
        d.e0 = torch.cat([struct, bond_onehot], dim=1)
    else:
        d.e0 = bond_onehot
    d.edge_attr = d.e0
    return d


def featurize_zinc_list(graphs, normalizer, n_jobs=-1):
    def _job(data):
        G = data_to_simple_undirected_nx(data)
        gd = compute_edge_girth(G)
        return onehot_featurize(data, normalizer, gd)

    return Parallel(n_jobs=n_jobs)(delayed(_job)(d) for d in graphs)


# ---------------------------------------------------------------------------
# Structural-baseline additions (Task 1, the project's development history): the existing table
# has no structural-edge-feature baseline other than edge-girth itself, so a
# reviewer can't tell whether the gain is "edge-girth" or "any structural
# edge feature." Adds GSN(k=4), GSN(k=6, the a-priori choice for molecular
# ring sizes.
#
# GSN(k) reuses the same GIN-backbone-plus-node-features design already used
# for BREC/CSL (edgegirth/models/baselines.py's GSNModel), which does NOT read
# edge_attr -- consistent with GCN/GIN/GatedGCN's existing precedent in this
# ZINC pipeline (see the project's development history: none of GCN/GIN/GatedGCN/GatedGCN-MLP
# actually consume bond-type edge features either, a pre-existing asymmetry
# with EGAGNN this task surfaces but does not fix, since fixing it would mean
# touching the frozen methods).
#
# triangle-count reuses the EGAGNN architecture with e0 = [triangle_count] +
# bond one-hot -- the same template as edge-girth's e0, differing only in the
# structural content, which is exactly the controlled comparison this task
# needs (architecture held fixed, structural feature swapped).
# ---------------------------------------------------------------------------
def onehot_featurize_triangle(data, G):
    d = data.clone()
    x_ids = d.x.view(-1).long().clamp(max=ATOM_TYPES - 1)
    d.x = torch.nn.functional.one_hot(x_ids, ATOM_TYPES).float()
    bond_ids = d.edge_attr.view(-1).long().clamp(max=BOND_TYPES - 1)
    bond_onehot = torch.nn.functional.one_hot(bond_ids, BOND_TYPES).float()
    tri = triangle_count_edge_attr(G, d.edge_index)
    d.e0 = torch.cat([tri, bond_onehot], dim=1)
    d.edge_attr = d.e0
    return d


def featurize_zinc_triangle(graphs, n_jobs=-1):
    def _job(data):
        G = data_to_simple_undirected_nx(data)
        return onehot_featurize_triangle(data, G)

    return Parallel(n_jobs=n_jobs)(delayed(_job)(d) for d in graphs)


def onehot_featurize_gsn(data, G, k):
    d = data.clone()
    x_ids = d.x.view(-1).long().clamp(max=ATOM_TYPES - 1)
    d.x = torch.nn.functional.one_hot(x_ids, ATOM_TYPES).float()
    return attach_gsn_features_k(d, G, k=k)


def featurize_zinc_gsn(graphs, k, n_jobs=-1):
    def _job(data):
        G = data_to_simple_undirected_nx(data)
        return onehot_featurize_gsn(data, G, k)

    return Parallel(n_jobs=n_jobs)(delayed(_job)(d) for d in graphs)


# ---------------------------------------------------------------------------
# GSN + bond types: fixes the confound flagged in
# the project's development history -- gsn4/gsn6 above never see ZINC's bond-type edge features,
# so their comparison against edgegirth (which does) mixes "structural descriptor"
# with "access to chemistry." gsn4_bonds/gsn6_bonds keep the exact same
# node-level GSN cycle counts, but additionally one-hot-encode the bond type
# into d.edge_attr and route it through GSNModel's new GINEConv path (see
# baselines.GSNModel's edge_dim argument) -- the minimal change that gives GSN
# access to the same chemistry as edgegirth without altering the rest of the
# architecture. gsn4/gsn6 (no bonds) are kept, unmodified, as the "before" for
# this comparison; the two are reported side by side, not replaced.
# ---------------------------------------------------------------------------
def onehot_featurize_gsn_bonds(data, G, k):
    d = onehot_featurize_gsn(data, G, k)
    bond_ids = d.edge_attr.view(-1).long().clamp(max=BOND_TYPES - 1)
    d.edge_attr = torch.nn.functional.one_hot(bond_ids, BOND_TYPES).float()
    return d


def featurize_zinc_gsn_bonds(graphs, k, n_jobs=-1):
    def _job(data):
        G = data_to_simple_undirected_nx(data)
        return onehot_featurize_gsn_bonds(data, G, k)

    return Parallel(n_jobs=n_jobs)(delayed(_job)(d) for d in graphs)


def build_model(method, node_dim, e0_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS):
    if method == "egagnn":
        return EGAGNNRegressor(node_dim, e0_dim, hidden_dim, num_layers)
    if method == "gcn":
        return make_regressor(GCNModel, node_dim, hidden_dim, num_layers)
    if method == "gin":
        return make_regressor(GINModel, node_dim, hidden_dim, num_layers)
    if method == "gatedgcn":
        return make_regressor(GatedGCNModel, node_dim, hidden_dim, num_layers)
    if method == "gatedgcn_mlp":
        return make_regressor(GatedGCNMLPModel, node_dim, hidden_dim, num_layers)
    if method == "triangle":
        return EGAGNNRegressor(node_dim, e0_dim, hidden_dim, num_layers)
    if method in ("gsn4", "gsn6"):
        k = int(method[3:])
        return make_regressor(GSNModel, node_dim, hidden_dim, num_layers, struct_dim=k - 2)
    if method in ("gsn4_bonds", "gsn6_bonds"):
        k = int(method[3 : method.index("_")])
        return make_regressor(
            GSNModel, node_dim, hidden_dim, num_layers, struct_dim=k - 2, edge_dim=BOND_TYPES
        )
    raise ValueError(method)


def match_param_budget(
    method, node_dim, e0_dim, target_params, num_layers=NUM_LAYERS, tol=0.10, lo=4, hi=1024
):
    """Binary-searches hidden_dim so the model's parameter count lands within
    +/-tol of target_params. Parameter count is monotone increasing in
    hidden_dim for all four architectures here, so binary search applies.
    Returns (hidden_dim, n_params)."""
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        n = build_model(method, node_dim, e0_dim, hidden_dim=mid, num_layers=num_layers).num_parameters()
        if best is None or abs(n - target_params) < abs(best[1] - target_params):
            best = (mid, n)
        if n < target_params * (1 - tol):
            lo = mid + 1
        elif n > target_params * (1 + tol):
            hi = mid - 1
        else:
            return mid, n
    return best


def _train_eval(
    method,
    train_graphs,
    val_graphs,
    test_graphs,
    node_dim,
    e0_dim,
    epoch,
    seed,
    batch_size=32,
    lr=1e-3,
    hidden_dim=HIDDEN_DIM,
    verbose_seed=None,
):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model = build_model(method, node_dim, e0_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)

    log_every = max(1, epoch // 10)  # ~10 progress lines, only for the designated verbose seed
    t_train = time.perf_counter()
    for ep in range(epoch):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for data in train_loader:
            optimizer.zero_grad()
            pred = model(data)
            loss = loss_fn(pred, data.y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if (
            verbose_seed is not None
            and seed == verbose_seed
            and ((ep + 1) % log_every == 0 or ep + 1 == epoch)
        ):
            elapsed = time.perf_counter() - t_train
            eta = elapsed / (ep + 1) * (epoch - ep - 1)
            print(
                f"    [{method}, seed {seed} (representative)] epoch {ep + 1}/{epoch}, "
                f"train_MAE~{epoch_loss / max(n_batches, 1):.4f}, elapsed={elapsed:.0f}s, ETA={eta:.0f}s",
                flush=True,
            )

    model.eval()

    def mae(graphs):
        loader = DataLoader(graphs, batch_size=64)
        errs, n = 0.0, 0
        with torch.no_grad():
            for data in loader:
                pred = model(data)
                errs += (pred - data.y.view(-1)).abs().sum().item()
                n += data.y.size(0)
        return errs / n

    return dict(test_mae=mae(test_graphs), val_mae=mae(val_graphs), n_params=model.num_parameters())


DEFAULT_ZINC_METHODS = ["egagnn", "gcn", "gin", "gatedgcn"]
# gatedgcn_mlp, triangle, gsn4, gsn6, gsn4_bonds, gsn6_bonds: addressable via
# --methods only, not run by default
ALL_ZINC_METHODS = [
    "egagnn",
    "gcn",
    "gin",
    "gatedgcn",
    "gatedgcn_mlp",
    "triangle",
    "gsn4",
    "gsn6",
    "gsn4_bonds",
    "gsn6_bonds",
]
STRUCTURAL_BASELINE_METHODS = {"triangle", "gsn4", "gsn6", "gsn4_bonds", "gsn6_bonds"}


def run_zinc(mode: str = "smoke", methods=None, n_jobs: int = -1, seeds=None):
    methods = methods or list(DEFAULT_ZINC_METHODS)
    t_start = time.perf_counter()
    print(f"\n{'=' * 70}\n[ZINC] mode={mode}, methods={methods}, n_jobs={n_jobs}\n{'=' * 70}", flush=True)
    train, val, test = load_zinc(subset=True)

    if mode == "smoke":
        seeds = seeds or [13]
        train = subsample_zinc_train(train, 2000, seed=13)
        val = subsample_zinc_train(val, 300, seed=13)
        test = subsample_zinc_train(test, 300, seed=13)
        epoch = 20
    else:
        seeds = seeds or [13, 14, 15, 16]
        epoch = 200

    node_dim = ATOM_TYPES
    needs_girth_pathway = any(m not in STRUCTURAL_BASELINE_METHODS for m in methods)
    train_f = val_f = test_f = None
    e0_dim = None
    preproc_time = 0.0

    if needs_girth_pathway:
        # unchanged from before this task -- the frozen methods' exact pathway
        fit_dicts = [
            compute_edge_girth(data_to_simple_undirected_nx(train[i])) for i in range(min(300, len(train)))
        ]
        normalizer = EdgeGirthNormalizer().fit(fit_dicts)

        t0 = time.perf_counter()
        train_f = featurize_zinc_list(list(train), normalizer, n_jobs=n_jobs)
        val_f = featurize_zinc_list(list(val), normalizer, n_jobs=n_jobs)
        test_f = featurize_zinc_list(list(test), normalizer, n_jobs=n_jobs)
        preproc_time = time.perf_counter() - t0
        print(
            f"[ZINC/{mode}] preprocessing (edge-girth, {len(train_f) + len(val_f) + len(test_f)} graphs): "
            f"{preproc_time:.1f}s"
        )
        e0_dim = train_f[0].e0.size(1)

    # per-method featurized datasets for the new structural-baseline methods
    # (Task 1/2, the project's development history); each caches its own cold-cache preprocessing time.
    structural_cache = {}
    structural_preproc_time = {}

    def get_featurized(method):
        if method not in STRUCTURAL_BASELINE_METHODS:
            return train_f, val_f, test_f, e0_dim
        if method in structural_cache:
            return structural_cache[method]
        t0 = time.perf_counter()
        if method == "triangle":
            tr = featurize_zinc_triangle(list(train), n_jobs=n_jobs)
            va = featurize_zinc_triangle(list(val), n_jobs=n_jobs)
            te = featurize_zinc_triangle(list(test), n_jobs=n_jobs)
        elif method.endswith("_bonds"):  # gsn4_bonds, gsn6_bonds
            k = int(method[3 : method.index("_")])
            tr = featurize_zinc_gsn_bonds(list(train), k, n_jobs=n_jobs)
            va = featurize_zinc_gsn_bonds(list(val), k, n_jobs=n_jobs)
            te = featurize_zinc_gsn_bonds(list(test), k, n_jobs=n_jobs)
        else:  # gsn4, gsn6
            k = int(method[3:])
            tr = featurize_zinc_gsn(list(train), k, n_jobs=n_jobs)
            va = featurize_zinc_gsn(list(val), k, n_jobs=n_jobs)
            te = featurize_zinc_gsn(list(test), k, n_jobs=n_jobs)
        m_preproc_time = time.perf_counter() - t0
        m_e0_dim = tr[0].e0.size(1) if hasattr(tr[0], "e0") else 0
        print(
            f"[ZINC/{mode}] preprocessing ({method}, {len(tr) + len(va) + len(te)} graphs): "
            f"{m_preproc_time:.1f}s",
            flush=True,
        )
        structural_preproc_time[method] = m_preproc_time
        structural_cache[method] = (tr, va, te, m_e0_dim)
        return tr, va, te, m_e0_dim

    PARAM_BUDGET = 100_000  # target for all ZINC-table methods, +/-10% (Task 2)
    results_path = os.path.join(RESULTS_DIR, "zinc_results.pkl")
    results = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "rb") as f:
                prior = pickle.load(f)
            if prior.get("mode") == mode and prior.get("seeds") == seeds and prior.get("epoch") == epoch:
                results = dict(prior.get("results", {}))
                if results:
                    print(
                        f"[ZINC/{mode}] found {len(results)} method(s) already on disk for this "
                        f"regime: {list(results.keys())}. Running/overwriting: {methods}"
                    )
        except Exception as e:
            print(f"[ZINC/{mode}] could not read prior results.pkl for merging ({e}); starting fresh.")

    for method_i, method in enumerate(methods, 1):
        m_train_f, m_val_f, m_test_f, m_e0_dim = get_featurized(method)
        matched_hidden_dim, matched_n_params = match_param_budget(method, node_dim, m_e0_dim, PARAM_BUDGET)
        t_block = announce_block(
            f"ZINC method {method_i}/{len(methods)}: '{method}' ({len(seeds)} seeds={seeds}, "
            f"epoch={epoch}, hidden_dim={matched_hidden_dim} -> {matched_n_params} params)"
        )
        t1 = time.perf_counter()
        seed_results = Parallel(n_jobs=min(n_jobs if n_jobs > 0 else os.cpu_count(), len(seeds)))(
            delayed(_train_eval)(
                method,
                m_train_f,
                m_val_f,
                m_test_f,
                node_dim,
                m_e0_dim,
                epoch,
                s,
                hidden_dim=matched_hidden_dim,
                verbose_seed=seeds[0],
            )
            for s in seeds
        )
        elapsed = time.perf_counter() - t1
        finish_block(f"ZINC method '{method}'", t_block)
        test_maes = [r["test_mae"] for r in seed_results]
        results[method] = dict(
            seed_results=seed_results,
            seeds=seeds,
            mean_test_mae=float(np.mean(test_maes)),
            std_test_mae=float(np.std(test_maes)),
            n_params=seed_results[0]["n_params"],
            hidden_dim=matched_hidden_dim,
            param_budget_target=PARAM_BUDGET,
            preprocessing_seconds=structural_preproc_time.get(method, preproc_time),
            elapsed_seconds=elapsed,
        )
        print(
            f"[ZINC/{mode}] {method}: test_MAE={np.mean(test_maes):.4f} +/- {np.std(test_maes):.4f} "
            f"(n_params={seed_results[0]['n_params']}, hidden_dim={matched_hidden_dim}), time={elapsed:.1f}s"
        )

        os.makedirs(RESULTS_DIR, exist_ok=True)
        checkpoint_save(
            results_path,
            dict(
                mode=mode,
                seeds=seeds,
                epoch=epoch,
                n_train=len(train),
                n_val=len(val),
                n_test=len(test),
                preprocessing_seconds=preproc_time,
                results=dict(results),
                total_elapsed_seconds=time.perf_counter() - t_start,
                methods_completed=list(results.keys()),
                methods_requested=methods,
                partial=(method_i < len(methods)),
            ),
        )

    total_elapsed = time.perf_counter() - t_start
    out = dict(
        mode=mode,
        seeds=seeds,
        epoch=epoch,
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        preprocessing_seconds=preproc_time,
        results=results,
        total_elapsed_seconds=total_elapsed,
        methods_completed=list(results.keys()),
        methods_requested=methods,
        partial=False,
    )
    checkpoint_save(results_path, out)
    print(
        f"\n[ZINC/{mode}] ALL METHODS DONE. total time: {total_elapsed:.1f}s "
        f"({total_elapsed / 60:.1f} min). Saved to results/zinc_results.pkl"
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--methods", nargs="+", default=None, choices=ALL_ZINC_METHODS)
    args = parser.parse_args()
    run_zinc(mode=args.mode, n_jobs=args.n_jobs, methods=args.methods)
