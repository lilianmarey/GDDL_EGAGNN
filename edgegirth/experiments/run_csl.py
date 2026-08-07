"""CSL (Circulant Skip Link) 5-fold classification experiment.

CSL is 150 graphs over 10 skip-value classes (15 graphs/class) -- see
the project's development history for a note on the task brief's "15 classes" vs. the actual dataset.

Usage:
    python -m edgegirth.experiments.run_csl --mode smoke
    python -m edgegirth.experiments.run_csl --mode full
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

from edgegirth.data.loaders import load_csl
from edgegirth.experiments.progress import announce_block, checkpoint_save, finish_block
from edgegirth.features.edge_girth import compute_edge_girth
from edgegirth.features.featurize import EdgeGirthNormalizer, attach_edge_girth, data_to_simple_undirected_nx
from edgegirth.models.baselines import (
    GatedGCNModel,
    GCNModel,
    GINModel,
    GSNModel,
    PPGNModel,
    attach_gsn_features,
    edge_girth_sequence_embedding,
    folklore_2wl_classification_embedding,
)
from edgegirth.models.gnn import EGAGNN
from edgegirth.paths import RESULTS_DIR

NUM_CLASSES = 10
OUT_DIM = 32
HIDDEN_DIM = 32
NUM_LAYERS = 3


class ClassifierHead(nn.Module):
    def __init__(self, backbone, out_dim=OUT_DIM, n_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(out_dim, n_classes)

    def forward(self, data):
        emb = self.backbone(data)
        if not torch.is_tensor(emb):
            emb = torch.as_tensor(emb)
        return self.head(emb)


def build_model(method, node_dim, e0_dim):
    if method == "egagnn":
        return ClassifierHead(EGAGNN(node_dim, e0_dim, HIDDEN_DIM, NUM_LAYERS, OUT_DIM))
    if method == "gcn":
        return ClassifierHead(GCNModel(node_dim, HIDDEN_DIM, NUM_LAYERS, OUT_DIM))
    if method == "gin":
        return ClassifierHead(GINModel(node_dim, HIDDEN_DIM, NUM_LAYERS, OUT_DIM))
    if method == "gatedgcn":
        return ClassifierHead(GatedGCNModel(node_dim, HIDDEN_DIM, NUM_LAYERS, OUT_DIM))
    if method == "gsn":
        return ClassifierHead(GSNModel(node_dim, HIDDEN_DIM, NUM_LAYERS, OUT_DIM))
    if method == "ppgn":
        return ClassifierHead(PPGNModel(node_dim, HIDDEN_DIM // 2, num_layers=2, out_dim=OUT_DIM))
    raise ValueError(method)


def featurize_csl(dataset, method, normalizer):
    out = []
    for data in dataset:
        data = data.clone()
        data.x = torch.ones(data.num_nodes, 1)
        G = data_to_simple_undirected_nx(data)
        if method == "egagnn":
            girth_dict = compute_edge_girth(G)
            data = attach_edge_girth(
                data, girth_dict, normalizer, feature_mode="full", keep_dataset_attr=False
            )
        elif method == "gsn":
            data = attach_gsn_features(data, G)
        out.append(data)
    return out


def _run_fold(method, train_idx, test_idx, feat_graphs, labels, node_dim, e0_dim, epoch, seed, fold_idx):
    """Trains/evaluates one fold. Returns (accuracy, loss_curve), where
    loss_curve is the per-epoch training loss for fold 0 of `ppgn` only
    (None otherwise) -- used by `run_csl` to reconstruct the reliability
    check (does training loss ever drop below chance-level cross-entropy?)
    without a separate diagnostic script."""
    torch.set_num_threads(1)
    torch.manual_seed(seed + fold_idx)
    if method == "folklore2wl":
        embs = np.stack(
            [
                folklore_2wl_classification_embedding(data_to_simple_undirected_nx(g), OUT_DIM)
                for g in feat_graphs
            ]
        )
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=2000).fit(embs[train_idx], labels[train_idx])
        acc = clf.score(embs[test_idx], labels[test_idx])
        return acc, None
    if method == "edge_girth_seq":
        embs = []
        for g in feat_graphs:
            G = data_to_simple_undirected_nx(g)
            gd = compute_edge_girth(G)
            embs.append(edge_girth_sequence_embedding(gd, OUT_DIM))
        embs = np.stack(embs)
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(max_iter=2000).fit(embs[train_idx], labels[train_idx])
        acc = clf.score(embs[test_idx], labels[test_idx])
        return acc, None

    model = build_model(method, node_dim, e0_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    y = torch.tensor(labels, dtype=torch.long)
    train_graphs = [feat_graphs[i] for i in train_idx]
    train_y = y[train_idx]
    test_graphs = [feat_graphs[i] for i in test_idx]
    test_y = y[test_idx]

    for g, lab in zip(train_graphs, train_y):
        g.y = lab.view(1)
    for g, lab in zip(test_graphs, test_y):
        g.y = lab.view(1)

    loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    model.train()
    log_every = max(1, epoch // 10)  # ~10 progress lines per fold when epoch counts are large
    t_fold = time.perf_counter()
    track_loss = method == "ppgn" and fold_idx == 0
    loss_curve = [] if track_loss else None
    for ep in range(epoch):
        epoch_loss, n_batches = 0.0, 0
        for data in loader:
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        mean_epoch_loss = epoch_loss / max(n_batches, 1)
        if track_loss:
            loss_curve.append(mean_epoch_loss)
        if fold_idx == 0 and ((ep + 1) % log_every == 0 or ep + 1 == epoch):
            elapsed = time.perf_counter() - t_fold
            eta = elapsed / (ep + 1) * (epoch - ep - 1)
            print(
                f"    [{method}, fold 0 (representative)] epoch {ep + 1}/{epoch}, "
                f"loss={mean_epoch_loss:.4f}, elapsed={elapsed:.0f}s, ETA={eta:.0f}s",
                flush=True,
            )

    model.eval()
    test_loader = DataLoader(test_graphs, batch_size=32)
    correct, total = 0, 0
    with torch.no_grad():
        for data in test_loader:
            pred = model(data).argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return correct / total, loss_curve


ALL_CSL_METHODS = ["egagnn", "gcn", "gin", "gatedgcn", "gsn", "ppgn", "folklore2wl", "edge_girth_seq"]


def _build_unreliable_methods_note(results: dict) -> dict[str, str]:
    """Reconstructs the CSL reliability caveats for `ppgn` and `folklore2wl`.

    `ppgn` is flagged only if this run's own fold-0 training loss never drops
    below the chance-level cross-entropy (a runtime check on this run's
    numbers, not a restated claim). `folklore2wl` is flagged unconditionally:
    its CSL readout is a lossy per-graph summary fed to a classifier, not the
    exact pairwise 2-FWL test this codebase uses correctly on BREC -- a
    structural limitation of the method's use on this dataset, not something
    a particular run's numbers could pass or fail.
    """
    notes = {}
    if "ppgn" in results and results["ppgn"].get("loss_curve_fold0"):
        curve = results["ppgn"]["loss_curve_fold0"]
        chance_entropy = math.log(NUM_CLASSES)
        if min(curve) >= chance_entropy:
            notes["ppgn"] = (
                f"Training loss moves ({curve[0]:.1f} -> {curve[-1]:.2f} over {len(curve)} epochs) "
                f"but plateaus at/above the chance-level cross-entropy (ln({NUM_CLASSES})="
                f"{chance_entropy:.3f}) -- the model trains but does not learn a discriminative "
                f"representation on CSL. Accuracy ({results['ppgn']['mean_acc']:.3f}) has no "
                "evidential value."
            )
    if "folklore2wl" in results:
        notes["folklore2wl"] = (
            "CSL uses folklore_2wl_classification_embedding: a canonical per-graph embedding "
            "(sorted histogram of stable-colour class sizes) fed to logistic regression. This is "
            "a lossy summary of the 2-FWL colouring, not the exact pairwise 2-FWL test used "
            "correctly for BREC -- it is not a valid 3-WL-equivalent classifier, so its CSL "
            f"accuracy ({results['folklore2wl']['mean_acc']:.3f}) does not reflect 3-WL power and "
            "should not be read as a 3-WL benchmark result."
        )
    return notes


def run_csl(mode: str = "smoke", methods=None, n_jobs: int = -1, seed: int = 13):
    methods = methods or list(ALL_CSL_METHODS)
    t_start = time.perf_counter()
    print(f"\n{'=' * 70}\n[CSL] mode={mode}, methods={methods}, n_jobs={n_jobs}\n{'=' * 70}", flush=True)
    dataset = load_csl()
    labels = np.array([int(d.y) for d in dataset])
    node_dim = 1
    epoch = 5 if mode == "smoke" else 100
    n_splits = 5

    fit_dicts = [compute_edge_girth(data_to_simple_undirected_nx(d)) for d in dataset]
    normalizer = EdgeGirthNormalizer().fit(fit_dicts)

    results_path = os.path.join(RESULTS_DIR, "csl_results.pkl")
    results = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "rb") as f:
                prior = pickle.load(f)
            if prior.get("mode") == mode and prior.get("epoch") == epoch:
                results = dict(prior.get("results", {}))
                if results:
                    print(
                        f"[CSL/{mode}] found {len(results)} method(s) already on disk for this "
                        f"regime: {list(results.keys())}. Running/overwriting: {methods}"
                    )
        except Exception as e:
            print(f"[CSL/{mode}] could not read prior results.pkl for merging ({e}); starting fresh.")

    for method_i, method in enumerate(methods, 1):
        t_block = announce_block(
            f"CSL method {method_i}/{len(methods)}: '{method}' ({n_splits}-fold, epoch={epoch})"
        )
        t0 = time.perf_counter()
        feat_graphs = (
            featurize_csl(dataset, method, normalizer)
            if method not in ("folklore2wl", "edge_girth_seq")
            else list(dataset)
        )
        preproc_time = time.perf_counter() - t0
        e0_dim = feat_graphs[0].e0.size(1) if hasattr(feat_graphs[0], "e0") else 0

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(skf.split(np.zeros(len(labels)), labels))

        t1 = time.perf_counter()
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(_run_fold)(method, tr, te, feat_graphs, labels, node_dim, e0_dim, epoch, seed, i)
            for i, (tr, te) in enumerate(folds)
        )
        accs = [acc for acc, _ in fold_results]
        elapsed = time.perf_counter() - t1
        finish_block(f"CSL method '{method}'", t_block)

        results[method] = dict(
            fold_accuracies=accs,
            mean_acc=float(np.mean(accs)),
            std_acc=float(np.std(accs)),
            elapsed_seconds=elapsed,
            preprocessing_seconds=preproc_time,
        )
        if method == "ppgn":
            loss_curve = fold_results[0][1]  # fold 0's tracked loss curve
            results[method]["loss_curve_fold0"] = loss_curve
        print(
            f"[CSL/{mode}] {method}: acc={np.mean(accs):.3f} +/- {np.std(accs):.3f} "
            f"(chance=1/{NUM_CLASSES}={1 / NUM_CLASSES:.3f}), time={elapsed:.1f}s"
        )

        os.makedirs(RESULTS_DIR, exist_ok=True)
        checkpoint_save(
            results_path,
            dict(
                mode=mode,
                seed=seed,
                n_splits=n_splits,
                epoch=epoch,
                num_classes=NUM_CLASSES,
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
        seed=seed,
        n_splits=n_splits,
        epoch=epoch,
        num_classes=NUM_CLASSES,
        results=results,
        total_elapsed_seconds=total_elapsed,
        methods_completed=list(results.keys()),
        methods_requested=methods,
        partial=False,
        unreliable_methods=_build_unreliable_methods_note(results),
    )
    checkpoint_save(results_path, out)
    print(
        f"\n[CSL/{mode}] ALL METHODS DONE. total time: {total_elapsed:.1f}s "
        f"({total_elapsed / 60:.1f} min). Saved to results/csl_results.pkl"
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--methods", nargs="+", default=None, choices=ALL_CSL_METHODS)
    args = parser.parse_args()
    run_csl(mode=args.mode, n_jobs=args.n_jobs, methods=args.methods)
