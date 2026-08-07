"""ZINC ablation: same EGAGNN architecture, e^(0) structural part swapped.

Variants (dataset edge attributes -- bond one-hot -- always kept, only the
structural triple changes):
  full        [g, lambda, bridge]
  g_only      [g, 0, bridge]
  lambda_only [0, lambda, bridge]
  constant    [0, 0, 0]              -- isolates the gated architecture alone
  noise       iid Gaussian, same dim -- isolates the gated architecture alone

Usage:
    python -m edgegirth.experiments.run_ablation --mode smoke
    python -m edgegirth.experiments.run_ablation --mode full
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import torch
from joblib import Parallel, delayed

from edgegirth.data.loaders import load_zinc, subsample_zinc_train
from edgegirth.experiments.progress import announce_block, checkpoint_save, finish_block
from edgegirth.experiments.run_zinc import ATOM_TYPES, HIDDEN_DIM, _train_eval, match_param_budget
from edgegirth.features.edge_girth import compute_edge_girth
from edgegirth.features.featurize import EdgeGirthNormalizer, data_to_simple_undirected_nx
from edgegirth.paths import RESULTS_DIR

BOND_TYPES = 4
VARIANTS = ["full", "g_only", "lambda_only", "constant", "noise"]


def featurize_variant(data, normalizer, girth_dict, variant, noise_seed):
    d = data.clone()
    x_ids = d.x.view(-1).long().clamp(max=ATOM_TYPES - 1)
    d.x = torch.nn.functional.one_hot(x_ids, ATOM_TYPES).float()
    bond_ids = d.edge_attr.view(-1).long().clamp(max=BOND_TYPES - 1)
    bond_onehot = torch.nn.functional.one_hot(bond_ids, BOND_TYPES).float()

    ei = d.edge_index
    n_edges = ei.size(1)
    if variant == "noise":
        g = torch.Generator().manual_seed(noise_seed)
        struct = torch.randn(n_edges, 3, generator=g)
    else:
        rows = []
        for k in range(n_edges):
            u, v = int(ei[0, k]), int(ei[1, k])
            g_e, lam_e = girth_dict.get(frozenset((u, v)), (float("inf"), 0))
            g_norm, lam_norm, bridge = normalizer.transform(g_e, lam_e)
            if variant == "g_only":
                lam_norm = 0.0
            elif variant == "lambda_only":
                g_norm = 0.0
            elif variant == "constant":
                g_norm, lam_norm, bridge = 0.0, 0.0, 0.0
            rows.append([g_norm, lam_norm, bridge])
        struct = torch.tensor(rows, dtype=torch.float32)

    d.e0 = torch.cat([struct, bond_onehot], dim=1)
    d.edge_attr = d.e0
    return d


def run_ablation(mode: str = "smoke", variants=None, n_jobs: int = -1, seeds=None):
    variants = variants or VARIANTS
    t_start = time.perf_counter()
    print(
        f"\n{'=' * 70}\n[Ablation] mode={mode}, variants={variants}, n_jobs={n_jobs}\n{'=' * 70}", flush=True
    )
    train, val, test = load_zinc(subset=True)

    if mode == "smoke":
        seeds = seeds or [13]
        train = subsample_zinc_train(train, 2000, seed=13)
        val = subsample_zinc_train(val, 300, seed=13)
        test = subsample_zinc_train(test, 300, seed=13)
        epoch = 20
    else:
        # full-scale regime matching the ZINC full run exactly (the project's development history
        # Task 1): full official 10k/1k/1k, 200 epochs, seeds 13-16, ~100k
        # parameter budget. Variant definitions themselves are untouched.
        seeds = seeds or [13, 14, 15, 16]
        epoch = 200

    fit_dicts = [
        compute_edge_girth(data_to_simple_undirected_nx(train[i])) for i in range(min(300, len(train)))
    ]
    normalizer = EdgeGirthNormalizer().fit(fit_dicts)

    def _girth(d):
        return compute_edge_girth(data_to_simple_undirected_nx(d))

    print(
        f"[Ablation/{mode}] computing edge-girth for {len(train) + len(val) + len(test)} graphs...",
        flush=True,
    )
    t0 = time.perf_counter()
    train_gd = Parallel(n_jobs=n_jobs)(delayed(_girth)(d) for d in train)
    val_gd = Parallel(n_jobs=n_jobs)(delayed(_girth)(d) for d in val)
    test_gd = Parallel(n_jobs=n_jobs)(delayed(_girth)(d) for d in test)
    preproc_time = time.perf_counter() - t0
    print(f"[Ablation/{mode}] edge-girth preprocessing: {preproc_time:.1f}s", flush=True)

    node_dim = ATOM_TYPES
    PARAM_BUDGET = 100_000 if mode == "full" else None

    results_path = os.path.join(RESULTS_DIR, "ablation_results.pkl")
    results = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "rb") as f:
                prior = pickle.load(f)
            if prior.get("mode") == mode and prior.get("seeds") == seeds and prior.get("epoch") == epoch:
                results = dict(prior.get("results", {}))
                if results:
                    print(
                        f"[Ablation/{mode}] found {len(results)} variant(s) already on disk for this "
                        f"regime: {list(results.keys())}. Running/overwriting: {variants}"
                    )
        except Exception as e:
            print(f"[Ablation/{mode}] could not read prior results.pkl for merging ({e}); starting fresh.")

    for variant_i, variant in enumerate(variants, 1):
        train_f = [
            featurize_variant(d, normalizer, gd, variant, noise_seed=13 + i)
            for i, (d, gd) in enumerate(zip(train, train_gd))
        ]
        val_f = [
            featurize_variant(d, normalizer, gd, variant, noise_seed=1013 + i)
            for i, (d, gd) in enumerate(zip(val, val_gd))
        ]
        test_f = [
            featurize_variant(d, normalizer, gd, variant, noise_seed=2013 + i)
            for i, (d, gd) in enumerate(zip(test, test_gd))
        ]
        e0_dim = train_f[0].e0.size(1)

        if PARAM_BUDGET is not None:
            hidden_dim, n_params_matched = match_param_budget("egagnn", node_dim, e0_dim, PARAM_BUDGET)
        else:
            hidden_dim, n_params_matched = HIDDEN_DIM, None

        t_block = announce_block(
            f"Ablation variant {variant_i}/{len(variants)}: '{variant}' ({len(seeds)} seeds={seeds}, "
            f"epoch={epoch}, hidden_dim={hidden_dim}"
            + (f" -> {n_params_matched} params" if n_params_matched else "")
            + ")"
        )
        t1 = time.perf_counter()
        seed_results = Parallel(n_jobs=min(n_jobs if n_jobs > 0 else os.cpu_count(), len(seeds)))(
            delayed(_train_eval)(
                "egagnn",
                train_f,
                val_f,
                test_f,
                node_dim,
                e0_dim,
                epoch,
                s,
                hidden_dim=hidden_dim,
                verbose_seed=seeds[0],
            )
            for s in seeds
        )
        elapsed = time.perf_counter() - t1
        finish_block(f"Ablation variant '{variant}'", t_block)
        test_maes = [r["test_mae"] for r in seed_results]
        results[variant] = dict(
            seed_results=seed_results,
            seeds=seeds,
            mean_test_mae=float(np.mean(test_maes)),
            std_test_mae=float(np.std(test_maes)),
            n_params=seed_results[0]["n_params"],
            hidden_dim=hidden_dim,
            param_budget_target=PARAM_BUDGET,
            elapsed_seconds=elapsed,
        )
        print(
            f"[Ablation/{mode}] {variant}: test_MAE={np.mean(test_maes):.4f} +/- {np.std(test_maes):.4f} "
            f"(seed MAEs: {[round(r['test_mae'], 4) for r in seed_results]}), "
            f"n_params={seed_results[0]['n_params']}, time={elapsed:.1f}s",
            flush=True,
        )

        os.makedirs(RESULTS_DIR, exist_ok=True)
        checkpoint_save(
            results_path,
            dict(
                mode=mode,
                seeds=seeds,
                epoch=epoch,
                preprocessing_seconds=preproc_time,
                results=dict(results),
                total_elapsed_seconds=time.perf_counter() - t_start,
                variants_completed=list(results.keys()),
                variants_requested=variants,
                partial=(variant_i < len(variants)),
            ),
        )

    total_elapsed = time.perf_counter() - t_start
    out = dict(
        mode=mode,
        seeds=seeds,
        epoch=epoch,
        preprocessing_seconds=preproc_time,
        results=results,
        total_elapsed_seconds=total_elapsed,
        variants_completed=list(results.keys()),
        variants_requested=variants,
        partial=False,
    )
    checkpoint_save(results_path, out)
    print(
        f"\n[Ablation/{mode}] ALL VARIANTS DONE. total time: {total_elapsed:.1f}s "
        f"({total_elapsed / 60:.1f} min). Saved to results/ablation_results.pkl"
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()
    run_ablation(mode=args.mode, n_jobs=args.n_jobs)
