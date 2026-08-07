"""Descriptor-vs-descriptor study, architecture and bond types held fixed.

The central comparison the paper's introduction promises: same architecture
(EGAGNNRegressor), same access to ZINC's bond-type edge features, only the
structural descriptor injected in e0 changes. This isolates the descriptor
itself from confounds a naive GSN-baseline comparison carries (a different
architecture entirely, and GSN not consuming bond types at all -- see
`edgegirth.models.baselines.GSNModel`).

Six variants, all trained with EGAGNNRegressor, all keeping the bond one-hot:
  constant  no structural signal (zeroed [g,lambda,bridge] triple) -- reused
            verbatim from the existing frozen ablation (not retrained: same
            architecture, same regime, same numbers already exist there)
  tri       triangle count per edge (dim 1)
  cyc4      cycle counts per edge, lengths 3-4 (dim 2)
  cyc6      cycle counts per edge, lengths 3-6 (dim 4)
  cyc8      cycle counts per edge, lengths 3-8 (dim 6)
  full      (g_e, lambda_e) + bridge indicator (dim 3) -- reused verbatim from
            the existing frozen ablation, same reason as `constant`

Deliberately NOT called "GSN": GSN counts per-node orbits with its own
GIN-style architecture (see baselines.GSNModel); `bounded_cycle_count_edge_attr`
here counts cycles per EDGE and is injected into the exact same architecture
as edge-girth. GSN is the inspiration for the bounded-k enumeration mechanism,
not the method being reproduced.

Usage:
    python -m edgegirth.experiments.run_descriptor_study --mode smoke
    python -m edgegirth.experiments.run_descriptor_study --mode full
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
from edgegirth.experiments.run_zinc import ATOM_TYPES, _train_eval, match_param_budget
from edgegirth.features.featurize import data_to_simple_undirected_nx
from edgegirth.models.baselines import bounded_cycle_count_edge_attr, triangle_count_edge_attr
from edgegirth.paths import RESULTS_DIR

BOND_TYPES = 4
NEW_VARIANTS = ["tri", "cyc4", "cyc6", "cyc8"]  # variants this script actually trains
REUSED_VARIANTS = ["constant", "full"]  # pulled from the frozen ablation, not retrained
ALL_VARIANTS = ["constant", "tri", "cyc4", "cyc6", "cyc8", "full"]


def featurize_descriptor(data, G, variant):
    d = data.clone()
    x_ids = d.x.view(-1).long().clamp(max=ATOM_TYPES - 1)
    d.x = torch.nn.functional.one_hot(x_ids, ATOM_TYPES).float()
    bond_ids = d.edge_attr.view(-1).long().clamp(max=BOND_TYPES - 1)
    bond_onehot = torch.nn.functional.one_hot(bond_ids, BOND_TYPES).float()

    if variant == "tri":
        struct = triangle_count_edge_attr(G, d.edge_index)
    elif variant.startswith("cyc"):
        k = int(variant[3:])
        struct = bounded_cycle_count_edge_attr(G, d.edge_index, k=k)
    else:
        raise ValueError(f"featurize_descriptor only handles new variants, got {variant}")

    d.e0 = torch.cat([struct, bond_onehot], dim=1)
    d.edge_attr = d.e0
    return d


def featurize_split(graphs, variant, n_jobs=-1):
    def _job(data):
        G = data_to_simple_undirected_nx(data)
        return featurize_descriptor(data, G, variant)

    return Parallel(n_jobs=n_jobs)(delayed(_job)(d) for d in graphs)


def _cold_cache_preproc_seconds(train, val, test, variant, n_jobs):
    """Isolated, cold-by-construction timing (no disk cache exists for any of
    these descriptors), matching the methodology of
    run_zinc_cold_cache_preprocessing.py."""
    t0 = time.perf_counter()
    _ = featurize_split(train, variant, n_jobs=n_jobs)
    _ = featurize_split(val, variant, n_jobs=n_jobs)
    _ = featurize_split(test, variant, n_jobs=n_jobs)
    return time.perf_counter() - t0


def run_descriptor_study(mode: str = "smoke", variants=None, n_jobs: int = -1, seeds=None):
    variants = variants or list(NEW_VARIANTS)
    t_start = time.perf_counter()
    print(
        f"\n{'=' * 70}\n[DescriptorStudy] mode={mode}, variants={variants}, n_jobs={n_jobs}\n{'=' * 70}",
        flush=True,
    )
    train, val, test = load_zinc(subset=True)

    if mode == "smoke":
        seeds = seeds or [13]
        train = subsample_zinc_train(train, 2000, seed=13)
        val = subsample_zinc_train(val, 300, seed=13)
        test = subsample_zinc_train(test, 300, seed=13)
        epoch = 20
    else:
        # same regime as the frozen ablation (the project's development history/v7): full official
        # 10k/1k/1k, 200 epochs, seeds 13-16, ~100k parameter budget -- so the
        # reused `constant`/`full` rows are directly comparable to the newly
        # trained ones.
        seeds = seeds or [13, 14, 15, 16]
        epoch = 200

    node_dim = ATOM_TYPES
    PARAM_BUDGET = 100_000 if mode == "full" else None

    results_path = os.path.join(RESULTS_DIR, "descriptor_study_results.pkl")
    results = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "rb") as f:
                prior = pickle.load(f)
            if prior.get("mode") == mode and prior.get("seeds") == seeds and prior.get("epoch") == epoch:
                results = dict(prior.get("results", {}))
                if results:
                    print(
                        f"[DescriptorStudy/{mode}] found {len(results)} variant(s) already on disk for this "
                        f"regime: {list(results.keys())}. Running/overwriting: {variants}"
                    )
        except Exception as e:
            print(f"[DescriptorStudy/{mode}] could not read prior results.pkl ({e}); starting fresh.")

    for variant_i, variant in enumerate(variants, 1):
        t_pre = time.perf_counter()
        train_f = featurize_split(train, variant, n_jobs=n_jobs)
        val_f = featurize_split(val, variant, n_jobs=n_jobs)
        test_f = featurize_split(test, variant, n_jobs=n_jobs)
        preproc_time = time.perf_counter() - t_pre
        e0_dim = train_f[0].e0.size(1)

        if PARAM_BUDGET is not None:
            hidden_dim, n_params_matched = match_param_budget("egagnn", node_dim, e0_dim, PARAM_BUDGET)
        else:
            hidden_dim, n_params_matched = 64, None

        t_block = announce_block(
            f"DescriptorStudy variant {variant_i}/{len(variants)}: '{variant}' (e0_dim={e0_dim}, "
            f"{len(seeds)} seeds={seeds}, epoch={epoch}, hidden_dim={hidden_dim}"
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
        finish_block(f"DescriptorStudy variant '{variant}'", t_block)
        test_maes = [r["test_mae"] for r in seed_results]
        results[variant] = dict(
            seed_results=seed_results,
            seeds=seeds,
            mean_test_mae=float(np.mean(test_maes)),
            std_test_mae=float(np.std(test_maes)),
            n_params=seed_results[0]["n_params"],
            hidden_dim=hidden_dim,
            param_budget_target=PARAM_BUDGET,
            e0_dim=e0_dim,
            elapsed_seconds=elapsed,
            preprocessing_seconds=preproc_time,
        )
        print(
            f"[DescriptorStudy/{mode}] {variant}: "
            f"test_MAE={np.mean(test_maes):.4f} +/- {np.std(test_maes):.4f} "
            f"(seed MAEs: {[round(r['test_mae'], 4) for r in seed_results]}), "
            f"n_params={seed_results[0]['n_params']}, e0_dim={e0_dim}, "
            f"preproc={preproc_time:.1f}s, train_time={elapsed:.1f}s",
            flush=True,
        )

        os.makedirs(RESULTS_DIR, exist_ok=True)
        checkpoint_save(
            results_path,
            dict(
                mode=mode,
                seeds=seeds,
                epoch=epoch,
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
        results=results,
        total_elapsed_seconds=total_elapsed,
        variants_completed=list(results.keys()),
        variants_requested=variants,
        partial=False,
    )
    checkpoint_save(results_path, out)
    print(
        f"\n[DescriptorStudy/{mode}] ALL VARIANTS DONE. total time: {total_elapsed:.1f}s "
        f"({total_elapsed / 60:.1f} min). Saved to results/descriptor_study_results.pkl"
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--methods", "--variants", dest="variants", nargs="+", default=None)
    args = parser.parse_args()
    run_descriptor_study(mode=args.mode, variants=args.variants, n_jobs=args.n_jobs)
