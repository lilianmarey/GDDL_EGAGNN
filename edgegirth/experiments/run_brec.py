"""BREC isomorphism-comparison experiment, official RPC protocol.

Usage:
    python -m edgegirth.experiments.run_brec --mode smoke
    python -m edgegirth.experiments.run_brec --mode full
"""

from __future__ import annotations

import argparse
import os
import pickle
import time

import torch
import torch_geometric
from joblib import delayed

from edgegirth.data.loaders import (
    BREC_NUM_RELABEL,
    BREC_PART_DICT,
    load_brec_raw,
    select_smoke_pair_ids,
)
from edgegirth.eval.brec_rpc import evaluate_pair
from edgegirth.experiments.progress import (
    announce_block,
    checkpoint_save,
    estimate_full_time,
    finish_block,
    run_parallel_with_progress,
)
from edgegirth.features.edge_girth import compute_edge_girth_batch
from edgegirth.features.featurize import (
    EdgeGirthNormalizer,
    attach_edge_girth,
    data_to_simple_undirected_nx,
)
from edgegirth.models.baselines import (
    EdgeGirthSequenceOnlyModel,
    GatedGCNModel,
    GCNModel,
    GINModel,
    GSNModel,
    PPGNModel,
    attach_gsn_features,
    folklore_2wl_distinguishable,
    triangle_count_edge_attr,
)
from edgegirth.models.gnn import EGAGNN
from edgegirth.paths import RESULTS_DIR

OUT_DIM = 16
HIDDEN_DIM = 32
NUM_LAYERS = 3


def _build_model_factory(method: str, e0_dim: int, node_dim: int = 1, extra=None):
    if method == "egagnn":
        return lambda: EGAGNN(node_dim, e0_dim, HIDDEN_DIM, NUM_LAYERS, OUT_DIM)
    if method == "triangle":
        return lambda: EGAGNN(node_dim, 1, HIDDEN_DIM, NUM_LAYERS, OUT_DIM)
    if method == "gcn":
        return lambda: GCNModel(node_dim, HIDDEN_DIM, NUM_LAYERS, OUT_DIM)
    if method == "gin":
        return lambda: GINModel(node_dim, HIDDEN_DIM, NUM_LAYERS, OUT_DIM)
    if method == "gatedgcn":
        return lambda: GatedGCNModel(node_dim, HIDDEN_DIM, NUM_LAYERS, OUT_DIM)
    if method == "gsn":
        return lambda: GSNModel(node_dim, HIDDEN_DIM, NUM_LAYERS, OUT_DIM)
    if method == "ppgn":
        return lambda: PPGNModel(node_dim, HIDDEN_DIM // 2, num_layers=2, out_dim=OUT_DIM)
    if method == "edge_girth_seq":
        return lambda: EdgeGirthSequenceOnlyModel(extra, OUT_DIM)
    raise ValueError(method)


def featurize_brec_subset(
    subset_graphs, nx_graphs, girth_dicts, method: str, normalizer: EdgeGirthNormalizer
):
    """Attaches whatever per-method structural features are needed to a
    pre-selected subset of graphs (nx_graphs / girth_dicts precomputed once,
    in parallel, and shared across methods). Returns (feat_graphs, girth_dicts_by_key)."""
    girth_dicts_by_key = {}
    out = []
    for i, (data, G, girth_dict) in enumerate(zip(subset_graphs, nx_graphs, girth_dicts)):
        if method in ("egagnn", "edge_girth_seq"):
            d = attach_edge_girth(data, girth_dict, normalizer, feature_mode="full", keep_dataset_attr=False)
        elif method == "triangle":
            d = data.clone()
            d.e0 = triangle_count_edge_attr(G, data.edge_index)
            d.edge_attr = d.e0
        elif method == "gsn":
            d = attach_gsn_features(data, G)
        else:
            d = data.clone()
        d.brec_key = i
        girth_dicts_by_key[i] = girth_dict
        out.append(d)
    return out, girth_dicts_by_key


def _run_one_pair(
    method,
    cat_name,
    pid,
    feat_graphs,
    index_map,
    e0_dim,
    node_dim,
    girth_dicts_by_key,
    epoch,
    num_relabel,
    seed,
):
    torch.set_num_threads(1)
    torch_geometric.seed_everything(seed)
    extra = girth_dicts_by_key if method == "edge_girth_seq" else None
    model_factory = _build_model_factory(method, e0_dim, node_dim, extra)
    learned = method not in ("folklore2wl", "edge_girth_seq")

    lo_main = pid * BREC_NUM_RELABEL * 2
    graphs_main = [feat_graphs[index_map[lo_main + k]] for k in range(num_relabel * 2)]
    lo_rel = (pid + 400) * BREC_NUM_RELABEL * 2
    graphs_rel = [feat_graphs[index_map[lo_rel + k]] for k in range(num_relabel * 2)]

    t0 = time.perf_counter()
    res = evaluate_pair(model_factory, graphs_main, graphs_rel, learned=learned, epoch=epoch, batch_size=16)
    res["elapsed_seconds"] = time.perf_counter() - t0
    return cat_name, pid, res


def _run_one_pair_folklore2wl(cat_name, pid, nx_graphs, index_map, num_relabel):
    """Exact 2-FWL: no training, no relabeling-averaging, no T^2 test. Uses a
    single relabeling of each graph (index 0) since the multiset comparison is
    a deterministic invariant -- further relabelings cannot change the result.
    """
    t0 = time.perf_counter()
    lo_main = pid * BREC_NUM_RELABEL * 2
    GA_main = nx_graphs[index_map[lo_main + 0]]
    GB_main = nx_graphs[index_map[lo_main + 1]]
    lo_rel = (pid + 400) * BREC_NUM_RELABEL * 2
    GA_rel = nx_graphs[index_map[lo_rel + 0]]
    GB_rel = nx_graphs[index_map[lo_rel + 1]]

    dist_main, rounds_main = folklore_2wl_distinguishable(GA_main, GB_main)
    dist_rel, rounds_rel = folklore_2wl_distinguishable(GA_rel, GB_rel)
    # reliability group = a control pair of graphs that ARE isomorphic to each
    # other (independent relabelings of the same underlying graph); a correct
    # test must NOT distinguish them.
    res = dict(
        isomorphic_flag=dist_main,
        reliable=not dist_rel,
        t2_main=None,
        t2_reliability=None,
        rounds_main=rounds_main,
        rounds_reliability=rounds_rel,
        elapsed_seconds=time.perf_counter() - t0,
    )
    return cat_name, pid, res


ALL_BREC_METHODS = [
    "egagnn",
    "gcn",
    "gin",
    "gatedgcn",
    "gsn",
    "ppgn",
    "folklore2wl",
    "triangle",
    "edge_girth_seq",
]


def _load_smoke_timings():
    """Best-effort load of smoke-mode per-method timings, for full-mode ETA
    estimates. Returns {} if no prior smoke run is on disk."""
    path = os.path.join(RESULTS_DIR, "brec_results.pkl")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            prior = pickle.load(f)
        if prior.get("mode") != "smoke":
            return {}
        return dict(
            num_relabel=prior["num_relabel"],
            epoch=prior["epoch"],
            n_pairs=sum(prior["results"][m]["n_pairs"] for m in prior["results"] if m in prior["results"])
            // max(len(prior["results"]), 1),
            per_method={m: r["elapsed_seconds"] for m, r in prior["results"].items()},
        )
    except Exception:
        return {}


def run_brec(mode: str = "smoke", methods=None, n_jobs: int = -1, seed: int = 13):
    methods = methods or list(ALL_BREC_METHODS)
    t_start = time.perf_counter()
    print(f"\n{'=' * 70}\n[BREC] mode={mode}, methods={methods}, n_jobs={n_jobs}\n{'=' * 70}", flush=True)

    all_graphs = load_brec_raw()
    node_dim = 1

    smoke_timings = {}
    if mode == "smoke":
        pair_ids_by_cat = select_smoke_pair_ids(n_per_category=40, seed=seed)
        num_relabel = 8
        epoch = 5
    else:
        pair_ids_by_cat = {c: list(range(lo, hi)) for c, (lo, hi) in BREC_PART_DICT.items()}
        num_relabel = BREC_NUM_RELABEL
        epoch = 20
        smoke_timings = _load_smoke_timings()
        if smoke_timings:
            print(
                f"[BREC/full] found prior smoke timings (num_relabel={smoke_timings['num_relabel']}, "
                f"epoch={smoke_timings['epoch']}) -- using them for per-method ETA estimates below."
            )
        else:
            print(
                "[BREC/full] no prior smoke-mode brec_results.pkl found -- no ETA estimates available "
                "for this run (they'll just show elapsed time, no prediction)."
            )

    # Only touch the graphs actually needed for the selected pairs (BREC has
    # no train/test split, so featurize once up front and reuse across methods).
    all_pair_ids = sorted({pid for ids in pair_ids_by_cat.values() for pid in ids})
    touched_idx = []
    for pid in all_pair_ids:
        for grp in (0, 400):
            lo = (pid + grp) * BREC_NUM_RELABEL * 2
            touched_idx.extend(range(lo, lo + num_relabel * 2))
    touched_idx = sorted(set(touched_idx))
    index_map = {gidx: pos for pos, gidx in enumerate(touched_idx)}
    subset_graphs = [all_graphs[i] for i in touched_idx]

    t_pre = time.perf_counter()
    nx_graphs = [data_to_simple_undirected_nx(d) for d in subset_graphs]
    girth_results = compute_edge_girth_batch(nx_graphs, n_jobs=n_jobs)
    girth_dicts = [r.values for r in girth_results]
    girth_preproc_time = time.perf_counter() - t_pre
    print(
        f"[BREC/{mode}] edge-girth preprocessing for {len(subset_graphs)} graphs: "
        f"{girth_preproc_time:.1f}s (parallel, cached)"
    )

    normalizer = EdgeGirthNormalizer().fit(girth_dicts[:300])

    results_path = os.path.join(RESULTS_DIR, "brec_results.pkl")
    n_total_pairs = sum(len(v) for v in pair_ids_by_cat.values())

    # merge with any existing on-disk results for this same mode/regime, so running
    # methods in separate invocations (e.g. priority-1 non-learned methods first,
    # then the rest later) accumulates instead of clobbering earlier work.
    results = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "rb") as f:
                prior = pickle.load(f)
            if (
                prior.get("mode") == mode
                and prior.get("num_relabel") == num_relabel
                and prior.get("epoch") == epoch
                and prior.get("pair_ids_by_cat") == pair_ids_by_cat
            ):
                results = dict(prior.get("results", {}))
                if results:
                    already_done = [m for m in results if m not in methods]
                    print(
                        f"[BREC/{mode}] found {len(results)} method(s) already completed on disk "
                        f"for this exact regime: {list(results.keys())}. Keeping "
                        f"{already_done or '(none)'} as-is; running/overwriting: {methods}"
                    )
        except Exception as e:
            print(f"[BREC/{mode}] could not read prior results.pkl for merging ({e}); starting fresh.")
    preprocessing_times = {"edge_girth_shared": girth_preproc_time}

    for method_i, method in enumerate(methods, 1):
        learned = method not in ("folklore2wl", "edge_girth_seq")
        est_seconds = None
        if smoke_timings.get("per_method", {}).get(method):
            est_seconds = estimate_full_time(
                smoke_timings["per_method"][method],
                smoke_timings["n_pairs"],
                smoke_timings["num_relabel"],
                smoke_timings["epoch"],
                n_total_pairs,
                num_relabel,
                epoch,
                learned=learned,
            )
        t_block = announce_block(
            f"BREC method {method_i}/{len(methods)}: '{method}' ({n_total_pairs} pairs, "
            f"num_relabel={num_relabel}, epoch={epoch if learned else 'n/a (non-learned)'})",
            estimated_seconds=est_seconds,
        )

        t0 = time.perf_counter()
        jobs = []
        if method == "folklore2wl":
            # exact test: operates directly on nx_graphs, no featurization, no
            # model, no training -- see Task 1 fix in the project's development history.
            preprocessing_times[method] = 0.0
            for cat_name, pair_ids in pair_ids_by_cat.items():
                for pid in pair_ids:
                    jobs.append(
                        delayed(_run_one_pair_folklore2wl)(cat_name, pid, nx_graphs, index_map, num_relabel)
                    )
        else:
            feat_graphs, girth_dicts_by_key = featurize_brec_subset(
                subset_graphs, nx_graphs, girth_dicts, method, normalizer
            )
            preprocessing_times[method] = time.perf_counter() - t0
            e0_dim = feat_graphs[0].e0.size(1) if hasattr(feat_graphs[0], "e0") else 0
            for cat_name, pair_ids in pair_ids_by_cat.items():
                for pid in pair_ids:
                    jobs.append(
                        delayed(_run_one_pair)(
                            method,
                            cat_name,
                            pid,
                            feat_graphs,
                            index_map,
                            e0_dim,
                            node_dim,
                            girth_dicts_by_key,
                            epoch,
                            num_relabel,
                            seed,
                        )
                    )

        t1 = time.perf_counter()
        log_every = max(1, len(jobs) // 20)  # ~20 progress lines per method block
        pair_results = run_parallel_with_progress(jobs, n_jobs, label=method, log_every=log_every)
        elapsed = time.perf_counter() - t1
        finish_block(f"BREC method '{method}'", t_block, expected_seconds=est_seconds)

        by_category = {cat: {"per_pair": {}} for cat in pair_ids_by_cat}
        for cat_name, pid, res in pair_results:
            by_category[cat_name]["per_pair"][pid] = res
        for cat_name, cat_res in by_category.items():
            per_pair = cat_res["per_pair"]
            cat_res["n_pairs"] = len(per_pair)
            cat_res["n_correct"] = sum(1 for r in per_pair.values() if r["isomorphic_flag"])
            cat_res["n_reliable_fail"] = sum(1 for r in per_pair.values() if not r["reliable"])
            cat_res["accuracy"] = (
                cat_res["n_correct"] / cat_res["n_pairs"] if cat_res["n_pairs"] else float("nan")
            )

        n_correct = sum(r["n_correct"] for r in by_category.values())
        n_pairs = sum(r["n_pairs"] for r in by_category.values())
        n_reliable_fail = sum(r["n_reliable_fail"] for r in by_category.values())
        results[method] = dict(
            by_category=by_category,
            overall_accuracy=n_correct / n_pairs if n_pairs else float("nan"),
            n_correct=n_correct,
            n_pairs=n_pairs,
            n_reliable_fail=n_reliable_fail,
            elapsed_seconds=elapsed,
            preprocessing_seconds=preprocessing_times[method],
        )
        print(
            f"[BREC/{mode}] {method}: acc={results[method]['overall_accuracy']:.2f} "
            f"({n_correct}/{n_pairs}), reliable_fail={n_reliable_fail}, "
            f"time={elapsed:.1f}s, preproc={preprocessing_times[method]:.1f}s"
        )
        has_smoke_ref = smoke_timings.get("per_method", {}).get(method)
        smoke_note = "N/A, no smoke reference found" if not has_smoke_ref else "see brec_results.pkl history"
        print(
            f"  --> vs. smoke ({smoke_timings.get('num_relabel', '?')} relabel, "
            f"{smoke_timings.get('epoch', '?')} epoch): {smoke_note}"
        )

        # incremental checkpoint: overwrite the results file after EVERY method,
        # not just at the very end, so a crash mid-campaign loses at most the
        # block currently in flight.
        os.makedirs(RESULTS_DIR, exist_ok=True)
        partial_out = dict(
            mode=mode,
            seed=seed,
            num_relabel=num_relabel,
            epoch=epoch,
            pair_ids_by_cat=pair_ids_by_cat,
            results=dict(results),
            total_elapsed_seconds=time.perf_counter() - t_start,
            methods_completed=list(results.keys()),
            methods_requested=methods,
            partial=(method_i < len(methods)),
        )
        checkpoint_save(results_path, partial_out)

    total_elapsed = time.perf_counter() - t_start
    out = dict(
        mode=mode,
        seed=seed,
        num_relabel=num_relabel,
        epoch=epoch,
        pair_ids_by_cat=pair_ids_by_cat,
        results=results,
        total_elapsed_seconds=total_elapsed,
        methods_completed=list(results.keys()),
        methods_requested=methods,
        partial=False,
    )
    checkpoint_save(results_path, out)
    print(
        f"\n[BREC/{mode}] ALL METHODS DONE. total time: {total_elapsed:.1f}s "
        f"({total_elapsed / 60:.1f} min). Saved to results/brec_results.pkl"
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=ALL_BREC_METHODS,
        help="Subset of methods to run (default: all 9). E.g. "
        "--methods edge_girth_seq folklore2wl for the cheap, "
        "non-learned, contingency-table-critical pair first.",
    )
    args = parser.parse_args()
    run_brec(mode=args.mode, n_jobs=args.n_jobs, methods=args.methods)
