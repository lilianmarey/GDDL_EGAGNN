"""Task 2: cold-cache preprocessing time on the full 12,000
ZINC-12k graphs, for edgegirth (edge-girth) and the three new structural
baselines (triangle, gsn4, gsn6).

"Cold cache" matters only for edgegirth: `compute_edge_girth` is disk-cached
under `.cache/edge_girth/` (content-addressed by graph hash), so a second run
on the same graphs would read from disk instead of recomputing -- this script
deletes that cache directory before timing edgegirth specifically, to get a
number that doesn't depend on what happened to run before it (the
138.7s figure in the paper's numbers table was flagged there as a warm-cache
measurement from a rerun; this replaces it with a clean, reproducible
cold-cache number).

`triangle`, `gsn4`, `gsn6` have no disk cache at all (their features --
triangle counts, bounded cycle counts -- are recomputed from scratch every
call, via networkx, with no caching layer implemented) -- so they are
"cold" by construction every time; this script times them anyway for a
like-for-like comparison, and says so explicitly rather than silently
treating "no cache" the same as "cache cleared."

Usage:
    python -m edgegirth.experiments.run_zinc_cold_cache_preprocessing
"""

from __future__ import annotations

import os
import pickle
import shutil
import time

from edgegirth.data.loaders import load_zinc
from edgegirth.experiments.run_zinc import (
    featurize_zinc_gsn,
    featurize_zinc_list,
    featurize_zinc_triangle,
)
from edgegirth.features.edge_girth import CACHE_DIR, compute_edge_girth
from edgegirth.features.featurize import EdgeGirthNormalizer, data_to_simple_undirected_nx
from edgegirth.paths import RESULTS_DIR

N_JOBS = 10


def run_measurement(n_jobs: int = N_JOBS):
    train, val, test = load_zinc(subset=True)
    n_total = len(train) + len(val) + len(test)
    print(f"Full ZINC-12k: {len(train)}/{len(val)}/{len(test)} = {n_total} graphs total", flush=True)

    results = {}

    # --- edgegirth: clear the edge-girth disk cache first, genuinely cold ---
    if os.path.exists(CACHE_DIR):
        n_cached = len(os.listdir(CACHE_DIR))
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared {CACHE_DIR} ({n_cached} cached files removed)", flush=True)
    else:
        print(f"{CACHE_DIR} did not exist -- already cold", flush=True)

    fit_dicts = [
        compute_edge_girth(data_to_simple_undirected_nx(train[i])) for i in range(min(300, len(train)))
    ]
    normalizer = EdgeGirthNormalizer().fit(fit_dicts)
    # the 300 graphs used to fit the normalizer are now warm in cache; that's
    # unavoidable (fitting needs them) and negligible (300/12000 = 2.5%) --
    # noted here rather than hidden.

    t0 = time.perf_counter()
    _ = featurize_zinc_list(list(train), normalizer, n_jobs=n_jobs)
    _ = featurize_zinc_list(list(val), normalizer, n_jobs=n_jobs)
    _ = featurize_zinc_list(list(test), normalizer, n_jobs=n_jobs)
    t_edgegirth = time.perf_counter() - t0
    print(
        f"edgegirth (edge-girth, cold cache except the 300-graph normalizer-fit subset): "
        f"{t_edgegirth:.2f}s for {n_total} graphs",
        flush=True,
    )
    results["egagnn"] = dict(
        preprocessing_seconds=t_edgegirth,
        n_graphs=n_total,
        cold_cache=True,
        note="300/12000 graphs (2.5%) were touched once already, to fit the normalizer",
    )

    # --- triangle: no cache exists for this at all ---
    t0 = time.perf_counter()
    _ = featurize_zinc_triangle(list(train), n_jobs=n_jobs)
    _ = featurize_zinc_triangle(list(val), n_jobs=n_jobs)
    _ = featurize_zinc_triangle(list(test), n_jobs=n_jobs)
    t_triangle = time.perf_counter() - t0
    print(f"triangle (no disk cache exists, always cold): {t_triangle:.2f}s for {n_total} graphs", flush=True)
    results["triangle"] = dict(
        preprocessing_seconds=t_triangle,
        n_graphs=n_total,
        cold_cache=True,
        note="no disk cache implemented for this feature -- always computed fresh",
    )

    # --- gsn4, gsn6: same, no cache ---
    for k in (4, 6):
        method = f"gsn{k}"
        t0 = time.perf_counter()
        _ = featurize_zinc_gsn(list(train), k, n_jobs=n_jobs)
        _ = featurize_zinc_gsn(list(val), k, n_jobs=n_jobs)
        _ = featurize_zinc_gsn(list(test), k, n_jobs=n_jobs)
        t_gsn = time.perf_counter() - t0
        print(f"{method} (no disk cache exists, always cold): {t_gsn:.2f}s for {n_total} graphs", flush=True)
        results[method] = dict(
            preprocessing_seconds=t_gsn,
            n_graphs=n_total,
            cold_cache=True,
            note="no disk cache implemented for this feature -- always computed fresh",
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "zinc_cold_cache_preprocessing.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(dict(n_jobs=n_jobs, n_total=n_total, results=results), f)
    print(f"\nSaved to {out_path}")
    return results


if __name__ == "__main__":
    run_measurement()
