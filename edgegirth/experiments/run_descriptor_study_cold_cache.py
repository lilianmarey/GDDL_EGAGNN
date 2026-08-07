"""Cold-cache preprocessing time for the six descriptor-vs-descriptor
variants, on the full 12,000-graph ZINC-12k set, measured in isolation --
same methodology as `run_zinc_cold_cache_preprocessing.py`.

`constant` has no computation at all (zeros). `full` is edge-girth, disk
cached under `.cache/edge_girth/`, so that cache is cleared first. `tri`,
`cyc4`, `cyc6`, `cyc8` have no disk cache implemented and are cold by
construction on every call.

Usage:
    python -m edgegirth.experiments.run_descriptor_study_cold_cache
"""

from __future__ import annotations

import os
import pickle
import shutil
import time

import torch

from edgegirth.data.loaders import load_zinc
from edgegirth.experiments.run_descriptor_study import featurize_split
from edgegirth.experiments.run_zinc import ATOM_TYPES, BOND_TYPES, featurize_zinc_list
from edgegirth.features.edge_girth import CACHE_DIR, compute_edge_girth
from edgegirth.features.featurize import EdgeGirthNormalizer, data_to_simple_undirected_nx
from edgegirth.paths import RESULTS_DIR

N_JOBS = 10


def _time_constant(train, val, test):
    """No computation -- zeroing a fixed-width vector is O(1) per edge and
    dominated by tensor allocation overhead; timed anyway for completeness,
    same one-hot/clone bookkeping as the other variants."""

    def _job(data):
        d = data.clone()
        x_ids = d.x.view(-1).long().clamp(max=ATOM_TYPES - 1)
        d.x = torch.nn.functional.one_hot(x_ids, ATOM_TYPES).float()
        bond_ids = d.edge_attr.view(-1).long().clamp(max=BOND_TYPES - 1)
        bond_onehot = torch.nn.functional.one_hot(bond_ids, BOND_TYPES).float()
        struct = torch.zeros(d.edge_index.size(1), 3)
        d.e0 = torch.cat([struct, bond_onehot], dim=1)
        d.edge_attr = d.e0
        return d

    t0 = time.perf_counter()
    for split in (train, val, test):
        for d in split:
            _job(d)
    return time.perf_counter() - t0


def run_measurement(n_jobs: int = N_JOBS):
    train, val, test = load_zinc(subset=True)
    n_total = len(train) + len(val) + len(test)
    print(f"Full ZINC-12k: {len(train)}/{len(val)}/{len(test)} = {n_total} graphs total", flush=True)

    results = {}

    # --- constant: no computation ---
    t_const = _time_constant(train, val, test)
    print(f"constant (no computation, zeroed vector): {t_const:.2f}s for {n_total} graphs", flush=True)
    results["constant"] = dict(
        preprocessing_seconds=t_const,
        n_graphs=n_total,
        cold_cache=True,
        note="no structural computation at all -- a zeroed fixed-width vector",
    )

    # --- full (edge-girth): clear the disk cache first, genuinely cold ---
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

    t0 = time.perf_counter()
    _ = featurize_zinc_list(list(train), normalizer, n_jobs=n_jobs)
    _ = featurize_zinc_list(list(val), normalizer, n_jobs=n_jobs)
    _ = featurize_zinc_list(list(test), normalizer, n_jobs=n_jobs)
    t_full = time.perf_counter() - t0
    print(
        f"full (edge-girth, cold cache except the 300-graph normalizer-fit subset): "
        f"{t_full:.2f}s for {n_total} graphs",
        flush=True,
    )
    results["full"] = dict(
        preprocessing_seconds=t_full,
        n_graphs=n_total,
        cold_cache=True,
        note="300/12000 graphs (2.5%) were touched once already, to fit the normalizer",
    )

    # --- tri, cyc4, cyc6, cyc8: no cache exists for any of these ---
    for variant in ("tri", "cyc4", "cyc6", "cyc8"):
        t0 = time.perf_counter()
        _ = featurize_split(train, variant, n_jobs=n_jobs)
        _ = featurize_split(val, variant, n_jobs=n_jobs)
        _ = featurize_split(test, variant, n_jobs=n_jobs)
        t_v = time.perf_counter() - t0
        print(f"{variant} (no disk cache exists, always cold): {t_v:.2f}s for {n_total} graphs", flush=True)
        results[variant] = dict(
            preprocessing_seconds=t_v,
            n_graphs=n_total,
            cold_cache=True,
            note="no disk cache implemented for this feature -- always computed fresh",
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "descriptor_study_cold_cache_preprocessing.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(dict(n_jobs=n_jobs, n_total=n_total, results=results), f)
    print(f"\nSaved to {out_path}")
    return results


if __name__ == "__main__":
    run_measurement()
