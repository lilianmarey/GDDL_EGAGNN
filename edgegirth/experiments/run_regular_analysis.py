"""Task 3: is the Regular category's cross-method agreement explained by
edge-girth-regularity (Jajcay et al. 2018), as Proposition 1 predicts?

For each pair in Regular (and, as a corollary check, 4-Vertex-Condition and
Distance-Regular): which methods resolve it, and are the underlying graphs
edge-girth-regular? Reads results/brec_results.pkl (must already contain the
Task-1-fixed folklore2wl), writes results/regular_analysis.pkl.

Usage:
    python -m edgegirth.experiments.run_regular_analysis
"""

from __future__ import annotations

import os
import pickle

from edgegirth.data.loaders import BREC_NUM_RELABEL, load_brec_raw
from edgegirth.features.edge_girth import compute_edge_girth, is_edge_girth_regular
from edgegirth.features.featurize import data_to_simple_undirected_nx
from edgegirth.paths import RESULTS_DIR

CATEGORIES = ["Regular", "4-Vertex_Condition", "Distance_Regular"]


def pair_egr_status(all_graphs, pid):
    """egr status of the two graphs in pair `pid` (first relabeling of each)."""
    lo = pid * BREC_NUM_RELABEL * 2
    GA = data_to_simple_undirected_nx(all_graphs[lo + 0])
    GB = data_to_simple_undirected_nx(all_graphs[lo + 1])
    egr_A, params_A = is_edge_girth_regular(GA, compute_edge_girth(GA))
    egr_B, params_B = is_edge_girth_regular(GB, compute_edge_girth(GB))
    both_egr = egr_A and egr_B
    matching_params = both_egr and params_A == params_B
    return dict(
        egr_A=egr_A,
        egr_B=egr_B,
        params_A=params_A,
        params_B=params_B,
        both_egr=both_egr,
        matching_params=matching_params,
    )


def run_regular_analysis():
    brec_path = os.path.join(RESULTS_DIR, "brec_results.pkl")
    with open(brec_path, "rb") as f:
        brec = pickle.load(f)

    all_graphs = load_brec_raw()
    methods = list(brec["results"].keys())

    per_category = {}
    for cat in CATEGORIES:
        pair_ids = brec["pair_ids_by_cat"][cat]
        rows = []
        for pid in pair_ids:
            resolved_by = {}
            for m in methods:
                per_pair = brec["results"][m]["by_category"][cat]["per_pair"]
                resolved_by[m] = bool(per_pair[pid]["isomorphic_flag"])
            egr_info = pair_egr_status(all_graphs, pid)
            rows.append(dict(pid=pid, resolved_by=resolved_by, **egr_info))
        per_category[cat] = rows

    # --- Step 1: is the *set* of resolved pairs identical across methods, for Regular?
    reg_rows = per_category["Regular"]
    resolved_sets = {m: {r["pid"] for r in reg_rows if r["resolved_by"][m]} for m in methods}
    non_trivial_methods = [m for m, s in resolved_sets.items() if len(s) > 0]
    reference_method = non_trivial_methods[0] if non_trivial_methods else None
    reference_set = resolved_sets.get(reference_method, set())
    agreement = {
        m: dict(
            resolved_set=sorted(resolved_sets[m]), identical_to_reference=(resolved_sets[m] == reference_set)
        )
        for m in methods
    }
    all_non_trivial_agree = all(agreement[m]["identical_to_reference"] for m in non_trivial_methods)

    # --- Step 3: 2x2 contingency (Regular), resolved (reference method) x egr (matching params)
    contingency = {
        "resolved_and_egr": 0,
        "resolved_and_not_egr": 0,
        "not_resolved_and_egr": 0,
        "not_resolved_and_not_egr": 0,
    }
    for r in reg_rows:
        resolved = r["pid"] in reference_set
        egr = r["matching_params"]
        key = ("resolved" if resolved else "not_resolved") + "_and_" + ("egr" if egr else "not_egr")
        contingency[key] += 1

    exact_match = (
        contingency["resolved_and_not_egr"] == len(reference_set)
        and contingency["not_resolved_and_egr"] == len(reg_rows) - len(reference_set)
        and contingency["resolved_and_egr"] == 0
        and contingency["not_resolved_and_not_egr"] == 0
    )

    # --- Corollary check: are ALL 4VC / Distance-Regular graphs edge-girth-regular?
    corollary_check = {}
    for cat in ["4-Vertex_Condition", "Distance_Regular"]:
        rows = per_category[cat]
        n_pairs = len(rows)
        n_both_egr = sum(1 for r in rows if r["both_egr"])
        n_matching = sum(1 for r in rows if r["matching_params"])
        violations = [r["pid"] for r in rows if not r["both_egr"]]
        corollary_check[cat] = dict(
            n_pairs=n_pairs,
            n_both_egr=n_both_egr,
            n_matching_params=n_matching,
            all_egr=(n_both_egr == n_pairs),
            violations=violations,
        )
        if violations:
            print(
                f"WARNING: {cat} has {len(violations)}/{n_pairs} pairs with a "
                f"non-edge-girth-regular graph: pair ids {violations}. "
                "This would invalidate Corollary 2 for these pairs."
            )
        else:
            print(
                f"{cat}: all {n_pairs} pairs are edge-girth-regular on both sides "
                f"({n_matching}/{n_pairs} with matching (k,g,lambda) params). Corollary 2 confirmed."
            )

    out = dict(
        methods=methods,
        reference_method=reference_method,
        per_category=per_category,
        method_agreement=agreement,
        all_non_trivial_methods_agree=all_non_trivial_agree,
        non_trivial_methods=non_trivial_methods,
        regular_contingency=contingency,
        regular_contingency_exact_match=exact_match,
        corollary_check=corollary_check,
    )

    print(
        f"\nRegular: reference method = {reference_method}, "
        f"{len(reference_set)}/{len(reg_rows)} pairs resolved"
    )
    print(f"All non-trivial methods agree on the resolved set: {all_non_trivial_agree}")
    if not all_non_trivial_agree:
        for m in non_trivial_methods:
            if not agreement[m]["identical_to_reference"]:
                print(
                    f"  DISAGREEMENT: {m} resolves {sorted(resolved_sets[m])} "
                    f"vs reference {sorted(reference_set)}"
                )
    print(f"Regular 2x2 contingency (resolved x edge-girth-regular): {contingency}")
    print(f"Contingency is an exact match (resolved <=> not egr): {exact_match}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "regular_analysis.pkl"), "wb") as f:
        pickle.dump(out, f)
    print("Saved to results/regular_analysis.pkl")
    return out


if __name__ == "__main__":
    run_regular_analysis()
