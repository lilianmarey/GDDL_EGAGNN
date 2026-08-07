"""Task 3 (v3): a single contingency table across all of sampled BREC.

For every pair actually evaluated in brec_results.pkl (all 6 categories):
category, egr (both graphs, matching (k,g,lambda) params), resolved by
edge_girth_seq, resolved by edgegirth, resolved by folklore2wl (exact 2-FWL, NOT
edge-girth-based -- reported for context, not as a Prop-1 check).

Explicitly lists any egr pair resolved by an edge-girth-based method
(edgegirth or edge_girth_seq) -- that would be a direct counterexample to
Proposition 1 and needs to be flagged immediately, not smoothed over.

Usage:
    python -m edgegirth.experiments.run_global_contingency
"""

from __future__ import annotations

import os
import pickle

from edgegirth.data.loaders import BREC_NUM_RELABEL, load_brec_raw
from edgegirth.features.edge_girth import compute_edge_girth, is_edge_girth_regular
from edgegirth.features.featurize import data_to_simple_undirected_nx
from edgegirth.paths import RESULTS_DIR

EDGE_GIRTH_METHODS = ["egagnn", "edge_girth_seq"]


def run_global_contingency():
    with open(os.path.join(RESULTS_DIR, "brec_results.pkl"), "rb") as f:
        brec = pickle.load(f)
    all_graphs = load_brec_raw()

    rows = []
    for cat, pair_ids in brec["pair_ids_by_cat"].items():
        for pid in pair_ids:
            lo = pid * BREC_NUM_RELABEL * 2
            GA = data_to_simple_undirected_nx(all_graphs[lo + 0])
            GB = data_to_simple_undirected_nx(all_graphs[lo + 1])
            egr_A, params_A = is_edge_girth_regular(GA, compute_edge_girth(GA))
            egr_B, params_B = is_edge_girth_regular(GB, compute_edge_girth(GB))
            matching = egr_A and egr_B and params_A == params_B

            row = dict(category=cat, pid=pid, egr=matching)
            for method in EDGE_GIRTH_METHODS + ["folklore2wl"]:
                if method in brec["results"]:
                    r = brec["results"][method]["by_category"][cat]["per_pair"][pid]
                    row[f"resolved_{method}"] = bool(r["isomorphic_flag"])
            rows.append(row)

    n_total = len(rows)
    n_egr = sum(1 for r in rows if r["egr"])
    print(
        f"Global sampled BREC: {n_total} pairs across {len(brec['pair_ids_by_cat'])} categories, "
        f"{n_egr} are edge-girth-regular (matching params, both graphs)"
    )

    contingency_by_method = {}
    violations_by_method = {}
    for method in EDGE_GIRTH_METHODS:
        col = f"resolved_{method}"
        c = {
            "resolved_and_egr": 0,
            "resolved_and_not_egr": 0,
            "not_resolved_and_egr": 0,
            "not_resolved_and_not_egr": 0,
        }
        violations = []
        for r in rows:
            if col not in r:
                continue
            key = ("resolved" if r[col] else "not_resolved") + "_and_" + ("egr" if r["egr"] else "not_egr")
            c[key] += 1
            if r[col] and r["egr"]:
                violations.append((r["category"], r["pid"]))
        contingency_by_method[method] = c
        violations_by_method[method] = violations
        print(f"\n{method}: {c}")
        if violations:
            print(
                f"  *** {len(violations)} PROPOSITION-1 VIOLATION(S): egr pair(s) resolved: {violations} ***"
            )
            print("  *** THIS IS A COUNTEREXAMPLE -- STOP AND INVESTIGATE THE CODE OR THE PROOF ***")
        else:
            print(f"  No violations: zero egr pairs resolved by {method}.")

    # per-category breakdown, for the table that goes in the paper
    per_category_summary = {}
    for cat in brec["pair_ids_by_cat"]:
        cat_rows = [r for r in rows if r["category"] == cat]
        per_category_summary[cat] = dict(
            n_pairs=len(cat_rows),
            n_egr=sum(1 for r in cat_rows if r["egr"]),
            n_resolved_egagnn=sum(1 for r in cat_rows if r.get("resolved_egagnn")),
            n_resolved_edge_girth_seq=sum(1 for r in cat_rows if r.get("resolved_edge_girth_seq")),
            n_resolved_folklore2wl=sum(1 for r in cat_rows if r.get("resolved_folklore2wl")),
        )

    out = dict(
        rows=rows,
        n_total=n_total,
        n_egr=n_egr,
        contingency_by_method=contingency_by_method,
        violations_by_method=violations_by_method,
        per_category_summary=per_category_summary,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "global_contingency.pkl"), "wb") as f:
        pickle.dump(out, f)
    print("\nSaved to results/global_contingency.pkl")
    return out


if __name__ == "__main__":
    run_global_contingency()
