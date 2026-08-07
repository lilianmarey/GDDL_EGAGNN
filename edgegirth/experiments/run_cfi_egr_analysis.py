"""Task 2 (v3): are BREC's CFI graphs edge-girth-regular? This is the decisive
test case: CFI is the only category where exact 2-FWL does well (0.57) and
edge-girth-based methods do badly (EGAGNN 0.03) -- the inverse of every other
category. If CFI graphs are (or mostly are) edge-girth-regular, Proposition 1
explains EGAGNN's near-zero score there directly, and does so on a category
that is NOT strongly-regular-graph-flavoured (CFI graphs are a completely
different, gadget-based construction), removing the Task-1 confound entirely.

Checks the FULL CFI category (100 pairs, all of them -- cheap, since egr status
needs no training), and cross-references against the resolved/unresolved status
already computed for the 40 CFI pairs actually evaluated in brec_results.pkl.

Usage:
    python -m edgegirth.experiments.run_cfi_egr_analysis
"""

from __future__ import annotations

import os
import pickle

from edgegirth.data.loaders import BREC_NUM_RELABEL, BREC_PART_DICT, load_brec_raw
from edgegirth.features.edge_girth import compute_edge_girth, is_edge_girth_regular
from edgegirth.features.featurize import data_to_simple_undirected_nx
from edgegirth.paths import RESULTS_DIR

EDGE_GIRTH_METHODS = ["egagnn", "edge_girth_seq"]  # methods Prop 1 / Remark 3 actually cover


def pair_egr_status(all_graphs, pid):
    lo = pid * BREC_NUM_RELABEL * 2
    GA = data_to_simple_undirected_nx(all_graphs[lo + 0])
    GB = data_to_simple_undirected_nx(all_graphs[lo + 1])
    egr_A, params_A = is_edge_girth_regular(GA, compute_edge_girth(GA))
    egr_B, params_B = is_edge_girth_regular(GB, compute_edge_girth(GB))
    both_egr = egr_A and egr_B
    matching = both_egr and params_A == params_B
    return dict(
        egr_A=egr_A,
        egr_B=egr_B,
        params_A=params_A,
        params_B=params_B,
        both_egr=both_egr,
        matching_params=matching,
    )


def run_cfi_egr_analysis():
    all_graphs = load_brec_raw()
    lo, hi = BREC_PART_DICT["CFI"]
    all_cfi_pids = list(range(lo, hi))

    rows = []
    for pid in all_cfi_pids:
        egr_info = pair_egr_status(all_graphs, pid)
        rows.append(dict(pid=pid, **egr_info))

    n_pairs = len(rows)
    n_matching = sum(1 for r in rows if r["matching_params"])
    n_both_egr_mismatched = sum(1 for r in rows if r["both_egr"] and not r["matching_params"])
    n_neither_egr = sum(1 for r in rows if not r["egr_A"] and not r["egr_B"])
    n_one_egr = sum(1 for r in rows if r["egr_A"] != r["egr_B"])

    print(f"CFI category: {n_pairs} pairs (full category, not sampled)")
    print(f"  both graphs egr, matching (k,g,lambda): {n_matching}")
    print(f"  both graphs egr but mismatched params: {n_both_egr_mismatched}")
    print(f"  exactly one graph egr: {n_one_egr}")
    print(f"  neither graph egr: {n_neither_egr}")

    # cross-reference with the sampled pairs actually evaluated in brec_results.pkl
    with open(os.path.join(RESULTS_DIR, "brec_results.pkl"), "rb") as f:
        brec = pickle.load(f)
    sampled_pids = set(brec["pair_ids_by_cat"]["CFI"])
    egr_by_pid = {r["pid"]: r for r in rows}

    cross_rows = []
    for method in EDGE_GIRTH_METHODS:
        if method not in brec["results"]:
            continue
        per_pair = brec["results"][method]["by_category"]["CFI"]["per_pair"]
        for pid, res in per_pair.items():
            cross_rows.append(
                dict(
                    method=method,
                    pid=pid,
                    resolved=res["isomorphic_flag"],
                    matching_params=egr_by_pid[pid]["matching_params"],
                )
            )

    print(f"\nCross-check against the {len(sampled_pids)} sampled CFI pairs actually evaluated:")
    exceptions_by_method = {}
    for method in EDGE_GIRTH_METHODS:
        method_rows = [r for r in cross_rows if r["method"] == method]
        if not method_rows:
            continue
        resolved_pids = {r["pid"] for r in method_rows if r["resolved"]}
        not_egr_pids = {r["pid"] for r in method_rows if not r["matching_params"]}
        # Proposition 1 predicts: resolved ⊆ not-egr (an egr pair can never be resolved)
        violations = sorted(resolved_pids - not_egr_pids)  # resolved AND egr -- should be empty
        exceptions_by_method[method] = violations
        print(
            f"  {method}: {len(resolved_pids)}/{len(method_rows)} resolved. "
            f"Resolved-but-egr violations of Prop 1: {violations if violations else 'NONE'}"
        )
        if resolved_pids and resolved_pids != (resolved_pids & not_egr_pids):
            pass  # already captured in violations
        # Does "resolved" == "not egr" exactly (as tight as the Regular case)?
        exact_match = resolved_pids == not_egr_pids
        print(f"    resolved set == not-egr set exactly: {exact_match}")
        if not exact_match:
            print(
                f"    not-egr-but-not-resolved (egr correctly predicts collapse, "
                f"but method ALSO failed on some non-egr pairs -- a separate, "
                f"non-Prop-1 limitation): {sorted(not_egr_pids - resolved_pids)}"
            )

    out = dict(
        all_cfi_pids=all_cfi_pids,
        rows=rows,
        n_pairs=n_pairs,
        n_matching=n_matching,
        n_both_egr_mismatched=n_both_egr_mismatched,
        n_one_egr=n_one_egr,
        n_neither_egr=n_neither_egr,
        sampled_pids=sorted(sampled_pids),
        cross_rows=cross_rows,
        exceptions_by_method=exceptions_by_method,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "cfi_egr_analysis.pkl"), "wb") as f:
        pickle.dump(out, f)
    print("\nSaved to results/cfi_egr_analysis.pkl")
    return out


if __name__ == "__main__":
    run_cfi_egr_analysis()
