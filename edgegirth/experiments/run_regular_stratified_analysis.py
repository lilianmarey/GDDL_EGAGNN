"""Task 1 (v3): does edge-girth-regularity coincide with strong regularity on
BREC's Regular category, or is there a confound with generic strongly-regular
hardness?

Regular mixes plain regular graphs with strongly regular ones. SRGs happen to
be edge-girth-regular whenever lambda > 0 (girth 3, lambda_e = lambda on every
edge), AND they are a classical source of 2-FWL/3-WL indistinguishability, for
reasons unrelated to Proposition 1. If the 21 unresolved Regular pairs from
the project's development history are exactly the SRG pairs, the edge-girth-regular / unresolved
correlation could be explained by "these are just hard graphs for everyone,"
not specifically by Proposition 1. This script checks whether egr and SRG
status coincide on this category, and isolates any pair where they diverge --
those pairs are the ones that actually isolate Proposition 1 from generic
2-FWL/3-WL hardness.

Usage:
    python -m edgegirth.experiments.run_regular_stratified_analysis
"""

from __future__ import annotations

import os
import pickle

from edgegirth.data.loaders import BREC_NUM_RELABEL, load_brec_raw
from edgegirth.features.edge_girth import is_strongly_regular
from edgegirth.features.featurize import data_to_simple_undirected_nx
from edgegirth.paths import RESULTS_DIR


def run_regular_stratified_analysis():
    with open(os.path.join(RESULTS_DIR, "regular_analysis.pkl"), "rb") as f:
        regular_analysis = pickle.load(f)
    all_graphs = load_brec_raw()

    reg_rows = regular_analysis["per_category"]["Regular"]
    reference_method = regular_analysis["reference_method"]
    reference_set = set(regular_analysis["method_agreement"][reference_method]["resolved_set"])

    augmented = []
    for row in reg_rows:
        pid = row["pid"]
        lo = pid * BREC_NUM_RELABEL * 2
        GA = data_to_simple_undirected_nx(all_graphs[lo + 0])
        GB = data_to_simple_undirected_nx(all_graphs[lo + 1])
        sr_A, sr_params_A = is_strongly_regular(GA)
        sr_B, sr_params_B = is_strongly_regular(GB)
        both_sr = sr_A and sr_B
        sr_matching = both_sr and sr_params_A == sr_params_B
        resolved = pid in reference_set
        augmented.append(
            dict(
                pid=pid,
                resolved=resolved,
                egr=row["matching_params"],
                egr_params_A=row["params_A"],
                egr_params_B=row["params_B"],
                sr=sr_matching,
                sr_params_A=sr_params_A,
                sr_params_B=sr_params_B,
            )
        )

    n = len(augmented)
    n_egr = sum(1 for r in augmented if r["egr"])
    n_sr = sum(1 for r in augmented if r["sr"])
    n_egr_and_sr = sum(1 for r in augmented if r["egr"] and r["sr"])
    n_egr_not_sr = [r["pid"] for r in augmented if r["egr"] and not r["sr"]]
    n_sr_not_egr = [r["pid"] for r in augmented if r["sr"] and not r["egr"]]
    egr_sr_coincide = len(n_egr_not_sr) == 0 and len(n_sr_not_egr) == 0

    print(f"Regular category: {n} pairs total")
    print(f"  egr (matching params, both graphs): {n_egr}")
    print(f"  strongly regular (matching params, both graphs): {n_sr}")
    print(f"  egr AND sr: {n_egr_and_sr}")
    print(f"  egr but NOT sr: {n_egr_not_sr}")
    print(f"  sr but NOT egr: {n_sr_not_egr}")
    print(f"  egr and sr coincide exactly on this category: {egr_sr_coincide}")

    # Stratified contingency: plain-regular (not sr) vs strongly-regular subsets
    plain = [r for r in augmented if not r["sr"]]
    strong = [r for r in augmented if r["sr"]]

    def contingency(rows):
        c = {
            "resolved_and_egr": 0,
            "resolved_and_not_egr": 0,
            "not_resolved_and_egr": 0,
            "not_resolved_and_not_egr": 0,
        }
        for r in rows:
            key = (
                ("resolved" if r["resolved"] else "not_resolved")
                + "_and_"
                + ("egr" if r["egr"] else "not_egr")
            )
            c[key] += 1
        return c

    plain_contingency = contingency(plain)
    strong_contingency = contingency(strong)
    print(f"\nPlain-regular subset (n={len(plain)}): {plain_contingency}")
    print(f"Strongly-regular subset (n={len(strong)}): {strong_contingency}")

    # The separating pairs: egr XOR sr -- these isolate Proposition 1 from
    # generic 2-FWL/3-WL hardness, if any exist.
    separating_pairs = [r for r in augmented if r["egr"] != r["sr"]]
    if separating_pairs:
        print(
            f"\n{len(separating_pairs)} pair(s) separate egr from strongly-regular "
            f"-- these are the most informative pairs in the category:"
        )
        for r in separating_pairs:
            print(f"  pid={r['pid']}: egr={r['egr']}, sr={r['sr']}, resolved={r['resolved']}")
    else:
        print(
            "\nNo pair separates egr from strongly-regular on this category: "
            "the two properties coincide exactly here, so this category alone "
            "cannot isolate Proposition 1 from generic strongly-regular-graph "
            "2-FWL/3-WL hardness. See Task 2 (CFI) for a category where they diverge."
        )

    out = dict(
        reference_method=reference_method,
        rows=augmented,
        n_egr=n_egr,
        n_sr=n_sr,
        n_egr_and_sr=n_egr_and_sr,
        egr_not_sr=n_egr_not_sr,
        sr_not_egr=n_sr_not_egr,
        egr_sr_coincide=egr_sr_coincide,
        plain_contingency=plain_contingency,
        strong_contingency=strong_contingency,
        separating_pairs=[r["pid"] for r in separating_pairs],
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "regular_stratified_analysis.pkl"), "wb") as f:
        pickle.dump(out, f)
    print("\nSaved to results/regular_stratified_analysis.pkl")
    return out


if __name__ == "__main__":
    run_regular_stratified_analysis()
