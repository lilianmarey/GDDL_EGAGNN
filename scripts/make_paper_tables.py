#!/usr/bin/env python3
"""Regenerates every markdown table in `paper_tables/` from the pickles in
`results/` (or `$EDGEGIRTH_RESULTS_DIR`). Read-only with respect to the
pickles: this script never runs an experiment, it only formats numbers
that are already on disk.

Usage:
    python scripts/make_paper_tables.py [--results-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import pickle


def _load(results_dir: str, name: str) -> dict | None:
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        print(f"  skip {name}: not found at {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _per_seed(r: dict) -> list[float]:
    return [round(s["test_mae"], 4) for s in r["seed_results"]]


def make_zinc_table(results_dir: str) -> str:
    d = _load(results_dir, "zinc_results.pkl")
    if d is None:
        return ""
    lines = [
        "# ZINC-12k main table\n",
        f"Regime: mode={d.get('mode')}, seeds={d.get('seeds')}, epoch={d.get('epoch')}.\n",
        "| Method | hidden_dim | n_params | Test MAE (mean +/- std) | Per-seed MAE |",
        "|---|---|---|---|---|",
    ]
    for method, r in sorted(d["results"].items(), key=lambda kv: kv[1]["mean_test_mae"]):
        lines.append(
            f"| {method} | {r['hidden_dim']} | {r['n_params']} | "
            f"{r['mean_test_mae']:.4f} +/- {r['std_test_mae']:.4f} | {_per_seed(r)} |"
        )
    return "\n".join(lines) + "\n"


def make_ablation_table(results_dir: str) -> str:
    d = _load(results_dir, "ablation_results.pkl")
    if d is None:
        return ""
    lines = [
        "# ZINC ablation table\n",
        f"Regime: mode={d.get('mode')}, seeds={d.get('seeds')}, epoch={d.get('epoch')}.\n",
        "| Variant | Test MAE (mean +/- std) | Per-seed MAE |",
        "|---|---|---|",
    ]
    order = ["full", "g_only", "lambda_only", "constant", "noise"]
    for variant in order:
        if variant not in d["results"]:
            continue
        r = d["results"][variant]
        lines.append(f"| {variant} | {r['mean_test_mae']:.4f} +/- {r['std_test_mae']:.4f} | {_per_seed(r)} |")
    return "\n".join(lines) + "\n"


def make_descriptor_study_table(results_dir: str) -> str:
    d = _load(results_dir, "descriptor_study_results.pkl")
    ablation = _load(results_dir, "ablation_results.pkl")
    cold = _load(results_dir, "descriptor_study_cold_cache_preprocessing.pkl")
    if d is None:
        return ""
    lines = [
        "# Descriptor-vs-descriptor study\n",
        f"Regime: mode={d.get('mode')}, seeds={d.get('seeds')}, epoch={d.get('epoch')}.\n",
        "`constant`/`full` are reused from ablation_results.pkl, not retrained here.\n",
        "| Variant | hidden_dim | n_params | Test MAE (mean +/- std) | Per-seed MAE | Cold-cache preproc. |",
        "|---|---|---|---|---|---|",
    ]
    rows = dict(d["results"])
    if ablation is not None:
        for variant in ("constant", "full"):
            if variant in ablation["results"]:
                rows[variant] = ablation["results"][variant]
    order = ["constant", "tri", "cyc4", "cyc6", "cyc8", "full"]
    cold_results = cold["results"] if cold is not None else {}
    for variant in order:
        if variant not in rows:
            continue
        r = rows[variant]
        preproc = cold_results.get(variant, {}).get("preprocessing_seconds")
        preproc_str = f"{preproc:.1f}s" if preproc is not None else "n/a"
        lines.append(
            f"| {variant} | {r['hidden_dim']} | {r['n_params']} | "
            f"{r['mean_test_mae']:.4f} +/- {r['std_test_mae']:.4f} | {_per_seed(r)} | {preproc_str} |"
        )
    return "\n".join(lines) + "\n"


def make_brec_table(results_dir: str) -> str:
    d = _load(results_dir, "brec_results.pkl")
    if d is None:
        return ""
    cats = list(next(iter(d["results"].values()))["by_category"].keys())
    lines = [
        "# BREC table\n",
        f"Regime: mode={d.get('mode')}, seed={d.get('seed')}, "
        f"num_relabel={d.get('num_relabel')}, epoch={d.get('epoch')}.\n",
        "| Method | " + " | ".join(cats) + " | Overall | Reliability |",
        "|---|" + "---|" * (len(cats) + 2),
    ]
    for method, r in sorted(d["results"].items(), key=lambda kv: -kv[1]["overall_accuracy"]):
        cat_accs = [f"{r['by_category'][c]['accuracy']:.4f}" for c in cats]
        reliability = 1.0 - r["n_reliable_fail"] / r["n_pairs"] if r["n_pairs"] else float("nan")
        lines.append(
            f"| {method} | " + " | ".join(cat_accs) + f" | {r['overall_accuracy']:.4f} | {reliability:.0%} |"
        )
    return "\n".join(lines) + "\n"


def make_csl_table(results_dir: str) -> str:
    d = _load(results_dir, "csl_results.pkl")
    if d is None:
        return ""
    unreliable = d.get("unreliable_methods", {})
    lines = [
        "# CSL table (annex)\n",
        f"Regime: mode={d.get('mode')}, seed={d.get('seed')}, n_splits={d.get('n_splits')}, "
        f"epoch={d.get('epoch')}, chance={1 / d.get('num_classes', 10):.3f}.\n",
        "| Method | Fold accuracies | Mean +/- std | Unreliable? |",
        "|---|---|---|---|",
    ]
    for method, r in sorted(d["results"].items(), key=lambda kv: -kv[1]["mean_acc"]):
        flag = "yes" if method in unreliable else "no"
        accs = [round(a, 2) for a in r["fold_accuracies"]]
        lines.append(f"| {method} | {accs} | {r['mean_acc']:.3f} +/- {r['std_acc']:.3f} | {flag} |")
    if unreliable:
        lines.append("\n## Why marked unreliable\n")
        for method, reason in unreliable.items():
            lines.append(f"- **{method}**: {reason}\n")
    return "\n".join(lines) + "\n"


def make_contingency_tables(results_dir: str) -> str:
    glob = _load(results_dir, "global_contingency.pkl")
    strat = _load(results_dir, "regular_stratified_analysis.pkl")
    lines = ["# Contingency tables (BREC annex)\n"]
    if glob is not None:
        lines.append("## Global contingency (400 pairs)\n")
        lines.append(f"n_total={glob['n_total']}, n_egr={glob['n_egr']}\n")
        lines.append("| Method | resolved & egr | resolved & not egr | not resolved & egr | not resolved & not egr |")
        lines.append("|---|---|---|---|---|")
        for method, c in glob["contingency_by_method"].items():
            lines.append(
                f"| {method} | {c['resolved_and_egr']} | {c['resolved_and_not_egr']} | "
                f"{c['not_resolved_and_egr']} | {c['not_resolved_and_not_egr']} |"
            )
    if strat is not None:
        lines.append("\n## Regular-category stratification (egr x strongly-regular x resolved)\n")
        lines.append(f"reference_method={strat['reference_method']}, "
                      f"n_egr={strat['n_egr']}, n_sr={strat['n_sr']}, "
                      f"egr_sr_coincide={strat['egr_sr_coincide']}\n")
        lines.append("| Subset | resolved & egr | resolved & not egr | not resolved & egr | not resolved & not egr |")
        lines.append("|---|---|---|---|---|")
        for label, c in (("Plain-regular (not SRG)", strat["plain_contingency"]),
                         ("Strongly-regular (SRG)", strat["strong_contingency"])):
            lines.append(
                f"| {label} | {c['resolved_and_egr']} | {c['resolved_and_not_egr']} | "
                f"{c['not_resolved_and_egr']} | {c['not_resolved_and_not_egr']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.environ.get("EDGEGIRTH_RESULTS_DIR", "results"))
    parser.add_argument("--out-dir", default="paper_tables")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    generators = {
        "zinc_table.md": make_zinc_table,
        "ablation_table.md": make_ablation_table,
        "descriptor_study_table.md": make_descriptor_study_table,
        "brec_table.md": make_brec_table,
        "csl_table.md": make_csl_table,
        "contingency_tables.md": make_contingency_tables,
    }
    print(f"Reading from {args.results_dir}, writing to {args.out_dir}")
    for filename, fn in generators.items():
        content = fn(args.results_dir)
        if not content:
            continue
        out_path = os.path.join(args.out_dir, filename)
        with open(out_path, "w") as f:
            f.write(content)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
