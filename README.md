# Edge-girth GNN: reproduction code

Code and frozen results accompanying the paper on EGAGNN, a graph neural
network using per-edge edge-girth features `(g_e, lambda_e)`.

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

## Tables and figures

Every command below writes to `results_reproduced/` (never to `results/`,
which holds the frozen numbers the paper reports). Run
`scripts/download_data.sh` first if `data/` is empty.

| Paper item | Command | Result file |
|---|---|---|
| Table 1 (ZINC-12k main table) | `scripts/reproduce_zinc.sh` | `zinc_results.pkl` |
| Table 2 (ZINC ablation) | `scripts/reproduce_zinc.sh` | `ablation_results.pkl` |
| Table 3 (descriptor-vs-descriptor study) | `scripts/reproduce_descriptor_study.sh` | `descriptor_study_results.pkl` (+ `ablation_results.pkl` for the `constant`/`full` rows) |
| Table 4 (BREC) | `scripts/reproduce_brec.sh` | `brec_results.pkl` |
| Cold-cache preprocessing costs | `scripts/reproduce_zinc.sh`, `scripts/reproduce_descriptor_study.sh` | `zinc_cold_cache_preprocessing.pkl`, `descriptor_study_cold_cache_preprocessing.pkl` |
| Appendix: global egr contingency | `scripts/reproduce_brec.sh` | `global_contingency.pkl` |
| Appendix: `Regular`-category stratification | `scripts/reproduce_brec.sh` | `regular_stratified_analysis.pkl` |
| Appendix: CFI edge-girth-regularity check | `scripts/reproduce_brec.sh` | `cfi_egr_analysis.pkl` |
| Appendix: CSL (2 methods marked unreliable, see `docs/REPRODUCIBILITY.md`) | `scripts/reproduce_brec.sh` | `csl_results.pkl` |

`scripts/reproduce_all.sh` runs all of the above in sequence. Every script
takes `--fast` for a smoke-mode run (a few minutes, verifies the pipeline
works end to end, **does not reproduce the paper's numbers** — do not cite
anything it prints).

After any of the above, regenerate the paper's markdown tables from whatever
is in a results directory:

```bash
python scripts/make_paper_tables.py --results-dir results_reproduced --out-dir paper_tables_reproduced
```

Or read the already-frozen numbers directly, no computation involved:
`notebooks/results.ipynb`, or the pre-generated `paper_tables/*.md`.

## Expected duration and hardware

Frozen results were produced on an 11-core Apple M3 Pro, 18 GB RAM, with
`n_jobs=10`. Approximate full-mode durations on that machine:

| Script | Duration |
|---|---|
| `reproduce_zinc.sh` | ~2.5 hours |
| `reproduce_descriptor_study.sh` | ~1 hour |
| `reproduce_brec.sh` | ~6 hours (BREC itself is ~5.7h of that) |
| `reproduce_all.sh` | ~8 hours total |
| any script with `--fast` | a few minutes to ~20 minutes (BREC) |

## Structure

```
edgegirth/            the package: features/ (edge-girth), models/, data/
                       (loaders), eval/ (BREC RPC protocol), experiments/
                       (one run_*.py per table), tests/
scripts/               download_data.sh, reproduce_*.sh, make_paper_tables.py
results/               frozen pickles the paper's numbers are read from
paper_tables/           markdown tables regenerated from results/
notebooks/results.ipynb  read-only pickle viewer, no computation
docs/REPRODUCIBILITY.md  seeds, hardware, and every known limitation
```

## Known limitations

See `docs/REPRODUCIBILITY.md` for full detail. In brief: BREC runs at a
single seed, not four, for cost reasons; preprocessing-time measurements
vary by up to 2.5x between otherwise-identical isolated runs on the same
machine; the `cyc8` descriptor-study variant lands at the opposite end of
the ±10% parameter-budget tolerance from edge-girth (90,351 vs. 104,113
params); `gcn`/`gin`/`gatedgcn`/`gatedgcn_mlp`/`gsn4`/`gsn6` do not consume
ZINC's bond-type edge features (`gsn4_bonds`/`gsn6_bonds` were added
specifically to correct this for the GSN comparison); and CSL's `ppgn` and
`folklore2wl` rows are not reliable (documented reasons, not just a flag).
