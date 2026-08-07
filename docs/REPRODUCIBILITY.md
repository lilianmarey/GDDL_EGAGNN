# Reproducibility notes

This document states plainly what is and isn't controlled in this codebase's
results, so a reader can judge how much to trust a rerun. Nothing below is
minimized for effect.

## Seeds and what remains stochastic

- **ZINC (main table, ablation, descriptor study)**: seeds `13, 14, 15, 16`,
  one model trained per seed, mean and std reported across the four. Every
  seed controls `torch.manual_seed` before model construction and training
  (`_train_eval` in `edgegirth/experiments/run_zinc.py`); the seed is always
  passed as an explicit function argument, never fixed at import time.
- **BREC**: a **single seed, 13**, for the entire 400-pair campaign. This is
  a departure from ZINC's 4-seed protocol, made for cost reasons: BREC's
  full run is roughly 9 GPU/CPU-hours already; running it at 4 seeds would
  cost proportionally more for a benchmark whose primary output (pass/fail
  per pair under the RPC test) is far less seed-sensitive than a regression
  MAE. Each of BREC's 32 relabelings per pair already provides an internal
  form of averaging (the RPC T² test operates over all 32 relabelings
  jointly), which is the main variance-reduction mechanism BREC relies on
  instead of outer-seed repetition. Rerunning BREC end to end with a
  different seed has not been done in this project; if you do, expect
  possible per-pair flips in categories with borderline T² statistics
  (chiefly `Regular`), not a change in the qualitative pattern (structural
  baselines resolving Basic/Extension well, all methods failing categories
  designed to be 3-WL/DRG-hard).
- **CSL**: seed 13, `StratifiedKFold(n_splits=5, shuffle=True, random_state=13)`.
  Every fold in the frozen results reports *identical* accuracy across all 5
  folds for every method (std = 0.000 throughout) — stated as-is; this
  reflects CSL's small size (150 graphs, 15 per class) rather than a
  computation error.
- **Non-learned methods** (`edge_girth_seq`, `folklore2wl`) are deterministic
  given the input graphs — there is nothing stochastic to seed.

## Preprocessing-time measurements vary substantially run to run

Two isolated, cache-cleared measurements of the *identical* computation
(edge-girth preprocessing on the full 12,000-graph ZINC-12k set, no
concurrent load, `.cache/edge_girth/` deleted immediately before timing) gave
**71.67s** and **182.3s** — a 2.5x spread with no methodology difference
between the two runs, only different points in time on the same machine.
Two earlier, non-isolated measurements of the same computation (taken as a
byproduct of a training run rather than a dedicated timing pass) gave 79.5s
and 138.7s. **Any single preprocessing-time number in this codebase's history
should be read as one sample from a wide distribution on this specific
machine, not a fixed cost of the algorithm.** If you rerun
`scripts/reproduce_zinc.sh` or `scripts/reproduce_descriptor_study.sh`, do
not expect your timing numbers to match the ones quoted in the paper closely
— expect them to be the right order of magnitude and to preserve the
relative ranking between methods measured in the same run, not to reproduce
a specific second count.

## Parameter-budget matching: ±10%, and where it bites

Every ZINC-table and descriptor-study row targets 100,000 parameters via a
binary search on `hidden_dim` (`match_param_budget` in
`edgegirth/experiments/run_zinc.py`), independently per method/descriptor —
there is no single shared architecture size. Most rows land within a few
percent of the target, but the **`cyc8` bounded-cycle-count variant reaches
only 90,351 parameters (−9.6%)**, against **104,113 for the edge-girth
`full` variant (+4.1%)** — both within the stated ±10% tolerance, but at
opposite ends of it, a roughly 14,000-parameter spread between two rows in
the same table. This happens because a wider structural descriptor (`cyc8`
has 6 informative dimensions vs. `full`'s 2+1) feeds into a wider edge
embedding and `psi` MLP at any given `hidden_dim`, so the binary search
settles on a *smaller* `hidden_dim` for `cyc8` to land back near the same
parameter target. This is disclosed, not hidden: if you need every row at
the exact same parameter count for a stricter comparison, this codebase does
not provide that; it matches the target within the stated tolerance only.

## Which ZINC baselines consume bond-type edge features, and which don't

| Method(s) | Consumes ZINC bond types? |
|---|---|
| `egagnn`, `triangle`, all six descriptor-study variants | **Yes** (`EGAGNNRegressor`'s `e0` always includes the bond one-hot) |
| `gsn4_bonds`, `gsn6_bonds` | **Yes** (`GSNModel` with `edge_dim` set, using `GINEConv`) |
| `gsn4`, `gsn6` | No (`GSNModel` with `edge_dim=None`, plain `GINConv`) |
| `gcn`, `gin`, `gatedgcn`, `gatedgcn_mlp` | **No** — none of these architectures' conv layers are ever called with `edge_attr` in this codebase's ZINC pipeline |

This means any comparison against `gcn`/`gin`/`gatedgcn`/`gatedgcn_mlp` in
the ZINC table is a comparison of *architecture*, not of *architecture with
equal access to chemistry* — those four baselines are structurally blind to
bond types regardless of what else is being varied. `gsn4`/`gsn6` carry the
same limitation; `gsn4_bonds`/`gsn6_bonds` were added specifically to remove
it for the GSN comparison (see the paper for why the original `gsn4`/`gsn6`
comparison was a confound, and why both the corrected and uncorrected
versions are kept in the table).

## CSL annex: two methods are not reliable, with the specific reason

CSL's frozen results (`results/csl_results.pkl['unreliable_methods']`) flag
two of the eight evaluated methods:

- **`ppgn`**: training loss decreases from 1962.4 to 2.73 over 100 epochs
  (`results/csl_ppgn_loss_check.pkl` is not included in this release — see
  below — but the loss trajectory's key values are captured verbatim in the
  quoted note) but never drops below the chance-level cross-entropy,
  `ln(10) = 2.3026` (its minimum over training is 2.462, still above chance).
  The model is training — the loss is moving — but it is not learning a
  discriminative representation; its reported accuracy (exactly 0.100, exact
  chance for 10 classes) has no evidential value one way or the other.
- **`folklore2wl`**: CSL evaluates it via a canonical per-graph embedding (a
  sorted histogram of stable-colour class sizes from a single-graph 2-FWL
  refinement) fed to logistic regression — a lossy summary of the 2-FWL
  colouring, not the exact pairwise 2-FWL test this codebase uses correctly
  on BREC (`folklore_2wl_distinguishable`, which does direct multiset
  comparison with no classifier in the loop). Two 2-FWL-inequivalent graphs
  can in principle produce the same colour-class-size histogram, so a wrong
  CSL prediction can come from the embedding's lossiness, the linear
  classifier, or both — not necessarily from a failure of 2-FWL itself. Its
  CSL accuracy (0.200) should not be read as a 3-WL power result.

Note on scope: the standalone PPGN loss-curve diagnostic script and its
output pickle (`run_csl_ppgn_loss_check.py` /
`results/csl_ppgn_loss_check.pkl`) were excluded from this release as a
one-off diagnostic that produces no table in the paper — the loss-curve
numbers it produced are preserved verbatim in
`csl_results.pkl['unreliable_methods']['ppgn']` (quoted above), so nothing
about the PPGN finding is lost, only the standalone script that first
surfaced it.

## Hardware

All frozen results in this repository were produced on:

| | |
|---|---|
| CPU | Apple M3 Pro |
| Physical / logical cores | 11 / 11 |
| RAM | 18 GB |
| OS | macOS (Darwin, arm64) |
| `n_jobs` used for full-mode runs | 10 |

What could differ elsewhere:

- **Fewer/more cores** changes wall-clock time roughly proportionally for
  the `Parallel(n_jobs=...)`-parallelized preprocessing and multi-seed
  training steps, but should not change the trained models' numbers (each
  seed's training is single-threaded, `torch.set_num_threads(1)`, so
  parallelism only affects how many seeds/pairs run concurrently, not any
  individual training run's arithmetic).
- **A different CPU architecture** (x86 vs. this project's arm64) could in
  principle produce tiny floating-point differences in the fourth decimal
  place or beyond, through different BLAS/vectorization paths in PyTorch —
  not tested in this project, since only one machine was used throughout.
- **Available RAM below ~8 GB** may not be sufficient for BREC's full run:
  `load_brec_raw()` loads and caches 51,200 small graphs in memory at once
  (see `edgegirth/data/loaders.py`); this was not a constraint on the 18 GB
  development machine and has not been tested at lower memory.

## Environment actually verified

Python 3.12.7, `torch==2.11.0`, `torch-geometric==2.7.0` (see the root
`README.md` for why these versions, not the higher versions a naive `pip
freeze` might suggest, were pinned). This exact combination was installed
into a brand-new virtual environment and both `python -m pytest` and a full
model-construction smoke test were run against it as part of preparing this
release — see `RELEASE_NOTES.md` in the working repository this release was
cut from for that test's output.
