#!/usr/bin/env bash
# Reproduces the ZINC-12k main table, the edge-girth ablation, and the
# cold-cache preprocessing-cost measurement.
#
# Usage:
#   ./scripts/reproduce_zinc.sh              # full run, ~2 hours on an 11-core CPU
#   ./scripts/reproduce_zinc.sh --fast       # smoke mode, a few minutes, NOT paper numbers
#
# Writes to results_reproduced/ by default (never to results/, which holds
# the frozen numbers this script's output should be compared against).
# Override the output directory with EDGEGIRTH_RESULTS_DIR.
set -euo pipefail

MODE="full"
if [ "${1:-}" = "--fast" ]; then
    MODE="smoke"
fi

export EDGEGIRTH_RESULTS_DIR="${EDGEGIRTH_RESULTS_DIR:-$(pwd)/results_reproduced}"
export EDGEGIRTH_DATA_DIR="${EDGEGIRTH_DATA_DIR:-$(pwd)/data}"
mkdir -p "$EDGEGIRTH_RESULTS_DIR"

echo "=== reproduce_zinc.sh (mode=${MODE}) ==="
echo "Output directory: ${EDGEGIRTH_RESULTS_DIR}"
if [ "$MODE" = "full" ]; then
    echo "Estimated duration: ~50 min (main table, 5 original methods) + ~20 min"
    echo "(6 added baselines) + ~70 min (ablation) + ~5 min (cold-cache timing)"
    echo "= roughly 2.5 hours total on an 11-core CPU. Resumable: rerunning this"
    echo "script skips any (method, regime) already checkpointed to disk."
else
    echo "FAST MODE: smoke-mode subsample, a few minutes. This checks that the"
    echo "pipeline runs end to end -- it does NOT reproduce the paper's numbers."
fi
echo

echo "--- ZINC main table (10 methods: egagnn + 5 architecture baselines + 4"
echo "    bounded-motif variants) ---"
python -m edgegirth.experiments.run_zinc --mode "$MODE" --n_jobs "${EDGEGIRTH_NJOBS:-10}" \
    --methods egagnn gcn gin gatedgcn gatedgcn_mlp triangle gsn4 gsn6 gsn4_bonds gsn6_bonds

echo "--- ZINC ablation (5 structural-feature variants) ---"
python -m edgegirth.experiments.run_ablation --mode "$MODE" --n_jobs "${EDGEGIRTH_NJOBS:-10}"

if [ "$MODE" = "full" ]; then
    echo "--- Cold-cache preprocessing time (isolated measurement, egagnn +"
    echo "    triangle + gsn4 + gsn6) ---"
    python -m edgegirth.experiments.run_zinc_cold_cache_preprocessing
else
    echo "--- Skipping cold-cache timing in fast mode (it is a wall-clock"
    echo "    measurement, meaningless on a subsampled run) ---"
fi

echo
echo "Done. Results written under ${EDGEGIRTH_RESULTS_DIR}/:"
echo "  zinc_results.pkl, ablation_results.pkl$( [ "$MODE" = "full" ] && echo ", zinc_cold_cache_preprocessing.pkl")"
