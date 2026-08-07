#!/usr/bin/env bash
# Reproduces the descriptor-vs-descriptor study (architecture and bond types
# held fixed, only the structural descriptor varies: constant / triangle /
# bounded cycle counts at k=4,6,8 / edge-girth) and its cold-cache
# preprocessing measurement.
#
# `constant` and `full` (edge-girth) are reused from the ablation results
# rather than retrained -- run reproduce_zinc.sh first if you want those two
# numbers regenerated from scratch too; this script only trains the four new
# variants (tri, cyc4, cyc6, cyc8).
#
# Usage:
#   ./scripts/reproduce_descriptor_study.sh              # full run, ~1 hour on an 11-core CPU
#   ./scripts/reproduce_descriptor_study.sh --fast       # smoke mode, a few minutes, NOT paper numbers
set -euo pipefail

MODE="full"
if [ "${1:-}" = "--fast" ]; then
    MODE="smoke"
fi

export EDGEGIRTH_RESULTS_DIR="${EDGEGIRTH_RESULTS_DIR:-$(pwd)/results_reproduced}"
export EDGEGIRTH_DATA_DIR="${EDGEGIRTH_DATA_DIR:-$(pwd)/data}"
mkdir -p "$EDGEGIRTH_RESULTS_DIR"

echo "=== reproduce_descriptor_study.sh (mode=${MODE}) ==="
echo "Output directory: ${EDGEGIRTH_RESULTS_DIR}"
if [ "$MODE" = "full" ]; then
    echo "Estimated duration: ~55 min (4 variants x 4 seeds x 200 epochs) + a"
    echo "few minutes for the cold-cache timing pass = roughly 1 hour on an"
    echo "11-core CPU. Resumable: rerunning skips variants already checkpointed."
else
    echo "FAST MODE: smoke-mode subsample, a few minutes. This checks that the"
    echo "pipeline runs end to end -- it does NOT reproduce the paper's numbers."
fi
echo

echo "--- Descriptor study training (tri, cyc4, cyc6, cyc8) ---"
python -m edgegirth.experiments.run_descriptor_study --mode "$MODE" --n_jobs "${EDGEGIRTH_NJOBS:-10}"

if [ "$MODE" = "full" ]; then
    echo "--- Cold-cache preprocessing time (isolated measurement, all 6 variants) ---"
    python -m edgegirth.experiments.run_descriptor_study_cold_cache
else
    echo "--- Skipping cold-cache timing in fast mode ---"
fi

echo
echo "Done. Results written under ${EDGEGIRTH_RESULTS_DIR}/:"
echo "  descriptor_study_results.pkl$( [ "$MODE" = "full" ] && echo ", descriptor_study_cold_cache_preprocessing.pkl")"
echo "(constant/full rows in the paper's table come from ablation_results.pkl -- see reproduce_zinc.sh)"
