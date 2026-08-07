#!/usr/bin/env bash
# Reproduces the BREC table, the four BREC-derived contingency/annex analyses
# (global contingency, Regular-category stratification, CFI edge-girth-
# regularity check), and the CSL annex table (grouped here for convenience;
# CSL does not depend on BREC).
#
# Usage:
#   ./scripts/reproduce_brec.sh              # full run, ~6 hours on an 11-core CPU
#   ./scripts/reproduce_brec.sh --fast       # smoke mode, ~20 min, NOT paper numbers
#
# BREC full mode is the single most expensive step in this whole repository.
# It is run here in two invocations, non-learned methods first: if the run is
# interrupted after the first invocation, the second can be re-launched alone
# without repeating the first, since results are checkpointed per method.
set -euo pipefail

MODE="full"
if [ "${1:-}" = "--fast" ]; then
    MODE="smoke"
fi

export EDGEGIRTH_RESULTS_DIR="${EDGEGIRTH_RESULTS_DIR:-$(pwd)/results_reproduced}"
export EDGEGIRTH_DATA_DIR="${EDGEGIRTH_DATA_DIR:-$(pwd)/data}"
mkdir -p "$EDGEGIRTH_RESULTS_DIR"

echo "=== reproduce_brec.sh (mode=${MODE}) ==="
echo "Output directory: ${EDGEGIRTH_RESULTS_DIR}"
if [ "$MODE" = "full" ]; then
    echo "Estimated duration: ~20 min (2 non-learned methods) + ~5.5 h (7 learned"
    echo "methods) + ~3 min (contingency analyses) + <1 min (CSL) = roughly 6"
    echo "hours total on an 11-core CPU. This is the slow step -- everything"
    echo "else in this repository is at most ~2.5 hours. Resumable: rerunning"
    echo "with the same --methods subset skips methods already checkpointed."
else
    echo "FAST MODE: smoke-mode subsample (40 pairs/category, 8 relabelings,"
    echo "5 epochs), ~20 min. This checks that the pipeline runs end to end --"
    echo "it does NOT reproduce the paper's numbers."
fi
echo

if [ "$MODE" = "full" ]; then
    echo "--- BREC, non-learned methods first (edge_girth_seq, folklore2wl) ---"
    python -m edgegirth.experiments.run_brec --mode full --n_jobs "${EDGEGIRTH_NJOBS:-10}" \
        --methods edge_girth_seq folklore2wl

    echo "--- BREC, remaining 7 learned methods ---"
    python -m edgegirth.experiments.run_brec --mode full --n_jobs "${EDGEGIRTH_NJOBS:-10}" \
        --methods egagnn gcn gin gatedgcn gsn ppgn triangle
else
    echo "--- BREC, all methods, smoke mode ---"
    python -m edgegirth.experiments.run_brec --mode smoke --n_jobs "${EDGEGIRTH_NJOBS:-4}"
fi

echo "--- BREC-derived contingency and annex analyses ---"
python -m edgegirth.experiments.run_regular_analysis
python -m edgegirth.experiments.run_regular_stratified_analysis
python -m edgegirth.experiments.run_cfi_egr_analysis
python -m edgegirth.experiments.run_global_contingency

echo "--- CSL annex table (5-fold classification, 8 methods) ---"
python -m edgegirth.experiments.run_csl --mode "$MODE" --n_jobs "${EDGEGIRTH_NJOBS:-10}"

echo
echo "Done. Results written under ${EDGEGIRTH_RESULTS_DIR}/:"
echo "  brec_results.pkl, regular_analysis.pkl, regular_stratified_analysis.pkl,"
echo "  cfi_egr_analysis.pkl, global_contingency.pkl, csl_results.pkl"
