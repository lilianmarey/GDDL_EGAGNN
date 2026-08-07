#!/usr/bin/env bash
# Runs reproduce_zinc.sh, reproduce_descriptor_study.sh, and reproduce_brec.sh
# in sequence -- everything needed to regenerate every table and figure in
# the paper and its appendix.
#
# Usage:
#   ./scripts/reproduce_all.sh              # full run, ~8 hours of CPU total
#   ./scripts/reproduce_all.sh --fast       # smoke mode, ~30 min, NOT paper numbers
#
# WARNING: full mode takes roughly 8 hours of CPU time on the 11-core machine
# this project was developed on (BREC alone is ~6 hours; see
# docs/REPRODUCIBILITY.md for the exact hardware). It is safe to interrupt
# and rerun: every underlying script checkpoints per method/variant, so a
# rerun of this script only redoes whatever was still in flight.
set -euo pipefail

FAST_FLAG=""
if [ "${1:-}" = "--fast" ]; then
    FAST_FLAG="--fast"
    echo "FAST MODE: smoke-mode subsample throughout, roughly 30 minutes total."
    echo "This verifies the full pipeline runs end to end -- it does NOT"
    echo "reproduce the paper's numbers. Do not cite any number it produces."
else
    echo "FULL MODE: this will take roughly 8 hours of CPU time in total"
    echo "(BREC: ~6h, ZINC + ablation: ~2.5h, descriptor study: ~1h -- some"
    echo "overlap is possible if you parallelize manually across machines,"
    echo "but this script runs them sequentially). Interrupt and rerun freely;"
    echo "every step is checkpointed."
fi
echo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "############################################################"
echo "# 1/3: ZINC main table + ablation"
echo "############################################################"
"${SCRIPT_DIR}/reproduce_zinc.sh" $FAST_FLAG

echo "############################################################"
echo "# 2/3: Descriptor-vs-descriptor study"
echo "############################################################"
"${SCRIPT_DIR}/reproduce_descriptor_study.sh" $FAST_FLAG

echo "############################################################"
echo "# 3/3: BREC + contingency analyses + CSL"
echo "############################################################"
"${SCRIPT_DIR}/reproduce_brec.sh" $FAST_FLAG

echo
echo "All reproduction steps finished. See results_reproduced/ (or"
echo "\$EDGEGIRTH_RESULTS_DIR if you set it) for every pickle, and"
echo "paper_tables/ (run scripts/make_paper_tables.py against that directory)"
echo "for the regenerated markdown tables."
