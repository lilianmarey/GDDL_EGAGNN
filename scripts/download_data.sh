#!/usr/bin/env bash
# Downloads BREC, ZINC, and CSL into a configurable data directory.
#
# ZINC and CSL are fetched by torch_geometric's own dataset classes, which
# know their own official download URLs; this script just triggers that
# download by instantiating them once. BREC has no PyG-native loader, so it
# is fetched directly from the official GraphPKU/BREC GitHub repository, from
# the commit this project's frozen results were produced against.
#
# Usage:
#   ./scripts/download_data.sh [DATA_DIR]
# DATA_DIR defaults to ./data (matching EDGEGIRTH_DATA_DIR's default in
# edgegirth/paths.py). Safe to re-run: skips any file whose checksum already
# matches.
set -euo pipefail

DATA_DIR="${1:-${EDGEGIRTH_DATA_DIR:-$(pwd)/data}}"
BREC_COMMIT="d09e8c349a8bbc0882d2932f7b37b2726f576ce9"
BREC_ZIP_URL="https://raw.githubusercontent.com/GraphPKU/BREC/${BREC_COMMIT}/BREC_data_all.zip"

echo "Data directory: ${DATA_DIR}"
echo "Expected total size: ~450 MB combined (BREC ~245 MB uncompressed from a"
echo "7 MB download, ZINC ~50 MB, CSL <5 MB). On a typical broadband"
echo "connection this takes 1-3 minutes; BREC's unzip step (three .npy files)"
echo "is the slowest part, well under a minute on any modern machine."
echo

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

verify_or_warn() {
    local file="$1" expected="$2"
    local actual
    actual="$(sha256_of "$file")"
    if [ "$actual" != "$expected" ]; then
        echo "  WARNING: checksum mismatch for $file"
        echo "    expected: $expected"
        echo "    got:      $actual"
        echo "    This means your copy of the data differs from the one this"
        echo "    project's frozen results were computed on -- reproduced"
        echo "    numbers may not match the paper exactly."
        return 1
    fi
    echo "  OK: $file"
    return 0
}

# --- BREC ---------------------------------------------------------------
BREC_DIR="${DATA_DIR}/brec"
BREC_NPY="${BREC_DIR}/brec_v3.npy"
BREC_NPY_SHA256="6975d172d27aedcdf4eb2c747450ede7928c111caeb2689fbc67e6b62fbd5562"
BREC_ZIP_SHA256="3a2c8e7ba068f774b1115422d1bc04749432dde9f19631e6625cba142659530f"

mkdir -p "$BREC_DIR"
if [ -f "$BREC_NPY" ] && verify_or_warn "$BREC_NPY" "$BREC_NPY_SHA256"; then
    echo "[BREC] already present and verified, skipping download."
else
    echo "[BREC] downloading BREC_data_all.zip from GraphPKU/BREC @ ${BREC_COMMIT} ..."
    curl -fL -o "${BREC_DIR}/BREC_data_all.zip" "$BREC_ZIP_URL"
    verify_or_warn "${BREC_DIR}/BREC_data_all.zip" "$BREC_ZIP_SHA256" || true
    echo "[BREC] unzipping (brec_v3.npy, brec_v3_3wl.npy, brec_v3_no4v_60cfi.npy) ..."
    unzip -o -j "${BREC_DIR}/BREC_data_all.zip" "brec_v3.npy" -d "$BREC_DIR"
    verify_or_warn "$BREC_NPY" "$BREC_NPY_SHA256"
fi

# --- ZINC and CSL ---------------------------------------------------------
# Delegated to torch_geometric's own dataset classes: they know their own
# official, versioned download URLs, so this script does not hardcode them.
echo "[ZINC] triggering download via torch_geometric.datasets.ZINC ..."
python3 -c "
from torch_geometric.datasets import ZINC
import os
root = os.path.join('${DATA_DIR}', 'ZINC')
for split in ('train', 'val', 'test'):
    ZINC(root=root, subset=True, split=split)
print('[ZINC] done.')
"

echo "[CSL] triggering download via torch_geometric.datasets.GNNBenchmarkDataset ..."
python3 -c "
from torch_geometric.datasets import GNNBenchmarkDataset
import os
root = os.path.join('${DATA_DIR}', 'CSL')
GNNBenchmarkDataset(root=root, name='CSL')
print('[CSL] done.')
"

echo
echo "Reference SHA-256 for the raw ZINC/CSL files this project used (not"
echo "re-verified above, since torch_geometric does its own integrity checks"
echo "on download; provided here only so you can cross-check independently):"
cat <<'EOF'
  ZINC/raw/train.pickle       9c76df5a1ab90aac200df22902893827168aeced3df87aab4b6bee100235d7f2
  ZINC/raw/val.pickle         4c6015ec483224c55593c862b0334ae4b64e14679eab041c0655c6a7b603e47a
  ZINC/raw/test.pickle        33c7d2bcf6332be422aaea4baaf056e36f2c80eb6f7d0f5d68a8264d1b94e65b
  ZINC/raw/atom_dict.pickle   c4c8e8625c9773bb725ec2771552eccb2b6cf6550c8fef0f6139627aa8fe51c6
  ZINC/raw/bond_dict.pickle   623359be444efd15db6eec412ea9ba9a12bf1afec57305f9f5abd4b4a52c541a
  CSL/raw/graphs_Kary_Deterministic_Graphs.pkl   356281ea02e70241ae462426025afa72d00f6645e82df836dc3f29ad69cf9461
EOF
echo
echo "Done. Set EDGEGIRTH_DATA_DIR=${DATA_DIR} before running any reproduce_*.sh script."
