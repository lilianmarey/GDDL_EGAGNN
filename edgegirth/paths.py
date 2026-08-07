"""Central, overridable filesystem locations for data, results, and caches.

Every location defaults to a subdirectory of the current working directory
(the convention used by every `scripts/*.sh` entry point: run from the
repository root) and can be overridden with an environment variable, so
nothing in this package hardcodes an absolute or user-specific path.
"""

from __future__ import annotations

import os

RESULTS_DIR = os.environ.get("EDGEGIRTH_RESULTS_DIR", os.path.join(os.getcwd(), "results"))
DATA_ROOT = os.environ.get("EDGEGIRTH_DATA_DIR", os.path.join(os.getcwd(), "data"))
CACHE_DIR = os.environ.get("EDGEGIRTH_CACHE_DIR", os.path.join(os.getcwd(), ".cache", "edge_girth"))

BREC_NPY = os.path.join(DATA_ROOT, "brec", "brec_v3.npy")
BREC_PYG_CACHE = os.path.join(DATA_ROOT, "brec", "brec_v3_pyg.pt")
