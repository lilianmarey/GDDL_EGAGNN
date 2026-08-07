"""Shared verbosity helpers for full-mode runs: live progress with ETA,
block-start announcements with a time estimate, and incremental checkpoint
writes so a crash doesn't lose completed work.
"""

from __future__ import annotations

import os
import pickle
import time

from joblib import Parallel


def run_parallel_with_progress(jobs, n_jobs, label="", log_every=10):
    """Runs joblib jobs, logging elapsed/ETA every `log_every` completions.

    Uses joblib's return_as="generator" (needs joblib >= 1.3) so progress can
    be reported as results stream back, instead of the default behaviour of
    blocking silently until every job is done.
    """
    total = len(jobs)
    if total == 0:
        return []
    t0 = time.perf_counter()
    results = []
    with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
        for i, res in enumerate(parallel(jobs), 1):
            results.append(res)
            if i % log_every == 0 or i == total:
                elapsed = time.perf_counter() - t0
                rate = elapsed / i
                eta = rate * (total - i)
                print(
                    f"  [{label}] {i}/{total} done -- elapsed {elapsed:.0f}s, "
                    f"ETA {eta:.0f}s ({rate:.2f}s/item avg)",
                    flush=True,
                )
    return results


def announce_block(label, estimated_seconds=None, extra=""):
    ts = time.strftime("%H:%M:%S")
    est = f", estimated ~{estimated_seconds / 60:.1f} min" if estimated_seconds else ""
    print(f"\n[{ts}] === STARTING: {label}{est} === {extra}", flush=True)
    return time.perf_counter()


def finish_block(label, t_start, expected_seconds=None):
    elapsed = time.perf_counter() - t_start
    ts = time.strftime("%H:%M:%S")
    msg = f"[{ts}] === DONE: {label} -- {elapsed:.1f}s ({elapsed / 60:.1f} min) ==="
    if expected_seconds and elapsed > expected_seconds * 1.5:
        msg += (
            f"\n  *** OVERRAN ESTIMATE by {elapsed / expected_seconds:.1f}x "
            f"(expected ~{expected_seconds:.0f}s) -- consider reducing scope for "
            f"remaining blocks rather than continuing silently ***"
        )
    print(msg, flush=True)
    return elapsed


def estimate_full_time(
    smoke_elapsed,
    smoke_pairs,
    smoke_num_relabel,
    smoke_epoch,
    full_pairs,
    full_num_relabel,
    full_epoch,
    learned=True,
):
    """Rough linear extrapolation from a measured smoke-mode timing to a full-mode
    estimate. Pair count scales linearly; for learned methods, relabel count and
    epoch count also scale roughly linearly (more graphs per batch, more passes).
    Non-learned methods (folklore2wl, edge_girth_seq) are insensitive to epoch and
    only weakly sensitive to relabel count (they use one relabeling regardless)."""
    pair_factor = full_pairs / smoke_pairs
    if learned:
        relabel_factor = full_num_relabel / smoke_num_relabel
        epoch_factor = full_epoch / smoke_epoch
        return smoke_elapsed * pair_factor * relabel_factor * epoch_factor
    return smoke_elapsed * pair_factor


def checkpoint_save(path, obj):
    """Overwrites the results pickle with the current (possibly partial) state.
    Called after each method/block completes so a crash mid-run only loses the
    block in progress, not everything already done."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)  # atomic on POSIX -- never leaves a half-written file
    print(f"  [checkpoint] wrote {os.path.basename(path)}", flush=True)
