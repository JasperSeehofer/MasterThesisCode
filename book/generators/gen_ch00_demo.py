"""Generator for the ch00 demo page — proves the offline data pipeline end to end.

Reads ONE delivered set of per-event, per-h posterior JSON files from a real
campaign run (results/campaign51_20260728/.../posteriors/) and reuses the
project's OWN combination logic
(:mod:`master_thesis_code.bayesian_inference.posterior_combination`) to build
combined H0 posteriors at a ladder of "number of stacked events" N.

This is deliberately NOT a re-derivation: the physics-floor zero-handling
strategy and the log-space combination (Sigma log L_i, no extra beta(h)^N
term -- see combine_log_space docstring, Loredo 2004 / Mandel et al. 2019)
are exactly the ratified production code path. The book only *slices* the
event list to a growing subset size and *re-normalizes*, so a reader can
watch a real posterior sharpen as N grows.

Output: book/site/data/ch00_demo.json

    {
      "h_grid": [...],                  # 41 h-values from the source run
      "h_true": 0.73,                   # constants.H fiducial value
      "n_events_total": <int>,
      "sizes": [1, 5, 10, 25, 50, 100, ...],
      "posteriors": {"<N>": [...41 density values...], ...},
      "source": {"run": "...", "strategy": "physics-floor"}
    }

Run from either worktree (or CI) with a synced venv, e.g.:
    /home/jasper/Repositories/MasterThesisCode/.venv/bin/python \\
        book/generators/gen_ch00_demo.py
or, once this repo has its own `.venv` (`uv sync --extra cpu --extra dev`):
    uv run python book/generators/gen_ch00_demo.py

``master_thesis_code/`` and ``results/`` are ordinary tracked files in this
git repository (verified: both are `git ls-files`-tracked and identical
across worktrees at a given commit) -- so REPO_ROOT is resolved relative to
this script, never hardcoded to one machine's checkout path. Only the
*Python interpreter* (the venv with `few`/numpy/etc. installed) needs to be
pointed at explicitly; the source it imports/reads is always local to
whichever checkout is running this generator.

Deterministic: the event ordering used to build nested subsets is a fixed
seeded permutation (seed=42), matching the convention already used by
master_thesis_code.plotting.interactive.interactive_h0_tension_explorer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# --- resolve the repo root relative to this file (book/generators/ -> book/ -> root) ---
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from master_thesis_code.bayesian_inference.posterior_combination import (  # noqa: E402
    CombinationStrategy,
    apply_strategy,
    build_likelihood_array,
    combine_log_space,
    load_posterior_jsons,
)
from master_thesis_code.constants import H as H_TRUE  # noqa: E402

SOURCE_POSTERIORS_DIR = (
    REPO_ROOT
    / "results"
    / "campaign51_20260728"
    / "realistic_20260729"
    / "seed61000"
    / "real_r1"
    / "posteriors"
)

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "site" / "data" / "ch00_demo.json"

SUBSET_SIZES = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]
SEED = 42


def _combined_for_subset(
    likelihoods: np.ndarray, order: np.ndarray, n: int, h_grid: list[float]
) -> list[float]:
    """Return the peak-normalized combined posterior over the first *n* (permuted) events."""
    idx = order[:n]
    subset = likelihoods[idx, :]
    processed, _excluded = apply_strategy(subset, CombinationStrategy.PHYSICS_FLOOR)
    posterior = combine_log_space(processed)
    norm = float(np.trapezoid(posterior, h_grid))
    if norm > 0:
        posterior = posterior / norm
    return [float(v) for v in posterior]


def main() -> None:
    if not SOURCE_POSTERIORS_DIR.is_dir():
        msg = f"Source posteriors directory not found: {SOURCE_POSTERIORS_DIR}"
        raise FileNotFoundError(msg)

    h_values, event_likelihoods = load_posterior_jsons(SOURCE_POSTERIORS_DIR)
    likelihoods, detection_indices = build_likelihood_array(h_values, event_likelihoods)
    n_events_total = len(detection_indices)

    # NaN-safe: any event missing an h-value cell is dropped entirely (rare;
    # keeps every kept row fully populated across the h-grid).
    complete_mask = ~np.any(np.isnan(likelihoods), axis=1)
    likelihoods = likelihoods[complete_mask, :]
    n_complete = int(likelihoods.shape[0])

    rng = np.random.default_rng(SEED)
    order = rng.permutation(n_complete)

    sizes = sorted({s for s in SUBSET_SIZES if s <= n_complete} | {n_complete})

    posteriors: dict[str, list[float]] = {
        str(n): _combined_for_subset(likelihoods, order, n, h_values) for n in sizes
    }

    payload = {
        "h_grid": [float(v) for v in h_values],
        "h_true": float(H_TRUE),
        "n_events_total": n_events_total,
        "n_events_complete": n_complete,
        "sizes": sizes,
        "posteriors": posteriors,
        "source": {
            "run": str(SOURCE_POSTERIORS_DIR.relative_to(REPO_ROOT)),
            "strategy": "physics-floor",
            "combination": "combine_log_space (Sigma log L_i, no beta(h)^N term)",
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {OUTPUT_PATH} ({n_complete}/{n_events_total} complete events, {len(sizes)} sizes)")


if __name__ == "__main__":
    main()
