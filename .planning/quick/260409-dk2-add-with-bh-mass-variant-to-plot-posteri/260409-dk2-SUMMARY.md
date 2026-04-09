---
phase: quick
plan: 260409-dk2
subsystem: plotting
tags: [paper-figures, convergence, dual-variant]
key-decisions:
  - Work on Phase 35 HEAD version of paper_figures.py (not the old working-tree version); worktree had pre-existing WD inconsistency from git reset --soft
  - Combine Tasks 1 and 2 into a single commit since pre-commit mypy checks both files together
key-files:
  created: []
  modified:
    - master_thesis_code/plotting/paper_figures.py
    - master_thesis_code_test/plotting/test_paper_figures.py
tech-stack:
  patterns:
    - _compute_convergence_stats() helper extracts loop body so both variants share identical sampling logic via the same rng instance
---

# Quick Task 260409-dk2: Add with-BH-mass variant to plot_posterior_convergence

**One-liner:** Added `_compute_convergence_stats()` helper and dual-variant errorbar plot (CYCLE[0] orange + CYCLE[3] yellow) to `plot_posterior_convergence`, removing the outdated delta-function docstring.

## Tasks Completed

| Task | Description | Status |
|------|-------------|--------|
| 1 | Add with-BH-mass variant to plot_posterior_convergence | Done |
| 2 | Add smoke test for dual-variant convergence plot | Done |

## Commits

| Hash | Message | Files |
|------|---------|-------|
| dc879d4 | feat(quick/260409-dk2): add with-BH-mass variant to plot_posterior_convergence | paper_figures.py, test_paper_figures.py |

## What Was Built

### paper_figures.py changes

- **Module docstring** (line 9): "without-BH-mass channel only" → "both analysis variants"
- **Section comment** (line 400): "without-BH-mass only" → "both analysis variants"
- **`_compute_convergence_stats()`** (new helper): Extracts the subset-sampling loop from `plot_posterior_convergence` into a reusable function. Takes `log_event_matrix`, `n_events_total`, `subset_sizes`, `n_subsets`, `rng`, `h_values` and returns `(used_sizes, medians, lo_pctiles, hi_pctiles)`.
- **`plot_posterior_convergence()`** updated docstring: Removed the paragraph claiming with-BH-mass posteriors collapse to delta functions. Added `posteriors_with_bh_mass/` to parameter description.
- **`plot_posterior_convergence()`** body: Loads `posteriors_with_bh_mass/` via `_load_per_event_with_mass_scalars`, calls `_compute_convergence_stats` twice (sharing the same `rng`), plots second errorbar with `CYCLE[3]` (yellow) and marker `"s"`. Reference line anchored to no-mass variant's largest-N median (unchanged).

### test_paper_figures.py changes

- Added `import json` at top level (needed by new test class)
- Added **`TestPosteriorConvergenceDualVariant`** class with:
  - `_write_per_event_json()` helper: creates `h_{int}_{frac}.json` files matching the loader's naming convention with random single-element list values
  - `test_posterior_convergence_with_synthetic_data()`: creates synthetic JSON files in both `posteriors/` and `posteriors_with_bh_mass/`, calls `plot_posterior_convergence(subset_sizes=[5, 10, 15], n_subsets=5)`, asserts `isinstance(fig, Figure)` and `len(ax.containers) >= 2`

## Deviations from Plan

### Worktree state deviation

The worktree had been set up via `git reset --soft` which left the working tree in a pre-Phase-35 state while HEAD was at Phase 35 (`4314d75`). The plan was written assuming work on the pre-Phase-35 version of `paper_figures.py`, but the correct base is HEAD (Phase 35). I restored `paper_figures.py` and `_helpers.py` from HEAD before applying changes to ensure consistency with the rest of the Phase 35 codebase (`compute_credible_interval`, no `_DATA_ROOT`, no `main()`).

This was an auto-fix (Rule 3 — blocking issue: committing against the wrong base would have caused mypy failures from HEAD's `test_paper_figures.py` referencing Phase 35 functions).

## Known Stubs

None — both variants are fully wired. `plot_posterior_convergence` now loads real data from both subdirectories. No placeholder text or hardcoded empty values.

## Threat Flags

None — pure plotting code with no new network endpoints, auth paths, or trust boundary crossings.

## Self-Check

Files to verify:
- `master_thesis_code/plotting/paper_figures.py` — modified with `_compute_convergence_stats` and dual-variant convergence
- `master_thesis_code_test/plotting/test_paper_figures.py` — modified with `TestPosteriorConvergenceDualVariant`
- Commit `dc879d4` — exists and contains both files
