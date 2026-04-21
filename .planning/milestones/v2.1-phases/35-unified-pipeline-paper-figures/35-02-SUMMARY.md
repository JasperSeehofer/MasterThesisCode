---
phase: 35-unified-pipeline-paper-figures
plan: "02"
subsystem: plotting
tags: [refactor, testing, paper-figures, manifest, data-threading]
dependency_graph:
  requires: [35-01 compute_credible_interval in _helpers.py]
  provides: [paper figure entries 16-19 in generate_figures() manifest]
  affects: [master_thesis_code/main.py, master_thesis_code/plotting/paper_figures.py]
tech_stack:
  added: []
  patterns: [manifest closure pattern, lazy import inside closures, graceful degradation on missing data]
key_files:
  created:
    - master_thesis_code_test/plotting/test_paper_figures.py
  modified:
    - master_thesis_code/plotting/paper_figures.py
    - master_thesis_code/main.py
decisions:
  - "Made data_dir a required parameter (no default) on all 4 public paper figure functions — callers always pass it explicitly"
  - "Removed apply_style and save_figure imports from paper_figures.py — only used by deleted main()"
metrics:
  duration: "~3 minutes"
  completed: "2026-04-08"
  tasks_completed: 2
  tasks_total: 2
  files_changed: 3
requirements:
  - PFIG-01
  - PFIG-02
---

# Phase 35 Plan 02: Wire Paper Figures into Unified Manifest Summary

Wired all 4 paper figure functions into the unified `generate_figures()` manifest in `main.py` as entries 16-19, removed `_DATA_ROOT` and the standalone `main()` from `paper_figures.py`, threaded `data_dir` through all internal loaders, replaced the private `_ci_width_from_log_posteriors` with the shared `compute_credible_interval` helper, and added 9 smoke tests confirming importability, signatures, graceful degradation, and absence of the standalone entry point.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Thread data_dir through paper_figures.py and replace CI function | 0fbc2c6 | paper_figures.py |
| 2 | Add paper figure entries 16-19 to manifest and write smoke tests | 2264bdf | main.py, test_paper_figures.py |

## What Was Built

**`paper_figures.py` refactored** (Task 1):
- Removed `from __future__ import annotations` (CLAUDE.md convention)
- Removed `_DATA_ROOT = Path("cluster_results/eval_corrected_full")` module-level constant (D-02)
- Updated `_load_combined_posterior(variant, data_dir)` — now requires `data_dir`, uses it for JSON paths
- Updated call sites in `plot_h0_posterior_comparison` to pass `data_dir` to `_load_combined_posterior`
- Replaced inline 68% CI CDF loop in `plot_h0_posterior_comparison` with `compute_credible_interval` (D-07)
- Deleted `_ci_width_from_log_posteriors` function — replaced call in `plot_posterior_convergence` with log-shift-then-exp + `compute_credible_interval`
- Deleted `main()` and `if __name__ == "__main__"` block (D-01)
- Removed unused `save_figure` and `apply_style` imports

**`main.py` manifest extended** (Task 2) — 4 new closure entries in `generate_figures()`:
```python
manifest.append(("paper_h0_posterior", _gen_paper_h0_posterior))   # entry 16
manifest.append(("paper_single_event", _gen_paper_single_event))   # entry 17
manifest.append(("paper_convergence", _gen_paper_convergence))      # entry 18
manifest.append(("paper_snr_distribution", _gen_paper_snr_distribution))  # entry 19
```
Each closure uses lazy imports and catches `(FileNotFoundError, KeyError[, ValueError])`, returning `None` on missing data (consistent with existing entries 1-15).

**9 smoke tests** in `test_paper_figures.py` (Task 2):
- `TestPaperFigureImports` (4 tests): importability + `data_dir` in signature
- `TestPaperFigureGracefulDegradation` (3 tests): FileNotFoundError on missing posteriors; placeholder figure for missing SNR CSV
- `TestNoStandaloneMain` (2 tests): `main` and `_DATA_ROOT` absent from module

## Verification

```
uv run pytest master_thesis_code_test/plotting/test_paper_figures.py -x -m "not slow and not gpu" -v --no-cov
# => 9 passed
grep -c "_DATA_ROOT" master_thesis_code/plotting/paper_figures.py   # => 0
grep -c "def main" master_thesis_code/plotting/paper_figures.py     # => 0
grep -c "paper_h0_posterior" master_thesis_code/main.py             # => 2
grep -c "paper_snr_distribution" master_thesis_code/main.py         # => 2
```

## Deviations from Plan

**1. [Rule 2 - Auto-fix] Made data_dir a required parameter instead of defaulting to removed _DATA_ROOT**

- **Found during:** Task 1 — after removing `_DATA_ROOT` constant, function default values still referenced it
- **Issue:** Four public functions had `data_dir: Path = _DATA_ROOT` as default; `_DATA_ROOT` no longer existed
- **Fix:** Replaced `= _DATA_ROOT` with no default, making `data_dir` required. The manifest closures always pass `Path(output_dir)`, and the test plan already uses `tmp_path` explicitly — so no callers relied on the default.
- **Files modified:** `master_thesis_code/plotting/paper_figures.py`
- **Commit:** 0fbc2c6

**2. [Rule 1 - Bug] Removed unused imports after main() deletion**

- **Found during:** Task 1 — after deleting `main()`, `save_figure` and `apply_style` were imported but never used, which would cause ruff F401 violations
- **Fix:** Removed both from the import line
- **Files modified:** `master_thesis_code/plotting/paper_figures.py`
- **Commit:** 0fbc2c6

## Known Stubs

None — all public functions accept `data_dir` explicitly. No hardcoded paths remain.

## Threat Flags

None — internal figure generation code; no network access, user input, or auth.

## Self-Check: PASSED

- `master_thesis_code/plotting/paper_figures.py` — FOUND, no `_DATA_ROOT`, no `main()`, imports `compute_credible_interval`
- `master_thesis_code/main.py` — FOUND, contains `paper_h0_posterior` and `paper_snr_distribution`
- `master_thesis_code_test/plotting/test_paper_figures.py` — FOUND, contains `TestNoStandaloneMain`
- Commit 0fbc2c6 — FOUND
- Commit 2264bdf — FOUND
