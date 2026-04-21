---
phase: "31"
plan: "01"
subsystem: bayesian-inference
tags: [diagnostic, cli-flag, catalog-only, csv-logging]
dependency_graph:
  requires: []
  provides: [catalog_only_flag, diagnostic_csv, completion_bypass]
  affects: [arguments.py, main.py, bayesian_statistics.py]
tech_stack:
  added: []
  patterns: [csv-append-mode, conditional-bypass]
key_files:
  created:
    - master_thesis_code_test/bayesian_inference/test_catalog_only_diagnostic.py
  modified:
    - master_thesis_code/arguments.py
    - master_thesis_code/main.py
    - master_thesis_code/bayesian_inference/bayesian_statistics.py
decisions:
  - Used csv.DictWriter with append mode and header-on-first-write pattern for diagnostic CSV
  - Wrapped float() cast around get_completeness_at_redshift to fix mypy type narrowing in if/else branches
  - Combined both plan tasks into a single commit since flag + bypass + CSV + tests are tightly coupled
metrics:
  duration_seconds: 386
  completed: "2026-04-08T11:10:33Z"
  tasks_completed: 2
  tasks_total: 2
---

# Phase 31 Plan 01: Catalog-Only Diagnostic Summary

**One-liner:** `--catalog_only` CLI flag that bypasses the completion integral (f_i=1, L_comp=0) with per-event diagnostic CSV logging for every evaluation run.

## What Was Done

### Task 1: --catalog_only flag + completion bypass (TDD)

Added `--catalog_only` argparse flag to `arguments.py` with a boolean `store_true` action and corresponding property accessor. Threaded the flag through `main.py:evaluate()` into `BayesianStatistics.evaluate()`, which stores it as `self.catalog_only`.

In `p_Di()`, added a conditional before the completion term block: when `self.catalog_only` is `True`, set `f_i = 1.0` and `L_comp = 0.0`, completely skipping the expensive `fixed_quad` integration and completeness lookup. The existing completion code is wrapped in the `else` branch, unchanged.

### Task 2: Diagnostic CSV logging (TDD)

Added `self._diagnostic_rows: list[dict[str, object]]` initialized in both `__init__` and `evaluate()`. Every call to `p_Di()` appends a row dict with columns: `event_idx`, `h`, `f_i`, `L_cat_no_bh`, `L_cat_with_bh`, `L_comp`, `combined_no_bh`, `combined_with_bh`.

Added `_write_diagnostic_csv()` method using `csv.DictWriter` with append mode: creates directory if needed, writes header only on first write, appends data rows. Called at the end of `evaluate()` writing to `simulations/diagnostics/event_likelihoods.csv`.

### Tests

7 tests in `test_catalog_only_diagnostic.py`:
- `TestCatalogOnlyFlag`: 3 tests for CLI flag presence/absence/independence from --evaluate
- `TestCatalogOnlyBypass`: 1 test verifying f_i=1.0, L_comp=0.0, completeness not called
- `TestDiagnosticCsv`: 3 tests for column structure, CSV write, and append mode without header duplication

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed mypy type narrowing for f_i**
- **Found during:** Task 1
- **Issue:** `get_completeness_at_redshift()` returns `float | NDArray`, causing mypy error when `f_i` is assigned in both if/else branches (if branch: `float`, else branch: `float | NDArray`)
- **Fix:** Added `float()` cast around the completeness call
- **Files modified:** `master_thesis_code/bayesian_inference/bayesian_statistics.py`
- **Commit:** e5f8de5

**2. [Rule 3 - Blocking] Fixed test type annotations for ruff/mypy compatibility**
- **Found during:** Task 1 (commit attempt)
- **Issue:** Test fixture return type `MagicMock` conflicted with actual `BayesianStatistics` return; forward string reference triggered ruff F821
- **Fix:** Added `TYPE_CHECKING` import for `BayesianStatistics`, used proper forward reference
- **Files modified:** `master_thesis_code_test/bayesian_inference/test_catalog_only_diagnostic.py`
- **Commit:** e5f8de5

## Commits

| Task | Commit | Message |
|------|--------|---------|
| 1+2 | e5f8de5 | feat(31-01): add --catalog_only flag with completion bypass and diagnostic CSV |

## Known Stubs

None - all functionality is fully wired.

## Self-Check: PASSED

All 4 files verified on disk. Commit e5f8de5 verified in git log. All 7 tests pass. mypy and ruff clean.
