---
phase: 31-catalog-only-diagnostic
plan: "02"
subsystem: bayesian-inference
tags: [diagnostic, evaluation-report, catalog-only, TDD]
dependency_graph:
  requires: [31-01]
  provides: [generate_diagnostic_summary, comparison-label-param]
  affects: [evaluation_report.py, main.py]
tech_stack:
  added: []
  patterns: [pandas-groupby-analysis, per-event-csv-summary]
key_files:
  created:
    - master_thesis_code_test/bayesian_inference/test_diagnostic_summary.py
  modified:
    - master_thesis_code/bayesian_inference/evaluation_report.py
    - master_thesis_code/main.py
decisions:
  - "Moved pandas to top-level import in evaluation_report.py (was conditional in _extract_per_event_summaries)"
  - "Used float(str(...)) pattern for mypy-safe object->float conversion in main.py logging"
metrics:
  duration_seconds: 196
  completed: "2026-04-08T11:17:29Z"
  tasks_completed: 1
  tasks_total: 2
  task_2_status: "checkpoint:human-verify (documented below)"
---

# Phase 31 Plan 02: Diagnostic Summary Generator

Diagnostic summary function that analyzes per-event CSV to explain WHY the H0 posterior bias changes, computing f_i statistics, L_comp contribution metrics, and bias direction fractions.

## Completed Tasks

### Task 1: Add generate_diagnostic_summary() and wire into comparison flow

**Commit:** facedb6

**Implementation:**

1. **evaluation_report.py** -- Added `generate_diagnostic_summary(diagnostic_csv_path, output_dir, label)` function that:
   - Reads per-event diagnostic CSV via pandas
   - Computes per-event f_i stats (mean, median, min, max) via groupby
   - Computes L_comp statistics (mean, median)
   - Calculates L_comp weight fraction in combined likelihood for events with f_i < 1.0
   - Determines fraction of events where L_comp(h_low) > L_comp(h_high) (bias direction)
   - Writes both JSON and Markdown summary files
   - Returns dict with all computed metrics

2. **main.py** -- Updated `_compare_baseline()`:
   - Added `label` parameter (default "current")
   - Passes label through to `generate_comparison_report()`
   - Calls `generate_diagnostic_summary()` when `event_likelihoods.csv` exists
   - Passes `label="catalog_only"` when `arguments.catalog_only` is True

3. **test_diagnostic_summary.py** -- 4 TDD tests:
   - `test_catalog_only_csv_returns_expected_summary` -- synthetic CSV with f_i=1.0, L_comp=0.0
   - `test_varied_csv_frac_pulls_low_h` -- 2 events, 1 pulls low, verifies frac=0.5
   - `test_frac_pulls_low_h_bounded` -- confirms frac in [0.0, 1.0]
   - `test_output_files_created` -- JSON and MD files created with expected keys

**Verification:** All 24 tests pass (20 existing + 4 new), mypy clean, ruff clean.

### Task 2: Human-verify checkpoint (not yet completed)

**Status:** Awaiting human verification.

**What was built:** Complete catalog-only diagnostic pipeline:
- `--catalog_only` CLI flag bypasses completion term (Plan 31-01)
- Per-event diagnostic CSV written on every `--evaluate` run (Plan 31-01)
- `--compare_baseline` now generates both comparison report and diagnostic summary
- Label parameter routes catalog_only vs standard comparison output

**Verification steps for human:**

1. Confirm CLI flag recognized:
   ```bash
   uv run python -m master_thesis_code --help | grep catalog_only
   ```

2. Run all new tests:
   ```bash
   uv run pytest master_thesis_code_test/bayesian_inference/test_catalog_only_diagnostic.py master_thesis_code_test/bayesian_inference/test_diagnostic_summary.py -v
   ```

3. Run full quality gate:
   ```bash
   uv run ruff check master_thesis_code/ && uv run mypy master_thesis_code/ master_thesis_code_test/ && uv run pytest -m "not gpu and not slow" -x
   ```

4. (On cluster) Run catalog-only evaluation:
   ```bash
   uv run python -m master_thesis_code <working_dir> --evaluate --catalog_only --h_value 0.73
   ```

5. (On cluster) Compare against baseline:
   ```bash
   uv run python -m master_thesis_code <working_dir> --compare_baseline .planning/debug/baseline.json --catalog_only
   ```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed mypy type error for dict[str, object] values**
- **Found during:** Task 1
- **Issue:** `float(diag_summary["frac_L_comp_pulls_low_h"])` fails mypy because dict values are typed as `object`
- **Fix:** Used `float(str(...))` pattern for safe conversion
- **Files modified:** master_thesis_code/main.py
- **Commit:** facedb6

**2. [Rule 2 - Missing] Removed duplicate pandas import**
- **Found during:** Task 1
- **Issue:** `_extract_per_event_summaries` had a local `import pandas as pd` while we added a top-level import
- **Fix:** Removed the redundant local import
- **Files modified:** master_thesis_code/bayesian_inference/evaluation_report.py
- **Commit:** facedb6

## Self-Check: PASSED

- All 3 modified/created files exist on disk
- Commit facedb6 exists in git log
- `generate_diagnostic_summary` function found in evaluation_report.py
- All 24 tests pass (20 existing + 4 new)
- mypy and ruff clean on modified files
