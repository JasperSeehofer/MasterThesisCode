---
phase: 30-baseline-evaluation-infrastructure
plan: "01"
subsystem: bayesian-inference
tags: [evaluation, baseline, cli, reporting]
dependency_graph:
  requires: []
  provides: [baseline-extraction, comparison-reporting, save-baseline-cli, compare-baseline-cli]
  affects: [phases/31, phases/32, phases/33, phases/34]
tech_stack:
  added: []
  patterns: [TDD-red-green, lazy-imports, dataclass-with-field-defaults]
key_files:
  created:
    - master_thesis_code/bayesian_inference/evaluation_report.py
    - master_thesis_code_test/bayesian_inference/test_evaluation_report.py
  modified:
    - master_thesis_code/arguments.py
    - master_thesis_code/main.py
decisions:
  - "Store baseline.json in .planning/debug/ (cross-phase reference per D-06)"
  - "dispatch functions use lazy imports to avoid CPU-only machine import failures"
  - "CI uses cumulative_trapezoid + linear interpolation, not percentile on discrete grid"
  - "from_json casts dict[str,object] to dict[str,Any] once instead of per-line type: ignore"
metrics:
  duration_seconds: 370
  completed_date: "2026-04-08"
  tasks_completed: 3
  tasks_total: 3
  files_created: 2
  files_modified: 2
requirements: [DIAG-03, EVAL-01, EVAL-02]
---

# Phase 30 Plan 01: Baseline Evaluation Infrastructure Summary

**One-liner:** `evaluation_report.py` module with `BaselineSnapshot` dataclass, `extract_baseline` / `generate_comparison_report` functions, and `--save_baseline` / `--compare_baseline` CLI flags wired into `main.py`.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Create evaluation_report.py (TDD) | 7eab0aa | evaluation_report.py, test_evaluation_report.py |
| 2 | Wire CLI flags and dispatch | e2d45cb | arguments.py, main.py |
| 3 | Quality gate (mypy, ruff, full suite) | a9cbc3d | evaluation_report.py |

## What Was Built

### `evaluation_report.py`

- **`BaselineSnapshot`** dataclass with `map_h`, `ci_lower`, `ci_upper`, `ci_width`, `bias_percent`, `n_events`, `h_values`, `log_posteriors`, `per_event_summaries`, `created_at`, `git_commit`. Full `to_json()` / `from_json()` round-trip.
- **`load_posteriors(posteriors_dir)`**: reads all `h_*.json` files, computes `log_posterior = sum(log(likelihood))` per h-value, returns sorted list with `n_detections`. Warns via logging if > 100 files found (T-30-02 DoS mitigation).
- **`compute_credible_interval(h_values, log_posteriors, level=0.68)`**: normalizes log-posteriors to probability, integrates CDF via `scipy.integrate.cumulative_trapezoid`, returns (lower, upper) by linear interpolation at (1-level)/2 and (1+level)/2 quantiles.
- **`extract_baseline(posteriors_dir, crb_csv_path=None, true_h=0.73)`**: raises `ValueError` for < 3 h-values, computes MAP h, 68% CI, bias %, event count, optional per-event summaries from CRB CSV.
- **`generate_comparison_report(baseline, current, output_dir, label)`**: writes `comparison_{label}.md` (table + ASCII chart + verdict) and `comparison_{label}.json` (metrics + deltas). Returns path to markdown.

### CLI

- `--save_baseline` (store_true): extracts baseline from `<working_dir>/simulations/posteriors/`, saves to `.planning/debug/baseline.json`.
- `--compare_baseline <path>` (str): loads baseline JSON, extracts current posteriors, generates comparison report to `.planning/debug/`.
- Both flags are fast-path (no model initialization required). When combined with `--evaluate`, evaluation runs first then comparison reads fresh posteriors.

### Tests (20 total, all passing)

- JSON round-trip, load/sort/log-posterior, detection count, large-file warning
- CI symmetry on Gaussian posterior, CI tuple type/ordering
- `extract_baseline` < 3 h-values raises `ValueError`, MAP h, bias %, event count, CI bounds, `created_at`
- `generate_comparison_report` markdown + JSON sidecar, all expected metrics, correct delta values
- Integration: `_save_baseline` end-to-end, `_compare_baseline` with shifted posteriors, standalone (without `--evaluate`)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `np.trapz` removed in NumPy 2.0**
- **Found during:** Task 3 (GREEN test run)
- **Issue:** `np.trapz` was removed in NumPy 2.0; codebase uses NumPy 2.x
- **Fix:** Replaced with `np.trapezoid`
- **Files modified:** evaluation_report.py
- **Commit:** a9cbc3d

**2. [Rule 1 - Bug] `datetime.utcnow()` deprecated in Python 3.12+**
- **Found during:** Task 3 (test warnings)
- **Issue:** `datetime.datetime.utcnow()` is deprecated; Python emits `DeprecationWarning`
- **Fix:** Replaced with `datetime.datetime.now(datetime.UTC)`
- **Files modified:** evaluation_report.py
- **Commit:** a9cbc3d

**3. [Rule 1 - Bug] mypy type errors in `from_json`**
- **Found during:** Task 3 (mypy run)
- **Issue:** `int()` and `list()` do not accept `object` in mypy strict mode; `type: ignore[arg-type]` covered wrong error code
- **Fix:** Cast `data: dict[str, object]` to `d: dict[str, Any]` once at top of method
- **Files modified:** evaluation_report.py
- **Commit:** a9cbc3d

## Known Stubs

None — all public functions are fully wired. `generate_comparison_report` produces real output; `extract_baseline` reads real files. No placeholder values flow to output.

## Threat Flags

None — no new network endpoints, auth paths, or schema changes beyond what was planned. `load_posteriors` and the CLI path read locally-generated simulation files only.

## Self-Check

- [x] `evaluation_report.py` exists at `master_thesis_code/bayesian_inference/evaluation_report.py`
- [x] `test_evaluation_report.py` exists at `master_thesis_code_test/bayesian_inference/test_evaluation_report.py`
- [x] Commits: 7eab0aa, e2d45cb, a9cbc3d all present
- [x] 20/20 tests pass
- [x] mypy clean on all 3 modified files
- [x] ruff check + format clean

## Self-Check: PASSED
