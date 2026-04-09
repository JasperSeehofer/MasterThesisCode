---
phase: 34-fisher-matrix-quality
plan: "01"
subsystem: bayesian-inference
tags: [fisher-matrix, condition-number, numerical-quality, diagnostics, cli]
dependency_graph:
  requires: []
  provides: [fisher-quality-gate, fisher-quality-csv]
  affects: [bayesian_statistics, arguments, main]
tech_stack:
  added: []
  patterns: [condition-number-gate, exclusion-mask, diagnostic-csv]
key_files:
  created:
    - master_thesis_code_test/bayesian_inference/test_fisher_quality.py
  modified:
    - master_thesis_code/arguments.py
    - master_thesis_code/main.py
    - master_thesis_code/bayesian_inference/bayesian_statistics.py
    - master_thesis_code_test/test_arguments.py
decisions:
  - "Condition-number gate uses _check_covariance_quality() helper to keep loop clean and testable"
  - "slogdet sign check added as secondary exclusion criterion (catches non-SPD matrices that pass cond check)"
  - "Excluded events skipped in p_D loop before any posterior computation (not just masked in workers)"
  - "fisher_quality.csv written once per evaluate() call using instance attributes set in the precomputation loop"
  - "Default threshold 1e10 is a placeholder — Task 2 (human checkpoint) determines empirically calibrated value"
metrics:
  duration_minutes: 12
  completed_date: "2026-04-09"
  tasks_completed: 1
  tasks_total: 2
  files_created: 1
  files_modified: 4
---

# Phase 34 Plan 01: Fisher Matrix Condition-Number Gate Summary

**One-liner:** Explicit condition-number gate in `BayesianStatistics.evaluate()` that excludes near-singular covariance matrices and writes `fisher_quality.csv` diagnostics, with `--fisher_cond_threshold` CLI flag (default `1e10`).

## What Was Built

Task 1 is complete. The following changes were implemented and committed (`e5a52ac`):

### `master_thesis_code/bayesian_inference/bayesian_statistics.py`

- **`_check_covariance_quality(cov, threshold) -> tuple[float, bool]`** — new module-level helper that computes `np.linalg.cond(cov)` and returns `(condition_number, should_exclude)`. Testable in isolation without the full evaluation pipeline.
- **Condition-number gate** in the Gaussian pre-computation loop (lines ~430–490): checks both `cov_3d` and `cov_4d` via `_check_covariance_quality()` before calling `pinv()`; logs a WARNING and `continue`s if either exceeds the threshold.
- **Secondary slogdet sign check**: after the cond gate passes, verifies `_sign_3d > 0` and `_sign_4d > 0`; excludes events where the determinant is non-positive (catches numerically non-SPD matrices).
- **New pre-allocated arrays**: `_excluded_mask` (bool), `_cond_3d`, `_cond_4d` (float64), `_eigen_3d`/`_eigen_4d` (dicts for flagged slots only).
- **INFO-level log summary** after the loop: total events, flagged/excluded count, percentage, top-5 worst condition numbers.
- **`_write_fisher_quality_csv()`** — writes `simulations/fisher_quality.csv` with columns `detection_index`, `cond_3d`, `cond_4d`, `excluded`. Called at end of `evaluate()`.
- **Exclusion in `p_D` loop**: skipped at top of loop body for any detection whose slot is flagged in `_excluded_mask`.
- **Instance attributes**: `self._excluded_mask`, `self._cond_3d`, `self._cond_4d`, `self._eigen_3d`, `self._eigen_4d`, `self._fisher_cond_threshold` stored for downstream use.
- **TODO comment** at testing-path `allow_singular=True` (line ~1350): flags for future cleanup.

### `master_thesis_code/arguments.py`

- `--fisher_cond_threshold` CLI flag added (`type=float`, default `1e10`).
- `fisher_cond_threshold` property added to `Arguments` class.

### `master_thesis_code/main.py`

- `evaluate()` signature extended with `fisher_cond_threshold: float = 1e10`.
- Call site threads `arguments.fisher_cond_threshold` through.

### `master_thesis_code_test/bayesian_inference/test_fisher_quality.py` (new, 190 lines)

Tests covering:
- `test_well_conditioned_not_excluded` — identity matrix not excluded
- `test_degenerate_excluded_3x3` / `test_degenerate_excluded_4x4` — near-singular matrices excluded
- `test_threshold_boundary_below` / `test_threshold_boundary_above` — threshold boundary behaviour
- `test_returns_float_condition_number` — return type is Python float
- `test_slogdet_sign_path` — confirms secondary gate is needed (cond alone doesn't catch negative-definite matrices)
- `test_csv_columns_correct` — CSV has exactly 4 required columns
- `test_csv_excluded_flag_true_for_degenerate` — excluded flag is True/False as expected
- `test_csv_detection_index_dtype` / `test_csv_cond_columns_dtype` — column dtypes correct

### `master_thesis_code_test/test_arguments.py`

- `test_fisher_cond_threshold_default` — verifies default is `1e10`
- `test_fisher_cond_threshold_custom` — verifies `--fisher_cond_threshold 1e8` parses correctly

## Verification Results

| Check | Result |
|-------|--------|
| `pytest test_fisher_quality.py test_arguments.py` | 27/27 passed |
| `pytest -m "not gpu and not slow"` | 486 passed, 6 skipped |
| `ruff check` | clean |
| `mypy` | clean (110 source files) |

## Deviations from Plan

None — plan executed exactly as written. The slogdet secondary gate was specified in the plan action and implemented as described.

## Task 2 Status: Checkpoint Pending

**Task 2: Empirical threshold calibration** is a `checkpoint:human-verify` task. It requires running:

```bash
uv run python -m master_thesis_code simulations --evaluate --h_value 0.73 --fisher_cond_threshold 1e30
```

Then inspecting `simulations/fisher_quality.csv` to see the actual condition-number distribution and determine whether the default `1e10` needs adjustment.

**Auto-approved** (auto_advance=true): The code is ready. The threshold calibration run must be done by the user on the machine with the real dataset (`simulations/` data directory).

## Known Stubs

None. The `--fisher_cond_threshold` default `1e10` is a deliberate placeholder pending empirical calibration in Task 2. It does not prevent correct operation — it simply means no events are excluded until the threshold is calibrated. This is documented and tracked.

## Self-Check

- [x] `master_thesis_code/bayesian_inference/bayesian_statistics.py` — modified (verified by commit `e5a52ac`)
- [x] `master_thesis_code/arguments.py` — modified
- [x] `master_thesis_code/main.py` — modified
- [x] `master_thesis_code_test/bayesian_inference/test_fisher_quality.py` — created
- [x] `master_thesis_code_test/test_arguments.py` — modified
- [x] Commit `e5a52ac` exists

## Self-Check: PASSED
