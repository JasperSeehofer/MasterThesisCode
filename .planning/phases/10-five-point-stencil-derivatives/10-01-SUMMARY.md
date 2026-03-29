---
phase: 10-five-point-stencil-derivatives
plan: 01
subsystem: parameter-estimation
tags: [physics, fisher-matrix, derivatives, numerical-methods, safety]
dependency_graph:
  requires: []
  provides: [five-point-stencil-default, condition-number-monitoring, crb-safety-checks, 90s-timeout]
  affects: [parameter_estimation.py, main.py]
tech_stack:
  added: []
  patterns: [toggle-dispatch, xp-cpu-gpu-pattern, condition-number-logging]
key_files:
  created: []
  modified:
    - master_thesis_code/parameter_estimation/parameter_estimation.py
    - master_thesis_code/main.py
    - master_thesis_code_test/parameter_estimation/parameter_estimation_test.py
decisions:
  - "D-06: use_five_point_stencil=True default on ParameterEstimation constructor"
  - "D-07: Both derivative methods retained, dispatch via toggle"
  - "D-01: CRB timeout increased from 30s to 90s"
  - "D-03: Condition number logged at INFO before every matrix inversion"
  - "D-04: Singular Fisher and negative CRB diagonals raise and skip event"
metrics:
  duration: 6m
  completed: 2026-03-29
  tasks_completed: 3
  tasks_total: 3
  files_modified: 3
---

# Phase 10 Plan 01: Five-Point Stencil Derivatives Summary

O(epsilon^4) five-point stencil wired as default Fisher matrix derivative with toggle, np.matrix replaced by np.linalg.inv, condition number monitoring, negative diagonal detection, and 90s CRB timeout.

## What Was Done

### Task 1: Write CPU-testable tests (TDD RED)
- Added 6 new test functions to `parameter_estimation_test.py`
- Tests cover: toggle dispatch (five-point default, forward-diff fallback), condition number logging, negative diagonal ParameterEstimationError, singular matrix LinAlgError, 90s alarm timeout
- All 6 tests failed in RED phase as expected
- Existing 5 CPU tests unaffected
- Commit: `7bccb68`

### Task 2: Refactor five_point_stencil_derivative, add toggle, CRB safety (GREEN)
- Added `use_five_point_stencil: bool = True` kwarg to `ParameterEstimation.__init__`
- Refactored `five_point_stencil_derivative()` from single-parameter to dict API (loops over all 14 parameters internally)
- Removed print() calls and unused multiprocessing import
- Added GPU memory logging before/after derivative loop
- Added toggle dispatch in `compute_fisher_information_matrix()`
- Replaced `np.matrix(cp.asnumpy(...)).I` with `np.linalg.inv()` + CPU/GPU guard
- Added condition number logging at INFO level
- Added negative CRB diagonal detection raising `ParameterEstimationError`
- Added `ParameterEstimationError` to imports
- Fixed `cp.zeros` to use xp pattern for CPU testability
- All 11 CPU tests pass (5 existing + 6 new)
- Commit: `a87eeab`

### Task 3: Update CRB timeout to 90s and add exception handling
- Changed both `signal.alarm(30)` to `signal.alarm(90)` in `data_simulation()`
- Updated timeout messages from 30s to 90s
- Added `np.linalg.LinAlgError` exception handler for singular Fisher matrices
- Added `ParameterEstimationError` exception handler for negative CRB diagonals
- Added `ParameterEstimationError` to imports in main.py
- Commit: `1e0cb21`

## Verification Results

- 198 CPU tests pass, 0 failures, 18 deselected (GPU/slow)
- Coverage: 40.8% (gate at 25%)
- `ruff check master_thesis_code/` -- clean
- `ruff format --check master_thesis_code/` -- clean
- `mypy master_thesis_code/` -- clean (0 issues in 44 files)
- `np.matrix(` : 0 occurrences in parameter_estimation.py
- `arXiv:gr-qc/0703086` : 3 reference comments present
- `signal.alarm(30)` : 0 occurrences in main.py
- `signal.alarm(90)` : 2 occurrences in main.py

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed cp.zeros for CPU test compatibility**
- **Found during:** Task 2
- **Issue:** `compute_fisher_information_matrix()` called `cp.zeros()` which fails on CPU (cp is None). Toggle dispatch tests could not run.
- **Fix:** Replaced `cp.zeros(...)` with `xp.zeros(...)` using the standard `xp = cp if (_CUPY_AVAILABLE and cp is not None) else np` pattern.
- **Files modified:** `master_thesis_code/parameter_estimation/parameter_estimation.py`
- **Commit:** `a87eeab`

**2. [Rule 2 - Missing] Removed unused Parameter import after refactor**
- **Found during:** Task 2 (ruff lint)
- **Issue:** `Parameter` was imported but no longer used after `five_point_stencil_derivative` refactor removed the parameter argument.
- **Fix:** Removed unused import.
- **Files modified:** `master_thesis_code/parameter_estimation/parameter_estimation.py`
- **Commit:** `a87eeab`

**3. [Rule 2 - Missing] Added _use_gpu=False to test helper**
- **Found during:** Task 2
- **Issue:** `_make_minimal_pe` did not set `_use_gpu`, which `compute_Cramer_Rao_bounds` now checks.
- **Fix:** Added `pe._use_gpu = False` and `pe._use_five_point_stencil = True` to the test helper.
- **Files modified:** `master_thesis_code_test/parameter_estimation/parameter_estimation_test.py`
- **Commit:** `a87eeab`

## Decisions Made

1. **Toggle default is True** -- five-point stencil is the production default; forward-difference available via `use_five_point_stencil=False` for regression testing
2. **No condition number threshold** -- kappa logged for observability; Phase 11 will determine if filtering is needed
3. **GPU memory logging added** -- INFO-level before/after derivative loop for diagnosing GPU OOM on cluster

## Known Stubs

None -- all functionality is fully wired.

## Self-Check: PASSED
