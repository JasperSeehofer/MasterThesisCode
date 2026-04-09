---
phase: 33-pdet-grid-resolution
plan: 01
subsystem: bayesian-inference
tags: [pdet, grid-resolution, cli, coverage-validation]
dependency_graph:
  requires: []
  provides: [configurable-pdet-bins, coverage-validation]
  affects: [bayesian_statistics, simulation_detection_probability, arguments, main]
tech_stack:
  added: []
  patterns: [instance-configurable-defaults, coverage-validation-logging]
key_files:
  created: []
  modified:
    - master_thesis_code/arguments.py
    - master_thesis_code/main.py
    - master_thesis_code/bayesian_inference/bayesian_statistics.py
    - master_thesis_code/bayesian_inference/simulation_detection_probability.py
    - master_thesis_code_test/bayesian_inference/test_simulation_detection_probability.py
decisions:
  - "D-01: --pdet_dl_bins (default 60) and --pdet_mass_bins (default 40) CLI flags added"
  - "D-03: SimulationDetectionProbability.__init__() accepts dl_bins and mass_bins keyword args"
  - "D-04: Module-level _DL_BINS/_M_BINS renamed to _DEFAULT_DL_BINS/_DEFAULT_M_BINS"
  - "D-05/D-06: Defaults increased from 30/20 to 60/40"
  - "D-08/D-09/D-10: validate_coverage() logs INFO always, WARNING if <95%, never aborts"
metrics:
  duration: 253s
  completed: 2026-04-09T12:55:13Z
  tasks_completed: 3
  tasks_total: 3
  files_modified: 5
---

# Phase 33 Plan 01: Configurable P_det Grid Bins Summary

Configurable P_det grid bins (60/40 defaults, up from 30/20) with CLI flags and 4-sigma d_L coverage validation logging.

## What Was Done

### Task 1+2: CLI flags, threading, and instance-level bin counts (combined commit)

- Added `--pdet_dl_bins` (int, default 60) and `--pdet_mass_bins` (int, default 40) to `arguments.py`
- Added `pdet_dl_bins` and `pdet_mass_bins` properties to `Arguments` class
- Threaded bin counts through `main.py:evaluate()` -> `BayesianStatistics.evaluate()` -> `SimulationDetectionProbability` constructor
- Renamed `_DL_BINS` -> `_DEFAULT_DL_BINS = 60` and `_M_BINS` -> `_DEFAULT_M_BINS = 40`
- Added `dl_bins` and `mass_bins` keyword args to `SimulationDetectionProbability.__init__()`
- Replaced all 10 references to module-level constants with `self._dl_bins` / `self._mass_bins`
- Added `validate_coverage()` method that computes fraction of events whose 4-sigma d_L bounds fall within the P_det grid
- Coverage validation logs INFO with percentage always, WARNING if below 95%
- Called `validate_coverage()` in `bayesian_statistics.py` after grid pre-warming

### Task 3: Tests for configurable bins and coverage validation

- `TestConfigurableBins`: 3 tests (custom grid shape, default values, pickle roundtrip)
- `TestCoverageValidation`: 4 tests (full coverage, partial coverage, warning log, info log)
- All 21 tests pass (14 existing + 7 new), no regressions

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1+2 | 0290956 | feat(33-01): configurable P_det grid bins with CLI flags and coverage validation |
| 3 | b6efbd3 | test(33-01): add tests for configurable bins and coverage validation |

## Deviations from Plan

### Task 1 and Task 2 combined into single commit

Tasks 1 and 2 had to be committed together because Task 1 passes `dl_bins` and `mass_bins` to the `SimulationDetectionProbability` constructor, which doesn't accept those parameters until Task 2 adds them. mypy (enforced by pre-commit hooks) rejects the intermediate state. This is a process deviation only -- all planned work was completed.

## Verification Results

- CLI flags: `Arguments.create(['dummy_dir', '--pdet_dl_bins', '100', '--pdet_mass_bins', '50'])` returns correct values
- Default values: `pdet_dl_bins=60`, `pdet_mass_bins=40` confirmed
- Tests: 21/21 passed
- mypy: Success, no issues in 4 source files
- `--help` output shows both new flags

## Known Stubs

None.
