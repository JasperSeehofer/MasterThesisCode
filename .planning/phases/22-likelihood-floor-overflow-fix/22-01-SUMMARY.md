---
phase: 22-likelihood-floor-overflow-fix
plan: 01
subsystem: bayesian-inference
tags: [posterior-combination, zero-handling, likelihood-floor, numerical-stability]

# Dependency graph
requires:
  - phase: 21-analysis-post-processing
    provides: "CombinationStrategy enum with PHYSICS_FLOOR stub, log-space accumulation, combine_posteriors entry point"
provides:
  - "Working physics-floor strategy: per-event min-nonzero likelihood as floor"
  - "check_overflow dead code removed from bayesian_statistics.py"
affects: [cluster-deployment, evaluation-pipeline, posterior-validation]

# Tech tracking
tech-stack:
  added: []
  patterns: ["per-event likelihood floor with logged diagnostics"]

key-files:
  created: []
  modified:
    - master_thesis_code/bayesian_inference/posterior_combination.py
    - master_thesis_code/bayesian_inference/bayesian_statistics.py
    - master_thesis_code_test/bayesian_inference/test_posterior_combination.py

key-decisions:
  - "Physics floor uses min(nonzero) directly per event, not divided by 100 like per-event-floor"
  - "All-zero events excluded (no nonzero value to derive floor from)"
  - "Floor application logged with event index, bin count, and floor value for traceability"

patterns-established:
  - "Per-event floor pattern: iterate rows, find min nonzero, replace zeros, log each event"

requirements-completed: [NFIX-02, NFIX-03]

# Metrics
duration: 3min
completed: 2026-04-02
---

# Phase 22 Plan 01: Likelihood Floor & Overflow Fix Summary

**Per-event min-nonzero likelihood floor replaces physics-floor stub; check_overflow dead code removed; 304 tests pass**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-02T17:14:27Z
- **Completed:** 2026-04-02T17:17:43Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Physics-floor strategy implemented: zeros replaced with per-event minimum nonzero likelihood value
- All-zero events excluded with logged warning (no nonzero value to derive floor from)
- Floor application logged per event with index, bin count, and floor value
- Dead check_overflow function removed from bayesian_statistics.py (no callers existed)
- 304 tests pass including 6 new physics-floor tests; ruff and mypy clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement physics-floor strategy and update tests** - `33d3aac` (test: RED), `db5eb2b` (feat: GREEN)
2. **Task 2: Remove check_overflow from bayesian_statistics.py** - `39279d3` (fix)

_Note: Task 1 used TDD (test -> feat commits)_

## Files Created/Modified
- `master_thesis_code/bayesian_inference/posterior_combination.py` - Added `_physics_floor` function, removed "not yet implemented" fallback
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` - Removed dead `check_overflow` function
- `master_thesis_code_test/bayesian_inference/test_posterior_combination.py` - 6 new physics-floor tests replacing fallback test, updated e2e test

## Decisions Made
- Physics floor uses `min(nonzero)` directly (not divided by 100 like per-event-floor) per D-01
- Floor scoped per-event (not global) per D-03
- Floor applied in combination step (not in single_host_likelihood) per D-04
- check_overflow removed entirely (not fixed) per D-05 -- log-space accumulation handles stability

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - no stubs remain in modified files.

## Next Phase Readiness
- Physics-floor strategy is ready for cluster validation against campaign data
- combine_posteriors with strategy="physics-floor" produces valid normalized posteriors
- Ready for deployment to cluster before pending evaluation jobs

---
*Phase: 22-likelihood-floor-overflow-fix*
*Completed: 2026-04-02*
