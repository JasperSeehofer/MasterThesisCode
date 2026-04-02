---
phase: 21-analysis-post-processing
plan: 01
subsystem: bayesian-inference
tags: [posterior-combination, log-space, zero-handling, numerical-stability, diagnostics]

requires:
  - phase: none
    provides: standalone module, no prior phase dependency
provides:
  - posterior_combination.py with 9 public functions for loading, combining, and diagnosing per-event posteriors
  - CombinationStrategy enum with 4 zero-handling strategies
  - Log-space accumulation with max-shift numerical stability
  - Diagnostic report and comparison table generation
affects: [21-02, phase-22]

tech-stack:
  added: []
  patterns: [log-space-accumulation, strategy-enum-dispatch, markdown-report-generation]

key-files:
  created:
    - master_thesis_code/bayesian_inference/posterior_combination.py
    - master_thesis_code_test/bayesian_inference/test_posterior_combination.py
  modified: []

key-decisions:
  - "Used StrEnum for CombinationStrategy (Python 3.13 ruff UP042 compliance)"
  - "Physics-floor strategy falls back to exclude with logged warning (Phase 22 placeholder)"
  - "NaN distinguishes missing events from zero-likelihood events in the array"

patterns-established:
  - "Strategy dispatch: StrEnum + match in apply_strategy for extensible zero-handling"
  - "Log-space accumulation: np.sum(np.log(...)) with max-shift before exp for 500+ event stability"
  - "Markdown report generation: string builder pattern for diagnostic and comparison outputs"

requirements-completed: [ANAL-01, ANAL-02, POST-01, NFIX-01]

duration: 5min
completed: 2026-04-02
---

# Phase 21 Plan 01: Posterior Combination Module Summary

**Log-space posterior combination with 4 zero-handling strategies, diagnostic reporting, and 20 passing tests**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-02T16:15:40Z
- **Completed:** 2026-04-02T16:20:51Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files created:** 2

## Accomplishments
- Complete posterior combination module with 9 public functions (541 lines)
- All 4 zero-handling strategies: naive (tiny floor), exclude (remove zero events), per-event-floor (min/100), physics-floor (fallback to exclude)
- Log-space accumulation using np.sum(np.log(...)) with max-shift for numerical stability -- handles 500+ events without underflow
- Diagnostic report generator: identifies zero patterns (all-zeros, low-h-only, partial), root cause analysis, impact assessment
- Comparison table generator: runs all 4 strategies, reports MAP estimates and event counts
- End-to-end combine_posteriors entry point: loads JSONs, applies strategy, writes combined_posterior.json + reports
- 20 passing unit tests (312 lines), mypy clean, ruff clean

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests** - `32e5382` (test)
2. **Task 1 GREEN: Implementation** - `7df0baa` (feat)

## Files Created/Modified
- `master_thesis_code/bayesian_inference/posterior_combination.py` - Complete combination logic: loading, array construction, 4 strategies, log-space combine, diagnostic report, comparison table, main entry point
- `master_thesis_code_test/bayesian_inference/test_posterior_combination.py` - 20 unit tests covering all functions, strategies, edge cases (all-zero events, 500-event stress test), and end-to-end pipeline

## Decisions Made
- Used `StrEnum` instead of `str, Enum` for CombinationStrategy (Python 3.13 / ruff UP042 compliance)
- Physics-floor strategy logs a warning and falls back to exclude (Phase 22 will implement the actual physics floor)
- NaN in the likelihood array distinguishes "missing event" (not in JSON) from "zero likelihood" (evaluated to 0.0)
- Diagnostic report categorizes zeros into three patterns: all-zeros (no hosts anywhere), low-h-only (redshift catalog gap), partial-zeros (coverage boundary)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed StrEnum inheritance for Python 3.13**
- **Found during:** Task 1 GREEN (ruff check)
- **Issue:** Plan specified `class CombinationStrategy(str, Enum)` but ruff UP042 flags this in Python 3.13
- **Fix:** Changed to `class CombinationStrategy(StrEnum)` using `from enum import StrEnum`
- **Files modified:** `master_thesis_code/bayesian_inference/posterior_combination.py`
- **Verification:** ruff check passes

**2. [Rule 1 - Bug] Fixed mypy type errors in diagnostic report**
- **Found during:** Task 1 GREEN (mypy check)
- **Issue:** Generic `dict[str, object]` for zero_events caused mypy iteration errors
- **Fix:** Replaced with parallel lists (zero_det_indices, zero_h_bins, zero_patterns) for type safety
- **Files modified:** `master_thesis_code/bayesian_inference/posterior_combination.py`
- **Verification:** mypy passes with no errors

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both auto-fixes necessary for type safety and linting compliance. No scope creep.

## Issues Encountered
None - implementation proceeded smoothly after mypy/ruff fixes.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functions are fully implemented. Physics-floor falls back to exclude by design (documented, Phase 22 scope).

## Next Phase Readiness
- posterior_combination.py is fully functional and tested, ready for CLI wiring in Plan 02
- Module exports all required public functions: CombinationStrategy, load_posterior_jsons, build_likelihood_array, apply_strategy, combine_log_space, generate_diagnostic_report, generate_comparison_table, combine_posteriors

---
*Phase: 21-analysis-post-processing*
*Completed: 2026-04-02*

## Self-Check: PASSED
- All created files exist on disk
- All commit hashes found in git log
