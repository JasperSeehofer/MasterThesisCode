---
phase: 21-analysis-post-processing
plan: 02
subsystem: cli
tags: [cli-wiring, integration-testing, posterior-combination, campaign-validation]

requires:
  - phase: 21-01
    provides: posterior_combination.py module with combine_posteriors entry point
provides:
  - "--combine and --strategy CLI arguments wired to combine_posteriors"
  - "6 integration tests validating against real h_sweep_20260401 campaign data"
  - "Run metadata captures combine and strategy parameters"
affects: [phase-22]

tech-stack:
  added: []
  patterns: [lazy-import-dispatch, skipif-data-availability]

key-files:
  created: []
  modified:
    - master_thesis_code/arguments.py
    - master_thesis_code/main.py
    - master_thesis_code_test/test_arguments.py
    - master_thesis_code_test/bayesian_inference/test_posterior_combination.py
    - master_thesis_code_test/test_main_metadata.py

key-decisions:
  - "Lazy import of combine_posteriors inside if-block matches existing generate_figures pattern"
  - "Integration tests use absolute path fallback to main repo for campaign data access from worktrees"
  - "Default posteriors_dir is working_directory/posteriors following existing convention"

patterns-established:
  - "CLI dispatch pattern: lazy import inside conditional block for optional pipeline stages"
  - "Campaign test pattern: skipif with Path.exists() for data-dependent integration tests"

requirements-completed: [POST-01, NFIX-01, ANAL-01, ANAL-02]

duration: 8min
completed: 2026-04-02
---

# Phase 21 Plan 02: CLI Wiring and Campaign Validation Summary

**CLI --combine/--strategy flags wired to posterior combination with 6 integration tests validating against real h_sweep campaign data**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-02T16:24:11Z
- **Completed:** 2026-04-02T16:32:27Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Added --combine flag and --strategy argument with 4 choices (naive, exclude, per-event-floor, physics-floor) to CLI
- Wired main.py dispatch to combine_posteriors with lazy import pattern
- 6 integration tests validate loading 15 h-value JSONs, naive/exclude strategies, diagnostic report (detects events 163/223/507), comparison table, and end-to-end output file generation
- 299 total tests pass (including 5 new argument tests + 6 campaign integration tests)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add --combine and --strategy CLI arguments** - `7837a44` (feat)
2. **Task 2: Integration test against real campaign data** - `1799e2d` (feat)

## Files Created/Modified
- `master_thesis_code/arguments.py` - Added --combine flag, --strategy with choices, and two properties
- `master_thesis_code/main.py` - Added combine dispatch block and metadata recording
- `master_thesis_code_test/test_arguments.py` - 5 new tests for combine/strategy arguments
- `master_thesis_code_test/bayesian_inference/test_posterior_combination.py` - TestCampaignIntegration class with 6 integration tests
- `master_thesis_code_test/test_main_metadata.py` - Fixed _make_arguments to include combine/strategy

## Decisions Made
- Lazy import of combine_posteriors inside `if arguments.combine:` block matches existing `generate_figures` pattern in main.py
- Integration tests try both relative and absolute paths for campaign data, enabling execution from worktrees
- Default posteriors_dir is `{working_directory}/posteriors` following the campaign output convention

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_main_metadata _make_arguments missing new attributes**
- **Found during:** Task 1 (full test suite verification)
- **Issue:** `_make_arguments` helper in test_main_metadata.py creates argparse.Namespace without combine/strategy, causing AttributeError when _write_run_metadata accesses arguments.combine
- **Fix:** Added `combine=False, strategy="physics-floor"` to the Namespace construction
- **Files modified:** `master_thesis_code_test/test_main_metadata.py`
- **Verification:** Full test suite passes (299 tests)
- **Committed in:** 1799e2d (Task 2 commit)

**2. [Rule 1 - Bug] Fixed ruff import sorting in test_posterior_combination.py**
- **Found during:** Task 2 (ruff check)
- **Issue:** `from __future__ import annotations` from Plan 01 caused I001 import sorting error
- **Fix:** Ran `ruff check --fix` to remove unnecessary future import
- **Files modified:** `master_thesis_code_test/bayesian_inference/test_posterior_combination.py`
- **Verification:** ruff check passes
- **Committed in:** 1799e2d (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both auto-fixes necessary for test suite compatibility. No scope creep.

## Issues Encountered
None - implementation proceeded smoothly.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functions are fully wired and tested. Physics-floor falls back to exclude by design (documented, Phase 22 scope).

## Next Phase Readiness
- Full CLI pipeline operational: `python -m master_thesis_code <dir> --combine --strategy exclude` produces combined_posterior.json, diagnostic_report.md, comparison_table.md
- Ready for Phase 22 to implement the actual physics-floor strategy
- All 299 tests pass, mypy clean, ruff clean

---
*Phase: 21-analysis-post-processing*
*Completed: 2026-04-02*

## Self-Check: PASSED
- All 5 modified files exist on disk
- All 2 commit hashes found in git log
