---
phase: 01-code-hardening
plan: 02
subsystem: cli
tags: [argparse, multiprocessing, gpu-flag, cli-threading]

requires:
  - phase: 01-code-hardening/01
    provides: CPU-safe MemoryManagement with use_gpu parameter
provides:
  - "--use_gpu CLI flag threaded to ParameterEstimation and MemoryManagement"
  - "--num_workers CLI flag threaded to BayesianStatistics.evaluate()"
  - "Affinity-expansion hack removed from bayesian_statistics.py"
affects: [02-slurm-scripts, 03-cluster-env, 04-integration]

tech-stack:
  added: []
  patterns: ["keyword-only use_gpu/num_workers parameters on pipeline functions"]

key-files:
  created:
    - master_thesis_code_test/test_arguments.py
  modified:
    - master_thesis_code/arguments.py
    - master_thesis_code/main.py
    - master_thesis_code/bayesian_inference/bayesian_statistics.py

key-decisions:
  - "num_workers default uses os.sched_getaffinity(0) - 2 with AttributeError fallback to os.cpu_count()"
  - "use_gpu threaded as keyword-only arg to data_simulation() and snr_analysis()"
  - "Removed affinity-expansion hack entirely rather than keeping as dead code"

patterns-established:
  - "CLI flags threaded as keyword-only parameters through pipeline dispatch functions"

requirements-completed: [CODE-01, CODE-03]

duration: 3min
completed: 2026-03-26
---

# Phase 01 Plan 02: CLI Flag Threading Summary

**--use_gpu and --num_workers CLI flags added and threaded through data_simulation, snr_analysis, and evaluate call chains**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-26T10:28:39Z
- **Completed:** 2026-03-26T10:31:37Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- Added --use_gpu flag (store_true) threaded to ParameterEstimation and MemoryManagement in both data_simulation() and snr_analysis()
- Added --num_workers flag with sched_getaffinity-based default, threaded to BayesianStatistics.evaluate()
- Removed affinity-expansion hack from bayesian_statistics.py (D-07)
- 9 unit tests covering both flags, all passing; 167 total tests green; mypy clean

## Task Commits

Each task was committed atomically:

1. **Task 1: Add CLI flags and thread through main.py** - `3c3fb19` (feat)
2. **Task 2: Update BayesianStatistics.evaluate() with num_workers** - `3052da2` (feat)
3. **Task 3: Add unit tests for CLI flags** - `5e954c9` (test)

## Files Created/Modified
- `master_thesis_code/arguments.py` - Added --use_gpu and --num_workers parser args and properties
- `master_thesis_code/main.py` - Threaded flags through data_simulation(), snr_analysis(), evaluate()
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` - Accept num_workers param, remove affinity hack
- `master_thesis_code_test/test_arguments.py` - 9 unit tests for both CLI flags

## Decisions Made
- num_workers default uses os.sched_getaffinity(0) - 2 with AttributeError fallback to os.cpu_count() - matches SLURM environments where affinity is set per job
- use_gpu kept as keyword-only parameter on pipeline functions to prevent positional-arg confusion
- Removed affinity-expansion hack entirely (was already commented out) rather than keeping as dead code

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed mypy error in num_workers property**
- **Found during:** Task 3 (test verification)
- **Issue:** `raw` from argparse.Namespace typed as `Any`, causing mypy `no-any-return` error
- **Fix:** Added explicit `int | None` type annotation on `raw` variable
- **Files modified:** master_thesis_code/arguments.py
- **Verification:** `uv run mypy master_thesis_code/` passes with 0 errors
- **Committed in:** 5e954c9 (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minimal - type annotation fix required for mypy compliance.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functionality is fully wired.

## Next Phase Readiness
- Phase 01 code-hardening complete: both plans (CPU-safe MemoryManagement + CLI flag threading) done
- Pipeline functions now accept explicit use_gpu and num_workers from CLI
- Ready for Phase 02 (SLURM scripts) and Phase 03 (cluster environment setup)

---
*Phase: 01-code-hardening*
*Completed: 2026-03-26*
