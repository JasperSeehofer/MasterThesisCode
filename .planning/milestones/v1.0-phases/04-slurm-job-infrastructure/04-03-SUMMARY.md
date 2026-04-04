---
phase: 04-slurm-job-infrastructure
plan: 03
subsystem: infra
tags: [slurm, bash, sbatch, pipeline, hpc, failure-recovery]

# Dependency graph
requires:
  - phase: 04-02
    provides: sbatch job scripts (simulate.sbatch, merge.sbatch, evaluate.sbatch)
provides:
  - Pipeline orchestrator (submit_pipeline.sh) chaining simulate -> merge -> evaluate via SLURM dependencies
  - Failure recovery helper (resubmit_failed.sh) for resubmitting only failed array tasks
affects: [05-documentation]

# Tech tracking
tech-stack:
  added: []
  patterns: [slurm-dependency-chain, sacct-failure-query, partial-output-cleanup]

key-files:
  created:
    - cluster/submit_pipeline.sh
    - cluster/resubmit_failed.sh
  modified: []

key-decisions:
  - "All three CLI flags (--tasks, --steps, --seed) required with no defaults for safety"
  - "resubmit_failed.sh takes 4 positional args rather than extracting seed/steps from metadata files"

patterns-established:
  - "SLURM dependency chain: sbatch --parsable captures job ID, --dependency=afterok chains next stage"
  - "Partial output cleanup before resubmission prevents corrupted data in merge step"

requirements-completed: [SLURM-04]

# Metrics
duration: 2min
completed: 2026-03-27
---

# Phase 4 Plan 3: Pipeline Orchestrator and Failure Recovery Summary

**Single-command SLURM pipeline submission with afterok dependency chaining and sacct-based failure recovery for array tasks**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-27T12:28:19Z
- **Completed:** 2026-03-27T12:29:54Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Pipeline orchestrator submits simulate (GPU array) -> merge -> evaluate with automatic SLURM dependency chaining
- Failure recovery script identifies failed array tasks via sacct, cleans partial output, and resubmits only failed indices
- Both scripts pass bash -n syntax validation and are executable

## Task Commits

Each task was committed atomically:

1. **Task 1: Create submit_pipeline.sh pipeline orchestrator** - `66fe45a` (feat)
2. **Task 2: Create resubmit_failed.sh failure recovery helper** - `aedf218` (feat)

## Files Created/Modified
- `cluster/submit_pipeline.sh` - Pipeline orchestrator: parses --tasks/--steps/--seed, creates run directory, submits 3 chained SLURM jobs
- `cluster/resubmit_failed.sh` - Failure recovery: queries sacct for failed tasks, deletes partial CSVs/metadata, resubmits failed indices only

## Decisions Made
- All three flags (--tasks, --steps, --seed) are required with no defaults to prevent accidental misconfigured submissions
- resubmit_failed.sh takes base_seed and sim_steps as positional arguments rather than parsing from run_metadata JSON files, keeping the script simple and avoiding jq dependency
- Log paths passed via CLI --output/--error (not in SBATCH headers) per research pitfall about variable expansion

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Known Stubs

None - both scripts are complete implementations.

## Next Phase Readiness
- All cluster infrastructure scripts are complete (modules.sh, setup.sh, simulate/merge/evaluate sbatch, submit_pipeline.sh, resubmit_failed.sh)
- Ready for Phase 5 documentation covering quickstart, monitoring, and troubleshooting

## Self-Check: PASSED

- FOUND: cluster/submit_pipeline.sh
- FOUND: cluster/resubmit_failed.sh
- FOUND: 04-03-SUMMARY.md
- FOUND: commit 66fe45a (Task 1)
- FOUND: commit aedf218 (Task 2)

---
*Phase: 04-slurm-job-infrastructure*
*Completed: 2026-03-27*
