---
phase: 05-documentation
plan: 01
subsystem: docs
tags: [cluster, slurm, hpc, bwunicluster, quickstart]

requires:
  - phase: 04-slurm-jobs
    provides: cluster scripts (simulate.sbatch, merge.sbatch, evaluate.sbatch, submit_pipeline.sh, etc.)
provides:
  - cluster/README.md quickstart and reference guide
affects: [05-02]

tech-stack:
  added: []
  patterns: []

key-files:
  created:
    - cluster/README.md
  modified: []

key-decisions:
  - "Used ASCII art for pipeline diagram instead of Mermaid for maximum portability"
  - "Included real parameter values in quickstart per D-05 decision"

patterns-established:
  - "Blockquote callouts for warnings and tips in cluster docs"

requirements-completed: [DOCS-01]

duration: 5min
completed: 2026-03-27
---

# Plan 05-01: Cluster README Summary

**Self-contained cluster/README.md with 5-command quickstart, ASCII pipeline diagram, worked example, troubleshooting, and script reference**

## Performance

- **Duration:** 5 min
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created 279-line cluster/README.md covering the full user journey
- Quickstart section with real parameter values (--tasks 100 --steps 50 --seed 42)
- ASCII pipeline diagram showing simulate -> merge -> evaluate dependency chain
- Troubleshooting section covering OOM, timeout, CUDA errors, and failed task resubmission
- Workspace expiration warnings with ws_extend command

## Task Commits

1. **Task 1: Create cluster/README.md** - `add47ae` (docs)

## Files Created/Modified
- `cluster/README.md` - Complete cluster workflow guide (279 lines)

## Decisions Made
None - followed plan as specified.

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- cluster/README.md ready for cross-referencing from CLAUDE.md and README.md (plan 05-02)

---
*Phase: 05-documentation*
*Completed: 2026-03-27*
