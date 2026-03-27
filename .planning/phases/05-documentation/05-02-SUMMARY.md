---
phase: 05-documentation
plan: 02
subsystem: docs
tags: [claude-code, readme, cluster, hpc]

requires:
  - phase: 05-documentation
    provides: cluster/README.md (plan 05-01)
provides:
  - CLAUDE.md Cluster Deployment section with CLI flags and script inventory
  - README.md Running on HPC section pointing to cluster/README.md
affects: []

tech-stack:
  added: []
  patterns: []

key-files:
  created: []
  modified:
    - CLAUDE.md
    - README.md

key-decisions:
  - "Kept CLAUDE.md section concise per D-09 (flags, inventory, pointer — no duplication)"
  - "Kept README.md section brief per D-11 (under 10 lines, just awareness + pointer)"

patterns-established: []

requirements-completed: [DOCS-02, DOCS-03]

duration: 5min
completed: 2026-03-27
---

# Plan 05-02: CLAUDE.md & README.md Cluster Sections Summary

**Cluster Deployment section in CLAUDE.md with CLI flags table and script inventory; Running on HPC section in README.md**

## Performance

- **Duration:** 5 min
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added Cluster Deployment section to CLAUDE.md between Running the Code and Running Tests
- CLI flags table (--use_gpu, --num_workers, --simulation_index, --seed)
- Script inventory table listing all 8 cluster/ scripts
- Quick reference code block with setup, submit, and resubmit commands
- Added Running on HPC section to README.md pointing to cluster/README.md

## Task Commits

1. **Task 1: Add Cluster Deployment section to CLAUDE.md** - `488e9d0` (docs)
2. **Task 2: Add Running on HPC section to README.md** - `488e9d0` (docs)

## Files Created/Modified
- `CLAUDE.md` - Added Cluster Deployment section (~40 lines)
- `README.md` - Added Running on HPC section (~10 lines)

## Decisions Made
None - followed plan as specified.

## Deviations from Plan
Tasks 1 and 2 were committed together in one commit since both were documentation-only changes to the same plan.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 5 documentation complete — all three requirement IDs (DOCS-01, DOCS-02, DOCS-03) satisfied

---
*Phase: 05-documentation*
*Completed: 2026-03-27*
