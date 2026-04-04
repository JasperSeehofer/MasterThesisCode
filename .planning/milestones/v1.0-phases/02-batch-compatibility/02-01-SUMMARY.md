---
phase: 02-batch-compatibility
plan: 01
subsystem: infra
tags: [argparse, cli, slurm, batch, entry-points]

# Dependency graph
requires:
  - phase: 01-code-hardening
    provides: CPU-safe imports and guarded GPU usage
provides:
  - Non-interactive merge script with --workdir and --delete-sources flags
  - Non-interactive prepare script with --workdir flag and main() entry point
  - Console entry points emri-merge and emri-prepare registered in pyproject.toml
affects: [04-slurm-jobs]

# Tech tracking
tech-stack:
  added: []
  patterns: [argparse CLI for batch scripts, --workdir path resolution pattern]

key-files:
  created:
    - scripts/__init__.py
    - master_thesis_code_test/scripts/__init__.py
    - master_thesis_code_test/scripts/test_merge_cramer_rao_bounds.py
    - master_thesis_code_test/scripts/test_prepare_detections.py
  modified:
    - scripts/merge_cramer_rao_bounds.py
    - scripts/prepare_detections.py
    - pyproject.toml

key-decisions:
  - "Console entry points registered in pyproject.toml [project.scripts] section for uv-installed CLI access"
  - "parse_args(argv) pattern allows both CLI and programmatic invocation (testable)"

patterns-established:
  - "Batch script pattern: parse_args(argv) + main(argv) for testable, non-interactive CLI scripts"
  - "--workdir flag for SLURM jobs to resolve paths relative to working directory"

requirements-completed: [BATCH-01, BATCH-02]

# Metrics
duration: 4min
completed: 2026-03-26
---

# Phase 02 Plan 01: Batch Script Compatibility Summary

**Argparse CLIs for merge and prepare scripts with --workdir/--delete-sources flags, zero interactive prompts, and emri-merge/emri-prepare console entry points**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-26T12:19:29Z
- **Completed:** 2026-03-26T12:23:08Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Removed all 4 `input()` calls from merge_cramer_rao_bounds.py, replaced with --delete-sources flag
- Extracted prepare_detections.py __main__ block into callable main(argv) with argparse --workdir
- Registered emri-merge and emri-prepare console entry points in pyproject.toml
- Added 13 new tests (8 merge, 5 prepare) covering batch behavior, all passing
- Full CPU test suite: 180 passed, 0 regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor merge_cramer_rao_bounds.py (RED)** - `d0517f2` (test)
2. **Task 1: Refactor merge_cramer_rao_bounds.py (GREEN)** - `4035060` (feat)
3. **Task 2: Refactor prepare_detections.py (RED)** - `6e38205` (test)
4. **Task 2: Refactor prepare_detections.py (GREEN)** - `795a041` (feat)

_Note: TDD tasks have separate test and implementation commits._

## Files Created/Modified
- `scripts/__init__.py` - Makes scripts a package for console entry points
- `scripts/merge_cramer_rao_bounds.py` - Non-interactive merge with argparse CLI (--workdir, --delete-sources)
- `scripts/prepare_detections.py` - Batch-callable prepare with main(argv) and argparse CLI (--workdir)
- `pyproject.toml` - Added [project.scripts] section with emri-merge and emri-prepare entry points
- `master_thesis_code_test/scripts/__init__.py` - Test package init
- `master_thesis_code_test/scripts/test_merge_cramer_rao_bounds.py` - 8 tests for merge batch behavior
- `master_thesis_code_test/scripts/test_prepare_detections.py` - 5 tests for prepare batch behavior

## Decisions Made
- Console entry points registered in pyproject.toml [project.scripts] for uv-installed CLI access
- parse_args(argv) pattern allows both CLI and programmatic invocation (testable without subprocess)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. After `uv sync --extra cpu --extra dev`, the `emri-merge` and `emri-prepare` commands are available on PATH.

## Next Phase Readiness
- Both scripts ready for SLURM job scripts in Phase 4
- emri-merge --workdir $WORKSPACE --delete-sources can be called directly from SLURM batch scripts
- emri-prepare --workdir $WORKSPACE can be called directly from SLURM batch scripts

---
*Phase: 02-batch-compatibility*
*Completed: 2026-03-26*
