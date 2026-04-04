---
phase: 11-validation-campaign
plan: 01
subsystem: validation
tags: [validation, comparison, crb, snr, fisher-matrix, pandas]

dependency_graph:
  requires:
    - phase: 10-five-point-stencil-derivatives
      provides: five-point stencil default, condition number monitoring, 90s timeout
    - phase: 09-confusion-noise
      provides: confusion noise in PSD
  provides:
    - merged branch with Phase 9 + 10 corrections on claudes_sidequests
    - comparison script for v1.1 vs v1.2 validation runs
  affects: [11-02-PLAN, cluster deployment]

tech_stack:
  added: []
  patterns: [cli-comparison-script, markdown-report-generation]

key_files:
  created:
    - scripts/compare_validation_runs.py
  modified: []

key_decisions:
  - "Merged Phase 10 with --no-ff to preserve branch history"
  - "Comparison script uses self-comparison for testing (baseline == new)"
  - "P90 fractional d_L error recommended as threshold"

patterns_established:
  - "Validation comparison reports as structured markdown with pass/fail checks"

requirements_completed: [SIM-01, SIM-03]

duration: 4min
completed: 2026-03-29
tasks_completed: 2
tasks_total: 2
files_modified: 1
---

# Phase 11 Plan 01: Merge and Comparison Script Summary

**Phase 10 five-point stencil merged into claudes_sidequests; CLI comparison script produces 8-section markdown validation report from CRB CSVs**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-29T17:45:22Z
- **Completed:** 2026-03-29T17:50:00Z
- **Tasks:** 2/2
- **Files modified:** 1 (+ merge commit touching 3 files)

## Accomplishments

- Merged Phase 10 (five-point stencil, 90s timeout, condition number logging) into claudes_sidequests via --no-ff merge
- Verified all 198 CPU tests pass, ruff and mypy clean after merge
- Created `scripts/compare_validation_runs.py` (586 lines) producing structured validation reports

## Task Commits

Each task was committed atomically:

1. **Task 1: Merge Phase 10 worktree branch** - `197d9f7` (merge)
2. **Task 2: Write validation comparison script** - `9b278ee` (feat)

## What Was Done

### Task 1: Merge Phase 10 worktree branch into claudes_sidequests

Merged `worktree-agent-a7dd0449` into `claudes_sidequests` with `--no-ff`. Verified:
- `signal.alarm(90)`: 2 occurrences in main.py
- `include_confusion_noise`: 3 occurrences in LISA_configuration.py
- `use_five_point_stencil`: 3 occurrences in parameter_estimation.py
- 198 tests pass, 18 deselected (GPU/slow), coverage 40.80%
- ruff check: clean
- mypy: clean (0 issues, 44 files)

### Task 2: Write validation comparison script

Created `scripts/compare_validation_runs.py` with 8 report sections:

1. **Run Metadata** -- git commit, seed, timestamp, task count from run_metadata_*.json
2. **Detection Rate** -- total events, detections (SNR >= 20), detection rate
3. **SNR Distribution** -- summary stats (min, median, mean, max, std) for detected events
4. **CRB Analysis** -- fractional d_L error stats + percentiles (P10-P99)
5. **d_L Threshold Recommendation** -- P90 percentile with D-07 note
6. **Wall Time Analysis** -- per-event generation_time stats
7. **Error Analysis** -- NaN counts, negative CRB diagonals, SLURM log references
8. **Pass/Fail Summary** -- 5 directional checks with PASS/FAIL/N/A

Self-comparison test (baseline == new) produces valid report with all 5 checks showing expected results (FAIL for check 1 since medians are equal in self-comparison, FAIL for check 2, PASS for checks 3-5).

## Files Created/Modified

- `scripts/compare_validation_runs.py` -- CLI validation comparison script (586 lines)

## Decisions Made

- Merged Phase 10 with `--no-ff` to preserve branch history and make the merge commit visible in git log
- Comparison script is pure pandas/numpy with no matplotlib dependency (markdown-only output)
- P90 fractional d_L error is the recommended threshold metric, with explicit note not to update constants.py

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed list.append() with two arguments**
- **Found during:** Task 2 (initial script run)
- **Issue:** Four calls to `lines.append("header", "")` passed two arguments to append, causing TypeError
- **Fix:** Changed to `lines.extend(["header", ""])` for all four occurrences
- **Files modified:** `scripts/compare_validation_runs.py`
- **Committed in:** `9b278ee`

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Trivial syntax fix, no scope change.

## Issues Encountered

None.

## Known Stubs

None -- all functionality is fully wired.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- claudes_sidequests now has both Phase 9 (confusion noise) and Phase 10 (five-point stencil) changes
- Comparison script ready for use once v1.2 cluster results arrive
- Next step: push to cluster and run validation campaign (Plan 02)

## Self-Check: PASSED

- scripts/compare_validation_runs.py: FOUND
- Commit 197d9f7 (Task 1 merge): FOUND
- Commit 9b278ee (Task 2 script): FOUND

---
*Phase: 11-validation-campaign*
*Completed: 2026-03-29*
