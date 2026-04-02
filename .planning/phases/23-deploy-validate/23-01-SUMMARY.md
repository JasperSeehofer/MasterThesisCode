---
phase: 23-deploy-validate
plan: 01
subsystem: bayesian-inference
tags: [posterior-combination, physics-floor, validation, h0-posterior, numerical-stability]

# Dependency graph
requires:
  - phase: 22-likelihood-floor-overflow-fix
    provides: physics-floor strategy implementation in posterior_combination.py
provides:
  - Committed validation record at results/v1.4-validation.md
  - Confirmed physics-floor MAP matches exclude MAP (diff=0.00, within +/-0.05)
  - Baselines reproduced: naive=0.86, exclude=0.66
affects: [23-02-cluster-deploy, thesis-writeup]

# Tech tracking
tech-stack:
  added: []
  patterns: [Direct Python import of combine_posteriors bypasses galaxy-catalog init for local validation runs]

key-files:
  created:
    - results/v1.4-validation.md
  modified:
    - .gitignore (added results/h_sweep_*/ exclusion)

key-decisions:
  - "Run combine_posteriors directly via uv run python -c import — bypasses galaxy-catalog init failure when GLADE+.txt absent"
  - "Physics-floor MAP=0.66 equals exclude MAP=0.66 (diff=0.00 < 0.05 threshold) — PASS verdict"
  - "Naive MAP=0.86 baseline reproduced confirming pipeline consistency"
  - "Deferred with-BH-mass validation: posteriors_with_bh_mass/ not present in h_sweep_20260401"

patterns-established:
  - "Validation reports committed to results/v1.4-validation.md as permanent thesis record"

requirements-completed: [DEPL-02]

# Metrics
duration: 3min
completed: 2026-04-02
---

# Phase 23 Plan 01: Deploy-Validate Summary

**Three-strategy posterior comparison (naive/exclude/physics-floor) on h_sweep_20260401 campaign data: physics-floor MAP=0.66 matches exclude MAP exactly (PASS), baselines reproduced**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-02T17:43:58Z
- **Completed:** 2026-04-02T17:46:40Z
- **Tasks:** 1
- **Files modified:** 2 (results/v1.4-validation.md created, .gitignore updated)

## Accomplishments
- Ran all three strategies (naive, exclude, physics-floor) on 534 local campaign events
- Reproduced both baselines: naive MAP=0.86, exclude MAP=0.66
- Confirmed physics-floor MAP=0.66 (diff=0.00 from exclude, well within +/-0.05)
- Committed permanent thesis validation record at results/v1.4-validation.md

## Task Commits

1. **Task 1: Run three-strategy comparison and write validation report** - `44635cb` (docs)

**Plan metadata:** (included in final state commit)

## Files Created/Modified
- `results/v1.4-validation.md` - Three-strategy comparison table, baseline check, acceptance criterion check, PASS verdict
- `.gitignore` - Added `results/h_sweep_*/` to exclude simulation campaign data from git

## Decisions Made
- Called `combine_posteriors` directly via `uv run python -c "..."` instead of the CLI `--combine` flag, because the CLI's `main()` initializes the galaxy catalog first and fails when `GLADE+.txt` is absent locally. This is a valid local-run approach; the full CLI works on the cluster where the catalog is present.
- Placed `results/h_sweep_*/` in `.gitignore` to keep large campaign data directories out of git (they are regenerable on the cluster).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Galaxy catalog missing, CLI --combine blocked**
- **Found during:** Task 1 (running naive strategy via CLI)
- **Issue:** `uv run python -m master_thesis_code results/h_sweep_20260401 --combine --strategy naive` failed because `main()` initializes `GalaxyCatalogueHandler` before the `--combine` block, and `GLADE+.txt` is absent locally.
- **Fix:** Invoked `combine_posteriors` directly via `uv run python -c "..."` import, bypassing the galaxy-catalog initialization entirely. The combination logic itself is independent of the catalog.
- **Files modified:** none (workaround, not a code change)
- **Verification:** All three strategies ran and produced valid MAP values
- **Committed in:** 44635cb (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 — blocking)
**Impact on plan:** Workaround is correct; the CLI works on cluster where catalog exists. No code change needed.

## Issues Encountered
- Campaign data in `results/h_sweep_20260401/` was untracked in git. Added `results/h_sweep_*/` to `.gitignore` to prevent accidental commits of large simulation output.

## Next Phase Readiness
- v1.4 physics-floor validation complete with PASS verdict
- Ready for Phase 23 Plan 02: cluster deployment
- Known deferred item: "with BH mass" variant validation (data not available locally)

---
*Phase: 23-deploy-validate*
*Completed: 2026-04-02*
