---
phase: quick
plan: 260330-ojq
subsystem: bayesian-inference
tags: [posterior, h0-bias, diagnostic, evaluation, pipeline-b]

requires:
  - phase: quick-260330-oaf
    provides: "Diagnostic bias fix (removed /d_L factor, disabled P_det)"
provides:
  - "11 diagnostic posterior JSONs for h-value grid 0.60-0.86"
  - "Comparison script: scripts/compare_posterior_bias.py"
  - "Comparison report showing bias shift from h=0.600 to h=0.678"
affects: [bayesian-inference, bias-investigation, phase-12]

tech-stack:
  added: []
  patterns: ["evaluation run directory with symlinked CRB data"]

key-files:
  created:
    - scripts/compare_posterior_bias.py
    - evaluation/run_v12_diagnostic/comparison_report.md
    - evaluation/run_v12_diagnostic/simulations/posteriors/ (11 JSON files)
  modified:
    - master_thesis_code_test/parameter_estimation/parameter_estimation_test.py

key-decisions:
  - "Symlinked galaxy catalog into diagnostic run dir to enable local CPU evaluation"
  - "Residual h=0.052 offset means /d_L and P_det are not the only bias sources"

requirements-completed: [diagnostic-bias-rerun]

duration: 26min
completed: 2026-03-30
---

# Quick Task 260330-ojq: Re-run Evaluation Pipeline Summary

**Diagnostic evaluation shows /d_L and P_det fix shifts posterior peak from h=0.600 to h=0.678, reducing bias by 60% but leaving a 0.052 residual offset from true h=0.73**

## Performance

- **Duration:** 26 min
- **Started:** 2026-03-30T15:44:59Z
- **Completed:** 2026-03-30T18:10:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Ran all 11 h-value evaluations (0.60 to 0.86, step 0.026) with the diagnostic bias fix
- Created reusable comparison script that computes log-posteriors and generates markdown reports
- Confirmed the /d_L factor and P_det were major bias contributors (peak shifted +0.078 toward true value)
- Identified that residual bias of 0.052 requires further investigation (additional bias sources)

## Key Results

| Metric | Biased Run | Diagnostic Run |
|--------|-----------|----------------|
| Peak h-value | 0.600 | 0.678 |
| Offset from h_true=0.73 | -0.130 | -0.052 |
| Bias reduction | -- | 60% |

The diagnostic fix (removing /d_L division and disabling P_det) accounts for approximately 60% of the total bias. The remaining 0.052 offset suggests at least one additional bias source exists in Pipeline B.

## Task Commits

1. **Task 1 + Task 2: Run diagnostic evaluations and create comparison script** - `8013749` (feat)

**Plan metadata:** (pending)

## Files Created/Modified

- `scripts/compare_posterior_bias.py` - Comparison tool for biased vs diagnostic posteriors
- `evaluation/run_v12_diagnostic/simulations/posteriors/*.json` - 11 diagnostic posterior files (gitignored)
- `evaluation/run_v12_diagnostic/comparison_report.md` - Full comparison report (gitignored)
- `master_thesis_code_test/parameter_estimation/parameter_estimation_test.py` - Fixed pre-existing mypy errors

## Decisions Made

- Symlinked `reduced_galaxy_catalogue.csv` (1.4 GB) into diagnostic run directory to enable local CPU evaluation (catalog path is relative to CWD)
- Also symlinked `undetected_events.csv` which is required by the evaluation pipeline
- Combined Task 1 and Task 2 into a single commit since the comparison script was ready before evaluations completed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Galaxy catalog path resolution for local evaluation**
- **Found during:** Task 1 (running evaluations)
- **Issue:** `GalaxyCatalogueHandler` reads from `./master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv` relative to CWD. When running from `evaluation/run_v12_diagnostic/`, the path doesn't resolve.
- **Fix:** Symlinked the galaxy catalog directory structure into the diagnostic run directory
- **Files modified:** evaluation/run_v12_diagnostic/master_thesis_code/galaxy_catalogue/ (symlink)
- **Verification:** All 11 evaluations completed successfully

**2. [Rule 3 - Blocking] Missing undetected_events.csv**
- **Found during:** Task 1 (running evaluations)
- **Issue:** Pipeline B also reads `simulations/undetected_events.csv` which wasn't symlinked
- **Fix:** Added symlink from validation run directory
- **Files modified:** evaluation/run_v12_diagnostic/simulations/undetected_events.csv (symlink)
- **Verification:** Evaluations proceeded after adding symlink

**3. [Rule 1 - Bug] Pre-existing mypy errors in parameter_estimation_test.py**
- **Found during:** Task 2 (commit attempt)
- **Issue:** 13 unused `type: ignore` comments blocked all commits via pre-commit hook
- **Fix:** Removed unnecessary comments, added `numpy.typing` import, used correct error codes
- **Files modified:** master_thesis_code_test/parameter_estimation/parameter_estimation_test.py
- **Verification:** `mypy master_thesis_code/ master_thesis_code_test/` reports 0 errors

**4. [Rule 3 - Blocking] uv.lock out of sync caused mypy hook failure**
- **Found during:** Task 2 (commit attempt)
- **Issue:** `uv run mypy` in pre-commit hook was modifying uv.lock, causing "files were modified" failure
- **Fix:** Ran `uv sync --extra cpu --extra dev` and committed updated uv.lock
- **Verification:** Pre-commit hooks pass cleanly

---

**Total deviations:** 4 auto-fixed (2 blocking path issues, 1 pre-existing bug, 1 blocking hook issue)
**Impact on plan:** All auto-fixes necessary for execution. No scope creep.

## Issues Encountered

- Initial evaluation attempts failed with `FileNotFoundError` because the pipeline reads data files relative to CWD. The cluster validation run had all files in place, but running locally required symlinks.
- The pre-commit mypy hook ran `uv run mypy` which modified `uv.lock` as a side effect, causing persistent commit failures until the lock file was synced and staged.

## Known Stubs

None.

## User Setup Required

None - no external service configuration required.

## Next Steps

The 60% bias reduction confirms /d_L and P_det are major contributors but not the only ones. Investigate:
1. Galaxy catalog weighting / volume element in likelihood integration
2. Prior normalization in `single_host_likelihood`
3. Detection selection effects beyond P_det
4. Redshift-dependent systematic errors

---
*Quick task: 260330-ojq*
*Completed: 2026-03-30*
