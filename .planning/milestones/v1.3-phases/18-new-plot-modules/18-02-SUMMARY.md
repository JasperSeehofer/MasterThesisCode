---
phase: 18-new-plot-modules
plan: 02
subsystem: plotting
tags: [matplotlib, astropy, wilson-score, convergence, detection-efficiency]

# Dependency graph
requires:
  - phase: 15-plot-style-infrastructure
    provides: "get_figure, _fig_from_ax, CYCLE, LABELS, TRUTH color constants"
provides:
  - "plot_h0_convergence factory function (two-panel: posteriors + CI width vs N)"
  - "plot_detection_efficiency factory function (binned efficiency + Wilson CI band)"
  - "_credible_interval_width helper for symmetric CI computation"
affects: [19-plot-wiring, thesis-figures]

# Tech tracking
tech-stack:
  added: [astropy.stats.binom_conf_interval]
  patterns: [log-sum-exp posterior combination, Wilson score CI for binomial proportions]

key-files:
  created:
    - master_thesis_code/plotting/convergence_plots.py
    - master_thesis_code_test/plotting/test_convergence_plots.py
  modified:
    - master_thesis_code_test/plotting/conftest.py

key-decisions:
  - "astropy binom_conf_interval type stubs are available -- no type: ignore needed"
  - "Log-sum-exp used for numerical stability when combining posteriors"

patterns-established:
  - "Two-panel convergence layout via get_figure(ncols=2, preset='double')"
  - "Wilson score CI via astropy for binomial efficiency plots"

requirements-completed: [CONV-01, CONV-02]

# Metrics
duration: 5min
completed: 2026-04-02
---

# Phase 18 Plan 02: Convergence Plots Summary

**H0 convergence two-panel plot (posterior narrowing + CI width vs N with 1/sqrt(N) reference) and detection efficiency curve with Wilson score CI band via astropy**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-02T18:52:22Z
- **Completed:** 2026-04-02T18:57:15Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Two-panel H0 convergence plot showing posterior curves narrowing with increasing event count (left) and CI width vs N with 1/sqrt(N) reference (right)
- Detection efficiency curve with Wilson score CI band computed via astropy.stats.binom_conf_interval
- 8 smoke tests covering return types, reproducibility, custom parameters, truth line, CI band presence
- Full quality gate pass: 312 tests, zero ruff warnings, zero mypy errors

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Add failing tests** - `0d56550` (test)
2. **Task 1 (GREEN): Implement convergence_plots.py** - `7dc1421` (feat)
3. **Task 2: Quality gate pass** - `1c8b049` (chore)

## Files Created/Modified
- `master_thesis_code/plotting/convergence_plots.py` - H0 convergence and detection efficiency factory functions
- `master_thesis_code_test/plotting/test_convergence_plots.py` - 8 smoke tests for both functions
- `master_thesis_code_test/plotting/conftest.py` - Added sample_event_posteriors and sample_injection_campaign fixtures

## Decisions Made
- astropy.stats type stubs are available in the project, so no `type: ignore` comment needed on import
- Log-sum-exp used for numerical stability when combining multiple posteriors (avoids underflow in np.prod)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functions are fully wired with real data processing logic.

## Next Phase Readiness
- Convergence plot functions ready to be wired into the figure generation pipeline
- Both functions follow the `(fig, ax)` return convention compatible with `save_figure`

---
*Phase: 18-new-plot-modules*
*Completed: 2026-04-02*
