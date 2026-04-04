---
phase: 19-campaign-dashboards
plan: 01
subsystem: plotting
tags: [matplotlib, mosaic, mollweide, rasterized, dashboard]

# Dependency graph
requires:
  - phase: 18-new-plot-modules
    provides: sky_plots.py Mollweide factory, bayesian_plots.py SNR distribution, simulation_plots.py detection yield
provides:
  - plot_campaign_dashboard 4-panel composite factory function
  - rasterized scatter optimization for PDF output
affects: [19-02-batch-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns: [subplot_mosaic with per_subplot_kw for mixed projections, constrained_layout for colorbar compatibility]

key-files:
  created: [master_thesis_code/plotting/dashboard_plots.py, master_thesis_code_test/plotting/test_dashboard_plots.py]
  modified: [master_thesis_code/plotting/sky_plots.py, master_thesis_code/plotting/evaluation_plots.py]

key-decisions:
  - "Used constrained_layout instead of tight_layout for colorbar compatibility with subplot_mosaic"
  - "Figure height set to width*0.75 (5.25in) to give Mollweide panel room"

patterns-established:
  - "subplot_mosaic with per_subplot_kw for mixed-projection composite figures"

requirements-completed: [CAMP-01, CAMP-03]

# Metrics
duration: 5min
completed: 2026-04-02
---

# Phase 19 Plan 01: Campaign Dashboard Summary

**4-panel campaign dashboard factory with Mollweide sky map and rasterized scatter for PDF size optimization**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-02T19:48:01Z
- **Completed:** 2026-04-02T19:53:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created `plot_campaign_dashboard()` factory assembling posterior, SNR, yield, and sky panels in a 2x2 mosaic
- Sky panel uses Mollweide projection via `per_subplot_kw` in `plt.subplot_mosaic`
- Added `rasterized=True` to 3 scatter calls (1 sky, 2 evaluation) for PDF file size optimization
- 4 smoke tests covering return types, projection, figure size, and content presence

## Task Commits

Each task was committed atomically:

1. **Task 1: Dashboard factory + smoke tests** - `08ed6b8` (feat) -- TDD: tests + implementation in single commit
2. **Task 2: Add rasterized=True to scatter calls** - `ce2910e` (feat)

## Files Created/Modified
- `master_thesis_code/plotting/dashboard_plots.py` - Campaign dashboard 4-panel composite factory
- `master_thesis_code_test/plotting/test_dashboard_plots.py` - 4 smoke tests for dashboard factory
- `master_thesis_code/plotting/sky_plots.py` - Added rasterized=True to Mollweide scatter
- `master_thesis_code/plotting/evaluation_plots.py` - Added rasterized=True to injected-vs-recovered scatter (2 calls)

## Decisions Made
- Used `constrained_layout` instead of `tight_layout()` because the sky panel's colorbar conflicts with tight_layout engine switching
- Figure size (7.0, 5.25) gives Mollweide panel adequate vertical space while staying at double-column width

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Switched from tight_layout to constrained_layout**
- **Found during:** Task 1 (dashboard factory implementation)
- **Issue:** `fig.tight_layout()` raises RuntimeError when a colorbar has already been created by the sky panel factory -- matplotlib cannot switch layout engines mid-figure
- **Fix:** Replaced `fig.tight_layout()` with `layout="constrained"` in `plt.subplot_mosaic()` call
- **Files modified:** master_thesis_code/plotting/dashboard_plots.py
- **Verification:** All 4 tests pass
- **Committed in:** 08ed6b8 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary fix for matplotlib layout engine compatibility. No scope creep.

## Issues Encountered
None beyond the tight_layout deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Dashboard factory ready for batch pipeline (Plan 02) to generate per-campaign summary figures
- All 4 sub-plot factories tested individually and as composite

---
*Phase: 19-campaign-dashboards*
*Completed: 2026-04-02*
