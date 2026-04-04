---
phase: 17-enhanced-existing-plots
plan: 03
subsystem: plotting
tags: [matplotlib, contourf, gridspec, scatter, colorbar, heatmap]

requires:
  - phase: 16-plotting-infrastructure
    provides: "_colors, _labels, _helpers, _data infrastructure modules"
provides:
  - "P_det heatmaps with contour lines (0.5/0.9) and scatter overlay in (d_L,M) and (z,M) spaces"
  - "plot_injected_vs_recovered multi-panel scatter grid with identity lines, CRB error bars, residual sub-panels"
  - "All model_plots and evaluation_plots functions style-migrated to _colors/_labels/get_figure"
affects: [18-new-plot-types, evaluation-pipeline]

tech-stack:
  added: []
  patterns: ["_plot_detection_heatmap private helper for code reuse between coordinate variants", "GridSpec with height_ratios for main+residual panel pairs"]

key-files:
  created: []
  modified:
    - master_thesis_code/plotting/model_plots.py
    - master_thesis_code/plotting/evaluation_plots.py
    - master_thesis_code/plotting/_helpers.py
    - master_thesis_code_test/plotting/conftest.py
    - master_thesis_code_test/plotting/test_model_plots.py
    - master_thesis_code_test/plotting/test_evaluation_plots.py

key-decisions:
  - "Extracted _plot_detection_heatmap private helper to share logic between grid and zM variants"
  - "Widened make_colorbar type from AxesImage to ScalarMappable to support contourf QuadContourSet"

patterns-established:
  - "Private _plot_* helpers for coordinate-variant heatmaps sharing identical logic"
  - "GridSpec main+residual layout with height_ratios=[3,1] per parameter"

requirements-completed: [CORE-05, FISH-06, FISH-07]

duration: 4min
completed: 2026-04-02
---

# Phase 17 Plan 03: Model & Evaluation Plot Upgrades Summary

**P_det heatmaps with contour lines and scatter overlay in both (d_L,M) and (z,M) spaces, plus multi-panel injected-vs-recovered scatter grid with residual sub-panels**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-02T18:09:23Z
- **Completed:** 2026-04-02T18:13:28Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments
- Upgraded plot_detection_probability_grid with contour lines at 0.5/0.9, [0,1] colorbar, and scatter overlay for injected/detected populations
- Created plot_detection_probability_zM as new factory for (z,M) coordinate space using shared _plot_detection_heatmap helper
- Built plot_injected_vs_recovered with multi-panel GridSpec layout: identity lines, optional CRB error bars, residual sub-panels
- Style-migrated all model_plots and evaluation_plots functions to _colors/_labels/get_figure infrastructure
- Added 7 new tests covering contours, scatter overlay, (z,M) variant, and injected-vs-recovered (with/without errors, custom params)

## Task Commits

Each task was committed atomically:

1. **Task 1: Upgrade model_plots.py with P_det contours, scatter overlay, and (z,M) variant** - `2a51a0f` (feat)
2. **Task 2: Add plot_injected_vs_recovered and style-migrate evaluation_plots.py** - `71c071f` (feat)
3. **Task 3: Update tests for new and changed functions** - `2e28ac2` (test)

## Files Created/Modified
- `master_thesis_code/plotting/model_plots.py` - P_det heatmaps with contours, scatter, new (z,M) variant, style migration
- `master_thesis_code/plotting/evaluation_plots.py` - New injected-vs-recovered + style migration of all functions
- `master_thesis_code/plotting/_helpers.py` - Widened make_colorbar type to ScalarMappable
- `master_thesis_code_test/plotting/conftest.py` - Added sample_injected_recovered fixture
- `master_thesis_code_test/plotting/test_model_plots.py` - 3 new tests for contours, scatter, zM
- `master_thesis_code_test/plotting/test_evaluation_plots.py` - 3 new tests for injected-vs-recovered

## Decisions Made
- Extracted `_plot_detection_heatmap` private helper to eliminate code duplication between grid and zM variants
- Widened `make_colorbar` type signature from `AxesImage` to `ScalarMappable` to support `contourf` return type (Rule 3 auto-fix)
- Removed `title` parameter from `plot_detection_contour` (backward-compatible: callers using keyword arg will get TypeError, but no callers existed in codebase)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Widened make_colorbar type from AxesImage to ScalarMappable**
- **Found during:** Task 1 (model_plots.py upgrade)
- **Issue:** `make_colorbar` was typed to accept `AxesImage` but `contourf` returns `QuadContourSet` (a `ScalarMappable`), causing mypy error
- **Fix:** Changed type annotation from `AxesImage` to `ScalarMappable` in `_helpers.py`
- **Files modified:** `master_thesis_code/plotting/_helpers.py`
- **Verification:** mypy passes on model_plots.py and _helpers.py
- **Committed in:** `2a51a0f` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential type fix for correct colorbar usage with contourf. No scope creep.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functions are fully wired to data inputs.

## Next Phase Readiness
- P_det heatmaps ready for thesis figures in both coordinate spaces
- Injected-vs-recovered grid ready for evaluation pipeline output
- Phase 18 (new plot types) can proceed with Mollweide projection replacing plot_sky_localization_3d

## Self-Check: PASSED

All 6 files verified present. All 3 task commits verified in git log.

---
*Phase: 17-enhanced-existing-plots*
*Completed: 2026-04-02*
