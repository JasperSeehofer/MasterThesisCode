---
phase: 18-new-plot-modules
plan: 01
subsystem: plotting
tags: [mollweide, corner-plot, fisher-matrix, sky-map, matplotlib]

# Dependency graph
requires:
  - phase: 16-data-layer-fisher
    provides: "_data.py PARAMETER_NAMES, label_key, reconstruct_covariance"
  - phase: 15-style-infrastructure
    provides: "_helpers.py get_figure/make_colorbar, _colors.py palette, _labels.py LABELS"
  - phase: 17-enhanced-existing-plots
    provides: "fisher_plots.py _ellipse_params helper"
provides:
  - "plot_sky_localization_mollweide factory function in sky_plots.py"
  - "plot_fisher_corner factory function in fisher_plots.py"
  - "Widened make_colorbar to accept ScalarMappable (not just AxesImage)"
affects: [18-new-plot-modules, plotting-callbacks]

# Tech tracking
tech-stack:
  added: [corner]
  patterns: [mollweide-projection-factory, corner-wrapper-with-rc-context]

key-files:
  created:
    - master_thesis_code/plotting/sky_plots.py
    - master_thesis_code_test/plotting/test_sky_plots.py
  modified:
    - master_thesis_code/plotting/fisher_plots.py
    - master_thesis_code/plotting/_helpers.py
    - master_thesis_code_test/plotting/test_fisher_plots.py
    - master_thesis_code_test/plotting/conftest.py
    - pyproject.toml

key-decisions:
  - "Widened make_colorbar mappable type from AxesImage to ScalarMappable for scatter/contourf support"
  - "Added corner to mypy ignore_missing_imports (no py.typed marker)"

patterns-established:
  - "Mollweide projection via get_figure(subplot_kw={'projection': 'mollweide'})"
  - "corner.corner wrapped in matplotlib.rc_context to disable constrained_layout"

requirements-completed: [SKY-01, FISH-03]

# Metrics
duration: 7min
completed: 2026-04-02
---

# Phase 18 Plan 01: Sky Map & Fisher Corner Plot Summary

**Mollweide sky localization map with SNR colorbar and error ellipses, plus corner plot wrapping corner.corner with thesis styling and multi-event overlay**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-02T18:52:21Z
- **Completed:** 2026-04-02T18:59:42Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Mollweide sky map factory function with ecliptic-to-Mollweide coordinate transform, SNR-colored scatter, optional localization ellipses from Fisher sky sub-covariance
- Fisher corner plot factory wrapping corner.corner with thesis color palette, truth lines, quantile labels, and up to 4-event overlay
- Full quality gate pass: 310 tests, zero ruff/mypy errors, 59% coverage

## Task Commits

Each task was committed atomically:

1. **Task 1 (RED): Failing tests** - `f318354` (test)
2. **Task 1 (GREEN): Implementation** - `26a53cc` (feat)
3. **Task 2: Quality gate fixes** - `debd26a` (chore)

## Files Created/Modified
- `master_thesis_code/plotting/sky_plots.py` - Mollweide sky localization map factory
- `master_thesis_code/plotting/fisher_plots.py` - Added plot_fisher_corner function
- `master_thesis_code/plotting/_helpers.py` - Widened make_colorbar type to ScalarMappable
- `master_thesis_code_test/plotting/test_sky_plots.py` - 3 smoke tests for sky map
- `master_thesis_code_test/plotting/test_fisher_plots.py` - 3 smoke tests for corner plot
- `master_thesis_code_test/plotting/conftest.py` - sample_sky_data fixture
- `pyproject.toml` - corner dependency + mypy override

## Decisions Made
- Widened make_colorbar mappable type from AxesImage to ScalarMappable to support scatter PathCollection
- Added both "corner" and "corner.*" to mypy ignore_missing_imports overrides (corner has no py.typed)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Widened make_colorbar type signature**
- **Found during:** Task 1 (sky map implementation)
- **Issue:** make_colorbar accepted only AxesImage, but scatter returns PathCollection (a ScalarMappable)
- **Fix:** Changed type annotation from AxesImage to ScalarMappable, removed unused AxesImage import
- **Files modified:** master_thesis_code/plotting/_helpers.py
- **Verification:** Sky map colorbar renders correctly, all existing tests pass
- **Committed in:** 26a53cc (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Type widening was necessary for correctness and already noted in Phase 17 decisions. No scope creep.

## Issues Encountered
- mypy cache retained stale type info for corner module after adding ignore override; cleared .mypy_cache to resolve

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Sky map and corner plot factories ready for integration into PlottingCallback
- Both functions follow (fig, ax/ndarray) return convention
- All existing tests continue to pass

## Self-Check: PASSED

All 5 created/modified source files verified present. All 3 commit hashes (f318354, 26a53cc, debd26a) found in git log.

---
*Phase: 18-new-plot-modules*
*Completed: 2026-04-02*
