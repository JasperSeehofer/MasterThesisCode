---
phase: 17-enhanced-existing-plots
plan: 01
subsystem: plotting
tags: [matplotlib, credible-intervals, colorbar, posterior, cosmology]

requires:
  - phase: 15-style-infrastructure
    provides: "_colors, _labels, _helpers with get_figure/make_colorbar"
  - phase: 16-color-label-constants
    provides: "LABELS dict, CYCLE/CMAP/TRUTH/EDGE color constants"
provides:
  - "plot_combined_posterior with 68%/95% credible intervals and Planck/SH0ES reference bands"
  - "plot_event_posteriors with color_by mapping (snr/redshift/dl_error) and combined overlay"
  - "plot_distance_redshift with multi-H0 comparison curves"
  - "Style-migrated plot_subset_posteriors, plot_detection_redshift_distribution, plot_number_of_possible_hosts"
affects: [17-02, 17-03, evaluation-plots, thesis-figures]

tech-stack:
  added: []
  patterns: [credible-interval-shading, scalar-mappable-colorbar, reference-band-overlay]

key-files:
  created: []
  modified:
    - master_thesis_code/plotting/bayesian_plots.py
    - master_thesis_code/plotting/physical_relations_plots.py
    - master_thesis_code_test/plotting/test_bayesian_plots.py
    - master_thesis_code_test/plotting/test_physical_relations_plots.py

key-decisions:
  - "Used np.trapezoid instead of deprecated np.trapz for NumPy 2.x compatibility"
  - "Title param kept for backward compat but not set by default (thesis uses captions)"

patterns-established:
  - "Credible intervals via CDF searchsorted + fill_between with alpha layering"
  - "ScalarMappable colorbar pattern for color_by metadata mapping"
  - "Reference bands via axvspan + center vline + ax.text inline label"

requirements-completed: [CORE-01, CORE-02, CORE-07]

duration: 3min
completed: 2026-04-02
---

# Phase 17 Plan 01: Enhanced Bayesian and Physical Relations Plots Summary

**H0 posterior plots upgraded with 68%/95% credible interval shading, Planck/SH0ES reference bands, SNR/redshift/dl_error color mapping with colorbar, combined posterior overlay, and multi-H0 distance-redshift comparison curves**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-02T18:09:11Z
- **Completed:** 2026-04-02T18:12:43Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- plot_combined_posterior now shows shaded 68%/95% credible intervals with boundary lines, plus Planck (h=0.674) and SH0ES (h=0.73) reference bands with 1-sigma shading
- plot_event_posteriors accepts color_by parameter (snr/redshift/dl_error) with automatic colorbar, and combined_posterior as thick overlay line
- plot_distance_redshift accepts h0_values list with distance_fn callback for multi-H0 comparison curves
- All five bayesian_plots functions and plot_distance_redshift migrated to get_figure/LABELS/_colors infrastructure
- Both peak and density normalization modes supported via normalize parameter

## Task Commits

Each task was committed atomically:

1. **Task 1: Upgrade bayesian_plots.py and physical_relations_plots.py** - `06ea76d` (feat)
2. **Task 2: Update tests for changed signatures and new parameters** - `adc2768` (test)

## Files Created/Modified
- `master_thesis_code/plotting/bayesian_plots.py` - Added credible intervals, reference bands, color_by mapping, combined overlay, normalization modes; migrated all 5 functions to style infrastructure
- `master_thesis_code/plotting/physical_relations_plots.py` - Added multi-H0 curves via h0_values/distance_fn; migrated to style infrastructure
- `master_thesis_code_test/plotting/test_bayesian_plots.py` - Added 5 new tests: credible intervals, density normalization, reference bands, color_by SNR, combined overlay
- `master_thesis_code_test/plotting/test_physical_relations_plots.py` - Added multi-H0 test with toy distance function

## Decisions Made
- Used `np.trapezoid` instead of deprecated `np.trapz` for NumPy 2.x compatibility (mypy caught it)
- Title parameter kept in signatures for backward compat but not set on axes unless caller passes non-default value

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] np.trapz replaced with np.trapezoid**
- **Found during:** Task 1 (verification)
- **Issue:** NumPy 2.x removed `np.trapz`; mypy flagged `Module has no attribute "trapz"`
- **Fix:** Changed to `np.trapezoid` which is the NumPy 2.x replacement
- **Files modified:** master_thesis_code/plotting/bayesian_plots.py
- **Verification:** mypy passes clean
- **Committed in:** 06ea76d (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Trivial API rename for NumPy 2.x compat. No scope creep.

## Issues Encountered
None

## Known Stubs
None - all functions are fully wired with real computation.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Style infrastructure fully applied to bayesian_plots.py and physical_relations_plots.py
- Ready for Plan 02 (new diagnostic plots) and Plan 03 (remaining plot modules)

---
*Phase: 17-enhanced-existing-plots*
*Completed: 2026-04-02*
