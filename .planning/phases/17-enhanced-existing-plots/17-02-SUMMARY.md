---
phase: 17-enhanced-existing-plots
plan: 02
subsystem: plotting
tags: [matplotlib, histogram, cdf, psd-decomposition, style-migration]

requires:
  - phase: 17-01
    provides: "Style infrastructure (_colors, _labels, _helpers), upgraded bayesian_plots.py"
provides:
  - "plot_snr_distribution factory function with histogram + CDF + threshold annotation"
  - "plot_detection_yield factory function with injected/detected histograms + detection fraction"
  - "plot_lisa_psd decompose mode with S_inst, S_gal, S_total curves"
  - "Style-migrated catalog_plots.py (all 4 functions use _colors/_labels/get_figure)"
  - "Style-migrated remaining simulation_plots.py functions"
affects: [17-03, evaluation-plots, thesis-figures]

tech-stack:
  added: []
  patterns:
    - "twinx pattern for dual-axis plots (CDF overlay, detection fraction)"
    - "Deferred import of LisaTdiConfiguration to avoid CPU import issues"
    - "np.maximum guard for floating-point subtraction in PSD confusion noise"

key-files:
  created: []
  modified:
    - master_thesis_code/plotting/bayesian_plots.py
    - master_thesis_code/plotting/simulation_plots.py
    - master_thesis_code/plotting/catalog_plots.py
    - master_thesis_code_test/plotting/test_bayesian_plots.py
    - master_thesis_code_test/plotting/test_simulation_plots.py

key-decisions:
  - "Used list[float] for bin_edges to satisfy mypy Axes.hist type expectations"
  - "LisaTdiConfiguration works on CPU (guarded cupy import) so decompose test runs without GPU"
  - "Kept plot_glade_completeness xlabel as 'Distance [Mpc]' since LABELS has d_L not distance"

patterns-established:
  - "twinx dual-axis pattern: primary axes returned to caller, secondary axes created internally"
  - "Combined legend from multiple axes: get_legend_handles_labels from both axes"

requirements-completed: [CORE-03, CORE-04, CORE-06]

duration: 5min
completed: 2026-04-02
---

# Phase 17 Plan 02: New Plot Factories and Style Migration Summary

**Added SNR distribution (histogram+CDF), detection yield (dual histogram+fraction), PSD decomposition (3-curve), and style-migrated all catalog/simulation plots to centralized infrastructure**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-02T18:15:57Z
- **Completed:** 2026-04-02T18:20:11Z
- **Tasks:** 2/2
- **Files modified:** 5

## Accomplishments

- Added `plot_snr_distribution` to bayesian_plots.py with histogram on left y-axis, CDF step function on right y-axis, and threshold annotation with fraction-above text
- Added `plot_detection_yield` to simulation_plots.py with injected (outline) and detected (filled) histograms plus detection fraction curve on secondary axis
- Upgraded `plot_lisa_psd` with backward-compatible `decompose=True` mode that computes and plots S_total (solid), S_inst (dashed), and S_gal (dash-dot) using deferred LisaTdiConfiguration import
- Style-migrated all 4 catalog_plots.py functions: replaced plt.subplots with get_figure, hardcoded colors with EDGE, plain labels with LABELS dict, removed titles
- Style-migrated simulation_plots.py: plot_gpu_usage and plot_lisa_noise_components now use get_figure and LABELS
- Added 5 new smoke tests covering all new functionality

## Task Commits

1. **Task 1: Add new plot functions and style-migrate** - `a0c7129` (feat)
2. **Task 2: Add tests for all new functions** - `04e1dba` (test)

## Files Created/Modified

- `master_thesis_code/plotting/bayesian_plots.py` - Added plot_snr_distribution, imported MEAN and REFERENCE colors
- `master_thesis_code/plotting/simulation_plots.py` - Added plot_detection_yield, upgraded plot_lisa_psd with decompose mode, style-migrated all functions
- `master_thesis_code/plotting/catalog_plots.py` - Style-migrated all 4 functions to use _colors/_labels/get_figure
- `master_thesis_code_test/plotting/test_bayesian_plots.py` - Added test_plot_snr_distribution and test_plot_snr_distribution_custom_threshold
- `master_thesis_code_test/plotting/test_simulation_plots.py` - Added test_plot_detection_yield and test_plot_lisa_psd_decompose

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None - all functions are fully implemented with real data processing.

## Verification

- All 22 plotting tests pass
- Full suite: 320 passed, 18 deselected (GPU/slow), 3 warnings (pre-existing)
- ruff check: all checks passed
- mypy: no errors in all 3 modified source files

## Self-Check: PASSED

- All 5 modified files exist on disk
- Commit a0c7129 (Task 1) found in git log
- Commit 04e1dba (Task 2) found in git log
- SUMMARY.md created at expected path
