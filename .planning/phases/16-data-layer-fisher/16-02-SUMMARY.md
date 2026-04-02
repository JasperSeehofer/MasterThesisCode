---
phase: 16-data-layer-fisher
plan: 02
subsystem: plotting
tags: [matplotlib, fisher-matrix, error-ellipses, strain-curve, uncertainty]

# Dependency graph
requires:
  - phase: 16-data-layer-fisher
    plan: 01
    provides: CRB data layer (_data.py), PARAMETER_NAMES, reconstruct_covariance, label_key
  - phase: 15-style-infra
    provides: _helpers.py (get_figure, _fig_from_ax), _colors.py (CYCLE, EDGE, REFERENCE), _labels.py (LABELS), _style.py

provides:
  - artifact: master_thesis_code/plotting/fisher_plots.py
    exports: [plot_fisher_ellipses, plot_characteristic_strain, plot_parameter_uncertainties]
    description: "Three Fisher-based visualization factories: error ellipses, strain sensitivity, parameter uncertainties"
---

## Summary

Created three Fisher matrix visualization factory functions in `fisher_plots.py`:
1. **Error ellipses** (`plot_fisher_ellipses`) — 1-sigma/2-sigma filled contours for parameter pairs with multi-event overlay support
2. **Characteristic strain** (`plot_characteristic_strain`) — log-log h_c(f) sensitivity curve showing total, instrument, and galactic foreground noise + example EMRI signal
3. **Parameter uncertainties** (`plot_parameter_uncertainties`) — violin plot (multi-event, >= 10 rows) or horizontal bar chart (single-event) of fractional uncertainties grouped by intrinsic/extrinsic

## Key Decisions

| ID | Decision | Rationale |
|----|----------|-----------|
| D-04 | Default pairs: (M, mu), (d_L, qS), (qS, phiS) | Most physically interesting correlations |
| D-05 | Filled ellipses with alpha=0.4/level, CYCLE colors | Consistent with project style, clear visual hierarchy |
| D-07 | Deferred import of LisaTdiConfiguration | Avoids CPU import failure from cupy dependency |
| D-09 | Violin threshold set to >= 10 rows | Below that, violin bodies degenerate; bar chart more readable |
| D-11 | Log scale on uncertainty axes | Fractional errors span 5+ orders of magnitude; linear makes small values unreadable |

## Deviations

| Plan | Actual | Reason |
|------|--------|--------|
| Violin threshold >= 5 | Threshold >= 10 | User feedback: 5 rows produce degenerate violins |
| strain preset="single" | preset="double" | User feedback: legend dominated small plot |
| Linear uncertainty scale | Log scale on both bar and violin | User feedback: M fractional error unreadable on linear scale |

## key-files

### created
- `master_thesis_code/plotting/fisher_plots.py` (357 lines) -- 3 factory functions + _ellipse_params helper
- `master_thesis_code_test/plotting/test_fisher_plots.py` (108 lines) -- 7 smoke tests

### modified
- `master_thesis_code_test/plotting/conftest.py` -- sample_crb_dataframe now generates 12 varied rows

## Verification

- 7/7 smoke tests pass (`test_fisher_plots.py`)
- 15/15 data layer tests pass (`test_data.py`)
- mypy: clean (0 errors)
- ruff: clean (0 violations)
- Visual verification: user approved all four plot types after log-scale and layout fixes

## Self-Check: PASSED
