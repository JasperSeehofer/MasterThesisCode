---
phase: 29-style-foundation
plan: 01
subsystem: plotting
tags: [style, colors, matplotlib, publication-quality]
dependency_graph:
  requires: []
  provides: [emri_thesis.mplstyle, _colors.py, _style.py]
  affects: [all-plotting-modules]
tech_stack:
  added: []
  patterns: [okabe-ito-palette, type42-fonts, truncated-colormap]
key_files:
  created: []
  modified:
    - master_thesis_code/plotting/emri_thesis.mplstyle
    - master_thesis_code/plotting/_colors.py
    - master_thesis_code/plotting/_style.py
    - master_thesis_code_test/plotting/test_style.py
    - master_thesis_code_test/plotting/test_colors.py
decisions:
  - "Okabe-Ito palette (Wong 2011) for colorblind-safe figures"
  - "7-9pt font range for REVTeX column widths (3.375in single, 7.0in double)"
  - "Type 42 fonts to prevent Type 3 bitmap fonts in PDFs"
  - "SEQUENTIAL_BLUES truncated 0.1-0.85 to avoid near-white/near-black extremes"
metrics:
  duration_seconds: 150
  completed: "2026-04-07T14:59:36Z"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 5
---

# Phase 29 Plan 01: Style Foundation Summary

Publication-quality matplotlib style with Okabe-Ito colorblind-safe palette, Type 42 fonts, 7-9pt font sizes for REVTeX columns, clean spines, and truncated sequential colormap.

## What Was Done

### Task 1: Update mplstyle, _style.py latex branch, and style regression tests (STYL-01 + STYL-03)

**Commit:** b6a4834

Updated `emri_thesis.mplstyle` with publication-quality rcParams:
- Font sizes reduced to 7-9pt range (font.size=8, axes.titlesize=9, tick/legend=7)
- pdf.fonttype=42, ps.fonttype=42 to prevent Type 3 bitmap fonts
- axes.spines.top=False, axes.spines.right=False for clean publication style
- xtick.direction=in, ytick.direction=in for inward ticks
- legend.frameon=False for frameless legends
- Okabe-Ito color cycle via axes.prop_cycle

Updated `_style.py` latex branch to use matching 7-9pt font sizes (was 9-12pt).

Updated `test_style.py`:
- `test_apply_style_default_unchanged` pinned to new 7-9pt values
- `test_rcparams_snapshot` expanded from 18 to 26 keys (added spines, fonttype, ticks, legend.frameon)
- Added `test_no_type3_fonts_in_pdf` with graceful skip when pdffonts unavailable

### Task 2: Replace _colors.py with Okabe-Ito palette and update test_colors.py (STYL-02)

**Commit:** 9d9e963

Replaced `_colors.py` with Okabe-Ito edition (Wong 2011, Nature Methods):
- CYCLE: 7 Okabe-Ito colors (orange, sky blue, bluish green, yellow, blue, vermillion, reddish purple)
- TRUTH=#009E73 (bluish green), MEAN=#D55E00 (vermillion), REFERENCE=#56B4E9 (sky blue), ACCENT=#E69F00 (orange)
- EDGE=#1a1a1a unchanged (near-black)
- SEQUENTIAL_BLUES: LinearSegmentedColormap truncated from Blues (0.1-0.85)
- CMAP="viridis" unchanged for backward compatibility

Updated `test_colors.py` with 3 new tests:
- test_accent_is_hex, test_sequential_blues_is_cmap_object, test_cycle_is_okabe_ito

## Verification Results

- `uv run pytest master_thesis_code_test/plotting/ -v --no-cov --tb=short`: 113 passed
- `uv run pytest -m "not gpu and not slow" --tb=short -q`: 382 passed, 6 skipped
- `uv run ruff check _colors.py _style.py`: All checks passed
- `uv run mypy _colors.py _style.py`: Success, no issues found

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added pdffonts availability guard to test_no_type3_fonts_in_pdf**
- **Found during:** Task 1
- **Issue:** Plan's test_no_type3_fonts_in_pdf did not handle missing pdffonts binary
- **Fix:** Added `shutil.which("pdffonts")` check with `pytest.skip()` for graceful degradation
- **Files modified:** master_thesis_code_test/plotting/test_style.py
- **Commit:** b6a4834

## Decisions Made

1. **Okabe-Ito palette**: Wong (2011) Nature Methods doi:10.1038/nmeth.1618 -- verified colorblind-safe for deuteranopia, protanopia, tritanopia
2. **7-9pt font range**: Sized for REVTeX single-column (3.375in) and double-column (7.0in) widths
3. **Type 42 fonts**: pdf.fonttype=42 prevents Type 3 bitmap fonts that cause journal rejection
4. **SEQUENTIAL_BLUES truncation**: 0.1-0.85 range avoids near-white and near-black extremes

## Self-Check: PASSED
