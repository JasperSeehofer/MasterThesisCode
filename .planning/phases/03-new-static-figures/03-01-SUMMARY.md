---
phase: 03-new-static-figures
plan: 01
subsystem: ui
tags: [matplotlib, plotting, composite-figure, selection-function, mollweide, color_by, revtex]

# Dependency graph
requires:
  - phase: 02-colormap-heatmap-modernization
    provides: "plot_pdet_surface (fig20) cividis pcolormesh + 0.5/0.9 horizon contours; sky SNR norm; get_figure presets; _colors v2 palette"
  - phase: 01-posterior-factory-consolidation
    provides: "plot_combined_posterior canonical stacked-posterior factory + load_canonical_combined_posterior loader"
provides:
  - "selection_plots.plot_selection_function_explainer (fig21): 1x2 survival-curve + p_det heatmap composite"
  - "population_plots.plot_population_constraint_view (fig22): 2x2 driver / sky / spaghetti / stacked composite"
  - "fig02 now surfaces the previously-never-enabled color_by=snr per-event coloring"
  - "fig21 + fig22 wired into generate_figures manifest with graceful skip on missing/misaligned data"
affects: [05-annotation-decluttering-rollout, 04-data-gated-figures]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Composite figure = orchestrate tested single-panel factories via ax= delegation; only the panel layout is novel"
    - "Mollweide-in-a-grid: get_figure(preset) for SIZE -> fig.clf() -> GridSpec(2,2) with projection='mollweide' on one cell only"
    - "Alignment-guard skip: enable cross-source coloring/composition only on exact length match; log + skip otherwise (never guess)"

key-files:
  created:
    - master_thesis_code/plotting/selection_plots.py
    - master_thesis_code/plotting/population_plots.py
    - master_thesis_code_test/plotting/test_selection_plots.py
    - master_thesis_code_test/plotting/test_population_plots.py
  modified:
    - master_thesis_code/main.py
    - master_thesis_code_test/plotting/test_bayesian_plots.py

key-decisions:
  - "Two NEW topic modules (selection_plots, population_plots) rather than bloating evaluation_plots/bayesian_plots — composites span multiple topic modules, so a cross-topic import into any one would break the one-concern-per-module convention"
  - "fig21 LEFT panel = 1D p_det(d_L) survival marginal (histogram ratio of saved SNR); RIGHT = delegate plot_pdet_surface unchanged"
  - "fig22 2x2 = driver SNRxz (TL) / Mollweide sky (TR) / de-emphasized color_by spaghetti (BL) / canonical stacked posterior (BR)"
  - "fig22 color_by driven by SNR by default (shared visual key with the driver panel); redshift selectable via color_by='redshift'"
  - "_labels.py was NOT modified — every needed symbol (z, SNR, h, M) already existed in LABELS; Gpc d_L label carried literally as fig20 does"

patterns-established:
  - "Composite orchestration: import + ax=delegate to tested factories; reimplement nothing"
  - "Preset-size invariant under custom GridSpec: inherit figsize from get_figure(preset) then clf()+GridSpec so the size-from-preset test still holds with a mixed-projection layout"

requirements-completed: [VR-NEW-03, VR-NEW-04]

# Metrics
duration: ~40min
completed: 2026-06-21
---

# Phase 3 Plan 01: New Static Figures (Selection & Population) Summary

**Added the two highest-value no-new-data static figures the pipeline lacked — a selection-function / detection-horizon explainer (fig21) and a population constraint-provenance view (fig22) — each a composite orchestrating already-tested single-panel encodings, plus surfacing fig02's previously-dead `color_by` SNR coloring.**

## Performance

- **Duration:** ~40 min
- **Started:** 2026-06-21 (this session)
- **Completed:** 2026-06-21
- **Tasks:** 3/3
- **Files modified:** 6 (4 created, 2 modified)

## Accomplishments

### Task 1 — fig21 selection-function / detection-horizon explainer (commit 13d200b)
New module `selection_plots.py`:
- `_pdet_survival_curve(d_l, snr, snr_threshold, n_bins)` — the 1D `p_det(d_L)` marginal of the fig20 surface, computed as `N(SNR>=thr)/N_total` per d_L bin (histogram ratio of saved SNR values; NO new data source, NO physics). Empty bins -> NaN.
- `plot_selection_function_explainer(injection_csv_glob, *, snr_threshold, h_inj_filter, n_survival_bins, axes)` — 1x2 composite.
- Wired `fig21_selection_function` into `generate_figures`, mirroring `_gen_pdet_surface`'s data resolution (project-root `simulations/injections/` then `<output_dir>/injections/`, else log + return None).
- 6 tests: smoke / monotone-survival / sorted-centers / horizon-contour-present / double-preset-size / graceful-no-horizon.

### Task 2 — surface fig02 latent color_by (commit 9257269)
- `_gen_event_posteriors` now passes `color_by="snr"` + the CRB SNR array into `plot_event_posteriors` **only when** the SNR column length matches the per-event posterior count; logs an info line and falls back to monochrome on a mismatch (never guesses an alignment).
- No change to `plot_event_posteriors`' body — caller wiring + one stronger content test (`test_plot_event_posteriors_color_by_adds_colorbar`) asserting exactly one colorbar whose norm spans the SNR range.

### Task 3 — fig22 population constraint-provenance view (commit 30bce20)
New module `population_plots.py`:
- `plot_population_constraint_view(h_values, event_posteriors, combined_posterior, true_h, theta_s, phi_s, snr, redshift, *, color_by, snr_threshold)` — 2x2 composite.
- Wired `fig22_population_view` into `generate_figures` from preloaded `post_data` + `crb_df` + canonical loader, with an alignment guard (skip when event count != CRB row count; T-03-02).
- 5 tests: smoke / double-preset-size / color_by-colorbar / stacked-dominant-over-spaghetti / stacked-MAP-matches-canonical.

## Panel Layouts Chosen (composite orchestration)

### fig21 selection-function explainer — `get_figure(1, 2, preset="double")` (1x2)
| Cell | Panel | Source (reused) |
|------|-------|-----------------|
| LEFT | `p_det(d_L)` survival curve (VARIANT_NO_MASS line, 0.5 REFERENCE-gray guide, Gpc x-label) | new `_pdet_survival_curve` helper (histogram ratio) |
| RIGHT | 2D `p_det(d_L, M_z)` heatmap + 0.5/0.9 EDGE horizon contours | `evaluation_plots.plot_pdet_surface` (ax=, unchanged) |

### fig22 population view — `get_figure(preset="double")` SIZE -> `fig.clf()` -> `GridSpec(2,2)` (2x2)
| Cell | Panel | Source (reused) |
|------|-------|-----------------|
| TOP-LEFT (driver) | SNR x z scatter, colored by the shared per-event scalar + SNR-threshold reference | `ax.scatter` (new orchestration) |
| TOP-RIGHT (sky) | Mollweide sky map (projection on this cell only) | `sky_plots.plot_sky_localization_mollweide` (ax=) |
| BOTTOM-LEFT (spaghetti) | de-emphasized per-event posteriors (alpha 0.5 / lw 0.5), colored by scalar | `bayesian_plots.plot_event_posteriors` color_by (ax=) |
| BOTTOM-RIGHT (stacked) | canonical combined posterior, hero line lw 2.0 (> spaghetti), MAP anchored to canonical | `bayesian_plots.plot_combined_posterior` (ax=) |

**color_by scalar:** SNR by default for both new figures (fig22 driver + spaghetti share SNR as the visual key); `color_by="redshift"` is selectable on fig22.

## FLAG FOR USER REVIEW (best-effort reading of proposal §5.3 / §5.4 — confirm before Phase 5)

These are NEW figures, so the **panel layout / composition of both composites is a best-effort interpretation** of the viz-redesign proposal:
- **fig21 (proposal §5.3):** I read the selection explainer as a 1x2 "survival curve + heatmap". Open questions for you: (a) is a 1x2 the layout you want, or a single overlaid panel? (b) should the survival curve be `p_det(d_L)` only, or also a `p_det(M_z)` marginal as a third panel?
- **fig22 (proposal §5.4):** I read the population view as a 2x2 "driver / sky / spaghetti / stacked". Open questions: (a) is 2x2 the arrangement you want, or a 1x4 strip? (b) should `color_by` default to SNR or redshift? (c) is the SNR x z driver the right "driver" view, or would M_z x z (or SNR x M_z) be more informative?

**Low risk:** every SUB-PANEL is a reuse of an existing well-tested single-panel encoding (fig20 heatmap, fig05 Mollweide, fig02 spaghetti, the canonical combined-posterior factory). Only the COMPOSITION (panel count / arrangement / which scalar drives color) is novel — so changing the layout in a follow-up is cheap and contained.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test bug] Composite dict-return flattening in test_population_plots.py**
- **Found during:** Task 3 GREEN phase.
- **Issue:** The factory returns axes as a `dict` (keyed driver/sky/spaghetti/stacked). The first draft of the test flattened with `np.asarray(dict, dtype=object).ravel()`, which yields a 0-d object array containing the dict (1 element), not the four Axes.
- **Fix:** Added a `_axes_list` test helper that branches on `isinstance(axes, dict)` and extracts `.values()`.
- **Files modified:** master_thesis_code_test/plotting/test_population_plots.py
- **Commit:** 30bce20

### Plan-faithful notes (not deviations)
- `_labels.py` (listed in plan `files_modified`) was NOT modified — every needed symbol already existed in `LABELS`; adding nothing is correct per the plan's "Add a LABELS entry only if a new symbol is needed".

## Deferred Issues (out of scope — pre-existing)

- **DI-03-01:** `astropy.stats.binom_conf_interval` raises "n must be positive" on the fig09 detection-efficiency path (`convergence_plots.py:332`) when a bin has zero injections, against `results/figures_seed200`. PRE-EXISTING (reproduces on `main` without any Phase-3 change), unrelated to fig21/fig22. Logged to `.planning/phases/03-new-static-figures/deferred-items.md`; surfaces as the "1 failed" in the manifest summary. Not fixed (Phase 3 is plotting-only for the two new figures).

## Verification

- **Check gate:** ruff check PASS (all source+tests); ruff format PASS (121 files clean); mypy PASS (123 source files); pytest `-m "not gpu and not slow"` = **630 passed, 6 skipped**, coverage 62.58% (>25% gate).
- **New tests:** 11 added (6 fig21 + 5 fig22) + 1 fig02 content assertion = 12 new tests, all green.
- **Render-verify (no traceback / graceful skip):**
  - `--generate_figures results/figures_seed200` -> fig21 + fig22 skip cleanly ("no injection campaign CSVs" / "required data not found") because seed200 has no injections and no posteriors. CORRECT graceful behavior.
  - fig21 factory rendered against REAL injection data (`simulations/archive/injections_partial_mar31_262files/`) -> valid 24 kB vector PDF, double-preset size, 2 axes.
  - fig22 factory rendered against REAL baseline posteriors (`simulations/_archive_v2_1_baseline/posteriors`, 38 h-values) with aligned synthetic sky/SNR -> valid 492 kB vector PDF, all four panels.
  - fig22 manifest path against the archive baseline correctly **skips** (417 events != 42 CRB rows) via the alignment guard — T-03-02 verified. In the current data layout no single directory has aligned CRB + per-event posteriors, so the manifest-route PDF awaits a future aligned dataset; the render path itself is verified directly.
- **No data-dir files committed:** all three commits stage only source/test files via explicit `git add`; no figure PDFs, no `.planning/debug/*` (other session's dirty files left untouched).

## Self-Check: PASSED

- FOUND: master_thesis_code/plotting/selection_plots.py
- FOUND: master_thesis_code/plotting/population_plots.py
- FOUND: master_thesis_code_test/plotting/test_selection_plots.py
- FOUND: master_thesis_code_test/plotting/test_population_plots.py
- FOUND commit 13d200b (fig21), 9257269 (fig02 color_by), 30bce20 (fig22)
- fig21_selection_function + fig22_population_view both present in main.py manifest (2 entries)
