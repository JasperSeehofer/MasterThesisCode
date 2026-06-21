---
phase: 01-posterior-factory-consolidation
plan: 01
subsystem: plotting
tags: [viz, refactor, consolidation, posterior, h0]
requires:
  - "bayesian_plots.plot_combined_posterior (canonical factory, pre-existing)"
  - "_helpers.load_canonical_combined_posterior / compute_hdi_interval / get_figure"
  - "paper_figures._kde_smooth_posterior / _load_combined_posterior"
provides:
  - "Single canonical plot_combined_posterior factory (kde + linestyle + truth-style + xlim/ylim/ylabel/legend superset)"
  - "All five combined-H0-posterior render paths delegate to one factory"
  - "Cross-path rendered-MAP regression (TestRenderedMapAgreesAcrossFigurePaths)"
affects:
  - master_thesis_code/plotting/bayesian_plots.py
  - master_thesis_code/plotting/paper_figures.py
  - master_thesis_code/plotting/convergence_analysis.py
  - master_thesis_code/plotting/convergence_plots.py
  - master_thesis_code/main.py
tech-stack:
  added: []
  patterns:
    - "Lazy local import for cross-module factory calls (circular-import dodge)"
    - "Factory superset with backward-compatible keyword-only defaults"
    - "ax= injection so multi-panel layouts delegate without restructuring"
key-files:
  created: []
  modified:
    - master_thesis_code/plotting/bayesian_plots.py
    - master_thesis_code/plotting/paper_figures.py
    - master_thesis_code/plotting/convergence_analysis.py
    - master_thesis_code/plotting/convergence_plots.py
    - master_thesis_code/main.py
    - master_thesis_code_test/plotting/test_canonical_map_consistency.py
decisions:
  - "fig01 §1.3-vs-VR-F3: normalize=density (area-norm PDF) + show_credible=False (no band under multi-variant overlay) — the two requirements pull on different switches, satisfied simultaneously"
  - "Removed dead _shade_nested_hdi / _annotate_inline_map (no remaining callers after delegation)"
  - "Theme passthrough governed by apply_style(); no new CLI theme flag this phase"
metrics:
  duration: "~22 min"
  completed: "2026-06-21"
requirements: [VR-CONS-01, VR-CONS-02]
---

# Phase 01 Plan 01: Posterior Factory Consolidation Summary

Collapsed the five duplicate combined-H0-posterior render paths into ONE canonical
factory (`bayesian_plots.plot_combined_posterior`) so every recolor/annotation edit
lands in one place; the quadruplicate-drift hazard (different MAPs from copy-pasted
plotting code) is now regression-anchored at the rendered-curve level.

## What shipped (commits)

| Task | Commit | Subject |
|------|--------|---------|
| 1 | `b3ee7dc` | refactor(viz): extend plot_combined_posterior to canonical superset + pin cross-path MAP |
| 2 | `16ce824` | refactor(viz): delegate paper_h0_posterior + KDE to plot_combined_posterior |
| 3 | `83bec7e` | refactor(viz): route convergence panels through plot_combined_posterior |
| 4 | `1a43709` | viz(fig01): area-normalize the two-variant overlay (§1.3-vs-VR-F3 reconciliation) |

## (a) The §1.3-vs-VR-F3 fig01 reconciliation

The ROADMAP asks fig01 to use the VR-F3 area-normalized headline treatment; VR-F4 §1.3
forbids shading an HDI band under a many-variant overlay. These pull on the **band**, not
on the **normalization**, so both are satisfied at once:

- **`normalize="density"`** on BOTH fig01 calls — honors VR-F3 (honest, area-normalized PDF).
- **`show_credible=False`** on BOTH calls — honors §1.3 (no shaded HDI band under the
  two-variant overlay). The single-variant paper figures (`paper_h0_posterior`,
  `paper_h0_posterior_kde`) keep their nested 68/95% HDI bands.
- The headline (Without $M_z$) call keeps the inline MAP annotation + dashed truth line +
  Planck/SH0ES references; the secondary (With $M_z$) call suppresses truth/MAP/references.

## (b) Final canonical superset of plot_combined_posterior kwargs

Pre-existing: `label, normalize ("peak"|"density"), show_credible, show_references,
annotate_map, show_truth, color, ax`.

Added this phase (all keyword-only, backward-compatible defaults — existing callers unchanged):

| kwarg | default | purpose |
|-------|---------|---------|
| `linestyle` | `"-"` | secondary overlay variants pass `"--"` |
| `linewidth` | `None` | `None` → stylesheet default; KDE/panels pass `1.4` |
| `truth_linestyle` | `"dashed"` | paper figs pass `":"`; fig08 passes `"dashed"` |
| `truth_label` | `None` | `None` → `f"True $h = {true_h}$"`; callers pass `"Injected"`/`"Truth"` |
| `xlim` | `None` | paper figs + panel pass `(0.59, 0.87)` |
| `ylim` | `None` | panel passes `(-0.05, 1.15)` |
| `ylabel` | `None` | `None` → `p(h|data)`; fig08 passes `"Posterior density"` |
| `kde` | `False` | `True` smooths via `_kde_smooth_posterior` BEFORE normalize/HDI/MAP |
| `legend` | `True` | overlay callers pass `False` so the figure owns one legend |

`kde=True` imports `_kde_smooth_posterior` via a **lazy local import** inside the function
(paper_figures imports from convergence_analysis, and bayesian_plots must not import
paper_figures at module top — the local import dodges the circular dependency). The HDI and
MAP are computed on the smoothed curve so the rendered MAP equals what the panel shows.

## (c) _shade_nested_hdi / _annotate_inline_map: REMOVED

After delegation, both helpers had no remaining callers (`grep` confirmed: only
`plot_h0_posterior_comparison` + `plot_h0_posterior_kde` used them, and both now delegate).
Removed, along with the now-orphaned `compute_hdi_interval` import in `paper_figures.py`.
The factory does the shading + annotation centrally.

## (d) The five consolidated call sites (grep)

```
$ grep -rn "plot_combined_posterior(" master_thesis_code/ | grep -v "def plot_combined_posterior"
master_thesis_code/main.py:932                          # fig01 — Without M_z (headline)
master_thesis_code/main.py:949                          # fig01 — With M_z (secondary)
master_thesis_code/plotting/paper_figures.py:208        # paper_h0_posterior — Without M_z
master_thesis_code/plotting/paper_figures.py:228        # paper_h0_posterior — With M_z
master_thesis_code/plotting/paper_figures.py:801        # paper_h0_posterior_kde — Without M_z
master_thesis_code/plotting/paper_figures.py:823        # paper_h0_posterior_kde — With M_z
master_thesis_code/plotting/convergence_analysis.py:758 # M_z panel top-middle — Without M_z
master_thesis_code/plotting/convergence_analysis.py:777 # M_z panel top-middle — With M_z
master_thesis_code/plotting/convergence_plots.py:170    # fig08 left panel — primary
master_thesis_code/plotting/convergence_plots.py:218    # fig08 left panel — alt
master_thesis_code/plotting/dashboard_plots.py:76       # (pre-existing 6th consumer, bonus)
```

All five target paths delegate (2 calls each). `dashboard_plots.py:76` was already a delegate
before this phase — a bonus 6th consumer, not one of the five targets.

### fig08-left handling

`plot_h0_convergence` left panel (`ax_post`) routes BOTH the primary and alt posterior draws
through the factory with `normalize="density"` (the factory area-normalizes once; the raw
`combined`/`combined_alt` are passed so the rendered curve is identical to today). The right
CI-width panel (`ax_ci`), bootstrap HDI band, 1/sqrt(N) reference, and the panel's own
`"Truth"` `axvline` + legend are untouched. The legacy bootstrap-subset fallback (when no
`canonical_*` is supplied) is preserved.

The M_z improvement top-middle panel keeps its own `"Injected"` `axvline`, title, and
small-font legend; the factory's truth/legend/references are suppressed there. The legacy
`representative_posteriors_*` fallback branch is left as-is.

## (e) Theme-passthrough mechanism

Governed by `apply_style()` — the single style entrypoint in `generate_figures`
(main.py ~819-825), which sets `text.usetex`/rcParams once. The factory inherits the active
rcParams (it uses `get_figure(preset=...)` + the stylesheet and hardcodes no figsize/base
fontsize), so a theme "passes through" automatically. The intentional inline reference-band
labels (`fontsize=6`) and inline MAP text (`fontsize=7`) are deliberate sub-axis annotations
and were left as literals. **No new CLI theme flag was added this phase** (arguments.py
untouched — out of scope).

## Tests added/updated

`test_canonical_map_consistency.py` — new class `TestRenderedMapAgreesAcrossFigurePaths`
(5 tests + a `_line0_map` helper) pinning the rendered-curve line-0 MAP against the canonical
discrete MAP for all five paths:
- `test_combined_posterior_rendered_map_matches_canonical` (exact)
- `test_paper_comparison_rendered_map_matches_canonical` (< 0.012, one grid step)
- `test_paper_kde_rendered_map_matches_canonical` (< 0.02, KDE sub-grid tolerance)
- `test_convergence_panel_rendered_map_matches_canonical` (< 0.012)
- `test_fig08_left_rendered_map_matches_canonical` (< 0.012)

The pre-existing `TestCanonicalMapAgreesAcrossFigurePaths` (loader contract) and
`TestHeadlinePosteriorNormalization` (area≈1, ylabel-not-peak, ≥2 HDI fills) stay green.

## Verification

- **Check gate green:** `ruff check` clean; `mypy` clean (119 source files);
  `pytest -m "not gpu and not slow"` → **594 passed, 6 skipped, 15 deselected**.
- **Figure regen** against `simulations/_archive_v2_1_baseline`: 21 generated, 2 skipped,
  3 failed. The 3 failures (`fig09_detection_efficiency`, `paper_single_event`,
  `paper_convergence`) are the PRE-EXISTING stale-data-availability errors from the
  quick-wins SUMMARY — NONE of the five consolidated figures
  (`fig01_h0_posterior_combined`, `paper_h0_posterior`, `paper_h0_posterior_kde`,
  `fig08_h0_convergence`, `paper_m_z_improvement`) are among them.
- **Vector + fonts:** regenerated PDFs contain 0 raster images and 6 embedded fonts
  (vector); stylesheet `emri_thesis.mplstyle` unchanged, no rasterization introduced.
- **Loaders untouched:** `git diff` shows NO change to `_helpers.py`; loader bodies
  (`load_canonical_combined_posterior`, `compute_m_z_improvement_bank`,
  `_load_combined_posterior`) unmodified — only render-path bodies changed.
- **No physics file changed** — diff confined to plotting factories, two convergence files,
  main.py manifest, and the one test file. `/physics-change` NOT triggered (GSD software work).

## Deviations from Plan

None — plan executed exactly as written. The Task 4 convergence-panel + fig08-left render
tests called for in part (3) were already added in Task 1's `TestRenderedMapAgreesAcrossFigurePaths`
class (the plan placed the test class scaffold in Task 1 and the two panel tests in Task 4;
they were written together in Task 1 as one coherent class covering all five paths). The
`LABELS["h"]` value resolves to `$h$`, matching the paper figures' xlabel, so no xlabel
override was needed (as the Task 2 plan note anticipated).

## Self-Check: PASSED

- bayesian_plots.py / paper_figures.py / convergence_analysis.py / convergence_plots.py /
  main.py / test_canonical_map_consistency.py — all present and modified.
- Commits b3ee7dc, 16ce824, 83bec7e, 1a43709 — all present in `git log`.
