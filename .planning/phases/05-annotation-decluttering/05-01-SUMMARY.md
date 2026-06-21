---
phase: 05-annotation-decluttering
plan: 01
subsystem: plotting
tags: [viz, horizon, declutter, annotation, labels, corner, presentation-only]
requires:
  - get_figure / save_figure / make_colorbar / make_heatmap_norm (_helpers)
  - LABELS provider (_labels)
  - HORIZON palette (_colors: VARIANT_NO_MASS/WITH_MASS, TRUTH, EDGE, REFERENCE, PLANCK, SH0ES, CMAP)
  - corner 2.2.3 (already present; no new dependency)
provides:
  - LABELS["d_L_gpc"] (Gpc) alongside LABELS["d_L"] (Mpc)
  - plot_fisher_diagnostics returning (Figure, ndarray) — caller saves
  - 2D Mollweide replacement for plot_sky_localization_3d
  - 2D marginal-panel replacement for plot_cramer_rao_coverage
  - filled-contour HORIZON corner (plot_fisher_corner)
  - on-curve N^-1/2 scaling-law annotation; active declarative titles
affects:
  - master_thesis_code/bayesian_inference/bayesian_statistics.py (caller saves Fisher diagnostic)
  - master_thesis_code/main.py (fig14 axes-array return; fig01 title; fig16 layout)
tech-stack:
  added: []
  patterns:
    - "constrained_layout-only: no fig.tight_layout in factory bodies (corner-internal via rc_context excepted)"
    - "scaling-law / reference rules annotated AT the line, not in a legend box"
    - "unit-explicit distance labels (Mpc vs Gpc) via two LABELS keys"
key-files:
  created:
    - master_thesis_code_test/plotting/test_labels.py
    - .planning/phases/05-annotation-decluttering/deferred-items.md
  modified:
    - master_thesis_code/plotting/convergence_plots.py
    - master_thesis_code/plotting/convergence_analysis.py
    - master_thesis_code/plotting/paper_figures.py
    - master_thesis_code/plotting/evaluation_plots.py
    - master_thesis_code/plotting/simulation_plots.py
    - master_thesis_code/plotting/fisher_plots.py
    - master_thesis_code/plotting/single_event_detail.py
    - master_thesis_code/plotting/selection_plots.py
    - master_thesis_code/plotting/catalog_plots.py
    - master_thesis_code/plotting/_labels.py
    - master_thesis_code/main.py
    - master_thesis_code/bayesian_inference/bayesian_statistics.py
    - master_thesis_code_test/plotting/test_fisher_plots.py
    - master_thesis_code_test/plotting/test_simulation_plots.py
    - master_thesis_code_test/plotting/test_evaluation_plots.py
decisions:
  - "Mpc/Gpc reconciled by ADDING d_L_gpc key (not converting data) — both units kept, always stated"
  - "corner modernized via corner 2.2.3 own kwargs only — no arviz/chainconsumer"
  - "drop edgecolor from corner hist_kwargs once smooth1d set (Line2D rejects it)"
  - "two-variant convergence legends kept (2 entries each); only the scaling law converted to a direct on-curve label"
metrics:
  tasks_completed: 5
  commits: 5
  files_changed: 16
  completed: 2026-06-21
---

# Phase 5 Plan 01: Annotation & Decluttering Rollout Summary

One-liner: package-wide HORIZON/Dispatch presentation sweep — stripped every
layout-fighting `tight_layout`, dropped smooth-curve marker speckle, restored the
`(fig, ax)` factory contract (incl. two 3D→2D rewrites), reconciled Mpc/Gpc axis
labels through a single `LABELS` provider, modernized the Fisher corner to filled
navy contours, and installed active titles + on-curve N^-1/2 scaling-law labels.

## Tasks shipped (atomic commits)

| # | SHA | Subject |
|---|-----|---------|
| 1 | 43b559a | viz(declutter): strip layout-fighting tight_layout + drop smooth-curve markers (VR-ANNO-02/03) |
| 2 | 958b6dc | viz(declutter): restore (fig,ax) factory contract + replace 3D scatters with 2D views (VR-ANNO-04/05) |
| 3 | ba56a68 | viz(annotate): route Gpc axis labels through LABELS + reconcile Mpc/Gpc (VR-ANNO-06) |
| 4 | 64a5dfb | viz(corner): modernize Fisher corner with filled HORIZON contours (VR-ANNO-07) |
| 5 | 8d5ab0d | viz(annotate): active titles + on-curve N^-1/2 scaling-law labels (VR-ANNO-01) |

## tight_layout sites removed (VR-ANNO-03) — grep gate = ZERO

All 11 surveyed `fig.tight_layout(...)` calls removed:
`convergence_plots.py:280`, `convergence_analysis.py:955`, `paper_figures.py:381 (h_pad=0.3)`,
`paper_figures.py:592`, `paper_figures.py:699 (w_pad=1.0)`, `paper_figures.py:917`,
`evaluation_plots.py:408`, `fisher_plots.py:153`, `fisher_plots.py:620 (in diagnostics)`,
`single_event_detail.py:378 (rect=...)`, `main.py:1315 (fig16)`.

`! grep -rn "fig\.tight_layout(" master_thesis_code/plotting/ master_thesis_code/main.py`
→ **zero hits.** (The corner-internal layout is handled by the retained
`rc_context({"figure.constrained_layout.use": False})` wrapper, which is not a
`fig.tight_layout(` token; the explanatory comment was reworded so it does not
trip the naive grep gate.) No panel needed `set_constrained_layout_pads`; regen
showed acceptable spacing everywhere (suptitle in single_event_detail packs fine
under constrained_layout). Per-point markers dropped on the smooth/near-delta
curves (convergence_plots o-/s--, convergence_analysis ×4, paper_figures o-/s-),
preserving the navy-solid / gold-dashed variant linestyle law; discrete-scatter
markers (model/catalog plots, errorbar fmt, flagged-event scatter) kept.

## The two 3D→2D replacements (VR-ANNO-05)

- **plot_sky_localization_3d** (evaluation_plots): `projection="3d"` scatter →
  **2D Mollweide** (lat/lon transform matching sky_plots, cividis + robust norm,
  `make_colorbar(..., label="Sky-localization error [sr]")`). Function name and
  `(Figure, Any)` return kept; ax is now a `mollweide` axes (verified, not Axes3D).
- **plot_cramer_rao_coverage** (simulation_plots): `scatter3D` → **three 2D
  marginal panels** (M–qS, M–phiS, qS–phiS) via `get_figure(1, 3, preset="double")`,
  EDGE points (s=9, alpha=0.5), axis labels via `LABELS["M"/"qS"/"phiS"]`, per-axis
  limits honored. Returns `(Figure, axes-array)`; verified all three panels are
  rectilinear (no Axes3D). main.py fig14 wiring unchanged (the `_save` helper uses
  only `fig`).

Also: `plot_fisher_diagnostics` now returns `(Figure, ndarray)` and no longer
saves internally; the bayesian_statistics caller captures `(fig, _axes)` and calls
`save_figure(fig, "simulations/fisher_quality_diagnostic")`. `output_dir` retained
for call-site compatibility (now unused). `plot_injected_vs_recovered` figure
creation routed through `get_figure(figsize=...)`. Dropped now-unused `plt`/`os`/
`save_figure` imports in the touched factories.

## Corner modernization (VR-ANNO-07) — final recipe

`plot_fisher_corner` now passes to `corner.corner` (corner 2.2.3, no new dep):

```
fill_contours=True, plot_datapoints=False, plot_density=False,
smooth=1.0, smooth1d=1.0, levels=(0.393, 0.865, 0.989),
contour_kwargs={"colors": EDGE, "linewidths": 0.8},
contourf_kwargs={"alpha": 0.7},
color=VARIANT_NO_MASS (navy), truths=sub_mean, truth_color=TRUTH (vermillion),
quantiles=[0.16,0.5,0.84], show_titles=True, title_fmt=".3f",
hist_kwargs={"color": VARIANT_NO_MASS}
```

Overlay events: first overlay HORIZON gold (`VARIANT_WITH_MASS`), remaining fall
back to `CYCLE`, all as filled families with the same recipe. The
`rc_context(constrained_layout=False)` wrapper is intentionally retained and
documented as the accepted handling for corner's self-managed figure (NOT a stray
tight_layout). Render verified: filled navy 1/2/3-sigma contours with near-black
edge lines, navy 1D marginals, vermillion truth crosshairs, no datapoint speckle.

## LABELS / Mpc–Gpc reconciliation (VR-ANNO-06)

Added `LABELS["d_L_gpc"] = r"$d_L \, [\mathrm{Gpc}]$"` next to `LABELS["d_L"]`
(Mpc); documented both in the module docstring. **No data value converted** — only
the labeling is made unit-explicit. Replaced the six surveyed hardcoded Gpc
literals with `LABELS["d_L_gpc"]`: fisher_plots scatter, paper_figures closure
scatter, evaluation_plots fig20 P_det surface, selection_plots P_det horizon,
catalog_plots completeness (×2). Fisher diagnostic scatter y-label →`LABELS["M"]`.
test_labels asserts d_L→Mpc / d_L_gpc→Gpc and no residual `[Gpc]` axis-label
literal in the six factory files.

## Taste choices for user confirmation

### Active declarative titles (figure → wording)
- **fig01 (main.py) + paper h0_posterior comparison + paper h0_posterior KDE:**
  "Adding the redshifted BH mass tightens the $H_0$ posterior"
- **convergence_analysis ax_w panel:** "Posterior $h$ uncertainty shrinks as $N^{-1/2}$"
- **paper convergence (plot_convergence_improvement):**
  "$H_0$ uncertainty shrinks as $N^{-1/2}$ with the event count"
- **closure test (paper_figures):** existing "Closure test: pipeline recovers each
  injection truth" — kept (already an active, finding-stating title).

### Named reference-rule labels and placement
- **Planck / SH0ES rules** on the H0 posterior figures are ALREADY labeled at the
  line ("Planck", "SH0ES" text at the rule, reserved PLANCK/SH0ES colors) by the
  canonical `plot_combined_posterior` factory (bayesian_plots.py:220/241). No new
  labeling needed; fig01/paper posteriors inherit it. **No new reference lines
  invented.**
- **Closure-test per-run truth axvlines:** kept color-matched to each posterior
  curve + legend-labeled by `h_true`. Direct text labels at each rule were SKIPPED
  (N overlapping vertical truths would collide) — taste choice, flagged.

### Scaling-law (N^-1/2) — legend → direct on-curve label
Converted in all three convergence factories: reference drawn `_nolegend_` and a
direct `ax.annotate(r"$\propto N^{-1/2}$", xy=(last point), xytext=(4,0) offset)`
in REFERENCE gray at the end of the curve. **Wording unified to `$\propto N^{-1/2}$`
across convergence_plots, convergence_analysis, and paper_figures** (previously
`$1/\sqrt{N}$ ref`, `$\propto N^{-1/2}$`, `$\propto N_\mathrm{det}^{-1/2}$`).
convergence_plots reference recolored CYCLE[5] → REFERENCE gray for consistency.

### Other legend → direct-label conversions
- **Two-variant convergence panels (Without/With $M_z$):** legends KEPT (each now
  has only 2 entries after the scaling law moved on-curve; the dense log-log
  small-multiples have no clean right-margin for end-of-line labels). Taste choice
  — the navy/gold + solid/dashed redundant encoding already disambiguates; flag for
  user if end-of-line labels are preferred.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1/3 - Blocking] corner hist_kwargs edgecolor rejected once smooth1d set**
- **Found during:** Task 4 (corner modernization).
- **Issue:** With `smooth1d=1.0`, corner draws the 1D marginal as a `Line2D`
  (not a stepped histogram patch), so `hist_kwargs={"edgecolor": EDGE}` raised
  `AttributeError: Line2D.set() got an unexpected keyword argument 'edgecolor'`,
  failing all 4 corner smoke tests.
- **Fix:** dropped `edgecolor` from `hist_kwargs` (kept `color`); the 2D-panel
  edge readability (grayscale/CB redundant channel) comes from
  `contour_kwargs={"colors": EDGE}` instead.
- **Files:** master_thesis_code/plotting/fisher_plots.py
- **Commit:** 64a5dfb

**2. [Rule for grep gate] reworded corner-layout comment to keep gate at zero**
- The Task-4 explanatory comment originally contained the literal `fig.tight_layout()`,
  which tripped the VR-ANNO-03 grep gate. Reworded to "the tight-layout pass" so
  the gate (`grep "fig\.tight_layout("`) stays zero. Commit 8d5ab0d.

### Deferred (out of scope) — see deferred-items.md
**DEFER-05-01:** `fig09_detection_efficiency` raises `ValueError: n must be positive`
from `astropy.stats.binom_conf_interval` when a redshift bin has zero injections.
Pre-existing (line introduced commit 7dc1421f, 2026-04-02); the
`plot_detection_efficiency` function was NOT touched by any Phase-5 task. Logged,
not fixed (scope boundary). Surfaces during figure regen on both data dirs.

## Verification

- **grep gates:** `fig.tight_layout(` → ZERO; `projection="3d"|scatter3D|plt.figure(figsize`
  in evaluation_plots/simulation_plots → ZERO.
- **LABELS:** d_L→Mpc, d_L_gpc→Gpc; no hardcoded [Gpc] axis-label literal in the six factories.
- **Full check gate (GREEN):** `ruff check` clean (source + tests); `mypy` clean
  (128 source files); `pytest -m "not gpu and not slow"` → **655 passed, 6 skipped,
  15 deselected**, coverage 63.95% (≥25% threshold).
- **Figure regen (no traceback in touched code):** against
  `simulations/_archive_v2_1_baseline` and `results/figures_seed200` — 14 generated,
  rest skipped for missing data; ONLY pre-existing fig09 failure (DEFER-05-01).
- **Vector + ≥7pt:** regenerated PDFs carry Tj/TJ text-show operators (fig07 corner
  273, fig14 CRB 72, fig05 sky 46, paper_snr 4) — text is selectable/vector, not
  rasterized; fonts subset-embedded (font-subsetting log confirms). REVTeX presets
  unchanged (single 3.375", double 7.0"); annotation/title fontsizes ≥7pt.
- **Visual taste read:** corner = filled navy contours + vermillion truth, no speckle;
  CRB coverage = three clean 2D marginal panels; sky-loc = 2D Mollweide; convergence
  law reads at the curve. (Note: corner 1D marginals render stepped despite smooth1d;
  acceptable — flag if smooth 1D curves preferred.)

## Self-Check: PASSED
- created files exist: test_labels.py, deferred-items.md, this SUMMARY.
- all 5 commits present (43b559a, 958b6dc, ba56a68, 64a5dfb, 8d5ab0d).
