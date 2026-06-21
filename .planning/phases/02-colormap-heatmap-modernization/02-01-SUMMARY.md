---
phase: 02-colormap-heatmap-modernization
plan: 01
subsystem: ui
tags: [matplotlib, colormap, cividis, lognorm, twoslopenorm, pcolormesh, plotting]

# Dependency graph
requires:
  - phase: 01-posterior-factory-consolidation
    provides: consolidated plotting factories + (fig,ax) contract this phase recolors/renorms
provides:
  - "cividis house sequential cmap (CMAP) package-wide; mplstyle default flipped"
  - "NO_DATA gray + DIVERGING_CMAP (RdBu_r) constants in _colors"
  - "diverging_norm(vcenter=0.73) + make_heatmap_norm(robust/log) + credible_contour_levels(68/95) helpers"
  - "explicit norm + set_bad on every sequential heatmap in model_plots + evaluation_plots"
  - "fig20 pdet surface via pcolormesh with true log mass axis + Gpc label + 0.5/0.9 horizon contour"
  - "fig05 sky SNR scatter with explicit LogNorm/robust norm"
  - "reusable diverging bias-map recipe (documented; no new dependency)"
affects: [03-new-static-figures, 04-data-gated-figures, 05-annotation-decluttering, 06-interactive-plotly]

# Tech tracking
tech-stack:
  added: []  # NO new runtime dependency — matplotlib built-ins only (cividis, RdBu_r, LogNorm, TwoSlopeNorm)
  patterns:
    - "Every heatmap: cmap = plt.get_cmap(CMAP).copy(); cmap.set_bad(NO_DATA); pass explicit norm"
    - "LogNorm safety: mask NaN/<=0 before LogNorm (make_heatmap_norm) so log(0) never crashes"
    - "pcolormesh on length-(n+1) bin EDGES (true axes); contour on bin CENTERS"
    - "Diverging bias maps: DIVERGING_CMAP + diverging_norm(0.73) + redundant vcenter iso-line"

key-files:
  created: []
  modified:
    - master_thesis_code/plotting/_colors.py
    - master_thesis_code/plotting/emri_thesis.mplstyle
    - master_thesis_code/plotting/_helpers.py
    - master_thesis_code/plotting/model_plots.py
    - master_thesis_code/plotting/evaluation_plots.py
    - master_thesis_code/plotting/sky_plots.py
    - master_thesis_code/plotting/fisher_plots.py
    - master_thesis_code/plotting/paper_figures.py
    - master_thesis_code_test/plotting/test_colors.py
    - master_thesis_code_test/plotting/test_helpers.py
    - master_thesis_code_test/plotting/test_model_plots.py
    - master_thesis_code_test/plotting/test_evaluation_plots.py
    - master_thesis_code_test/plotting/test_sky_plots.py
    - master_thesis_code_test/plotting/test_style.py

key-decisions:
  - "CMAP = cividis (perceptually uniform + deuteranopia-safe redesign of viridis, Nuñez et al. 2018)"
  - "NO_DATA = #D9D9D9 set_bad gray on every heatmap; empty/NaN bins read as no-data, not blank"
  - "DIVERGING_CMAP = RdBu_r (built-in, NO new dep); cmcrameri-vik documented as future upgrade"
  - "fig20 d_L stays Gpc (CSV luminosity_distance ~0.4..11 Gpc; c/H0~4.1 Gpc) — distinct from LABELS['d_L'] (Mpc)"
  - "fig05 stays a Mollweide scatter; the washed-out range was a normalization bug, fixed by an explicit norm (not a projection change)"
  - "pdet horizon contour drawn only for attainable levels — never fabricate a 0.5 line the data does not contain"

patterns-established:
  - "make_heatmap_norm: robust percentile clip vs LogNorm; one entry point for every heatmap norm"
  - "credible_contour_levels: 2D analogue of compute_hdi_interval for future 2D-posterior overlays"

requirements-completed: [VR-CMAP-01, VR-CMAP-02, VR-CMAP-03, VR-CMAP-04, VR-CMAP-05]

# Metrics
duration: ~70min
completed: 2026-06-21
---

# Phase 2 Plan 01: Colormap & Heatmap Modernization Summary

**Migrated the plotting package to a cividis house cmap with an explicit norm + `set_bad` no-data gray on every heatmap, rebuilt the pdet surface as a true-log `pcolormesh` with a detection-horizon contour, fixed the sky-map SNR norm, and established a reusable diverging convention — all matplotlib built-ins, no new dependency, no physics change.**

## Performance

- **Duration:** ~70 min
- **Tasks:** 4/4 (all `type=auto tdd=true`)
- **Files modified:** 14 (8 source, 6 test)
- **Commits:** 4 atomic task commits, all gate-green

## Accomplishments

### Task 1 — cividis + constants + helpers + route hardcoded cmaps (`6192594`)
- `_colors.CMAP` `viridis` -> `cividis`; mplstyle `image.cmap` -> `cividis`; rcparams snapshot updated.
- Added `NO_DATA = "#D9D9D9"` and `DIVERGING_CMAP = "RdBu_r"` (built-in, cmcrameri-vik upgrade path documented, NOT added).
- Added `diverging_norm(vcenter=0.73)` (clamps/nudges vcenter inside (vmin,vmax) so `TwoSlopeNorm` never raises) and `make_heatmap_norm(mode="robust"|"log")` (masks NaN/<=0 before LogNorm; robust = finite-percentile clip).
- **fisher_plots:604 `"plasma"` -> `CMAP` (D-CMAP-04)** and **paper_figures:675 `"viridis"` -> `CMAP` (D-CMAP-05)** — both sequential scatters now share cividis; presentation-only recolor, no value change.

### Task 2 — explicit norm + set_bad on every sequential heatmap (`487e733`)
- model_plots: detection heatmap pdet -> explicit `Normalize(0,1)` + masked NaN (0.5 horizon stays linear; LogNorm would distort it); emri_distribution -> robust norm; emri_sampling counts -> LogNorm. All via `plt.get_cmap(CMAP).copy()` + `cmap.set_bad(NO_DATA)`.
- evaluation_plots: CRB covariance imshow -> robust norm (orders-of-magnitude spread) + set_bad; detection_contour hist2d -> LogNorm + set_bad; sky_localization_3d scatter -> explicit robust norm (norm-consistency only; 3D->2D replacement is Phase 5).
- NaN/empty bins now render as NO_DATA gray; colorbars preserved; signatures + (fig,ax) contract unchanged.

### Task 3 — fig20 pcolormesh + horizon contour; fig05 SNR norm (`78b60e6`)
- **fig20**: replaced the fake-index `imshow` + hand-formatted ticks with `pcolormesh` on the real length-(n+1) (d_L, M) bin EDGES + `set_yscale("log")` -> a true log mass y-axis. Explicit `Normalize(0,1)` + masked NaN -> `set_bad(NO_DATA)`. P_det = 0.5/0.9 detection-horizon contour on bin CENTERS in EDGE color (only attainable levels; guarded for all-NaN / too-few-points; faithfully skipped when the data never reaches 0.5).
- **fig05**: explicit SNR norm (LogNorm when all SNR>0, else robust clip) so the near-threshold cluster no longer washes out the dynamic range; stays a Mollweide scatter.

### Task 4 — credible-contour helper + diverging recipe + audit closeout (`512f66f`)
- Added `credible_contour_levels(density, levels=(0.68,0.95))` — 2D analogue of `compute_hdi_interval`; returns iso-density levels enclosing the requested HDR mass (68% level encloses ~0.68 on a clean 2D Gaussian, unit-tested).
- `diverging_norm` docstring carries the canonical bias-map recipe + cmcrameri-vik upgrade note.
- VR-CMAP-01 audit sweep verified clean.

## VR-CMAP-01 closeout — call-site audit sweep

`grep -rEn 'cmap="(viridis|plasma)"' master_thesis_code/plotting/` -> **CLEAN (no matches).**

Full `cmap=` inventory across `master_thesis_code/plotting/` — every sequential heatmap routes through an approved name:

| Module | cmap routing |
|--------|--------------|
| model_plots.py (x3 heatmaps) | `plt.get_cmap(CMAP).copy()` |
| evaluation_plots.py (CRB, detection_contour, fig20) | `plt.get_cmap(CMAP).copy()` |
| evaluation_plots.py (sky_localization_3d) | `cmap=CMAP` |
| sky_plots.py (fig05) | `cmap=CMAP, norm=snr_norm` |
| fisher_plots.py:604 | `cmap=CMAP` (was `"plasma"`) |
| paper_figures.py:675 | `cmap=CMAP` (was `"viridis"`) |
| bayesian_plots.py (color_by spaghetti) | `plt.get_cmap(CMAP)` (inherits cividis) |
| _colors.py | `colormaps["Blues"]` -> `SEQUENTIAL_BLUES` (documented exception) |
| _helpers.py (diverging recipe docstring) | `plt.get_cmap(DIVERGING_CMAP).copy()` (documented diverging exception) |

Only string-literal colormap name remaining anywhere is `"Blues"` (the documented `SEQUENTIAL_BLUES`). The only non-CMAP cmaps are `DIVERGING_CMAP` and `SEQUENTIAL_BLUES`, both documented.

## Decision records

### fisher_plots:604 / paper_figures:675 cmap routing
Both were orphan hardcoded sequential cmaps (`"plasma"` for a Fisher conditioning-number scatter; `"viridis"` for a paper SNR-vs-redshift scatter). Both routed to the house `CMAP` (cividis) per D-CMAP-04/05. Presentation-only recolor — the `c=` data and norms are unchanged, so no computed value moves.

### fig20 Mpc/Gpc unit resolution
The injection-CSV `luminosity_distance` values span ~0.4..10.9 for z in [0.087, 1.499]. With `c/H0 ≈ 4.1 Gpc` (H0=73), z~1.5 gives d_L ~ 11 Gpc, so the CSV values are in **Gpc**. The existing `$d_L\,[\mathrm{Gpc}]$` label is therefore **correct for fig20** — the "Mpc/Gpc bug" is that fig20 must NOT route through `LABELS["d_L"]` (which is `[\mathrm{Mpc}]`, used by the per-source recovery plots whose data is in Mpc). fig20 keeps its own Gpc label with a documenting comment; no relabel needed.

### fig05 scatter-vs-pcolormesh
Kept a Mollweide **scatter** (D-CMAP-06). The washed-out dynamic range was a *normalization* failure (default linear autoscale-from-zero with a dense near-threshold cluster), not a primitive problem — fixed by an explicit LogNorm. A linear `pcolormesh` does not map trivially onto the curved Mollweide grid, and a scatter is the right primitive for discrete sources. Documented inline.

### VR-CMAP-04 — diverging convention (fallback record)
**No pre-existing bias/residual heatmap exists** in the package (confirmed: zero `TwoSlopeNorm` usages outside `_helpers`; `cmcrameri` not installed). The convention is therefore satisfied by the reusable pieces + test + doc note: `DIVERGING_CMAP` (RdBu_r, built-in, no new dep) + `diverging_norm(vcenter=0.73)` + the documented bias-map recipe in the `diverging_norm` docstring. **cmcrameri-vik upgrade path documented** (one-line `uv add cmcrameri` + swap `DIVERGING_CMAP = "cmc.vik"`) — NOT added this phase.

### VR-CMAP-05 — 2D-posterior host check
Two halves:
- **P_det = 0.5/0.9 horizon contour** — landed on the pdet maps (Task 3 fig20 + the model_plots detection heatmaps already carry [0.5, 0.9] contour levels).
- **68/95% credible contours** — `credible_contour_levels` helper provided + unit-tested. Grep confirms **no 2D-posterior DENSITY heatmap exists today** (no `imshow`/`pcolormesh`/`contourf`/`contour` on a 2D posterior grid; the combined-posterior factory is 1D line plots, `plot_fisher_corner` draws Fisher ellipses not a binned 2D posterior). The helper is therefore **recorded for the Phase-5 consumer** that builds the 2D-posterior heatmap.

## Render-verify evidence

- **fig20** (`plot_pdet_surface` on real `injections_partial_mar31_262files/injection_h_0p73_task_*.csv`, threshold=20): `ax.get_yscale() == "log"`, QuadMesh with `Normalize(0,1)` + `set_bad == NO_DATA`, Gpc xlabel, no traceback. Horizon contour: this archive slice's max P_det is 0.37 (never reaches 0.5) so the 0.5 line is faithfully skipped; on synthetic data crossing 0.5 the contour IS drawn (1 contour collection) — verified.
- **fig05** (`plot_sky_localization_mollweide`): SNR scatter norm is `LogNorm` (vmin/vmax set from data), stays a Mollweide `PathCollection` scatter, ellipse path intact, no traceback.
- **CRB heatmap / detection heatmap**: render with explicit norm + `set_bad == NO_DATA`, no traceback.
- **PDF vector contract**: regenerated fig05 (39 KB) and fig20 (23 KB) PDFs both start with `%PDF-` and are vector (sky markers rasterized by design, axes/colorbar vector).
- **Archive figure-gen path** (`--generate_figures simulations/_archive_v2_1_baseline`): identical before/after — "21 generated, 2 skipped, 3 failed"; the recolor/renorm introduced **zero** new failures.

## Deviations from Plan

### 1. [Rule 1 - Bug] rcparams snapshot pinned viridis
- **Found during:** Task 1 (full pytest gate)
- **Issue:** `test_style.py::test_rcparams_snapshot` pins `image.cmap` (was `"viridis"`); flipping the mplstyle default broke it.
- **Fix:** Updated the snapshot to `"cividis"` — exactly the intentional-update the test's own docstring mandates. Committed in Task 1.

### 2. [Rule 3 - Blocking] pre-commit ruff UP038 stricter than bare `ruff check`
- **Found during:** Task 2 (first commit attempt blocked)
- **Issue:** `isinstance(coll, (QuadMesh, ScalarMappable))` in new test helpers tripped pre-commit's ruff UP038 ("use `X | Y`"), though a bare `uv run ruff check` passed.
- **Fix:** Simplified to `isinstance(coll, ScalarMappable)` (QuadMesh is a ScalarMappable subclass). Adopted `uv run pre-commit run ruff --files ...` as the per-task pre-commit gate thereafter.

### 3. [Scope boundary] pdet horizon contour cannot be fabricated on real data
- **Found during:** Task 3 render-verify
- **Issue:** The plan's verify command expects ">=1 contour collection" on the real h_0p73 data, but that archive slice's max P_det is 0.37 — the 0.5 horizon genuinely does not exist there.
- **Resolution:** Draw the contour only for attainable levels (within the data range); never fabricate a 0.5 line absent from the data (scientific honesty on a presentation phase). The unit test asserts the contour appears when the horizon IS present (synthetic data crossing 0.5), satisfying VR-CMAP-05 faithfully.

### 4. [Out of scope] pre-existing failures not touched
- `pre-commit run --all-files` surfaced pre-existing ruff errors in `scripts/quick_snr_calibration.py` (F401, N818) and pre-existing data-gated render failures (fig09, paper_single_event, paper_convergence) + skips (fig16, fig20 in the empty `simulations/injections/` glob) in the archive dir. All independent of colormaps. Logged to `deferred-items.md`; not fixed (scope boundary).

## Known Stubs

None. Every heatmap is wired to real data norms; no placeholder/empty-data flows introduced.

## Threat Flags

None. All edits are colormap / norm / axis-presentation only — no new network endpoint, auth path, file-access pattern, or schema change. The T-02-01..04 mitigations in the plan's threat register are implemented (LogNorm zero/NaN masking, pcolormesh edges-vs-centers, TwoSlopeNorm vcenter clamp) and unit-tested.

## Self-Check: PASSED

- SUMMARY.md: FOUND
- All 5 spot-checked source files: FOUND (_colors, _helpers, model_plots, evaluation_plots, sky_plots)
- All 4 task commits: FOUND (6192594, 487e733, 78b60e6, 512f66f)
- No file deletions in any task commit
- No data-dir outputs committed (archive figures/logs gitignored)
