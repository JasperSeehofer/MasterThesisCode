# Roadmap: EMRI Parameter Estimation

## Milestones

- ✅ **v1.0 EMRI HPC Integration** — Phases 1-5 (shipped 2026-03-27)
- ✅ **v1.1 Clean Simulation Campaign** — Phases 6-8 (shipped 2026-03-29)
- ✅ **v1.2 Production Campaign & Physics Corrections** — Phases 9-13 (shipped 2026-04-01)
- 🚧 **v1.3 Visualization Overhaul** — Phases 14-19 (in progress)

## Phases

<details>
<summary>✅ v1.0 EMRI HPC Integration (Phases 1-5) — SHIPPED 2026-03-27</summary>

- [x] Phase 1: Code Hardening (2/2 plans) — CPU-safe imports, --use_gpu/--num_workers CLI flags
- [x] Phase 2: Batch Compatibility (1/1 plan) — Non-interactive merge/prepare scripts with entry points
- [x] Phase 3: Cluster Environment (1/1 plan) — modules.sh + setup.sh for bwUniCluster 3.0
- [x] Phase 4: SLURM Job Infrastructure (3/3 plans) — simulate/merge/evaluate pipeline with dependency chaining
- [x] Phase 5: Documentation (2/2 plans) — cluster/README.md quickstart, CLAUDE.md/README.md sections

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

<details>
<summary>✅ v1.1 Clean Simulation Campaign (Phases 6-8) — SHIPPED 2026-03-29</summary>

- [x] Phase 6: Data Cleanup (1/1 plan) — Removed stale artifacts, verified .gitignore
- [x] Phase 7: Cluster Access (1/1 plan) — SSH ControlMaster, environment preflight
- [x] Phase 8: Simulation Campaign (2/2 plans) — Smoke-test pipeline, validation, H0 posterior

Full details: `.planning/milestones/v1.1-ROADMAP.md`

</details>

<details>
<summary>✅ v1.2 Production Campaign & Physics Corrections (Phases 9-13) — SHIPPED 2026-04-01</summary>

- [x] Phase 9: Galactic Confusion Noise (1/1 plan) — Added galactic foreground to LISA PSD
- [x] Phase 10: Five-Point Stencil Derivatives (1/1 plan) — O(epsilon^4) Fisher matrix derivatives
- [x] Phase 11: Validation Campaign (2/2 plans) — Corrected physics validated at small scale
- [x] Phase 11.1: Simulation-Based Detection Probability (5/5 plans) — Replaced KDE P_det with simulation-based
- [x] Phase 12: Production Campaign (1/1 plan) — 100+ task campaign on bwUniCluster
- [x] Phase 13: H0 Posterior Sweep (1/1 plan) — Full H0 posterior over [0.6, 0.9]

Full details: `.planning/milestones/v1.2-ROADMAP.md`

</details>

### v1.3 Visualization Overhaul (In Progress)

**Milestone Goal:** Modernize the visualization stack to produce publication-quality, thesis-ready matplotlib figures with consistent styling, proper uncertainty visualization, and standard EMRI/LISA community plot types.

- [x] **Phase 14: Test Infrastructure & Safety Net** - Smoke tests and rcParams regression checks before any refactoring (completed 2026-04-01)
- [x] **Phase 15: Style Infrastructure** - Centralized colors, figure sizes, LaTeX toggle, and shared helpers (completed 2026-04-01)
- [x] **Phase 16: Data Layer & Fisher Visualizations** - Covariance reconstruction and Fisher ellipse/strain plots (completed 2026-04-02)
- [ ] **Phase 17: Enhanced Existing Plots** - Upgrade all existing plot modules to thesis quality
- [ ] **Phase 18: New Plot Modules** - Sky localization, corner plots, and convergence diagnostics
- [ ] **Phase 19: Campaign Dashboards & Batch Generation** - Multi-panel composites and automated figure pipeline

## Phase Details

### Phase 14: Test Infrastructure & Safety Net
**Goal**: A safety net of plot smoke tests exists so that style and infrastructure changes in later phases cannot silently break existing thesis-critical figures
**Depends on**: Phase 13 (v1.2 complete)
**Requirements**: TEST-01, TEST-02
**Success Criteria** (what must be TRUE):
  1. Every existing plot factory function has a smoke test that calls it with minimal valid data and asserts a (Figure, Axes) return without error
  2. An rcParams snapshot test calls `apply_style()` and verifies that key rcParams (font size, tick direction, figure DPI, color cycle) match expected values, failing if the mplstyle drifts
  3. Running `uv run pytest -m "not gpu and not slow"` passes with all new plot smoke tests green
**Plans:** 2/2 plans complete
Plans:
- [x] 14-01-PLAN.md — Shared conftest fixtures + smoke tests for bayesian, catalog, evaluation plots (14 tests)
- [ ] 14-02-PLAN.md — Smoke tests for model, physical_relations, simulation plots (9 tests) + rcParams regression

### Phase 15: Style Infrastructure
**Goal**: All downstream plot work builds on a consistent, centralized style system with proper figure sizing, LaTeX support, and shared color palette
**Depends on**: Phase 14
**Requirements**: STYLE-01, STYLE-02, STYLE-03, STYLE-04, STYLE-05
**Success Criteria** (what must be TRUE):
  1. Figures created via helpers use standardized thesis column widths (single ~3.5in, double ~7.0in) and existing tests still pass
  2. All axis labels in existing plot modules use LaTeX mathematical notation via mathtext (no raw ASCII for physics symbols)
  3. `apply_style(use_latex=True)` enables full LaTeX rendering; `apply_style()` (default) uses mathtext fallback and works on headless CI
  4. A `_colors.py` module provides a named color palette used by all plot modules, with no ad-hoc color strings remaining in production code
  5. `_fig_from_ax` is importable from `_helpers.py` and the old location in `simulation_plots.py` is removed or re-exports from helpers
**Plans:** 1/1 plans complete
Plans:
- [ ] 15-01-PLAN.md — Move _fig_from_ax, create _colors.py + _labels.py, add preset/latex params, tests

### Phase 16: Data Layer & Fisher Visualizations
**Goal**: CRB CSV data can be loaded and reconstructed into covariance matrices, and Fisher-based visualizations (error ellipses, characteristic strain) are available as factory functions
**Depends on**: Phase 15
**Requirements**: FISH-01, FISH-02, FISH-04, FISH-05
**Success Criteria** (what must be TRUE):
  1. `_data.py` reconstructs a symmetric 14x14 covariance matrix from the `delta_X_delta_Y` columns in any CRB CSV file produced by the simulation pipeline
  2. 2D Fisher error ellipses (1-sigma, 2-sigma contours) render for at least three key EMRI parameter pairs (e.g., M-mu, d_L-inclination, sky angles)
  3. A characteristic strain h_c(f) plot shows an example EMRI signal track overlaid on the LISA sensitivity curve with noise components
  4. Parameter uncertainty distributions display with intrinsic/extrinsic grouping and LaTeX-formatted parameter labels
**Plans:** 2/2 plans complete
Plans:
- [x] 16-01-PLAN.md — CRB data layer (_data.py): constants, covariance reconstruction, label mapping + unit tests
- [x] 16-02-PLAN.md — Fisher plot factories (error ellipses, characteristic strain, uncertainty distributions) + smoke tests + visual verification

### Phase 17: Enhanced Existing Plots
**Goal**: All existing plot modules produce thesis-quality figures with proper uncertainty visualization, reference annotations, and consistent styling
**Depends on**: Phase 16
**Requirements**: CORE-01, CORE-02, CORE-03, CORE-04, CORE-05, CORE-06, CORE-07, FISH-06, FISH-07
**Success Criteria** (what must be TRUE):
  1. H0 posterior plot shows shaded 68%/95% credible intervals with vertical reference lines for Planck and SH0ES values
  2. Individual event posteriors are distinguishable by SNR or redshift via color mapping, with the combined posterior visually prominent
  3. SNR distribution histogram includes a cumulative distribution overlay and a vertical threshold annotation line
  4. Detection yield vs redshift shows both injected and detected populations with a detection fraction curve
  5. Detection probability heatmap has a clean colorbar spanning [0,1] and detection contours in (z, M) space overlay the injected population
  6. LISA PSD plot shows the galactic confusion noise component as a separate curve alongside the instrument noise
  7. Luminosity distance d_L(z) plot includes comparison curves for different H0 values
  8. Injected vs recovered parameter scatter plots show measurement quality with identity lines and residual annotations
**Plans**: TBD
**UI hint**: yes

### Phase 18: New Plot Modules
**Goal**: Sky localization, multi-parameter corner plots, and convergence diagnostics are available as standard factory functions following the project's (Figure, Axes) pattern
**Depends on**: Phase 16, Phase 17
**Requirements**: SKY-01, FISH-03, CONV-01, CONV-02
**Success Criteria** (what must be TRUE):
  1. A Mollweide projection sky map displays EMRI sky positions with localization ellipses, replacing the non-standard 3D scatter plot
  2. A corner plot of an EMRI parameter subset renders from Fisher-derived Gaussian approximation using the `corner` library with thesis styling
  3. H0 convergence plot shows how the posterior width narrows as the number of detected events increases
  4. Detection efficiency curve shows a 1D P_det slice with confidence intervals
**Plans**: TBD
**UI hint**: yes

### Phase 19: Campaign Dashboards & Batch Generation
**Goal**: A single command produces all thesis figures from campaign data, with composite summary panels and size-optimized PDF output
**Depends on**: Phase 17, Phase 18
**Requirements**: CAMP-01, CAMP-02, CAMP-03
**Success Criteria** (what must be TRUE):
  1. A multi-panel composite figure combines key result plots (H0 posterior, SNR distribution, detection yield, sky map) into a single summary dashboard
  2. A batch generation script produces all thesis figures from a campaign working directory without manual intervention
  3. No single-figure PDF exceeds 2 MB; scatter plots with >1000 points use `rasterized=True` for vector/raster hybrid output
**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 14 -> 15 -> 16 -> 17 -> 18 -> 19

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Code Hardening | v1.0 | 2/2 | Complete | 2026-03-26 |
| 2. Batch Compatibility | v1.0 | 1/1 | Complete | 2026-03-26 |
| 3. Cluster Environment | v1.0 | 1/1 | Complete | 2026-03-27 |
| 4. SLURM Job Infrastructure | v1.0 | 3/3 | Complete | 2026-03-27 |
| 5. Documentation | v1.0 | 2/2 | Complete | 2026-03-27 |
| 6. Data Cleanup | v1.1 | 1/1 | Complete | 2026-03-27 |
| 7. Cluster Access | v1.1 | 1/1 | Complete | 2026-03-28 |
| 8. Simulation Campaign | v1.1 | 2/2 | Complete | 2026-03-29 |
| 9. Galactic Confusion Noise | v1.2 | 1/1 | Complete | - |
| 10. Five-Point Stencil | v1.2 | 1/1 | Complete | - |
| 11. Validation Campaign | v1.2 | 2/2 | Complete | - |
| 11.1 Simulation-Based P_det | v1.2 | 5/5 | Complete | 2026-04-01 |
| 12. Production Campaign | v1.2 | 1/1 | Complete | - |
| 13. H0 Posterior Sweep | v1.2 | 1/1 | Complete | - |
| 14. Test Infrastructure | v1.3 | 1/2 | Complete    | 2026-04-01 |
| 15. Style Infrastructure | v1.3 | 0/1 | Complete    | 2026-04-01 |
| 16. Data Layer & Fisher | v1.3 | 2/2 | Complete   | 2026-04-02 |
| 17. Enhanced Existing Plots | v1.3 | 0/0 | Not started | - |
| 18. New Plot Modules | v1.3 | 0/0 | Not started | - |
| 19. Campaign Dashboards | v1.3 | 0/0 | Not started | - |
