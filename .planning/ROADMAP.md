# Roadmap: EMRI Parameter Estimation

## Milestones

- ✅ **v1.0 EMRI HPC Integration** — Phases 1-5 (shipped 2026-03-27)
- ✅ **v1.1 Clean Simulation Campaign** — Phases 6-8 (shipped 2026-03-29)
- ✅ **v1.2 Production Campaign & Physics Corrections** — Phases 9-13 (shipped 2026-04-01)
- ✅ **v1.3 Visualization Overhaul** — Phases 14-19 (shipped 2026-04-02)
- ✅ **v1.4 Posterior Numerical Stability** — Phases 21-23 (shipped 2026-04-02)
- ✅ **v1.5 Galaxy Catalog Completeness Correction** — Phases 24-25 (shipped 2026-04-04, GPD-tracked)
- 🔄 **v2.0 Paper** — Phases 26-28 (in progress, GPD-tracked)
- 🔄 **v2.1 Publication Figures** — Phases 29-33 (in progress, GSD-tracked)

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

<details>
<summary>✅ v1.3 Visualization Overhaul (Phases 14-19) — SHIPPED 2026-04-02</summary>

- [x] Phase 14: Test Infrastructure & Safety Net (2/2 plans) — 23 smoke tests + rcParams regression test
- [x] Phase 15: Style Infrastructure (1/1 plan) — _colors.py, _labels.py, REVTeX presets, LaTeX toggle
- [x] Phase 16: Data Layer & Fisher Visualizations (2/2 plans) — _data.py CRB loader, fisher_plots.py factory functions
- [x] Phase 17: Enhanced Existing Plots (3/3 plans) — credible intervals, heatmaps with contours, injected-vs-recovered scatter
- [x] Phase 18: New Plot Modules (2/2 plans) — Mollweide sky map, corner plots, H0 convergence, efficiency curve
- [x] Phase 19: Campaign Dashboards & Batch Generation (2/2 plans) — 2x2 dashboard mosaic, 15-figure manifest pipeline

Full details: `.planning/milestones/v1.3-ROADMAP.md`

</details>

<details>
<summary>✅ v1.4 Posterior Numerical Stability (Phases 21-23) — SHIPPED 2026-04-02</summary>

- [x] Phase 21: Analysis & Post-Processing (2/2 plans) — Log-space combination script, 4 strategies, diagnostics, comparison table
- [x] Phase 22: Likelihood Floor & Overflow Fix (1/1 plan) — Physics-motivated floor in single_host_likelihood, underflow detection
- [x] Phase 23: Deploy & Validate (2/2 plans) — Deployed to cluster at 5793f70, validation PASS (|delta MAP|=0.00 < 0.05)

Full details: `.planning/milestones/v1.4-ROADMAP.md`

</details>

<details>
<summary>✅ v1.5 Galaxy Catalog Completeness Correction (Phases 24-25) — SHIPPED 2026-04-04 (GPD-tracked)</summary>

- [x] Phase 24: Completeness Estimation (1/1 plan) — GLADE+ f(z,h) from B-band luminosity comparison, 23 tests
- [x] Phase 25: Likelihood Correction (1/1 plan) — Gray et al. (2020) Eq. 9 combination formula, completion term, 11 tests

Full artifacts: `.gpd/phases/24-completeness-estimation/`, `.gpd/phases/25-likelihood-correction/`

**Note:** Originally scoped as 4 GSD phases (24-27). Phases 24-25 executed by GPD. Remaining scope (verification + deployment) rescoped into v2.0.

</details>

### v2.0 Paper (GPD-tracked)

- [x] **Phase 26: Paper Draft** — First complete PRD paper draft with all sections (1/1 plan, GPD)
- [ ] **Phase 27: Production Run & Figures** — Cluster evaluation + replace RESULT PENDING placeholders + publication figures
- [ ] **Phase 28: Review & Submission** — Internal review, finalize co-authors, submit to PRD + arXiv

### Phase Details (v2.0)

#### Phase 26: Paper Draft
**Goal**: First complete draft of the PRD paper "Constraints on the Hubble constant from EMRI dark sirens with LISA using the massive black hole mass"
**Status:** Complete (2026-04-05, GPD Phase 26)
**Result:** 11-page PDF builds with REVTeX4-2. 6 sections (Introduction, Method, Results, Discussion, Conclusions, Appendix A). 21 references. 25 RESULT PENDING markers awaiting production run.
See `.gpd/phases/` and `.gpd/ROADMAP.md` for full details.

#### Phase 27: Production Run & Figures
**Goal**: Run completeness-corrected evaluation on cluster, replace all RESULT PENDING placeholders with final numbers, generate publication figures
**Depends on**: Phase 25 (completeness code), Phase 26 (paper structure)
**Blocked on**: Cluster filesystem recovery
See `.gpd/ROADMAP.md` for full details.

#### Phase 28: Review & Submission
**Goal**: Internal peer review, resolve all TODO markers, finalize co-authors, submit to PRD + arXiv
**Depends on**: Phase 27 (final results and figures)
See `.gpd/ROADMAP.md` for full details.

### v2.1 Publication Figures (GSD-tracked)

- [ ] **Phase 29: Style Foundation** — Modernize mplstyle, colorblind-safe palette, font sizing, `pdf.fonttype: 42`
- [ ] **Phase 30: Unified Figure Pipeline & Paper Figures** — Merge manifests, fix CI bug, polish 4 existing + add contour/smoothed variants
- [ ] **Phase 31: Galaxy-Level Figures** — Pre-process 580MB JSONs, galaxy ranking, dominant fraction, BH mass impact, sky map
- [ ] **Phase 32: Interactive Figures** — Plotly/Sphinx integration, data pre-aggregation, MyST-NB notebooks, JupyterLite
- [ ] **Phase 33: Quality Assurance & New Science Figures** — P_det surface, completeness heatmap, parameter contours, colorblind/grayscale/pdffonts audit

### Phase Details (v2.1)

#### Phase 29: Style Foundation
**Goal**: Modernize the visual style infrastructure so all downstream figures automatically inherit publication-quality aesthetics
**Requirements**: STYL-01, STYL-02, STYL-03
**Depends on**: None
**Success criteria**:
1. `emri_thesis.mplstyle` updated: top/right spines removed, `pdf.fonttype: 42`, font sizes 7-9pt, inward ticks, frameless legends
2. `_colors.py` replaced with Okabe-Ito cycle + sequential Blues (0.1-0.85) + accent color
3. `get_figure` presets tuned for new font sizes at REVTeX widths
4. All existing tests pass with new style (no visual regression in smoke tests)
5. `pdffonts` on any generated PDF shows zero Type 3 fonts

#### Phase 30: Unified Figure Pipeline & Paper Figures
**Goal**: Merge the two disconnected figure pipelines and deliver polished paper figures with new style
**Requirements**: PFIG-01, PFIG-02, PFIG-03, PFIG-04
**Depends on**: Phase 29 (style)
**Success criteria**:
1. Single `--generate_figures <dir>` command generates all figures (paper + thesis + galaxy-level)
2. `paper_figures.py` functions integrated into unified manifest
3. CI calculation uses trapezoidal CDF everywhere (unit test)
4. 4 existing paper figures polished with new style
5. Contour-smoothed H0 posterior added as new variant (KDE, preserves MAP within grid spacing)
6. Auto-detect h-grid resolution (works with 15-pt and future finer grids)

#### Phase 31: Galaxy-Level Figures
**Goal**: Exploit the rich per-galaxy likelihood data from posteriors_with_bh_mass for novel visualizations
**Requirements**: GLXY-01, GLXY-02, GLXY-03, GLXY-04, GLXY-05
**Depends on**: Phase 29 (style), Phase 30 (manifest integration)
**Success criteria**:
1. Pre-processing script extracts per-event summaries from 580MB JSONs to CSVs (~1-5MB total)
2. Memory-aware loading: pre-process path for low-memory, direct-load fallback for 64GB+ machines
3. Galaxy likelihood ranking figure for representative events
4. Dominant galaxy fraction plot (top-1/5/10 across all events)
5. BH mass channel impact scatter (with vs without mass per galaxy)
6. Sky map with candidate host galaxies colored by likelihood weight
7. All galaxy figures integrated into unified manifest

#### Phase 32: Interactive Figures
**Goal**: Add interactive Plotly figures and Jupyter notebooks to GitHub Pages via Sphinx
**Requirements**: INTV-01, INTV-02, INTV-03, INTV-04, INTV-05
**Depends on**: Phase 29 (style), Phase 30 (data loaders)
**Success criteria**:
1. Plotly proof-of-concept: one interactive posterior plot embedded in Sphinx, no MathJax conflict
2. Data pre-aggregation script produces ~1MB summary JSONs in `docs/source/_static/data/`
3. Full interactive figure set: posterior explorer, sky map, galaxy browser
4. MyST-NB: at least one executed notebook in Sphinx docs with caching
5. JupyterLite: parameter exploration notebook (h-value slider) working in-browser
6. CI pipeline deploys interactive figures to GitHub Pages alongside existing docs

#### Phase 33: Quality Assurance & New Science Figures
**Goal**: Complete the figure set with science figures and run full accessibility/format audit
**Requirements**: PFIG-05, PFIG-06, PFIG-07, QUAL-01, QUAL-02, QUAL-03
**Depends on**: Phase 29-31 (all figures must exist before audit)
**Success criteria**:
1. P_det surface figure as 2D filled contour
2. Completeness f(z,h) standalone heatmap + P_det overlay version
3. Parameter correlation contours (d_L vs M, z vs SNR)
4. daltonize colorblind simulation passes for all paper figures
5. All paper figures readable in grayscale
6. `pdffonts` shows zero Type 3 fonts across all generated PDFs

## Progress

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
| 9. Galactic Confusion Noise | v1.2 | 1/1 | Complete | 2026-03-29 |
| 10. Five-Point Stencil | v1.2 | 1/1 | Complete | 2026-03-29 |
| 11. Validation Campaign | v1.2 | 2/2 | Complete | 2026-04-01 |
| 11.1 Simulation-Based P_det | v1.2 | 5/5 | Complete | 2026-04-01 |
| 12. Production Campaign | v1.2 | 1/1 | Complete | 2026-04-01 |
| 13. H0 Posterior Sweep | v1.2 | 1/1 | Complete | 2026-04-01 |
| 14. Test Infrastructure | v1.3 | 2/2 | Complete | 2026-04-01 |
| 15. Style Infrastructure | v1.3 | 1/1 | Complete | 2026-04-01 |
| 16. Data Layer & Fisher | v1.3 | 2/2 | Complete | 2026-04-02 |
| 17. Enhanced Existing Plots | v1.3 | 3/3 | Complete | 2026-04-02 |
| 18. New Plot Modules | v1.3 | 2/2 | Complete | 2026-04-02 |
| 19. Campaign Dashboards | v1.3 | 2/2 | Complete | 2026-04-02 |
| 21. Analysis & Post-Processing | v1.4 | 2/2 | Complete | 2026-04-02 |
| 22. Likelihood Floor & Overflow Fix | v1.4 | 1/1 | Complete | 2026-04-02 |
| 23. Deploy & Validate | v1.4 | 2/2 | Complete | 2026-04-02 |
| 24. Completeness Estimation | v1.5 | 1/1 | Complete (GPD) | 2026-04-04 |
| 25. Likelihood Correction | v1.5 | 1/1 | Complete (GPD) | 2026-04-04 |
| 26. Paper Draft | v2.0 | 1/1 | Complete (GPD) | 2026-04-05 |
| 27. Production Run & Figures | v2.0 | 0/? | Not started (GPD) | - |
| 28. Review & Submission | v2.0 | 0/? | Not started (GPD) | - |
| 29. Style Foundation | v2.1 | 0/? | Not started | - |
| 30. Unified Pipeline & Paper Figs | v2.1 | 0/? | Not started | - |
| 31. Galaxy-Level Figures | v2.1 | 0/? | Not started | - |
| 32. Interactive Figures | v2.1 | 0/? | Not started | - |
| 33. QA & New Science Figures | v2.1 | 0/? | Not started | - |
