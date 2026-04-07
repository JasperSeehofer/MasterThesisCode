# Requirements: v2.1 Publication Figures

**Defined:** 2026-04-07
**Core Goal:** Unify and modernize the visualization pipeline with publication-quality style, exploit per-galaxy likelihood data, and add interactive figures for GitHub Pages.

## Primary Requirements

### Style Infrastructure

- [ ] **STYL-01**: Update `emri_thesis.mplstyle` — remove top/right spines, add `pdf.fonttype: 42` and `ps.fonttype: 42`, reduce font sizes to 7-9pt range for REVTeX column widths, set inward ticks, frameless legends
- [ ] **STYL-02**: Replace `_colors.py` palette with Okabe-Ito colorblind-safe cycle, sequential Blues emphasis (truncated 0.1-0.85), and a single accent color for truth/reference lines
- [ ] **STYL-03**: Update `get_figure` presets with font sizes tuned for REVTeX 3.375"/7.0" column widths

### Paper Figures

- [ ] **PFIG-01**: Unify `paper_figures.py` functions and `--generate_figures` manifest (`main.py`) into a single pipeline — one command generates all figures
- [ ] **PFIG-02**: Fix CI calculation inconsistency — replace `np.cumsum` CDF in `bayesian_plots.py` with trapezoidal CDF (matching `paper_figures.py`)
- [ ] **PFIG-03**: Polish existing 4 paper figures (H0 posterior comparison, single-event likelihoods, convergence, SNR distribution) with new style; add as new contour/smoothed variants alongside existing discrete-marker versions
- [ ] **PFIG-04**: Add contour-smoothed H0 posterior variant using `scipy.stats.gaussian_kde` on discrete grid
- [ ] **PFIG-05**: P_det surface as 2D filled contour plot (z vs M or d_L vs M_z)
- [ ] **PFIG-06**: Completeness f(z,h) standalone heatmap/contour + version integrated with P_det overlay
- [ ] **PFIG-07**: Parameter correlation contours (d_L vs M, z vs SNR, etc.) as 2D density contours

### Galaxy-Level Figures

- [ ] **GLXY-01**: Galaxy data pre-processing — extract per-event summary stats from 580MB JSONs to lightweight CSVs; memory-aware: pre-process path for low-memory machines, direct-load path for 64GB+ machines
- [ ] **GLXY-02**: Galaxy likelihood ranking — per-event bar chart or heatmap showing top contributing galaxies
- [ ] **GLXY-03**: Dominant galaxy fraction — across all events, what fraction of likelihood comes from top-1/5/10 galaxies
- [ ] **GLXY-04**: BH mass channel impact — scatter showing with-mass vs without-mass likelihood per galaxy, revealing how BH mass reranks candidates
- [ ] **GLXY-05**: Sky map with candidate host galaxies colored by likelihood weight (Mollweide or zoomed inset)

### Interactive Figures

- [ ] **INTV-01**: Plotly proof-of-concept — one interactive posterior plot embedded in Sphinx docs via raw HTML include; verify CDN mode, no MathJax conflict
- [ ] **INTV-02**: Data pre-aggregation script for interactive plots — read large JSONs, output ~1MB summary JSONs to `docs/source/_static/data/`
- [ ] **INTV-03**: Full interactive figure set in Plotly (posterior explorer, sky map, galaxy browser) deployed to GitHub Pages
- [ ] **INTV-04**: MyST-NB integration — executed Jupyter notebooks in Sphinx docs with caching
- [ ] **INTV-05**: JupyterLite parameter exploration notebook (h-value slider, SNR threshold) embedded in docs

### Quality Assurance

- [ ] **QUAL-01**: Colorblind simulation test (daltonize) on all paper figures — verify distinguishability under deuteranopia, protanopia, tritanopia
- [ ] **QUAL-02**: Grayscale safety check — verify all paper figures remain readable in grayscale print
- [ ] **QUAL-03**: `pdffonts` verification — confirm no Type 3 fonts in any generated PDF

## Follow-up Requirements

### Deferred

- **DEFR-01**: Dark/presentation theme variant for conference talks
- **DEFR-02**: Animated convergence GIF (posterior narrowing as events accumulate)
- **DEFR-03**: Observable/D3 custom JavaScript visualizations

## Out of Scope

| Topic | Reason |
|-------|--------|
| Physics formula changes | v2.1 is visualization only; physics changes require /physics-change protocol |
| New evaluation runs | Cluster runs are independent of this milestone |
| Waveform visualization | Requires GPU and `few` package; not visualization infrastructure |
| Pipeline A (bayesian_inference.py) plots | Development cross-check only; not for publication |

## Accuracy and Validation Criteria

| Requirement | Accuracy Target | Validation Method |
|-------------|-----------------|-------------------|
| STYL-01 | All rcParams applied | `apply_style()` smoke test + visual inspection |
| STYL-02 | Colorblind-safe | daltonize simulation (QUAL-01) |
| PFIG-02 | CI matches trapezoid method | Unit test: CI from cumsum == CI from trapezoid on known posterior |
| PFIG-04 | KDE preserves MAP location | MAP of smoothed == MAP of discrete within grid spacing |
| GLXY-01 | Summary matches full data | Spot-check: top-5 galaxies from CSV == top-5 from full JSON for 10 events |
| QUAL-03 | Zero Type 3 fonts | `pdffonts` output shows only Type 1 or TrueType |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| STYL-01 | TBD | Pending |
| STYL-02 | TBD | Pending |
| STYL-03 | TBD | Pending |
| PFIG-01 | TBD | Pending |
| PFIG-02 | TBD | Pending |
| PFIG-03 | TBD | Pending |
| PFIG-04 | TBD | Pending |
| PFIG-05 | TBD | Pending |
| PFIG-06 | TBD | Pending |
| PFIG-07 | TBD | Pending |
| GLXY-01 | TBD | Pending |
| GLXY-02 | TBD | Pending |
| GLXY-03 | TBD | Pending |
| GLXY-04 | TBD | Pending |
| GLXY-05 | TBD | Pending |
| INTV-01 | TBD | Pending |
| INTV-02 | TBD | Pending |
| INTV-03 | TBD | Pending |
| INTV-04 | TBD | Pending |
| INTV-05 | TBD | Pending |
| QUAL-01 | TBD | Pending |
| QUAL-02 | TBD | Pending |
| QUAL-03 | TBD | Pending |

**Coverage:**

- Primary requirements: 23 total
- Mapped to phases: 0 (roadmap pending)
- Unmapped: 23

---

_Requirements defined: 2026-04-07_
_Last updated: 2026-04-07 after initial definition_
