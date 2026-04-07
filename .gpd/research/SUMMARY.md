# Research Summary

**Project:** EMRI Dark Siren H0 Inference -- v2.1 Publication Figures
**Domain:** Scientific visualization for gravitational wave cosmology (PRD submission)
**Researched:** 2026-04-07
**Confidence:** HIGH

## Unified Notation

| Symbol | Quantity | Units/Dimensions | Convention Notes |
|--------|---------|-----------------|-----------------|
| h | Dimensionless Hubble parameter | dimensionless | H0 = 100h km/s/Mpc; fiducial h = 0.73; x-axis label is "$h$" throughout |
| P_det | Detection probability | dimensionless [0,1] | P(SNR >= threshold | z, M, h); visualized as 2D heatmap |
| z | Source redshift | dimensionless | Cosmological redshift of EMRI host |
| M | Source-frame MBH mass | M_sun | Logarithmic axis in P_det plots |
| M_z | Observer-frame mass | M_sun | M_z = M(1+z); used in "with BH mass" channel |
| d_L | Luminosity distance | Gpc | Scatter plot axis in SNR vs distance |
| SNR | Signal-to-noise ratio | dimensionless | Matched-filter SNR; threshold = 20 |
| CI | Credible interval | dimensionless | 68% and 95% levels; computed via trapezoidal CDF |

**Convention notes:**
- Posteriors are peak-normalized for visual comparison; density normalization used only for insets or when comparing constraining power. State normalization in y-axis labels.
- "Injected" (not "True") for reference H0 line in paper figures. "True" is reserved for internal diagnostics.
- Planck reference: h = 0.674 +/- 0.005 (arXiv:1807.06209). SH0ES reference: h = 0.73 +/- 0.01 (arXiv:2112.04510). Verify against latest values before submission.
- All figure sizes in inches: single-column = 3.375", double-column = 7.0" (REVTeX/Physical Review D).
- Color convention: Okabe-Ito palette replaces tab10. Primary = #0072B2 (blue), secondary = #D55E00 (vermillion), truth = #009E73 (bluish green).

## Executive Summary

This milestone produces publication-quality figures for a Physical Review D submission on EMRI dark siren H0 inference. The literature survey reveals that four figure types are mandatory and already implemented (H0 posterior comparison, single-event likelihoods, convergence vs N_det, SNR distribution), while three high-value figures are missing: a P_det(z, M) surface visualization, a per-galaxy likelihood decomposition showing how BH mass information breaks the distance-redshift degeneracy, and a channel information gain metric (effective number of hosts or KL divergence). The per-galaxy decomposition and channel comparison are genuinely novel -- no EMRI dark siren paper has shown these, and they directly illustrate the paper's central scientific contribution.

The visualization approach requires two parallel tracks: (1) modernizing the matplotlib static pipeline for print publication (REVTeX-compliant sizing, Okabe-Ito colorblind-safe palette, Type 42 font embedding, minimal L-frame spines, LaTeX typography), and (2) building an interactive web pipeline (Plotly with CDN mode, MyST-NB for Sphinx notebook integration, JupyterLite for serverless parameter exploration on GitHub Pages). The static pipeline is the priority -- it directly supports the paper submission. The interactive pipeline is a value-add for the thesis website and can proceed in parallel without blocking the paper.

The principal risks are: (a) the 15-point h-grid produces coarse posteriors that look unusual compared to MCMC-based papers -- honest disclosure with markers is the correct approach but requires explicit caption language; (b) the "with BH mass" posterior collapses to a near-delta on the coarse grid, which could look like an artifact without explanation; (c) the 580 MB per-galaxy JSON files require a pre-aggregation pipeline before any interactive visualization is possible; and (d) Type 3 fonts in PDF output will cause APS rejection if `pdf.fonttype: 42` is not set. All risks are addressable with known solutions documented in the research files.

## Key Findings

### Prior Work Landscape

The dark siren H0 inference visualization canon is well-established, with Gray et al. (2020), the LVK O3 paper (arXiv:2111.03604), and Laghi et al. (2021) setting the standard. [CONFIDENCE: HIGH]

**Must reproduce (standard figures):**

- H0 posterior with Planck/SH0ES reference bands and injected value -- every dark siren paper shows this
- Per-event likelihood waterfall (transparent overlaid curves) -- standard in LVK papers
- Posterior convergence vs N_det on log-log axes with N^{-1/2} reference line -- standard for forecasts
- SNR distribution histogram with threshold line -- standard characterization

**Novel contributions (differentiating):**

- Per-galaxy likelihood decomposition showing which galaxies dominate the event likelihood and how BH mass narrows the effective host set -- no EMRI paper has shown this [HIGH NOVELTY]
- Channel information gain visualization (effective N_hosts with vs without mass, or KL divergence per event) -- directly illustrates the central scientific contribution [HIGH NOVELTY]
- P_det(z, M) surface with detected events overlaid -- recommended but not always shown; critical for demonstrating well-behaved selection effects

**Defer (supplementary or future work):**

- Sky localization maps (EMRI localization is sub-degree; less dramatic than BBH sky maps)
- 2D cosmological contours in (H0, w0) plane (current analysis is 1D in h only)
- Full corner plots (meaningless for 1D inference)

### Methods and Tools

The recommended approach is a custom `.mplstyle` file targeting REVTeX single-column (3.375") with LaTeX/Computer Modern typography, Okabe-Ito qualitative palette, and Blues sequential palette for contour fills. Font sizes must be calibrated for column width: 8pt body, 7pt ticks, ensuring the 2mm minimum capital height after any production scaling. [CONFIDENCE: HIGH]

**Major components:**

1. **Custom mplstyle + `apply_style()`** -- Global rcParams: L-frame spines, inward ticks, frameless legends, near-black (#262626) axes, Type 42 font embedding. Replaces the current style entirely for paper figures.
2. **Okabe-Ito palette** -- Colorblind-safe qualitative colors replacing tab10. Blue (#0072B2) + vermillion (#D55E00) for the two-channel comparison. Green truth line (#2ca02c) replaced with bluish green (#009E73) or vermillion depending on context.
3. **KDE smoothing via scipy.stats.gaussian_kde** -- For 1D posteriors from discrete samples (Scott bandwidth). For 2D grids (P_det surface), use `scipy.ndimage.gaussian_filter` with sigma=1.0. Do NOT smooth the 15-point h-grid posterior -- markers + straight lines are the honest representation.
4. **Plotly (CDN mode) + MyST-NB** -- Interactive figures for GitHub Pages. Self-contained HTML with `include_plotlyjs='cdn'` to avoid 3-5 MB per figure. MyST-NB provides Sphinx integration with execution caching via jupyter-cache.
5. **corner library** -- For any multi-parameter visualization (e.g., if Fisher matrix projections are added). Configured with filled contours from Blues palette, `smooth=1.2`, `plot_datapoints=False`.

### Computational Approaches

The interactive pipeline requires solving the 580 MB JSON data problem through build-time pre-aggregation. [CONFIDENCE: HIGH]

**Core approach:**

- **Pre-aggregate at build time** into ~1 MB summary JSONs (binned posteriors, top-N events, HEALPix sky map, parameter histograms). Commit summaries to `docs/source/_static/data/`. No large files in git or CI.
- **Plotly over Bokeh** -- Better Sphinx integration, CDN option, WebGL for large datasets (`Scattergl` handles up to 200k points).
- **JupyterLite via jupyterlite-sphinx** -- Serverless notebooks on GitHub Pages using Pyodide. Limitations: no CuPy/GPU, ~2-4 GB browser memory, 5-15s cold start. Suitable only for lightweight analysis on pre-aggregated data.
- **MyST-NB over nbsphinx** -- Execution caching via jupyter-cache, glue mechanism for embedding outputs in arbitrary Sphinx pages.
- **Dual-output pattern** -- Keep matplotlib for thesis PDF (vector quality). Add Plotly equivalents for web. Never replace matplotlib with Plotly for the paper.

### Critical Pitfalls

1. **Type 3 fonts cause APS rejection [CRITICAL]** -- Set `pdf.fonttype: 42` and `ps.fonttype: 42` in mplstyle. The current `emri_thesis.mplstyle` does NOT set these. Detection: `pdffonts paper/figures/*.pdf` should show no Type 3.

2. **tab10 palette is not colorblind-safe [CRITICAL]** -- Green (TRUTH) and red (MEAN / "With Mz") are indistinguishable under deuteranopia (~6% of males). Replace with Okabe-Ito. The two-channel comparison should use blue + vermillion, which are maximally separated under all CVD types.

3. **15-point h-grid creates misleading visual impression [CRITICAL]** -- Do NOT smooth or spline the posterior. Show markers at grid points, connect with straight lines, state grid resolution in caption. CI boundaries should be rounded to grid spacing (0.02 in h); do not quote sub-grid precision from interpolation.

4. **Inconsistent CI calculation between bayesian_plots.py and paper_figures.py [MODERATE]** -- `bayesian_plots.py` uses `np.cumsum` (discrete sum), `paper_figures.py` uses `np.trapezoid` (trapezoidal integral). These give different CI boundaries for the same posterior. Unify on trapezoidal integration throughout.

5. **Peak normalization hides relative constraining power [MODERATE]** -- Both channels peak at 1.0 after normalization. Annotate 68% CI width directly on the figure, or add a density-normalized inset showing the height difference.

6. **580 MB JSON files cannot be loaded in CI or client-side [MODERATE]** -- Pre-aggregate locally, commit ~1 MB summaries. The tail-read approach in `_load_per_event_with_mass_scalars` is memory-efficient; do not regress by switching to full `json.load` for with-mass files.

7. **Legend placement obscures posterior peak [MINOR]** -- Move legend to upper-left or outside axes. Consider direct curve annotation with `ax.annotate()` instead of legend box.

## Approximation Landscape

| Method | Valid Regime | Breaks Down When | Controlled? | Complements |
|--------|------------|-----------------|------------|-------------|
| KDE smoothing (Scott bandwidth) | N > 100 samples, unimodal | Multi-modal or sharp features; N < 50 | Yes (h = N^{-1/5} for 1D) | Raw histogram for validation |
| Gaussian filter on 2D grid | Grid spacing << feature width; sigma < grid_dim/2 | Sharp boundaries (P_det transition); noisy grids | No formal bound; visual check | Unsmoothed grid comparison |
| Linear interpolation (markers + lines) | Any grid; honest representation | Misses curvature between grid points | N/A (exact at grid points) | Finer grid if available |
| Peak normalization | Visual comparison of shape | Hides relative constraining power (width differences) | N/A | Density normalization inset |
| Trapezoidal CDF | Monotonic posterior on regular grid | Highly irregular spacing; posterior not well-sampled | Yes (error ~ O(dh^2)) | Simpson's rule if higher accuracy needed |

**Coverage gap:** The 15-point h-grid is the fundamental limitation for posterior visualization. No smoothing or interpolation method can reliably recover sub-grid structure. If reviewers question the grid resolution, the only remedy is running a finer grid evaluation (computationally expensive but possible). The with-BH-mass channel is most affected because its posterior is narrow enough to be under-resolved.

## Theoretical Connections

### Honest Representation of Discrete Inference [ESTABLISHED]

The markers-on-grid approach for the 15-point posterior is more transparent than standard practice. Most dark siren papers show smooth MCMC posteriors from thousands of samples. This project's grid-based evaluation produces only 15 likelihood values per channel. The honest disclosure via markers is scientifically superior to artificial smoothing, but requires explicit caption language to avoid appearing as an artifact. This connects to the broader methodological point that grid-based posterior evaluation trades resolution for exact likelihood computation at each point.

### BH Mass as Degeneracy Breaker [ESTABLISHED]

The per-galaxy likelihood decomposition (proposed Figure 6) directly visualizes how M_z measurement narrows the effective host galaxy set. This is the mechanism by which the "with BH mass" channel achieves a narrower posterior. The effective number of hosts is an entropy-based measure (N_eff = exp(H), where H is the Shannon entropy of the normalized galaxy weights). Connecting per-event N_eff reduction to the combined posterior width provides a quantitative bridge between the micro (galaxy-level) and macro (H0 posterior) scales of the analysis.

### Dual Pipeline as Cross-Validation [ESTABLISHED]

The matplotlib (static/print) and Plotly (interactive/web) pipelines produce the same figures in different formats. Visual consistency between the two serves as an implicit cross-check that the data processing is correct. Any discrepancy between static and interactive versions of the same figure indicates a bug in one rendering path.

### Selection Effects Visualization as Scientific Argument [ESTABLISHED]

The P_det(z, M) surface figure serves dual purpose: it is both a standard characterization figure and a scientific argument that the selection function is well-behaved (monotonic in z, sensible boundary location). This connects to the injection campaign work (v1.2.2) where the P_det grid quality was assessed with Wilson CIs. The visualization should overlay quality flag information (e.g., hatch unreliable bins) to make this argument visually.

## Critical Claim Verification

| # | Claim | Source | Verification | Result |
|---|-------|--------|--------------|--------|
| 1 | APS requires Type 42 fonts (no Type 3) | PITFALLS.md | APS Style Basics page confirms font embedding requirements | CONFIRMED |
| 2 | Okabe-Ito palette is distinguishable under all CVD types | METHODS.md, PITFALLS.md | Okabe & Ito (2002), Wong Nature Methods 2011 | CONFIRMED |
| 3 | REVTeX single-column width is 3.375 inches | METHODS.md | APS Author Guidelines | CONFIRMED |
| 4 | `pdf.fonttype: 42` forces TrueType embedding | PITFALLS.md | matplotlib docs on font handling | CONFIRMED |
| 5 | Plotly CDN mode reduces per-figure HTML from ~5MB to ~2KB | COMPUTATIONAL.md | Plotly documentation on `include_plotlyjs='cdn'` | CONFIRMED |
| 6 | JupyterLite supports scipy/numpy via Pyodide | COMPUTATIONAL.md | Pyodide package list and JupyterLite docs | CONFIRMED |
| 7 | Gray et al. (2020) is the foundational dark siren method paper | PRIOR-WORK.md | arXiv:1908.06050, widely cited | CONFIRMED |

### Cross-Validation Matrix

|                    | Matplotlib PDF | Plotly HTML | Grayscale print | CVD simulation |
|--------------------|:---:|:---:|:---:|:---:|
| **Matplotlib PDF** | --- | Same data, different renderer | Convert to L-channel | daltonize/Coblis |
| **Plotly HTML**    | Same data | --- | N/A (screen only) | N/A (screen only) |
| **Grayscale**      | Visual check | N/A | --- | N/A |
| **CVD sim**        | daltonize | N/A | N/A | --- |

### Input Quality -> Roadmap Impact

| Input File | Quality | Affected Recommendations | Impact if Wrong |
|------------|---------|------------------------|-----------------|
| PRIOR-WORK.md | GOOD | Figure selection, priority ordering, novelty assessment | Could miss a standard expected figure or overestimate novelty |
| METHODS.md | GOOD | Style configuration, palette choice, smoothing approach | Wrong font sizes or palette could cause APS rejection |
| COMPUTATIONAL.md | GOOD | Interactive pipeline architecture, data reduction strategy | Wrong tool choice adds unnecessary complexity |
| PITFALLS.md | GOOD | Risk mitigation for APS compliance, accessibility, CI consistency | Missing a pitfall could cause late-stage rejection or incorrect results |

## Implications for Roadmap

### Suggested Phase Structure

#### Phase 1: Style Foundation

**Rationale:** All downstream figures depend on the style configuration. Changing colors and fonts after figures are made wastes effort. Do this first.
**Delivers:** New `emri_thesis_pub.mplstyle` with REVTeX sizing, Okabe-Ito palette, Type 42 fonts, L-frame spines; updated `_colors.py` with semantic Okabe-Ito colors; updated `apply_style()` to load the new style.
**Validates:** `pdffonts` check on test PDF shows no Type 3; font size measurement confirms >= 2mm capitals at 3.375" width; daltonize simulation confirms two-channel distinguishability.
**Avoids:** Pitfall 1 (Type 3 fonts), Pitfall 2 (colorblind safety), Pitfall 6 (font size after scaling).
**Risk:** LOW -- well-documented matplotlib configuration.

#### Phase 2: Polish Existing Figures (1-4)

**Rationale:** Four figures already exist and are scientifically correct. Polish them to publication standard using the new style, fix the CI calculation inconsistency, and add missing annotations.
**Delivers:** Refined Figures 1-4 (H0 posterior, single-event likelihoods, convergence, SNR distribution) with consistent trapezoidal CI, CI width annotations, grid resolution disclosure in captions.
**Builds on:** Phase 1 style foundation.
**Avoids:** Pitfall 3 (misleading grid interpolation), Pitfall 4 (inconsistent CI), Pitfall 5 (grayscale collapse), Pitfall 8 (peak normalization).
**Risk:** LOW -- existing code with targeted modifications.

#### Phase 3: New Science Figures (5-7)

**Rationale:** The three new figures (P_det surface, per-galaxy decomposition, channel information gain) are the highest-impact additions. They require new plotting code and access to injection/per-galaxy data.
**Delivers:** Figure 5 (P_det heatmap with contours and event overlay), Figure 6 (per-galaxy likelihood bar chart or scatter for 1-2 representative events), Figure 7 (effective N_hosts or KL divergence per event showing BH mass information gain).
**Builds on:** Phase 1 style, Phase 2 data loading patterns.
**Uses:** `pcolormesh` for P_det surface, tail-read approach for 580MB JSON extraction, entropy-based N_eff calculation.
**Avoids:** Pitfall 7 (memory exhaustion from full JSON load), Pitfall 9 (non-reproducible event selection -- pin representative events).
**Risk:** MEDIUM -- new code, requires 580MB JSON data available locally, per-galaxy data extraction not yet fully implemented.

#### Phase 4: Interactive Pipeline

**Rationale:** Interactive figures add value for the thesis website but do not block the paper. Can proceed in parallel after Phase 1.
**Delivers:** Pre-aggregation script (`scripts/prepare_interactive_data.py`), Plotly versions of key figures with CDN mode, MyST-NB integration in Sphinx, JupyterLite embed for parameter exploration.
**Dependencies:** Pre-aggregated data from 580MB JSONs (overlaps with Phase 3 data access). Style foundation from Phase 1 (Plotly template should match matplotlib style).
**Avoids:** Anti-approaches: loading 580MB client-side, self-contained Plotly HTML (3-5MB each), replacing matplotlib with Plotly for paper.
**Risk:** MEDIUM -- MathJax conflict between Plotly and Sphinx is a known issue requiring early testing; JupyterLite cold start UX may be poor.

#### Phase 5: Supplementary Figures and Final QA

**Rationale:** Supplementary figures (waterfall plot, hosts vs redshift, injection summary) and final quality assurance (grayscale check, CVD simulation, `pdffonts` audit) complete the figure set.
**Delivers:** Supplementary Figures S1-S4, comprehensive grayscale and CVD audit of all figures, final PDF validation.
**Builds on:** All previous phases.
**Risk:** LOW -- incremental additions and quality checks.

### Phase Ordering Rationale

- Phase 1 must come first: style changes cascade to every figure; retrofitting is expensive.
- Phase 2 before Phase 3: polishing existing figures builds familiarity with the data loading patterns and identifies any data availability issues before attempting new figures.
- Phase 3 is the creative/science-heavy phase: requires the most design decisions and data wrangling.
- Phase 4 can run in parallel with Phases 2-3 after Phase 1 completes: interactive pipeline is independent of static figure refinement.
- Phase 5 is final: quality assurance must run on the complete figure set.

### Phases Requiring Deep Investigation

- **Phase 3 (New Science Figures):** Per-galaxy data extraction from 580MB JSONs needs the tail-read approach extended to galaxy-level entries. The effective N_hosts metric requires defining the entropy measure and deciding on visualization (scatter, histogram, or heatmap). Representative event selection must be pinned for reproducibility.

Phases with established methodology (straightforward execution):

- **Phase 1 (Style Foundation):** All rcParams are documented; Okabe-Ito hex codes are fixed; mplstyle syntax is standard.
- **Phase 2 (Polish Existing Figures):** Code exists; changes are targeted (CI method, annotations, style application).
- **Phase 5 (Supplementary + QA):** Standard quality checks with known tools.

## Confidence Assessment

| Area | Confidence | Notes |
|------|-----------|-------|
| Prior Work | HIGH | Dark siren figure canon is well-established; 7 reference papers surveyed comprehensively |
| Methods | HIGH | matplotlib publication techniques are mature; Okabe-Ito and REVTeX sizing are standard |
| Computational | HIGH | Plotly/MyST-NB/JupyterLite are production tools; data reduction strategy is straightforward |
| Pitfalls | HIGH | APS requirements are documented; colorblind and grayscale issues are well-characterized |

**Overall confidence:** HIGH

### Gaps to Address

- **Grid resolution sensitivity:** If reviewers demand finer h-grid resolution, a new evaluation run is needed. This is a computational cost issue, not a methodology gap. Consider preparing a response explaining why the 15-point grid is sufficient for the current analysis.
- **With-BH-mass near-delta:** The posterior collapse on the coarse grid needs careful presentation. A finer grid would resolve whether it is genuinely narrow or an artifact of under-sampling. This may block full confidence in Figure 1.
- **Per-galaxy data extraction:** The tail-read approach for 580MB files works for event-level scalars but has not been tested for galaxy-level entries. Phase 3 may discover that a different approach (pre-processing to intermediate CSV) is needed.
- **MathJax conflict:** Plotly and Sphinx both load MathJax; double-loading may cause rendering issues. Needs early testing in Phase 4.

## Open Questions

1. **Is the 15-point h-grid resolution sufficient for publication?** [HIGH priority] -- Reviewers may question the coarse grid. The markers-with-lines approach is honest, but a finer grid would strengthen the paper. Decision: proceed with 15-point grid, prepare response to reviewer objection, consider running finer grid if time permits.

2. **How to present the with-BH-mass near-delta posterior?** [HIGH priority] -- The posterior is so narrow on the 15-point grid that it may look like an artifact. Options: (a) finer grid for this channel only, (b) inset zoom, (c) explicit discussion in caption and text.

3. **Can per-galaxy likelihood data be extracted efficiently from 580MB JSONs?** [MEDIUM priority, blocks Phase 3] -- The tail-read approach needs extension; alternatively, a one-time pre-processing step could create intermediate CSV files.

4. **Will ipywidgets work reliably in JupyterLite?** [LOW priority] -- Start with manual cell re-execution; add widgets only if UX warrants it.

5. **Should the Plotly template match the matplotlib style exactly?** [LOW priority] -- Exact matching is difficult (different rendering engines). Consistent color palette and general aesthetic are sufficient.

## Sources

### Primary (HIGH)

- Gray et al. (2020), arXiv:1908.06050 -- Foundational dark siren method; figure conventions
- LVK GWTC-3 (2021), arXiv:2111.03604 -- Official H0 constraints; waterfall plot standard
- Laghi et al. (2021), arXiv:2102.01708 -- EMRI dark siren forecast; convergence plots
- APS Style Basics, https://journals.aps.org/authors/style-basics -- Figure requirements
- Wong, B. (2011), Nature Methods 8(6):441 -- Colorblind-safe palette
- Okabe & Ito (2002), https://jfly.uni-koeln.de/color/ -- Color Universal Design palette

### Secondary (MEDIUM)

- Finke et al. (2021), arXiv:2101.12660 -- GWTC-2 dark siren H0; galaxy weighting figures
- Gray et al. (2022), arXiv:2212.08694 -- Hitchhiker's Guide to galaxy catalog method
- Borghi et al. (2024), arXiv:2404.16092 -- O4a dark siren H0; sky maps with galaxies
- Laghi et al. (2026), arXiv:2603.23612 -- Joint EMRI+MBHB LISA cosmology
- Inchauspe et al. (2025), arXiv:2406.09228 -- Target aesthetic reference
- Scott (1992), "Multivariate Density Estimation," Wiley -- KDE bandwidth selection
- Rougier et al. (2014), PLoS Comput Biol 10(9):e1003833 -- Ten Simple Rules for Better Figures

### Tertiary (LOW)

- Plotly documentation, https://plotly.com/python/ -- Interactive plotting API
- MyST-NB documentation, https://github.com/executablebooks/MyST-NB -- Sphinx notebook integration
- JupyterLite documentation, https://jupyterlite.readthedocs.io/ -- Serverless notebooks
- corner.py documentation, https://corner.readthedocs.io/ -- Triangle plot library

---

_Research analysis completed: 2026-04-07_
_Ready for research plan: yes_

```yaml
# --- ROADMAP INPUT (machine-readable, consumed by gpd-roadmapper) ---
synthesis_meta:
  project_title: "EMRI Dark Siren H0 Inference -- v2.1 Publication Figures"
  synthesis_date: "2026-04-07"
  input_files: [METHODS.md, PRIOR-WORK.md, COMPUTATIONAL.md, PITFALLS.md]
  input_quality: {METHODS: good, PRIOR-WORK: good, COMPUTATIONAL: good, PITFALLS: good}

conventions:
  unit_system: "mixed: dimensionless h, Gpc (distances), M_sun (masses)"
  metric_signature: "N/A (visualization milestone)"
  fourier_convention: "N/A"
  coupling_convention: "N/A"
  renormalization_scheme: "N/A"

methods_ranked:
  - name: "Custom mplstyle with LaTeX/REVTeX typography"
    regime: "All static figures for PRD submission"
    confidence: HIGH
    cost: "O(1) setup; applies globally via rcParams"
    complements: "Plotly template for interactive figures"
  - name: "Okabe-Ito colorblind-safe palette"
    regime: "All figures with categorical color encoding"
    confidence: HIGH
    cost: "O(1) palette swap in _colors.py"
    complements: "Blues sequential colormap for density/contour fills"
  - name: "scipy.stats.gaussian_kde for 1D/2D smoothing"
    regime: "Posterior samples with N > 100; NOT for 15-point grid posteriors"
    confidence: HIGH
    cost: "O(N*M) for N samples, M grid points"
    complements: "scipy.ndimage.gaussian_filter for pre-computed 2D grids"
  - name: "Plotly with CDN mode for interactive web figures"
    regime: "GitHub Pages / Sphinx HTML output; datasets < 200k points"
    confidence: HIGH
    cost: "~2KB per figure HTML + CDN load"
    complements: "matplotlib for print/PDF output"
  - name: "Build-time pre-aggregation for 580MB JSON data"
    regime: "Any interactive visualization or CI-built figure"
    confidence: HIGH
    cost: "One-time local script; ~1MB output committed to repo"
    complements: "Tail-read approach for targeted extraction of specific events"
  - name: "MyST-NB with jupyter-cache for Sphinx notebook integration"
    regime: "Notebook-based interactive exploration pages"
    confidence: MEDIUM
    cost: "~3-4 min added to CI build time"
    complements: "Raw HTML include for pre-built Plotly figures (faster build)"

phase_suggestions:
  - name: "Style Foundation"
    goal: "Establish REVTeX-compliant mplstyle with Okabe-Ito palette, Type 42 fonts, and L-frame spines"
    methods: ["Custom mplstyle with LaTeX/REVTeX typography", "Okabe-Ito colorblind-safe palette"]
    depends_on: []
    needs_research: false
    risk: LOW
    pitfalls: ["type3-fonts-aps-rejection", "tab10-not-colorblind-safe", "font-size-scaling"]
  - name: "Polish Existing Figures"
    goal: "Refine Figures 1-4 to publication standard with consistent CI calculation and annotations"
    methods: ["Custom mplstyle with LaTeX/REVTeX typography", "Okabe-Ito colorblind-safe palette"]
    depends_on: ["Style Foundation"]
    needs_research: false
    risk: LOW
    pitfalls: ["grid-interpolation-misleading", "ci-cumsum-vs-trapezoid", "peak-normalization-hides-width", "grayscale-collapse"]
  - name: "New Science Figures"
    goal: "Create P_det surface, per-galaxy decomposition, and channel information gain figures"
    methods: ["scipy.stats.gaussian_kde for 1D/2D smoothing", "Build-time pre-aggregation for 580MB JSON data"]
    depends_on: ["Style Foundation"]
    needs_research: false
    risk: MEDIUM
    pitfalls: ["memory-exhaustion-json-load", "non-reproducible-event-selection"]
  - name: "Interactive Pipeline"
    goal: "Build Plotly/MyST-NB/JupyterLite interactive figures for GitHub Pages"
    methods: ["Plotly with CDN mode for interactive web figures", "MyST-NB with jupyter-cache for Sphinx notebook integration", "Build-time pre-aggregation for 580MB JSON data"]
    depends_on: ["Style Foundation"]
    needs_research: false
    risk: MEDIUM
    pitfalls: ["mathjax-double-loading", "580mb-client-side-crash"]
  - name: "Supplementary Figures and QA"
    goal: "Complete supplementary figure set and run grayscale/CVD/pdffonts quality audit"
    methods: ["Custom mplstyle with LaTeX/REVTeX typography", "Okabe-Ito colorblind-safe palette"]
    depends_on: ["Polish Existing Figures", "New Science Figures"]
    needs_research: false
    risk: LOW
    pitfalls: ["eps-transparency-incompatibility", "legend-obscures-data"]

critical_benchmarks:
  - quantity: "Minimum font capital height at single-column width"
    value: ">= 2mm (8pt minimum at 3.375 inches)"
    source: "APS Style Basics"
    confidence: HIGH
  - quantity: "PDF font type"
    value: "Type 42 (TrueType) only; zero Type 3 fonts"
    source: "APS submission requirements"
    confidence: HIGH
  - quantity: "Colorblind distinguishability"
    value: "All data series distinguishable under deuteranopia, protanopia, tritanopia"
    source: "Wong (2011), Okabe & Ito (2002)"
    confidence: HIGH
  - quantity: "Planck h reference value"
    value: "0.674 +/- 0.005"
    source: "Planck 2018 (arXiv:1807.06209)"
    confidence: HIGH
  - quantity: "SH0ES h reference value"
    value: "0.73 +/- 0.01"
    source: "Riess et al. 2022 (arXiv:2112.04510)"
    confidence: HIGH

open_questions:
  - question: "Is the 15-point h-grid resolution sufficient for publication, or will reviewers demand finer resolution?"
    priority: HIGH
    blocks_phase: "none"
  - question: "How to present the with-BH-mass near-delta posterior on the coarse grid without it looking like an artifact?"
    priority: HIGH
    blocks_phase: "Polish Existing Figures"
  - question: "Can per-galaxy likelihood data be extracted efficiently from 580MB JSONs for Figure 6?"
    priority: MEDIUM
    blocks_phase: "New Science Figures"
  - question: "Will MathJax double-loading between Plotly and Sphinx cause rendering issues?"
    priority: LOW
    blocks_phase: "Interactive Pipeline"

contradictions_unresolved:
  - claim_a: "METHODS.md recommends TRUTH color as vermillion (#D55E00) for contrast against blue fills"
    claim_b: "PITFALLS.md recommends TRUTH color as bluish green (#009E73) for distinction from both blue and vermillion"
    source_a: "METHODS.md Tier 3 Semantic Accent Colors"
    source_b: "PITFALLS.md Pitfall 2 Prevention section"
    investigation_needed: "Test both options with daltonize under all three CVD types; choose based on maximum contrast against the specific blue posterior fill used in Figure 1"
```
