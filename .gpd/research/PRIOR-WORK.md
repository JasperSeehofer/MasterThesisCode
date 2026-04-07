# Prior Work: Visualization in Dark Siren H0 Inference Papers

**Project:** EMRI Dark Siren H0 Inference -- Publication Figures
**Physics Domain:** Gravitational wave cosmology, dark siren methodology, EMRI parameter estimation
**Researched:** 2026-04-07

## Theoretical Framework

This document surveys figure types and visualization approaches used in state-of-the-art dark siren H0 inference papers, classifying each as standard (expected by referees), recommended (adds value), or novel (differentiating for an EMRI paper).

### Key Reference Papers

| Paper | arXiv | Relevance | Figure Focus |
|-------|-------|-----------|--------------|
| Gray et al. (2020) | 1908.06050 | Foundational dark siren mock data challenge | H0 posterior, per-event likelihoods, convergence |
| Finke et al. (2021) | 2101.12660 | GWTC-2 dark siren H0 with GLADE | Galaxy weighting, completeness, systematics |
| Laghi et al. (2021) | 2102.01708 | First EMRI dark siren cosmology with LISA | Corner plots, convergence vs N_det, sky localization |
| LVK GWTC-3 (2021) | 2111.03604 | Official LVK H0 constraints from O3 | Combined posterior, per-event contributions, sky maps |
| Borghi et al. (2024) | 2404.16092 | O4a dark siren H0 update | Sky maps with galaxy overlays, posterior stacking |
| Gray et al. (2022) | 2212.08694 | "Hitchhiker's Guide" to galaxy catalog method | Pedagogical figures: completeness, weighting, method |
| Laghi et al. (2026) | 2603.23612 | Joint EMRI+MBHB LISA cosmology | 2D cosmological contours, combined constraints |

**Confidence: HIGH** -- These are the canonical references; every dark siren paper cites Gray et al. (2020) and the LVK papers.

## Standard Figure Types in Dark Siren Papers

### Category 1: H0 Posterior Distribution (MANDATORY)

**What it is:** Combined posterior p(H0 | data) plotted against H0 (or h), with reference values marked.

**Standard elements:**
- Peak-normalized or density-normalized posterior curve
- Vertical lines or bands for Planck (h = 0.674 +/- 0.005) and SH0ES (h = 0.73 +/- 0.01)
- Shaded 68% and 95% credible intervals
- Injected/true value marked (for simulations)

**How leading papers do it:**
- Gray et al. (2020): Single panel, multiple curves for different analysis variants, shaded CI bands
- LVK GWTC-3 (2111.03604): Combined posterior with individual event contributions shown as thin transparent lines underneath, truth line
- Finke et al. (2021): Multiple curves comparing different galaxy catalog treatments on same axes
- Laghi et al. (2021): Corner-style 1D marginalized posterior for H0, often alongside w0 in a 2D corner

**Your existing implementation:** `plot_combined_posterior()` in `bayesian_plots.py` and `plot_h0_posterior_comparison()` in `paper_figures.py`. Already includes Planck/SH0ES bands, CI shading, and two-variant comparison (with/without M_z). Uses discrete markers ("o-", "s--") to honestly show grid resolution -- this is good practice for a 23-point grid.

**Assessment: IMPLEMENTED but needs refinement.** The existing figure is functional. For publication:
- Consider adding a flat/uninformative prior line to show that the posterior is informative
- The 23-point h-grid with markers is honest; add a note in the caption about grid resolution
- Ensure the x-axis range (currently 0.59-0.87) captures the full posterior support

**Status in your paper: Already Figure 1. Refine, do not redesign.**

### Category 2: Individual Event Likelihoods / Waterfall Plot (STANDARD)

**What it is:** Per-event likelihood curves overlaid, showing how individual events contribute to the combined posterior.

**How leading papers do it:**
- LVK papers: Hundreds of thin, semi-transparent lines (alpha ~0.1-0.3) in a single color, with the combined posterior as a thick line on top
- Gray et al. (2020): Selected representative events shown in separate panels or as colored lines
- Borghi et al. (2024): Individual contributions colored by metadata (SNR, redshift, distance error)

**Your existing implementation:** `plot_event_posteriors()` in `bayesian_plots.py` supports all three variants: uniform color, color-by-metadata (SNR/z/d_L error), and combined overlay. `plot_single_event_likelihoods()` in `paper_figures.py` shows 4 representative events in a 4x2 grid (without/with M_z).

**Assessment: IMPLEMENTED, two complementary versions.** The 4x2 representative grid is good for showing the effect of the BH mass channel. The waterfall plot (all events, semi-transparent) is standard and should also appear, either in the main text or supplementary material.

**Status in your paper: Already Figure 2 (representative events). Add a waterfall/spaghetti plot as supplementary or Figure 5.**

### Category 3: Posterior Convergence vs Number of Events (STANDARD for forecasts)

**What it is:** CI width (or relative H0 uncertainty) as a function of N_det, demonstrating the statistical power of the method and the expected sqrt(N) scaling.

**How leading papers do it:**
- Laghi et al. (2021): sigma(H0)/H0 vs N_det on log-log axes, with N^{-1/2} reference line
- Gray et al. (2020): Similar, showing convergence for different detection scenarios
- Typically shows median and 16th/84th percentile scatter bands from bootstrap resampling

**Your existing implementation:** `plot_posterior_convergence()` in `paper_figures.py`. Uses 50 random subsets at each of 9 subset sizes (10-500), log-log axes, N^{-1/2} reference line, error bars from 16th/84th percentiles. Only for without-BH-mass channel (correct: with-BH-mass collapses on coarse grid).

**Assessment: IMPLEMENTED and well-designed.** The bootstrap approach with scatter bands is exactly what the field expects. The deliberate omission of the with-BH-mass channel (explained in docstring) is scientifically appropriate.

**Status in your paper: Already Figure 3. Minor polish only.**

### Category 4: SNR Distribution and Detection Properties (STANDARD)

**What it is:** Histogram of detected-event SNR, often paired with SNR vs distance or SNR vs redshift scatter.

**How leading papers do it:**
- Two-panel: left = SNR histogram with threshold line, right = SNR vs d_L colored by redshift
- Sometimes includes a cumulative distribution overlay
- Laghi et al. (2021): SNR histogram for different EMRI population models

**Your existing implementation:** `plot_snr_distribution()` in `paper_figures.py`. Two-panel layout: SNR histogram + SNR vs d_L scatter colored by redshift. Falls back to placeholder if CRB data unavailable.

**Assessment: IMPLEMENTED.** Check that CRB CSV data is available locally before final figure generation.

**Status in your paper: Already Figure 4. Ensure data is copied from cluster.**

## Recommended Additional Figure Types

### Category 5: Detection Probability P_det Visualization (RECOMMENDED)

**What it is:** 2D heatmap or contour of P_det(z, M) or P_det(z, M_z), showing the selection function used in the inference.

**How leading papers do it:**
- Typically a filled contour or heatmap in (redshift, mass) space
- Color represents detection probability from 0 to 1
- Contour lines at P_det = 0.1, 0.5, 0.9
- Sometimes overlaid with detected event positions as scatter points

**Relevance to your work:** Your pipeline uses `SimulationDetectionProbability` (injection-based P_det with `RegularGridInterpolator`). Visualizing this selection function is critical for the paper because:
1. It demonstrates that your selection effects correction is well-behaved
2. It shows the (z, M) parameter space your analysis covers
3. Referees will want to see this -- it is the most common source of systematic error in dark siren analyses

**Your existing implementation:** `plot_detection_contour()` in `evaluation_plots.py` makes a 2D histogram of detections, but this is NOT the same as a P_det surface. You need a new figure that plots the actual P_det interpolation surface.

**Assessment: NOT YET IMPLEMENTED as a P_det surface. High priority.**

**Recommended approach:**
- Plot P_det(z, M) as a `pcolormesh` heatmap with logarithmic mass axis
- Overlay detected events as scatter points
- Add contour lines at P_det = {0.1, 0.5, 0.9}
- Use diverging or sequential colormap (viridis is fine)

### Category 6: Galaxy Catalog Completeness Visualization (RECOMMENDED)

**What it is:** Shows the galaxy catalog coverage and completeness as a function of redshift or apparent magnitude.

**How leading papers do it:**
- Apparent magnitude vs redshift scatter/density plot with completeness limit marked
- Number of galaxies per event vs redshift
- Completeness fraction vs redshift (from comparison with uniform-in-comoving-volume expectation)
- Gray et al. (2020), Finke et al. (2021): magnitude-redshift diagrams with survey depth overlaid

**Relevance to your work:** You use the GLADE+ catalog with BallTree lookups and a completeness correction. Showing the catalog's redshift-dependent completeness justifies your correction and helps readers understand at what redshifts the method becomes catalog-limited.

**Your existing implementation:** `plot_number_of_possible_hosts()` in `bayesian_plots.py` shows a histogram of host counts per event. This is useful but not sufficient.

**Assessment: PARTIALLY IMPLEMENTED.** The host count histogram exists. Additional figures needed:
- Number of candidate hosts vs event redshift (scatter plot)
- Completeness fraction vs redshift

### Category 7: Sky Localization / Localization Volume (CONTEXT-DEPENDENT)

**What it is:** Mollweide or orthographic projection showing sky position uncertainties of detected events, optionally overlaid with galaxy catalog positions.

**How leading papers do it:**
- LVK papers: Mollweide sky maps with 90% credible regions for each event, galaxies as dots colored by redshift
- Borghi et al. (2024): Sky map with galaxy positions and catalog completeness overlaid
- EMRI papers (Laghi et al. 2021): Typically quote sky localization in square degrees rather than showing full sky maps (EMRI localization is much better than BBH)

**Relevance to your work:** EMRI sky localization with LISA is typically sub-degree to few-degree, much better than LIGO BBH. A sky map would be less dramatic than for BBH events but could still illustrate the EMRI advantage.

**Assessment: LOW PRIORITY for main text.** EMRI sky localization is well-known to be excellent. A sky map could be informative in supplementary material but is not essential for the main paper. Your 3D sky localization plot (`plot_sky_localization_3d`) is a thesis figure, not publication-ready.

### Category 8: 2D Cosmological Parameter Contours (STANDARD for extended models)

**What it is:** Corner plot or 2D contour in (H0, w0) or (H0, Omega_m) parameter space.

**How leading papers do it:**
- Laghi et al. (2021, 2026): 2D Fisher ellipses or MCMC contours for H0 vs w0
- LVK papers: Focus on 1D H0 posterior (since they fix other cosmological parameters)
- Standard: 68% and 95% contour levels, with fiducial/true values marked

**Relevance to your work:** Your pipeline currently infers H0 only (grid-based), with other cosmological parameters fixed. If you only present H0 results, this figure is not applicable. If you extend to joint (H0, w0) or discuss future extensions, a forecast contour using Fisher matrix projections would be valuable.

**Assessment: NOT APPLICABLE to current analysis.** Your inference is 1D (h only). Mention as future work. If you want to include a Fisher forecast figure, you could project CRB results onto the (H0, w0) plane, but this would require additional analysis code.

## Novel / Differentiating Figure Types

### Category 9: Per-Galaxy Likelihood Decomposition (NOVEL for EMRI papers)

**What it is:** For a single representative event, show the per-galaxy likelihood contributions that sum to the event likelihood.

**How leading papers do it:**
- Gray et al. (2022, Hitchhiker's Guide): Pedagogical version showing how galaxy weights combine
- Finke et al. (2021): Galaxy weighting systematics comparison
- NOT commonly shown for individual events in the EMRI literature

**Relevance to your work:** Your with-BH-mass JSON files contain per-galaxy breakdowns (files are ~585 MB because of this). You have up to 22k candidate galaxies per event. This is a unique dataset that most papers do not have or do not show.

**Recommended figure:**
- For one well-localized event: horizontal bars showing per-galaxy likelihood contribution, colored by galaxy redshift
- Or: scatter plot of galaxy d_L vs likelihood weight, showing which galaxies dominate
- Demonstrates how the BH mass channel narrows the effective number of host candidates

**Assessment: HIGH NOVELTY, HIGH IMPACT.** This directly illustrates the physics of how the BH mass information breaks the d_L-z degeneracy. No EMRI dark siren paper has shown this decomposition.

### Category 10: Channel Comparison: Information Gain from BH Mass (NOVEL)

**What it is:** Direct visualization of how much information the BH mass channel adds, beyond just comparing posterior widths.

**Possible approaches:**
- KL divergence between per-event with-mass and without-mass likelihoods, plotted vs event properties
- Effective number of hosts (entropy-based) with vs without mass, showing the reduction
- Ratio of CI widths (with-mass / without-mass) for each event

**Relevance to your work:** Your two-channel analysis (with/without M_z) is the central scientific contribution. A figure that quantifies the per-event information gain would be powerful.

**Assessment: HIGH NOVELTY.** Requires modest additional computation from existing per-event data.

### Category 11: Detection Efficiency / Injection Campaign Summary (RECOMMENDED)

**What it is:** Summary of the injection campaign showing injected vs detected events in parameter space.

**How leading papers do it:**
- LVK papers (injection papers): 2D scatter of injected events colored by found/missed
- P_det contour overlaid with injection results for validation

**Relevance to your work:** You ran a large injection campaign. Showing the parameter space coverage validates the P_det estimate.

**Assessment: MODERATE PRIORITY.** Good for supplementary material or a methods section figure.

## Visualization Style Standards

### Color Conventions in the Field

| Element | Convention | Your Current |
|---------|-----------|-------------|
| Primary posterior | Blue (muted) | CYCLE[0] = #1f77b4 (blue) -- matches |
| Secondary/comparison | Red or orange | CYCLE[3] = #d62728 (red) -- matches |
| Truth/injected value | Green dashed | TRUTH = #2ca02c (green) -- matches |
| Planck band | Purple or teal | CYCLE[4] = #9467bd (purple) -- matches |
| SH0ES band | Orange or gold | CYCLE[1] = #ff7f0e (orange) -- matches |
| Reference/grid lines | Gray | REFERENCE = #7f7f7f -- matches |

**Assessment:** Your color palette is already consistent with field conventions. The tab10-based cycle is standard in astronomy.

### Figure Size and Layout

| Journal | Single column | Double column |
|---------|--------------|---------------|
| Physical Review D | 3.375 in | 6.75 in |
| MNRAS | 3.32 in | 6.97 in |
| ApJ (AAS) | 3.5 in | 7.1 in |

**Your presets:** `single = 3.375 in`, `double = 7.0 in` -- these target REVTeX (Physical Review D). Appropriate for a GW cosmology paper.

### Font and Rendering

- `text.usetex: False` in style sheet, but `apply_style(use_latex=True)` enables LaTeX for publication
- Font sizes (10pt body, 9pt ticks/legend) match REVTeX conventions
- **Recommendation:** Use `use_latex=True` for all final paper figures

### Rasterization

- Scatter plots with many points should use `rasterized=True` (you already do this in `plot_injected_vs_recovered`)
- PDF + rasterized scatter is standard for GW papers with O(1000) events

## What Existing Papers Do NOT Show (Gaps/Opportunities)

1. **Per-galaxy likelihood decomposition** -- papers aggregate, never show the galaxy-level structure
2. **Channel comparison at the event level** -- papers compare combined posteriors, not per-event information gain
3. **P_det validation against injection truth** -- injection papers exist but are usually separate from inference papers
4. **Effective number of hosts** -- quoted in text, rarely visualized as a function of event properties
5. **Grid resolution effects** -- your honest markers on the 23-point grid are more transparent than most papers, which interpolate onto smooth curves

## Recommended Paper Figure Set

Based on the literature survey, here is the recommended figure set:

### Main Text Figures (5-7 figures)

| # | Figure | Type | Status | Priority |
|---|--------|------|--------|----------|
| 1 | H0 posterior comparison (without/with M_z) | Standard | Implemented | Polish |
| 2 | Single-event likelihoods (4 representative, 2 channels) | Standard | Implemented | Polish |
| 3 | Posterior convergence vs N_det | Standard | Implemented | Polish |
| 4 | SNR distribution + SNR vs d_L | Standard | Implemented | Data needed |
| 5 | P_det(z, M) surface with detected events overlaid | Recommended | NOT implemented | **New** |
| 6 | Per-galaxy likelihood decomposition (1-2 events) | Novel | NOT implemented | **New** |
| 7 | Channel information gain (effective N_hosts or KL div) | Novel | NOT implemented | **New** |

### Supplementary / Appendix Figures

| # | Figure | Type | Status |
|---|--------|------|--------|
| S1 | All-event waterfall plot (spaghetti) | Standard | Partially implemented |
| S2 | Number of hosts vs redshift | Recommended | Partially implemented |
| S3 | Injection campaign summary (injected vs detected) | Recommended | NOT implemented |
| S4 | Fisher matrix / CRB heatmap | Context | Implemented |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Posterior plot style | Discrete markers on 23-pt grid | Smooth interpolated curve | Dishonest: hides grid resolution; 23 points is too few for smooth interpolation |
| Color scheme | tab10 subset (field standard) | Custom perceptually uniform | Unnecessary: tab10 is recognized by reviewers; custom adds friction |
| P_det visualization | 2D heatmap (pcolormesh) | 3D surface plot | 3D surfaces are harder to read and do not reproduce well in print |
| Per-galaxy figure | Bar chart or scatter | Treemap or sunburst | Over-engineered for the information content |
| Corner plots | Not recommended for 1D inference | Full corner plot | Only 1 parameter (h); corner plot is meaningless for 1D |

## Key References

| Reference | arXiv/DOI | Type | Relevance |
|-----------|-----------|------|-----------|
| Gray et al. (2020) | arXiv:1908.06050 | Foundational paper | Dark siren method, mock data, figure conventions |
| Finke et al. (2021) | arXiv:2101.12660 | GWTC-2 analysis | Galaxy catalog systematics, weighting figures |
| Laghi et al. (2021) | arXiv:2102.01708 | EMRI dark siren forecast | EMRI-specific figures, convergence, Fisher contours |
| LVK (2021) | arXiv:2111.03604 | Official O3 H0 | Combined posterior, per-event contributions |
| Gray et al. (2022) | arXiv:2212.08694 | Pedagogical review | Hitchhiker's Guide: completeness, weighting methods |
| Borghi et al. (2024) | arXiv:2404.16092 | O4a H0 update | Sky maps with galaxies, posterior stacking |
| Laghi et al. (2026) | arXiv:2603.23612 | Joint EMRI+MBHB | 2D cosmological contours, latest LISA forecast |
| Mukherjee et al. (2022) | DOI:10.1093/mnras/stac366 | Completeness | Pixelated catalog completeness approach |

## Open Questions

- **Grid resolution disclosure:** The 23-point h-grid is coarser than typical MCMC posteriors in the field. Reviewers may ask for finer resolution. The markers-on-grid approach is honest but may look unusual. Consider either (a) running a finer grid evaluation, or (b) adding an explicit discussion in the methods section.
- **With-BH-mass delta function:** The with-BH-mass posterior collapses to a near-delta on the coarse grid. This needs careful presentation -- it could look like an artifact rather than a genuine narrowing. A finer grid would resolve this.
- **Per-galaxy data size:** The 585 MB per-h-value JSON files for the with-BH-mass channel contain the raw per-galaxy data needed for Figure 6. Extracting this efficiently requires the tail-reading approach already implemented in `_load_per_event_with_mass_scalars()`, extended to read galaxy-level entries.
