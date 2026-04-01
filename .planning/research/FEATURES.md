# Feature Landscape: v1.3 Visualization Overhaul

**Domain:** Publication-quality visualizations for EMRI parameter estimation master thesis
**Researched:** 2026-04-01
**Overall confidence:** MEDIUM-HIGH (based on survey of published EMRI/LISA literature and existing pipeline data products)

## Table Stakes

Features that thesis examiners and GW community readers expect. Missing any of these makes the thesis feel incomplete or unprofessional.

| Feature | Why Expected | Complexity | Data Source | Notes |
|---------|--------------|------------|-------------|-------|
| **LISA sensitivity curve with confusion noise** | Every LISA paper includes this. Robson, Cornish & Liu (2019) arXiv:1803.01944 is the standard reference. Shows PSD components (S_OMS, S_TM, galactic confusion) and total sensitivity on log-log axes. | Low | `LISA_configuration.py` PSD functions | Already have `plot_lisa_psd` and `plot_lisa_noise_components` -- need to overlay confusion noise (implemented in Phase 9) and match Robson et al. convention (characteristic strain h_c vs ASD sqrt(S_n)) |
| **Characteristic strain plot with EMRI tracks** | Shows EMRI waveform tracks overlaid on LISA sensitivity curve. Standard in Babak et al. (2017), Moore, Cole & Berry (2015). Area between source track and noise curve visualizes SNR. | Medium | Waveform generator output + PSD | Requires computing h_c(f) from time-domain waveforms via FFT with running-average smoothing. The `eXtremeGravityInstitute/LISA_Sensitivity` GitHub repo provides reference implementations. |
| **SNR distribution histogram** | Basic output of any detection study. Shows how many events pass threshold, shape of SNR distribution. | Low | CRB CSV `SNR` column | Already have basic version. Needs: cumulative distribution overlay, SNR threshold line annotation. |
| **Detection yield vs redshift** | Standard in Babak et al. (2017), Gair et al. (2004). Shows how detection rate falls with distance. Essential for understanding LISA's EMRI horizon. | Low | Injection CSV `z`, `SNR` columns | Already have `plot_detection_redshift_distribution`. Needs: overlay of injected vs detected populations, detection fraction curve. |
| **Detection probability heatmap P_det(d_L, M)** | Standard selection function visualization. Shows where in parameter space LISA can detect EMRIs. Critical for explaining Bayesian inference selection bias correction. | Low | `SimulationDetectionProbability` grid | Already have `plot_detection_probability_grid`. Needs: Wilson CI contours from Phase 18, cleaner colorbar with probability [0,1] range. |
| **H0 posterior plot with credible intervals** | The central result of the thesis. Must show: combined posterior, injected value, shaded 68%/95% credible regions. Standard in dark siren papers (Schutz 1986, Gray et al. 2020, LIGO/Virgo/KAGRA O4a). | Low | Bayesian inference output | Already have `plot_combined_posterior`. Needs: `fill_between` for CI, Planck/SH0ES vertical bands, proper PDF normalization. |
| **Individual event posteriors (spaghetti plot)** | Shows contribution of each EMRI detection to the combined H0 measurement. Standard in dark siren analyses (Gray et al. 2020, arXiv:2404.16092). | Low | Per-event posterior data | Already have `plot_event_posteriors`. Needs: color coding by SNR or redshift, combined posterior highlighted. |
| **Parameter uncertainty distributions** | Histograms or violin plots of relative CRB uncertainties (sigma_X / X) for each EMRI parameter. Standard in Barack & Cutler (2004), Babak et al. (2017). | Low | CRB CSV delta columns | Already have `plot_uncertainty_violins`. Needs: parameter grouping (intrinsic vs extrinsic), proper LaTeX labels. |
| **Detection contour in (z, M) space** | 2D histogram or scatter showing where detected EMRIs live in redshift-mass space. Standard in population studies. | Low | CRB CSV `z`, `M` columns | Already have `plot_detection_contour`. Needs: injected population overlay, detection boundary curve. |
| **LaTeX rendering in all plots** | Thesis-quality figures require proper mathematical notation: $M_\bullet$, $\sigma_{d_L}/d_L$, $h = H_0/(100\,\mathrm{km\,s^{-1}\,Mpc^{-1}})$. | Low | Style sheet config | Currently `text.usetex: False` in mplstyle. Must enable or use mathtext consistently. |
| **Consistent figure sizing for thesis** | All figures must fit standard thesis column widths (single-column ~6.5in, double-column ~3.25in) with consistent font sizes readable at print scale. | Low | Style sheet | Current figsize varies wildly (10x6, 12x8, 12x9, 14x6, 16x9). Needs standardization to match LaTeX `\textwidth`. |
| **Proper axis labels with units** | Every axis must have physical units. Currently inconsistent: some use `"M_sun"`, some use `"M [M_sun]"`. | Low | All plot functions | Standardize to `"$M$ [$M_\odot$]"`, `"$f$ [Hz]"`, `"$d_L$ [Gpc]"` etc. |
| **PDF vector output** | All figures must be vector format for thesis embedding. | Already done | `save_figure()` | Already handles this. |
| **d_L(z) relation plot** | Luminosity distance vs redshift for the assumed cosmology. Basic context figure. | Low | `physical_relations.py` | Already have `plot_distance_redshift`. Needs: comparison curves for different H0 values. |

## Differentiators

Features that elevate the thesis from "adequate" to "impressive." Not strictly required but demonstrate deeper understanding and technical skill.

| Feature | Value Proposition | Complexity | Data Source | Notes |
|---------|-------------------|------------|-------------|-------|
| **Fisher matrix error ellipses (2D projections)** | Shows parameter correlations and degeneracies from the Fisher matrix. Standard in Barack & Cutler (2004), visually compelling for thesis defense. 2D confidence ellipses for key parameter pairs (M-mu, d_L-inclination, sky angles). | Medium | CRB covariance matrix (full matrix needed) | Currently only lower-triangular elements saved to CSV. Can reconstruct 2x2 sub-matrices from existing `delta_X_delta_Y` columns. Plot ellipses at 1-sigma, 2-sigma. |
| **Corner plot for EMRI parameters** | Multi-panel triangle plot showing all pairwise correlations and 1D marginals from Fisher matrix. Gold standard for PE visualization in modern papers. | Medium | Full covariance matrix | Use `corner.py` (dfm/corner) or `getdist` with Fisher-derived Gaussian approximation. Must clearly label as "Fisher approximation." |
| **Mollweide sky localization map** | Mollweide projection showing detected EMRI sky positions with error ellipses. Visually striking, standard in LIGO/Virgo papers. | Medium | CRB sky angle uncertainties (qS, phiS, delta_qS, delta_phiS) | Replace current `plot_sky_localization_3d` (non-standard, unreadable in print) with proper Mollweide. Use matplotlib `mollweide` projection. No need for full `ligo.skymap` (LIGO-specific). |
| **Injected vs recovered scatter** | Scatter of injected parameter value vs CRB uncertainty. Shows which regions of parameter space are well-constrained. E.g., sigma(d_L)/d_L vs z, sigma(M)/M vs M. | Low | Injection CSV + CRB CSV | Simple but informative. Shows selection effects and measurement quality across parameter space. |
| **H0 convergence plot** | Combined H0 posterior narrowing as N_events increases (N=1, 5, 10, 25, 50, ...). Shows statistical power scaling. | Medium | Re-run Bayesian inference with subsets | Already have `plot_subset_posteriors` concept. Needs systematic N-event subsampling with CI width vs N overlay. |
| **Parameter uncertainty scaling with SNR** | Verifies fundamental Fisher result: sigma(param) ~ 1/SNR. Scatter of relative uncertainty vs SNR. | Low | CRB CSV | Simple diagnostic but shows the pipeline produces physically consistent results. |
| **Fisher condition number vs parameter space** | Shows numerical stability across parameter space. Phase 10 added condition number logging. | Low | Run metadata/logs | Scatter plot of log10(cond) vs z or M. Unique diagnostic for thesis. |
| **Detection efficiency curve** | P_det as function of single variable (z or SNR) with Wilson CIs. Cleaner than 2D heatmap for showing detection horizon. | Low | Injection campaign data | 1D slice through detection probability grid with bootstrap/Wilson CIs from Phase 18. |
| **EMRI event rate model visualization** | Shows assumed dN/dz/dM. Contextualizes simulation inputs. | Low | `Model1CrossCheck` | Already have `plot_emri_distribution` and `plot_emri_rate`. Needs proper labels. |
| **Waveform strain time series** | Example EMRI waveform h(t) showing chirp and LISA orbital modulation. Good for introduction chapter. | Low-Medium | Single waveform generation | Show ~hours of signal with zoom inset. Requires one generation call (GPU or CPU). |
| **SNR vs redshift scatter with threshold** | Shows detection catalog in physical context. Horizontal line at SNR=20. | Low | CRB CSV or injection CSV | Common in EMRI papers. |
| **Multi-panel summary figure** | Single composite figure with 4-6 sub-panels summarizing key results. High impact for thesis defense. | Medium | All pipeline outputs | Requires `GridSpec` layout. Build after individual plots are finalized. |
| **GLADE catalog completeness** | Shows catalog completeness vs distance. Context for dark siren methodology. | Low | `GalaxyCatalogueHandler` | Already have `plot_glade_completeness`. |
| **Individual + combined posteriors overlay** | Shows how information accumulates with events. Pedagogically valuable. | Low | Bayesian output | Overlay `plot_event_posteriors` with `plot_combined_posterior` in one figure. |

## Anti-Features

Features to explicitly NOT build. Would waste time or add complexity without thesis value.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Interactive dashboards** (Plotly, Bokeh, Dash) | Thesis is PDF. Interactive plots cannot be embedded. Adds 50-200MB of dependencies for zero thesis value. | Static matplotlib figures exported as PDF vectors. |
| **Full MCMC corner plots** | Pipeline uses Fisher matrix, not MCMC. Generating fake MCMC chains from Fisher covariance is misleading. | Fisher ellipses and Fisher-derived corner plots with clear "Fisher approximation" labels. |
| **Animated waveform visualizations** | Cannot embed in thesis PDF. Time-consuming. Not standard in publications. | Single static waveform time series with zoom inset. |
| **3D surface plots** | `plot_sky_localization_3d` and `plot_cramer_rao_coverage` use 3D axes: hard to read in print, hide data behind surfaces, depend on viewing angle. | 2D: Mollweide for sky, 2D scatter/contour for parameter space. |
| **Real-time monitoring dashboard** | `PlottingCallback` already handles GPU usage. Full monitoring UI has no thesis value. | Keep existing callback. Post-hoc analysis plots are sufficient. |
| **Seaborn/Altair high-level wrappers** | Adds dependencies. Matplotlib is standard in GW community. Seaborn defaults clash with physics conventions. | Stay with matplotlib. `corner` library only exception (matplotlib-based). |
| **Figure captions in Python code** | Captions belong in LaTeX. Embedding them in matplotlib creates maintenance burden and coupling. | Minimal titles in figures. Full captions in thesis LaTeX source. |
| **Fake H0 likelihood comparison overlays** | Overlaying Planck/SH0ES likelihood curves requires careful treatment of priors and systematics. Easy to do misleadingly. | Simple vertical bands showing central values with 1-sigma uncertainties. No fake likelihood curves. |
| **Time-frequency spectrograms** | STFT spectrograms used in EMRI detection papers (time-frequency methods), not in Fisher-matrix PE studies. Adds complexity without relevance. | Waveform time series is sufficient for illustrative purposes. |
| **Multi-detector comparison** (LIGO/ET/LISA) | Out of thesis scope. | Focus on LISA-only sensitivity. |
| **seaborn-style statistical plots** | Not physics-standard. Risk of style inconsistency. | Raw matplotlib with thesis mplstyle. |

## Feature Dependencies

```
Style system (usetex, figsize, labels) --> ALL other features (must be first)

LISA PSD with confusion noise --> Characteristic strain plot with EMRI tracks
Injection campaign data --> Detection yield vs redshift
Injection campaign data --> Detection probability heatmap
Injection campaign data --> Injected vs recovered scatter
Injection campaign data --> Detection efficiency curve

CRB CSV covariance columns --> Fisher error ellipses (reconstruct from delta_X_delta_Y)
CRB CSV covariance columns --> Corner plot (Fisher-derived Gaussian)
corner library addition --> Corner plots

CRB CSV data --> Parameter uncertainty distributions
CRB CSV data --> Detection contour (z, M)
CRB CSV data --> SNR vs parameter scatter

Bayesian inference output --> H0 posterior plot
Bayesian inference output --> Individual event posteriors
Bayesian inference output --> H0 convergence plot
```

**Critical dependency note:** The full covariance matrix IS recoverable from the existing CRB CSV. The CSV contains all `delta_X_delta_Y` columns for the lower triangle. To get the full covariance matrix for a detection, read all `delta_{param_i}_delta_{param_j}` columns and reconstruct the symmetric matrix. No pipeline changes needed for Fisher ellipses/corner plots.

## MVP Recommendation

### Phase 1: Foundation (do first, everything depends on it)
1. **Style system upgrade** -- enable `text.usetex: True` (with graceful fallback), standardize figure sizes to match thesis `\textwidth`, consistent font sizes
2. **LaTeX parameter label mapping** -- dict from code names (`M`, `qS`, `phiS`, `d_L`) to LaTeX (`$M_\bullet$`, `$\theta_S$`, `$\phi_S$`, `$d_L$`)
3. **Systematic color palette** -- define consistent colors for injected/detected/threshold/true-value categories
4. **Infrastructure refactoring** -- move `_fig_from_ax` to `_helpers.py`, standardize all function signatures, remove ad-hoc figsize overrides

### Phase 2: Core result figures (the thesis hinges on these)
5. **H0 posterior with credible intervals** -- shaded 68%/95% regions, Planck/SH0ES vertical bands
6. **Individual event posteriors** -- color-coded by SNR or redshift, combined posterior highlighted
7. **LISA sensitivity curve** -- all PSD components + confusion noise overlay, matching Robson et al. convention
8. **Detection yield vs redshift** -- injected vs detected population overlay with detection fraction

### Phase 3: Parameter estimation showcase
9. **Fisher error ellipses** for key parameter pairs (reconstruct covariance from CRB CSV)
10. **Parameter uncertainty distributions** with grouping (intrinsic/extrinsic) and LaTeX labels
11. **Detection probability heatmap** with Wilson CIs from Phase 18
12. **Characteristic strain plot** with example EMRI track overlaid on sensitivity curve
13. **SNR vs redshift scatter** with threshold line

### Phase 4: Polish and differentiators
14. **Sky localization Mollweide** (replace 3D scatter)
15. **H0 convergence plot** (N-event scaling)
16. **Corner plot** of key EMRI parameter subsets (if time permits)
17. **Multi-panel summary figure**
18. **Injected vs recovered scatter**

### Defer
- **Waveform time series**: Nice for introduction chapter but low priority vs result figures
- **Fisher condition number**: Only if condition number analysis becomes a thesis section
- **Multi-panel summary**: Build after individual plots finalized

## Existing Code Assessment

### What works well
- Factory pattern (data in, fig/ax out) is clean, testable, and composable
- `emri_thesis.mplstyle` provides single source of truth for rcParams
- `save_figure()` helper handles format/path concerns
- `SimulationCallback` protocol decouples plotting from simulation loop
- Plot modules organized by topic (bayesian, evaluation, model, catalog, simulation)

### What needs improvement

| Issue | Current State | Target State | Severity |
|-------|--------------|--------------|----------|
| Figure sizes inconsistent | 10x6, 12x8, 12x9, 14x6, 16x9 across modules | Standard single/double column presets | HIGH |
| No LaTeX rendering | `text.usetex: False`, labels like `"M_sun"` | `text.usetex: True` or mathtext, `$M_\odot$` | HIGH |
| No credible intervals on H0 | Raw posterior curve only | Shaded 68%/95% regions with `fill_between` | HIGH |
| 3D plots non-standard | `plot_sky_localization_3d`, `plot_cramer_rao_coverage` | 2D Mollweide / 2D scatter | MEDIUM |
| No injected-vs-detected comparison | Detection plots show only detected events | Overlay injected population as background | MEDIUM |
| Missing confusion noise in PSD plot | `plot_lisa_psd` shows instrument noise only | Overlay galactic foreground component | MEDIUM |
| Color usage ad-hoc | "green", "red", "viridis" inconsistently | Systematic palette for data categories | MEDIUM |
| `_fig_from_ax` in wrong module | Defined in `simulation_plots.py`, imported by others | Move to `_helpers.py` | LOW |
| CRB heatmap hard to interpret | Raw covariance values as imshow | Correlation matrix (normalized) or 2D ellipses | LOW |

## Data Products Available for Visualization

| Data Product | Format | Key Columns/Fields | Produced By |
|-------------|--------|-------------------|-------------|
| CRB CSV | `cramer_rao_bounds_$index.csv` | 14 EMRI params + all `delta_X_delta_Y` covariance entries + T, dt, SNR, generation_time, host_galaxy_index | `ParameterEstimation.save_cramer_rao_bound()` |
| Injection CSV | `injection_results.csv` | z, M, phiS, qS, SNR, h_inj, luminosity_distance | `injection_campaign()` |
| SNR analysis CSV | `SNR_analysis.csv` | 14 params + T, SNR, generation_time | `ParameterEstimation.SNR_analysis()` |
| H0 posterior JSON | `simulations/posteriors/` | h_values, posterior array | `BayesianStatistics.evaluate()` |
| Detection probability grid | In-memory `RegularGridInterpolator` | d_L, M, P_det | `SimulationDetectionProbability` |
| Galaxy catalog | Reduced CSV | RA, Dec, z, stellar mass, BH mass estimate | `GalaxyCatalogueHandler` |
| Waveform | In-memory array | h(t) time series from `few`/`ResponseWrapper` | `ParameterEstimation.generate_lisa_response()` |
| PSD | In-memory array | S_n(f) from `LisaTdiConfiguration` | `power_spectral_density_a_channel()` |
| Run metadata | `run_metadata.json` | git_commit, seed, CLI args, SLURM info | `main.py` |

## Sources

- Robson, Cornish & Liu (2019), "The construction and use of LISA sensitivity curves," [arXiv:1803.01944](https://arxiv.org/abs/1803.01944)
- Babak et al. (2017), "Science with the space-based interferometer LISA. V. EMRIs," [arXiv:1703.09722](https://arxiv.org/abs/1703.09722)
- Barack & Cutler (2004), "LISA capture sources," [arXiv:gr-qc/0310125](https://arxiv.org/abs/gr-qc/0310125)
- Moore, Cole & Berry (2015), "Gravitational-wave sensitivity curves," [arXiv:1408.0740](https://arxiv.org/abs/1408.0740)
- Gray et al. (2020), "Cosmological inference using GW standard sirens," [arXiv:1908.06050](https://arxiv.org/abs/1908.06050)
- LIGO/Virgo/KAGRA O4a dark siren (2024), [arXiv:2404.16092](https://arxiv.org/abs/2404.16092)
- Babak et al. (2023), LISA PSD with confusion noise, [arXiv:2303.15929](https://arxiv.org/abs/2303.15929)
- [eXtremeGravityInstitute/LISA_Sensitivity](https://github.com/eXtremeGravityInstitute/LISA_Sensitivity) -- reference implementation
- [LISA Data Challenge](https://lisa-ldc.in2p3.fr/) -- community standards
- [dfm/corner](https://corner.readthedocs.io/) -- standard corner plot library
- [GetDist](https://getdist.readthedocs.io/) -- Fisher contour plots, [gallery](https://getdist.readthedocs.io/en/latest/plot_gallery.html)
- [ChainConsumer](https://samreay.github.io/ChainConsumer/) -- alternative corner plots
- [ligo.skymap](https://lscsoft.docs.ligo.org/ligo.skymap/) -- GW sky localization tools (reference, not dependency)
- [Recent EMRI PE](https://arxiv.org/html/2505.17814) -- 2025 EMRI PE visualization patterns
- [GW sensitivity curves blog](https://cplberry.com/2015/01/10/1408-0740/) -- characteristic strain conventions

---

*Feature analysis for v1.3 Visualization Overhaul: 2026-04-01*
