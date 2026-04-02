# Phase 17: Enhanced Existing Plots - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Upgrade all existing plot modules to thesis-quality figures with proper uncertainty visualization, reference annotations, and consistent styling. This phase enhances `bayesian_plots.py`, `evaluation_plots.py`, `model_plots.py`, `physical_relations_plots.py`, `simulation_plots.py`, and `catalog_plots.py` — it does NOT add new plot types (Phase 18) or batch generation (Phase 19).

Requirements: CORE-01, CORE-02, CORE-03, CORE-04, CORE-05, CORE-06, CORE-07, FISH-06, FISH-07

</domain>

<decisions>
## Implementation Decisions

### H0 Posterior (CORE-01, CORE-02)
- **D-01:** 68%/95% credible intervals shown as shaded bands (darker for 68%, lighter for 95%) PLUS thin boundary lines at interval edges for precise reading.
- **D-02:** Planck (67.4 +/- 0.5) and SH0ES (73.0 +/- 1.0) shown as labeled vertical bands with 1-sigma shading. Text labels inline on the plot.
- **D-03:** Individual event posteriors use a `color_by` parameter accepting three modes: `'snr'`, `'redshift'`, `'dl_error'` (fractional luminosity distance error). Combined posterior always rendered as a thick black line on top.
- **D-04:** Single factory function `plot_event_posteriors(h_values, posteriors, color_by='snr', color_values=..., ...)` — caller passes the color values array. Colorbar label updates automatically based on `color_by` selection.
- **D-05:** Posterior normalization: default peak-normalized (peak=1), optional `normalize='density'` parameter for proper probability density (integral=1).

### SNR Distribution (CORE-03)
- **D-06:** Histogram on left y-axis with cumulative fraction (CDF) as step function on right y-axis. SNR threshold as vertical dashed line with annotation showing fraction of events above threshold.

### Detection Yield vs Redshift (CORE-04)
- **D-07:** Injected population as outline histogram, detected population as filled histogram inside it. Detection fraction curve on right y-axis. Shows both absolute counts and efficiency in one figure.

### Detection Probability Heatmap (CORE-05, FISH-06)
- **D-08:** Two separate factory functions for both coordinate spaces: P_det(z, M) and P_det(d_L, M). Both include colorbar spanning [0,1], detection contour lines at 0.5 and 0.9. Injected population overlay as scatter (detected=filled, missed=open circles).

### LISA PSD (CORE-06)
- **D-09:** Three curves on log-log axes: S_inst(f) as dashed, S_gal(f) as dash-dot, total S_n(f) as solid thick line. Standard LISA frequency range. Matches Phase 9 confusion noise decomposition.

### Luminosity Distance (CORE-07)
- **D-10:** Factory accepts a configurable list of H0 values. Default includes Planck (67.4), SH0ES (73.0), and simulation true value (73.0). Each curve labeled in legend with H0 value.

### Injected vs Recovered Scatter (FISH-07)
- **D-11:** Multi-panel grid (2x3 or 3x3) with configurable parameter list. Default key parameters: M, mu, d_L, a, e0, sky angles. Identity line + scatter points + 1-sigma CRB error bars on each panel.
- **D-12:** Residual sub-panels below each scatter (like HEP ratio plots). Shows (recovered - injected) to reveal systematic bias.

### Claude's Discretion
- Exact alpha levels for credible interval shading
- Colormap choice for event posterior color mapping (viridis or similar sequential)
- Histogram bin counts and edge styling
- Aspect ratios for multi-panel figures
- Residual sub-panel height ratio relative to main scatter
- Which specific sky angle parameters to include in default grid
- Line widths and dash patterns for PSD curves
- Error bar cap styling on recovery scatter

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing plot modules (source files to upgrade)
- `master_thesis_code/plotting/bayesian_plots.py` — `plot_combined_posterior()`, `plot_event_posteriors()`, `plot_subset_posteriors()`, `plot_detection_redshift_distribution()`, `plot_number_of_possible_hosts()`
- `master_thesis_code/plotting/evaluation_plots.py` — `plot_mean_cramer_rao_bounds()`, `plot_uncertainty_violins()`, `plot_sky_localization_3d()`, `plot_detection_contour()`, `plot_generation_time_histogram()`
- `master_thesis_code/plotting/model_plots.py` — `plot_emri_distribution()`, `plot_emri_rate()`, `plot_emri_sampling()`, `plot_detection_probability_grid()`
- `master_thesis_code/plotting/physical_relations_plots.py` — `plot_distance_redshift()`
- `master_thesis_code/plotting/simulation_plots.py` — `plot_gpu_usage()`, `plot_lisa_psd()`, `plot_lisa_noise_components()`, `plot_cramer_rao_coverage()`
- `master_thesis_code/plotting/catalog_plots.py` — `plot_bh_mass_distribution()`, `plot_redshift_distribution()`, `plot_glade_completeness()`, `plot_comoving_volume_sampling()`

### Style infrastructure (Phase 15/16, in place)
- `master_thesis_code/plotting/_helpers.py` — `get_figure(preset=...)`, `save_figure()`, `make_colorbar()`, `_fig_from_ax()`
- `master_thesis_code/plotting/_colors.py` — `TRUTH`, `MEAN`, `EDGE`, `REFERENCE`, `CYCLE`, `CMAP`
- `master_thesis_code/plotting/_labels.py` — `LABELS` dict with LaTeX labels for all 14 EMRI parameters + observables
- `master_thesis_code/plotting/_style.py` — `apply_style()` with `use_latex` toggle

### Data layer (Phase 16, in place)
- `master_thesis_code/plotting/_data.py` — CRB CSV loading, covariance reconstruction, parameter grouping

### LISA PSD (for confusion noise decomposition)
- `master_thesis_code/LISA_configuration.py` — `power_spectral_density()`, galactic confusion noise functions
- `master_thesis_code/constants.py` — physical constants, `SNR_THRESHOLD`, `H` (true H0)

### Existing smoke tests (must stay green)
- `master_thesis_code_test/plotting/test_bayesian_plots.py`
- `master_thesis_code_test/plotting/test_evaluation_plots.py`
- `master_thesis_code_test/plotting/test_model_plots.py`
- `master_thesis_code_test/plotting/test_physical_relations_plots.py`
- `master_thesis_code_test/plotting/test_simulation_plots.py`
- `master_thesis_code_test/plotting/test_catalog_plots.py`

### Requirements
- `.planning/REQUIREMENTS.md` — CORE-01 through CORE-07, FISH-06, FISH-07

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_labels.py:LABELS` — LaTeX labels for all 14 parameters + observables; use for axis labels and colorbars
- `_colors.py:TRUTH` (green), `MEAN` (red), `REFERENCE` (gray) — for reference lines (Planck, SH0ES, threshold)
- `_colors.py:CYCLE` — 8-color cycle for multi-line plots
- `_colors.py:CMAP` — default viridis colormap for heatmaps and color mapping
- `get_figure(preset="single"|"double")` — REVTeX sizing for all upgraded figures
- `make_colorbar()` — standardized colorbar for heatmaps and color-mapped scatter
- `_data.py` — CRB data loading and covariance reconstruction for recovery scatter plots

### Established Patterns
- All plot factories follow `data in, (fig, ax) out` pattern — must be preserved
- `_fig_from_ax()` for extracting Figure from Axes
- Smoke tests assert `(Figure, Axes)` return — signature changes must update tests

### Integration Points
- Existing factory signatures will gain new optional parameters (e.g., `color_by`, `normalize`, `h0_values`)
- New factory functions may be added for new variants (e.g., two P_det heatmap functions)
- All upgrades must use `_colors.py` semantic colors instead of ad-hoc color strings
- All axis labels must use `_labels.py` LaTeX labels

</code_context>

<specifics>
## Specific Ideas

- Event posterior `color_by` supports three modes: SNR, redshift, fractional d_L error — all from the same factory function
- P_det heatmaps in two coordinate spaces (z,M) and (d_L,M) as separate factories
- d_L(z) comparison takes a configurable H0 list, defaulting to Planck + SH0ES + simulation true value
- Recovery scatter uses HEP-style residual sub-panels below each parameter scatter
- Credible intervals use shaded bands + boundary lines (both, not just one)
- Planck/SH0ES reference bands include 1-sigma uncertainty shading with inline text labels

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 17-enhanced-existing-plots*
*Context gathered: 2026-04-02*
