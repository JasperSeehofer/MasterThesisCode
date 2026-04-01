# Phase 16: Data Layer & Fisher Visualizations - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

CRB CSV data loading, covariance matrix reconstruction, and Fisher-based plot factory functions: error ellipses, characteristic strain h_c(f), and parameter uncertainty distributions. This phase creates the data layer (`_data.py`) and three new plot factory functions — it does NOT upgrade existing plots (Phase 17) or add corner plots (Phase 18).

</domain>

<decisions>
## Implementation Decisions

### Data Layer (_data.py) — FISH-01
- **D-01:** Row-level API: `reconstruct_covariance(row: pd.Series) -> np.ndarray` returns a symmetric 14x14 matrix from the `delta_X_delta_Y` columns. Callers iterate over the DataFrame.
- **D-02:** `PARAMETER_NAMES: list[str]` — ordered list `['M','mu','a','p0','e0','x0','luminosity_distance','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0']` as single source of truth for index-to-name mapping. Derived from `ParameterSpace._parameters_to_dict()` key order.
- **D-03:** Parameter grouping defined in `_data.py`:
  - `INTRINSIC = ['M','mu','a','p0','e0','x0']`
  - `EXTRINSIC = ['luminosity_distance','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0']`
  Reused by ellipse and uncertainty plot factories.

### Fisher Error Ellipses — FISH-02
- **D-04:** Three default parameter pairs: `(M, mu)`, `(luminosity_distance, qS)`, `(qS, phiS)` — covering mass, distance-inclination degeneracy, and sky localization.
- **D-05:** Filled contours with alpha transparency for 1-sigma and 2-sigma ellipses. Standard GW literature style.
- **D-06:** Factory supports both single-event and multi-event overlay via optional parameter. Single event by default; pass a list of (covariance, params) tuples for overlay mode with color cycle.

### Characteristic Strain — FISH-04
- **D-07:** Example EMRI waveform uses hardcoded reference parameters (representative event from CRB CSV, e.g., median-SNR). No waveform generation at plot time — precomputed h_c(f) curve embedded or loaded.
- **D-08:** Three noise curves on the sensitivity plot: instrument noise S_inst(f), galactic foreground S_gal(f), and their total sum S_n(f). Matches Phase 9's galactic confusion noise addition.

### Parameter Uncertainty Distributions — FISH-05
- **D-09:** Violin plot for multi-event mode showing distribution of fractional uncertainties (sigma_i / x_i) across events, grouped by intrinsic/extrinsic.
- **D-10:** Single-event fallback: horizontal bar chart of fractional uncertainties. Automatic switching based on input type.
- **D-11:** Factory supports both modes via optional parameter: single `pd.Series` for bar chart, `pd.DataFrame` for violin plot.

### Claude's Discretion
- Internal structure of `reconstruct_covariance()` (how it parses column names and fills the symmetric matrix)
- Exact default parameters for the hardcoded reference EMRI in h_c(f) plot
- Ellipse computation method (eigenvalue decomposition of 2x2 submatrix)
- Aspect ratios and subplot layout for multi-panel ellipse figures
- Violin plot styling details (width, kernel bandwidth, etc.)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### CRB CSV format (data layer source of truth)
- `master_thesis_code/parameter_estimation/parameter_estimation.py:410-418` — `independent_cramer_rao_bounds` dict construction: lower-triangle columns named `delta_{row}_delta_{col}`
- `master_thesis_code/datamodels/parameter_space.py:150-166` — `_parameters_to_dict()` defines the 14 parameter names and their order
- `evaluation/run_20260328_seed100_v3/simulations/cramer_rao_bounds.csv` — example CRB CSV for column inspection

### Style infrastructure (Phase 15, already in place)
- `master_thesis_code/plotting/_helpers.py` — `get_figure(preset=...)`, `save_figure()`, `make_colorbar()`, `_fig_from_ax()`
- `master_thesis_code/plotting/_colors.py` — `TRUTH`, `MEAN`, `EDGE`, `REFERENCE`, `CYCLE`, `CMAP`
- `master_thesis_code/plotting/_labels.py` — `LABELS` dict with LaTeX labels for all 14 EMRI parameters + observables
- `master_thesis_code/plotting/_style.py` — `apply_style()` with `use_latex` toggle

### LISA PSD (for characteristic strain plot)
- `master_thesis_code/LISA_configuration.py` — `power_spectral_density()`, galactic confusion noise functions
- `master_thesis_code/constants.py:77-83` — galactic confusion noise coefficients

### Requirements
- `.planning/REQUIREMENTS.md` — FISH-01, FISH-02, FISH-04, FISH-05

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_labels.py:LABELS` — all 14 parameter symbols already have LaTeX labels; use directly in axis labels and legends
- `_colors.py:CYCLE` — 8-color cycle for multi-event overlays
- `_colors.py:TRUTH`, `REFERENCE` — for reference lines on strain plot
- `get_figure(preset="single"|"double")` — REVTeX sizing for all new figures
- `make_colorbar()` — ready for any heatmap-style additions

### Established Patterns
- All plot factories follow `data in, (fig, ax) out` pattern
- `_fig_from_ax()` for extracting Figure from Axes
- Plot modules live in `master_thesis_code/plotting/`; tests mirror in `master_thesis_code_test/plotting/`

### Integration Points
- New `_data.py` module in `master_thesis_code/plotting/` (data loading utilities)
- New plot factory functions — likely in a new `fisher_plots.py` module or split across focused modules
- CRB CSV files produced by the simulation pipeline in `<working_dir>/simulations/cramer_rao_bounds.csv`
- `LISA_configuration.py` provides PSD functions for the characteristic strain plot

</code_context>

<specifics>
## Specific Ideas

- CRB CSV stores only the lower triangle (105 entries for 14x14). Reconstruction must fill the upper triangle by symmetry.
- The CSV also contains the 14 parameter values, T, dt, SNR, generation_time, and host_galaxy_index alongside the covariance entries.
- Characteristic strain plot decomposes noise into 3 curves (instrument, galactic, total) — matches the Phase 9 confusion noise work.
- For multi-event ellipse overlay, use `_colors.py:CYCLE` to distinguish events.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 16-data-layer-fisher*
*Context gathered: 2026-04-02*
