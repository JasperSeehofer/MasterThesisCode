# Phase 18: New Plot Modules - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Add new plot factory functions: Mollweide sky localization map, Fisher-derived corner plots, H0 convergence plot, and detection efficiency curve. These are entirely new visualizations — not upgrades to existing plots (Phase 17) or batch generation (Phase 19).

Requirements: SKY-01, FISH-03, CONV-01, CONV-02

</domain>

<decisions>
## Implementation Decisions

### Mollweide Sky Localization (SKY-01)
- **D-01:** Replace `plot_sky_localization_3d` (non-standard, unreadable in print) with a proper Mollweide projection using matplotlib's built-in `mollweide` projection.
- **D-02:** Show detected EMRI sky positions as scatter points, with localization error ellipses computed from the Fisher matrix sky angle uncertainties (delta_qS, delta_phiS, delta_qS_delta_phiS cross-term).
- **D-03:** Color points by SNR using the project colormap (CMAP = viridis). Include colorbar.
- **D-04:** Input coordinates are ecliptic (qS = colatitude, phiS = longitude). Convert to Mollweide-compatible (latitude = pi/2 - qS, longitude = phiS - pi for [-pi, pi] range).

### Fisher Corner Plot (FISH-03)
- **D-05:** Use the `corner` library (dfm/corner) for the triangle plot. Must add `corner` as a dependency via `uv add corner`.
- **D-06:** Input is the 14x14 covariance matrix from CRB CSV (reconstructed via `_data.py:reconstruct_covariance`). Generate synthetic samples from the multivariate Gaussian N(param_values, covariance) for the `corner` library.
- **D-07:** Default parameter subset: M, mu, a, d_L, qS, phiS (6 params = 15 panels). Full 14-parameter corner deferred — too dense for print.
- **D-08:** Apply thesis styling to corner output: use `_labels.py` LaTeX labels, `_colors.py` palette, match font sizes to mplstyle. Label as "Fisher approximation" in plot or caption.
- **D-09:** Support both single-event and multi-event overlay modes.

### H0 Convergence Plot (CONV-01)
- **D-10:** Show how the combined H0 posterior narrows as N_events increases. Build on existing `plot_subset_posteriors` concept.
- **D-11:** Left panel: stacked posterior curves for N = 1, 5, 10, 25, 50, 100, ... (subset sizes). Right panel or inset: 68% credible interval width vs N_events, showing 1/sqrt(N) scaling expectation.
- **D-12:** Subsets are randomly sampled from the detection catalog. Accept a `seed` parameter for reproducibility.

### Detection Efficiency Curve (CONV-02)
- **D-13:** 1D P_det as a function of a single variable (redshift z or luminosity distance d_L) with confidence intervals.
- **D-14:** Use Wilson score interval or bootstrap CI for the binomial detection probability in each bin.
- **D-15:** Accept injection campaign data (injected + detected arrays) and produce the efficiency curve with shaded CI band.

### Claude's Discretion
- Number of synthetic samples for corner plot (1000-10000)
- Exact subset sizes for convergence plot
- Wilson vs bootstrap CI for detection efficiency
- Error ellipse scaling on Mollweide (may need angular size conversion)
- Whether convergence right-panel is a separate axes or inset
- Marker size and alpha for sky map scatter
- Corner plot diagonal: histogram or KDE

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Style infrastructure (Phase 15/16/17, in place)
- `master_thesis_code/plotting/_helpers.py` — `get_figure(preset=...)`, `save_figure()`, `make_colorbar()`, `_fig_from_ax()`
- `master_thesis_code/plotting/_colors.py` — `TRUTH`, `MEAN`, `EDGE`, `REFERENCE`, `CYCLE`, `CMAP`
- `master_thesis_code/plotting/_labels.py` — `LABELS` dict with LaTeX labels for all 14 EMRI parameters + observables
- `master_thesis_code/plotting/_style.py` — `apply_style()` with `use_latex` toggle

### Data layer (Phase 16, in place)
- `master_thesis_code/plotting/_data.py` — CRB CSV loading, `reconstruct_covariance()`, `PARAMETER_NAMES`, `INTRINSIC`, `EXTRINSIC`

### Fisher plots (Phase 16, in place)
- `master_thesis_code/plotting/fisher_plots.py` — `_ellipse_params()` helper (reuse for sky ellipses), `plot_fisher_ellipses()`

### Existing functions to replace or extend
- `master_thesis_code/plotting/evaluation_plots.py:71` — `plot_sky_localization_3d()` (replace with Mollweide)
- `master_thesis_code/plotting/bayesian_plots.py:293` — `plot_subset_posteriors()` (extend concept for convergence)

### Existing smoke tests (must stay green)
- `master_thesis_code_test/plotting/test_evaluation_plots.py`
- `master_thesis_code_test/plotting/test_bayesian_plots.py`

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_data.py:reconstruct_covariance()` — builds 14x14 covariance matrix from CRB CSV row
- `_data.py:PARAMETER_NAMES` — ordered list of 14 parameter names matching CSV columns
- `fisher_plots.py:_ellipse_params()` — computes (width, height, angle) from 2x2 covariance sub-matrix
- `_labels.py:LABELS` — LaTeX labels for all parameters (use for corner plot axis labels)
- `_colors.py:CMAP` — viridis colormap for scatter coloring
- `get_figure(preset="single"|"double")` — REVTeX sizing

### Established Patterns
- All plot factories: data in, (fig, ax) out — must be preserved
- Smoke tests assert `(Figure, Axes)` return type
- `_fig_from_ax()` for extracting Figure from Axes
- Deferred imports for modules with GPU dependencies (see `fisher_plots.py:plot_characteristic_strain`)

### Integration Points
- New factory functions go in `evaluation_plots.py` (sky map), new `convergence_plots.py` (convergence + efficiency), or `fisher_plots.py` (corner)
- `corner` library must be added as dependency
- Sky map function replaces `plot_sky_localization_3d` but the old function stays for backward compat (deprecated)

</code_context>

<specifics>
## Specific Ideas

- Mollweide sky map uses matplotlib's `projection='mollweide'` — no external GW sky map library needed
- Corner plot wraps `corner.corner()` with thesis styling applied via rcParams context manager
- H0 convergence uses cumulative event subsets (not random) as primary mode, with random subsampling as optional
- Detection efficiency uses binomial proportion CI (Wilson score) — simpler and more principled than bootstrap for binary outcomes
- Sky localization ellipses projected onto Mollweide using angular covariance from Fisher matrix

</specifics>

<deferred>
## Deferred Ideas

- Full 14-parameter corner plot (too dense for print — keep 6-param default)
- Animated convergence (cannot embed in thesis PDF)
- Multi-detector sky maps (LISA-only scope)

</deferred>

---

*Phase: 18-new-plot-modules*
*Context gathered: 2026-04-02*
