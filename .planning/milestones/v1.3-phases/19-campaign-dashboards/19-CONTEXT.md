# Phase 19: Campaign Dashboards & Batch Generation - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning

<domain>
## Phase Boundary

A single command produces all thesis figures from campaign data, with composite summary panels and size-optimized PDF output. This phase wires the existing plot factory functions (Phases 14-18) into a batch pipeline and adds a multi-panel dashboard composite — it does NOT create new plot types or modify existing factory function signatures.

Requirements: CAMP-01, CAMP-02, CAMP-03

</domain>

<decisions>
## Implementation Decisions

### Dashboard Layout (CAMP-01)
- **D-01:** Summary dashboard uses a 2x2 grid via `plt.subplot_mosaic` composing: H0 posterior (top-left), SNR distribution (top-right), detection yield (bottom-left), sky localization Mollweide (bottom-right).
- **D-02:** Dashboard is a factory function `plot_campaign_dashboard(...)` in a new `master_thesis_code/plotting/dashboard_plots.py` module, returning `(Figure, dict[str, Axes])` where keys match the mosaic labels.
- **D-03:** Dashboard uses `preset="double"` (7.0in width) for thesis double-column layout.

### Batch Generation Interface (CAMP-02)
- **D-04:** Implement `generate_figures()` in `main.py` (stub already exists at line 608) to iterate a figure manifest and produce all thesis figures from a campaign working directory.
- **D-05:** Entry point is `--generate_figures <working_dir>` CLI flag (already wired in `arguments.py`). No separate script — uses existing CLI infrastructure.
- **D-06:** Figure manifest is a Python list of `(factory_function, data_loader, output_name)` tuples defined in `generate_figures()`. Not a YAML config — keep it simple and type-checked.
- **D-07:** Each figure is saved via `save_figure(fig, path, formats=("pdf",))` to `<working_dir>/figures/`. Directory created automatically by `save_figure`.

### File Size Optimization (CAMP-03)
- **D-08:** Scatter plots with >1000 points use `rasterized=True` on the scatter call. This is set inside the factory functions where needed (sky_plots, evaluation_plots scatter calls).
- **D-09:** After saving each PDF, check `os.path.getsize()` and log a warning if >2MB. No auto-compression — just a warning for manual review.
- **D-10:** Default output format is PDF only. PNG can be added via `formats=("pdf", "png")` but is not the default.

### Figure Manifest (CAMP-02)
- **D-11:** Batch generation produces all thesis-relevant figures: H0 posteriors (combined + individual events), SNR distribution, detection yield, sky localization, Fisher ellipses (3 parameter pairs), corner plot, H0 convergence, detection efficiency, LISA PSD with noise decomposition, d_L(z) multi-H0, CRB coverage, uncertainty violins.
- **D-12:** Data loading uses existing CRB CSV files in working directory. Factory functions receive pre-loaded DataFrames/arrays — generate_figures handles the I/O.
- **D-13:** Figures that require data not present in the working directory are skipped with a log warning (graceful degradation — not all campaigns produce all data types).

### Claude's Discretion
- Figure ordering in manifest (aesthetic choice)
- Exact subplot_mosaic layout string
- Whether to add a figure numbering scheme (e.g., `fig01_h0_posterior.pdf`)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Plotting Infrastructure
- `master_thesis_code/plotting/_helpers.py` — `get_figure`, `save_figure`, `make_colorbar`, `_fig_from_ax` utilities
- `master_thesis_code/plotting/_colors.py` — Color palette constants (TRUTH, CYCLE, CMAP, etc.)
- `master_thesis_code/plotting/_labels.py` — LaTeX label dictionary
- `master_thesis_code/plotting/_data.py` — CRB data loading, covariance reconstruction, parameter name mapping
- `master_thesis_code/plotting/_style.py` — `apply_style()` and mplstyle loading

### Existing Factory Functions (to wire into batch)
- `master_thesis_code/plotting/bayesian_plots.py` — H0 posteriors, event posteriors
- `master_thesis_code/plotting/evaluation_plots.py` — CRB bounds, uncertainty violins, sky 3D, detection contour, injected-vs-recovered
- `master_thesis_code/plotting/simulation_plots.py` — LISA PSD, detection yield, CRB coverage
- `master_thesis_code/plotting/model_plots.py` — Detection probability grids, EMRI distribution
- `master_thesis_code/plotting/fisher_plots.py` — Fisher ellipses, characteristic strain, uncertainty distributions, corner plot
- `master_thesis_code/plotting/sky_plots.py` — Mollweide sky localization
- `master_thesis_code/plotting/convergence_plots.py` — H0 convergence, detection efficiency
- `master_thesis_code/plotting/catalog_plots.py` — BH mass distribution, redshift distribution
- `master_thesis_code/plotting/physical_relations_plots.py` — d_L(z) curves

### Entry Point
- `master_thesis_code/main.py` lines 608-619 — `generate_figures()` stub to implement
- `master_thesis_code/arguments.py` line 73 — `generate_figures` CLI property

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `save_figure()` in `_helpers.py`: handles directory creation, multi-format output, auto-close
- `get_figure(preset="double")`: thesis-width figure creation
- All 20+ factory functions follow `(data_in) -> (Figure, Axes)` convention
- `apply_style()` in `_style.py`: ensures consistent styling before any rendering

### Established Patterns
- Factory functions are pure: data in, (fig, ax) out — no file I/O inside
- `_fig_from_ax` extracts Figure from existing Axes
- `make_colorbar` handles colorbar attachment uniformly
- `SimulationCallback` protocol in `callbacks.py` — not needed for batch (batch is post-hoc)

### Integration Points
- `main.py:generate_figures()` is the entry point — already wired to `--generate_figures` CLI
- CRB CSV files in working directory are the primary data source
- Posterior JSON files from `--combine` are secondary data source
- `_data.py` provides `reconstruct_covariance()` and `PARAMETER_NAMES` for CRB processing

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches. Follow the existing factory function pattern and thesis styling conventions established in Phases 14-18.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 19-campaign-dashboards*
*Context gathered: 2026-04-02*
