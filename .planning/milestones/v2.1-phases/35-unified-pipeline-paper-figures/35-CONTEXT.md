# Phase 35: Unified Figure Pipeline & Paper Figures - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Merge the two disconnected figure pipelines (`generate_figures()` 15-item manifest in `main.py` and `paper_figures.py` standalone 4-figure entry point) into a single unified manifest. Deliver polished, publication-quality paper figures with the Phase 29 style. Add a contour-smoothed H0 posterior variant. Consolidate duplicated CI calculation code.

</domain>

<decisions>
## Implementation Decisions

### Pipeline Merge Strategy
- **D-01:** Integrate `paper_figures.py` functions into the existing `generate_figures()` manifest in `main.py` as entries 16-19. Remove the standalone `main()` from `paper_figures.py`.
- **D-02:** Replace the hardcoded `_DATA_ROOT = Path("cluster_results/eval_corrected_full")` with the same `data_dir` parameter used by `generate_figures()`. All figures look in the same working directory passed via `--generate_figures <dir>`.
- **D-03:** Keep `--generate_interactive` as a separate flag. Interactive Plotly figures (HTML) remain decoupled from static PDF generation.

### Paper Figure Polish
- **D-04:** Full visual rework of all 4 paper figures. Redesign layouts, adjust spacing, add annotations, reconsider color choices. Publication-ready polish pass on each figure — not just style inheritance.
- **D-05:** Add a contour-smoothed H0 posterior variant using conservative Gaussian KDE (Scott's rule bandwidth). Must preserve MAP within one grid spacing of the discrete MAP.
- **D-06:** Auto-detect h-grid resolution from the data via `np.diff(h_values)`. No config parameter needed. Must work with 15-pt, 31-pt, and future finer grids.

### CI Calculation Consistency
- **D-07:** Extract the duplicated CDF/CI calculation into a shared `compute_credible_interval(h_values, posterior, level=0.68)` function in `_helpers.py`. Both `paper_figures.py` and `convergence_plots.py` call it.
- **D-08:** Unit test the shared CI function against two analytical distributions: (1) Gaussian where 68% CI = ±σ is known analytically, and (2) uniform distribution where 68% CI = 0.68 × range.

### Output Organization
- **D-09:** All figures (paper + thesis) go to `<dir>/figures/` in a single flat directory. Paper figures use a `paper_` prefix (e.g., `paper_h0_posterior.pdf`).

### Claude's Discretion
- Figure numbering scheme within the unified manifest (continue fig16-fig19 or renumber)
- Internal refactoring of data loaders in `paper_figures.py` to work with the manifest pattern
- Exact KDE bandwidth selection method details
- Specific visual rework choices (annotation text, spacing values, legend positioning) for each of the 4 paper figures

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Plotting Infrastructure
- `master_thesis_code/plotting/paper_figures.py` — The 4 paper figure functions to be integrated
- `master_thesis_code/plotting/_helpers.py` — `get_figure()`, `save_figure()`, figure presets
- `master_thesis_code/plotting/_style.py` — `apply_style()`, mplstyle loading
- `master_thesis_code/plotting/_colors.py` — Okabe-Ito CYCLE, TRUTH, REFERENCE, MEAN, EDGE colors
- `master_thesis_code/plotting/_labels.py` — Label constants for axis labels

### Existing Manifest Pipeline
- `master_thesis_code/main.py` lines 754-1072 — `generate_figures()` with 15-item manifest pattern
- `master_thesis_code/main.py` lines 1074+ — `generate_interactive_figures()` (separate, not touched)
- `master_thesis_code/arguments.py` — CLI flag definitions for `--generate_figures`, `--generate_interactive`

### CI Calculation Duplication
- `master_thesis_code/plotting/paper_figures.py` lines 397-419 — `_ci_width_from_log_posteriors()` with loop-based CDF
- `master_thesis_code/plotting/convergence_plots.py` — likely has similar CI calculation

### Data Sources
- `master_thesis_code/plotting/_data.py` — CRB data loading, `PARAMETER_NAMES`, `reconstruct_covariance`
- Cluster results in `cluster_results/eval_corrected_full/` — combined posteriors, per-event JSONs, CRB CSVs

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `get_figure(preset="single"|"double")` — REVTeX width presets already tuned for paper
- `save_figure(fig, path, formats=("pdf",))` — handles directory creation, multiple formats
- `apply_style()` — Phase 29 mplstyle with Okabe-Ito, Type 42 fonts, frameless legends
- `_load_combined_posterior()`, `_load_per_event_no_mass()`, `_load_per_event_with_mass_scalars()` — data loaders in `paper_figures.py` that need to accept `data_dir` parameter

### Established Patterns
- Manifest pattern: `list[tuple[str, Callable[[], tuple[object, object] | None]]]` — generator returns `(fig, ax)` or `None` if data missing
- Factory convention: data in, `(Figure, Axes)` out — all plotting functions follow this
- `_save()` helper in `generate_figures()` calls `save_figure()` + `_check_file_size()`

### Integration Points
- Paper figure functions integrate as manifest entries 16-19 in `generate_figures()`
- CI utility extracted to `_helpers.py` alongside `get_figure()` and `save_figure()`
- No new CLI flags needed — paper figures ride on existing `--generate_figures <dir>`

</code_context>

<specifics>
## Specific Ideas

- Conservative KDE for contour-smoothed posterior: Scott's rule bandwidth, MAP must stay within one grid spacing of discrete MAP
- Auto-detect grid resolution: `np.diff(h_values)` to infer spacing, no hardcoded grid size assumptions
- CI unit tests: Gaussian (68% CI = ±σ) and uniform (68% CI = 0.68 × range) as ground truth

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 35-unified-pipeline-paper-figures*
*Context gathered: 2026-04-08*
