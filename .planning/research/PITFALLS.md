# Domain Pitfalls: v1.3 Visualization Overhaul

**Domain:** Scientific Python visualization for gravitational wave parameter estimation thesis
**Researched:** 2026-04-01
**Overall confidence:** HIGH (based on direct codebase inspection, matplotlib documentation, community best practices)

## Critical Pitfalls

Mistakes that cause broken plots, wasted time, or thesis-quality regressions.

### Pitfall 1: Style System Global State Pollution

**What goes wrong:** Adding a library like seaborn that calls `sns.set_theme()` or `sns.set_context()` mutates matplotlib's global `rcParams`, overwriting the carefully tuned `emri_thesis.mplstyle` settings. Existing plots silently change appearance -- font sizes shift, grid lines appear or disappear, line widths change, constrained layout may be disabled. The thesis advisor sees different-looking plots in different chapters with no code change.

**Why it happens:** The project uses `apply_style()` which calls `matplotlib.style.use()` to load `emri_thesis.mplstyle` once at program entry. Seaborn's `set_theme()` writes directly to `matplotlib.rcParams`, overriding all previously set values. Even `import seaborn` in some versions runs `set_theme()` automatically. The session-scoped `_plotting_style` fixture in `conftest.py` calls `apply_style()` once, so any test that imports seaborn after that fixture runs will see polluted rcParams for all subsequent tests in the session.

**Consequences:** Visual inconsistency across thesis figures. Debugging is difficult because the pollution depends on import order and which plotting function runs first. Test results become non-deterministic based on test execution order.

**Prevention:**
1. Never call `sns.set_theme()` globally. Use seaborn only through its functional API with explicit `ax` parameters, letting the custom mplstyle control aesthetics.
2. If seaborn styling is needed for specific plots, use `plt.style.context()` as a context manager to temporarily override and then restore rcParams.
3. Add a regression test that asserts key rcParams (font.size=11, axes.titlesize=13, figure.constrained_layout.use=True) remain unchanged after importing and calling any new visualization function.
4. Re-call `apply_style()` defensively after any seaborn operation in production code.

**Detection:** Add a pytest fixture that snapshots `matplotlib.rcParams` before and after each test module, failing if any project-defined keys changed unexpectedly.

### Pitfall 2: LaTeX Font Rendering Breaks PDF Output

**What goes wrong:** Enabling `text.usetex: True` in the mplstyle (currently `False`) for publication-quality math labels requires a working LaTeX installation with specific packages (amsmath, type1cm, etc.) on every machine that generates plots. The dev machine, CI runner, and cluster all need identical LaTeX setups. Missing fonts produce blank text, fallback bitmaps, or `RuntimeError` crashes. Even with usetex=True working, mixing LaTeX-rendered and non-LaTeX text (e.g., legend entries vs axis labels) produces visually jarring font mismatches.

**Why it happens:** matplotlib shells out to `latex`, `dvipng`, and `ghostscript` when `text.usetex=True`. CI runners (GitHub Actions) and the bwUniCluster compute nodes may not have these installed. The current mplstyle has `text.usetex: False`, so switching it on is a cross-environment breaking change.

**Consequences:** CI fails on plot-generation tests. Cluster evaluation jobs crash when trying to save figures. Or worse: plots render with bitmap fonts that look terrible at any zoom level in the PDF thesis.

**Prevention:**
1. Keep `text.usetex: False` in the mplstyle for robustness. Instead, use matplotlib's built-in mathtext renderer (the default) which handles `$H_0$`, `$d_L$`, `$\sigma$` without any LaTeX dependency.
2. If true LaTeX rendering is required for specific publication figures, create a separate `emri_thesis_publication.mplstyle` that sets `text.usetex: True` and is only used in a dedicated publication-plot script, never in CI or cluster pipelines.
3. If enabling usetex, add `latex` and `dvipng` to the CI setup step, and verify availability on the cluster with a preflight check.
4. Use `matplotlib.checkdep_usetex(True)` programmatically to detect missing TeX before attempting to render.

**Detection:** A CI test that generates one figure with every text element type (title, xlabel, ylabel, legend, annotation, colorbar label) and verifies the PDF file is valid and non-empty.

### Pitfall 3: Over-Engineering Interactive Features for a Static PDF Thesis

**What goes wrong:** Developer adds plotly, bokeh, or holoviews for "interactive exploration" of parameter spaces and posteriors. These libraries add 50-200 MB of dependencies, require a running JavaScript runtime for output, produce HTML files instead of PDF-embeddable figures, and their static export (orca/kaleido) is fragile and version-sensitive. The thesis ultimately needs static PDFs. Six weeks of development produces beautiful dashboards that cannot be `\includegraphics{}`'d into LaTeX.

**Why it happens:** Interactive visualization is genuinely useful during development and exploration. The temptation is to build the exploration tool AND the thesis figure generator as one thing. But interactive and static visualization have fundamentally different output requirements.

**Consequences:** Dependency bloat (kaleido alone is 80+ MB). Build complexity. Two rendering paths to maintain. The interactive features are abandoned after the defense because the thesis PDF is all that persists.

**Prevention:**
1. Commit to matplotlib as the sole output renderer for all thesis figures. Period.
2. If interactive exploration is needed during development, use Jupyter notebooks with matplotlib's `%matplotlib widget` backend -- this provides pan/zoom/hover without adding any new library.
3. If a corner-plot or specific statistical visualization is needed, use libraries that build ON matplotlib (corner, arviz, chainconsumer) rather than libraries that replace it.
4. Any library added must support `fig.savefig("output.pdf")` natively. If it cannot produce a vector PDF, it does not belong in the thesis plotting stack.

**Detection:** Review `pyproject.toml` before any dependency addition. If the library's documentation mentions "interactive" more than "publication," it is the wrong tool for this project.

### Pitfall 4: Existing Plot Functions Break Silently During Refactoring

**What goes wrong:** The existing plotting module has 15+ factory functions across 6 files, all returning `(fig, ax)` tuples. During the visualization overhaul, someone refactors the style, changes default figsize, modifies colormap, or restructures the module. An existing function that worked correctly now produces subtly different output -- the log-scale axis on `plot_uncertainty_violins` clips data, the `plot_detection_contour` colorbar range changes, or `plot_combined_posterior` normalization produces a flat line because the y-axis range shifted.

**Why it happens:** There are no regression tests that verify plot content. The existing `test_style.py` tests verify that `save_figure` produces a file and that `apply_style` sets the backend, but no test checks that `plot_combined_posterior` actually draws a line at the correct position or that `plot_mean_cramer_rao_bounds` heatmap has the right value range. Changes to shared infrastructure (helpers, style) propagate silently.

**Consequences:** Thesis contains incorrect figures. The H0 posterior plot might show a flat line due to normalization error. The CRB heatmap might use a wrong colorscale. These are the most important figures in the thesis and they are untested.

**Prevention:**
1. Before any refactoring, add smoke tests for every existing plot function: call with known synthetic data, verify the function completes without error, and check basic properties (number of lines on axes, axis limits, colorbar exists).
2. Use `pytest-mpl` for image comparison tests on the 3-5 most critical thesis figures (posterior, CRB heatmap, detection distribution). Generate baselines before the overhaul begins.
3. Refactor in layers: style changes in one commit, function signatures in another, never both at once.
4. The `_fig_from_ax` helper is used cross-module (evaluation_plots imports it from simulation_plots). Moving or renaming it breaks two modules silently.

**Detection:** Run the full test suite after every plotting change. A missing import or broken function signature will be caught by mypy if type annotations are maintained on the `(Figure, Axes)` return types.

## Moderate Pitfalls

### Pitfall 5: Matplotlib Backend Must Be Set Before Any pyplot Import

**What goes wrong:** `apply_style()` calls `matplotlib.use("Agg")` to set the non-interactive backend. But if ANY module imports `matplotlib.pyplot` before `apply_style()` is called, the backend is locked to whatever the system default is (often TkAgg or Qt5Agg on dev machines). On the cluster or CI (no display), this crashes with `_tkinter.TclError: no display name`. The project currently handles this correctly (main.py imports apply_style before pyplot), but adding a new visualization library that eagerly imports pyplot at module level will break this.

**Why it happens:** matplotlib's backend selection is a one-shot operation. The first `import matplotlib.pyplot` or `plt.figure()` call locks it. New libraries (e.g., corner, chainconsumer) may import pyplot at the module level in their `__init__.py`.

**Consequences:** Crash on headless systems (CI, cluster). Works on dev machine with a display, fails everywhere else. The failure mode is an opaque Tcl/Tk error with no indication that the fix is import ordering.

**Prevention:**
1. Keep `apply_style()` as the very first call after entry point detection, before any other imports from the plotting module or third-party viz libraries.
2. In test conftest.py, the session-scoped `_plotting_style` fixture already handles this. Verify it runs before any test that imports a new visualization library.
3. Add `matplotlib.use("Agg")` as a safety fallback in `__init__.py` of the plotting module (guarded by `if not os.environ.get("DISPLAY")`), though the explicit `apply_style()` call is preferred.
4. When evaluating new libraries, check whether their top-level `__init__.py` imports pyplot.

**Detection:** CI will catch this immediately -- headless runners have no display server. But the crash will be in a test file that seems unrelated to the new library.

### Pitfall 6: Figure Memory Leaks in Batch Plot Generation

**What goes wrong:** The thesis needs 20-40 figures generated from campaign data. Each `plt.subplots()` or `plt.figure()` creates a figure object tracked by pyplot's global state. If figures are not closed after saving, memory grows linearly with figure count. With high-resolution data (thousands of posterior samples, full parameter space scatter), each figure can consume 50-200 MB. A batch script generating all thesis figures runs out of memory or slows to a crawl.

**Why it happens:** The existing `save_figure()` helper defaults to `close=True`, which is correct. But new plot functions added during the overhaul may use `plt.subplots()` directly (bypassing `get_figure()`), forget to close, or create intermediate figures for multi-panel layouts that are never closed.

**Consequences:** OOM on the dev machine or cluster evaluation node. Matplotlib emits "More than 20 figures have been opened" warnings that are easy to ignore.

**Prevention:**
1. Enforce that all new plot functions use `get_figure()` from `_helpers.py`, never raw `plt.subplots()`. This provides a single choke point for figure lifecycle management.
2. Add a pytest fixture that counts open figures before and after each test, warning if the count increases.
3. For batch generation scripts, wrap each figure in a `try/finally` that calls `plt.close(fig)`.
4. Set `matplotlib.rcParams["figure.max_open_warning"]` to a low number (5) during development to catch leaks early.

**Detection:** Monitor RSS memory during batch figure generation. A monotonically increasing memory profile indicates unclosed figures.

### Pitfall 7: Corner Plot Libraries Fight With Custom Styles

**What goes wrong:** Libraries like `corner` (for posterior corner plots) and `chainconsumer` (for chain visualization) apply their own styling to axes, overriding the project's mplstyle. Corner plots are a standard visualization in GW parameter estimation papers, so they will likely be needed. But `corner.corner()` sets its own tick parameters, label sizes, and spacing that conflict with `emri_thesis.mplstyle`.

**Why it happens:** These libraries are designed to produce self-contained, visually consistent multi-panel plots. They override rcParams locally to ensure their output looks correct regardless of the user's global style. This is good library design but bad for style consistency in a thesis.

**Consequences:** Corner plots look different from all other figures in the thesis -- different font sizes, different tick formatting, different line widths.

**Prevention:**
1. When using corner or similar libraries, pass explicit style overrides to match the thesis style: `corner.corner(..., label_kwargs={"fontsize": 12}, title_kwargs={"fontsize": 13})`.
2. Wrap corner plot calls in a function that applies the project style before and after the call.
3. Prefer libraries that accept `fig` and `axes` arguments so you can pre-create them with the project's style settings.
4. Test that the corner plot font sizes match the rest of the thesis by inspecting the returned axes objects.

**Detection:** Visual inspection of the compiled thesis PDF -- corner plots should not look "different" from other figures.

### Pitfall 8: PDF Vector Output Bloat From Dense Scatter Plots

**What goes wrong:** Parameter space scatter plots with 10,000+ points produce multi-megabyte PDF files because each point is a separate vector object. The thesis PDF becomes 100+ MB, LaTeX compilation slows down, and the PDF viewer struggles to render the page. The `plot_cramer_rao_coverage` function already creates 3D scatter plots that will scale poorly.

**Why it happens:** Vector formats (PDF, SVG) store each graphical element individually. A scatter plot with 50,000 points creates 50,000 path objects. The existing `plot_sky_localization_3d` and `plot_cramer_rao_coverage` functions will produce enormous PDFs with production-scale data.

**Consequences:** Thesis PDF exceeds university submission size limits. LaTeX crashes with "TeX capacity exceeded" on pages with dense plots. Reviewers cannot open the PDF.

**Prevention:**
1. For scatter plots with >1000 points, rasterize the scatter layer while keeping axes/labels as vectors: `ax.scatter(..., rasterized=True)` with `fig.savefig(..., dpi=300)`.
2. Alternatively, use `hexbin` or `hist2d` (density plots) instead of scatter for large datasets. The existing `plot_detection_contour` already uses `hist2d` -- follow this pattern.
3. Set `savefig.dpi: 300` (already in the mplstyle) so rasterized elements look sharp.
4. Add a `rasterized=True` default to `get_figure()` or to the style configuration for scatter-type plots.

**Detection:** Check the file size of each generated PDF. Any single-figure PDF over 2 MB is suspect and should be rasterized.

### Pitfall 9: The `bayesian_inference_mwe.py` Contains a Rogue `plt.show()`

**What goes wrong:** Line 177 of `bayesian_inference_mwe.py` calls `plt.show()`, which blocks on the Agg backend and does nothing useful, or on an interactive backend opens a window that blocks the script. This is the only `plt.show()` call in the codebase. During the visualization overhaul, if this module is imported or its code is reused, the `plt.show()` will cause hangs on the cluster or in CI.

**Why it happens:** This is legacy code from the MWE (minimum working example) that was designed for interactive Jupyter use. It was never cleaned up because the MWE is not part of the main pipeline.

**Consequences:** Script hangs if accidentally triggered in batch mode. On CI, the test either hangs indefinitely or crashes depending on the backend.

**Prevention:**
1. Remove the `plt.show()` call during the visualization overhaul. Replace with `save_figure()` if the output is needed, or delete if it is dead code.
2. Add a grep-based CI check that fails if `plt.show()` appears anywhere in the source tree (excluding test files and notebooks).

**Detection:** `grep -r "plt.show()" master_thesis_code/` -- this should return zero results after cleanup.

### Pitfall 10: Constrained Layout Incompatible With Some Multi-Panel Configurations

**What goes wrong:** The mplstyle sets `figure.constrained_layout.use: True`, which is generally superior to `tight_layout`. But constrained layout can fail with certain subplot configurations: colorbars on specific axes in grid layouts, axes with `projection="3d"`, or figures using `GridSpec` manually. When it fails, matplotlib emits a warning and falls back to overlapping elements. The existing `plot_cramer_rao_coverage` and `plot_sky_localization_3d` use `projection="3d"` which has known constrained layout issues.

**Why it happens:** Constrained layout's solver cannot always satisfy all constraints simultaneously, especially with 3D axes, inset axes, or complex GridSpec layouts. New multi-panel thesis figures (e.g., a 2x3 grid of posterior plots with shared colorbars) are likely to hit edge cases.

**Consequences:** Labels overlap, colorbars extend beyond figure bounds, or subplot spacing is wildly uneven. The figure looks broken but no error is raised -- only a UserWarning that is easy to miss.

**Prevention:**
1. For complex multi-panel figures, explicitly pass `constrained_layout=False` to `plt.subplots()` and use `fig.tight_layout()` or manual `subplots_adjust()` instead.
2. For 3D projection plots, always disable constrained layout.
3. Capture matplotlib warnings in tests: `with warnings.catch_warnings(record=True) as w:` and fail if constrained layout warnings appear.
4. Test every multi-panel figure configuration with the production data dimensions.

**Detection:** Matplotlib emits `UserWarning: constrained_layout not applied` -- configure the test suite to treat this warning as an error for plotting tests.

## Minor Pitfalls

### Pitfall 11: Colormap Accessibility (Colorblind Safety)

**What goes wrong:** The mplstyle uses `image.cmap: viridis` (colorblind-safe), but individual plot functions may override with non-accessible colormaps. Physics tradition sometimes uses jet or rainbow colormaps that are inaccessible to ~8% of male readers.

**Prevention:** Only use perceptually uniform colormaps: viridis, cividis, inferno. Never use jet, rainbow, or spectral. Add a lint check or code review guideline.

### Pitfall 12: DPI Mismatch Between Screen and Print

**What goes wrong:** The mplstyle sets `figure.dpi: 150` for screen display but `savefig.dpi: 300` for file output. If a figure's layout is tuned visually (in a notebook at 150 DPI), text sizes and line widths may look different in the saved 300 DPI PDF due to the DPI difference affecting relative sizing of rasterized vs vector elements.

**Prevention:** Always evaluate figure quality from the saved PDF, not from the notebook or screen rendering. Use a consistent DPI for both display and save when doing layout work.

### Pitfall 13: `_fig_from_ax` Helper Is Not Exported But Used Cross-Module

**What goes wrong:** `evaluation_plots.py` and `bayesian_plots.py` both import `_fig_from_ax` from `simulation_plots.py`. This underscore-prefixed "private" function is a cross-module dependency. Refactoring `simulation_plots.py` will break two other modules.

**Prevention:** Move `_fig_from_ax` to `_helpers.py` where it belongs alongside `get_figure` and `save_figure`. Export it from `__init__.py` if needed by external callers.

### Pitfall 14: No Type Stubs for Visualization Libraries

**What goes wrong:** mypy is configured with `disallow_untyped_defs = true`. Libraries like corner, seaborn, and chainconsumer have incomplete or no type stubs. Adding calls to these libraries produces mypy errors unless their modules are added to `ignore_missing_imports` in pyproject.toml.

**Prevention:** Add any new visualization library to the `[[tool.mypy.overrides]]` ignore list immediately when adding the dependency. Do not let mypy errors accumulate.

### Pitfall 15: Plot Function Signatures Inconsistent With New Pattern

**What goes wrong:** Existing plot functions use two patterns: some accept `ax: Axes | None = None` (and create a figure internally if None), others always create their own figure (like `plot_cramer_rao_coverage`). During the overhaul, if a third pattern is introduced (e.g., returning only `Figure` without `Axes`), the calling code becomes inconsistent and confusing.

**Prevention:** Standardize all plot functions on the `ax: Axes | None = None` pattern with `(Figure, Axes)` return type. Functions that need special projections (3D) should still follow this pattern but accept `Any` for the axes type.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Style system audit/rebuild | rcParams pollution (Pitfall 1), constrained layout failures (Pitfall 10) | Snapshot rcParams in tests, test 3D + multi-panel layouts explicitly |
| Adding new visualization deps | Backend import order (Pitfall 5), over-engineering interactive (Pitfall 3), type stubs (Pitfall 14) | Verify headless compatibility, matplotlib-only output, update mypy config |
| Refactoring existing plots | Silent breakage (Pitfall 4), _fig_from_ax coupling (Pitfall 13), pattern inconsistency (Pitfall 15) | Add smoke tests BEFORE refactoring, move shared helpers first |
| LaTeX/publication quality | Font rendering (Pitfall 2), PDF bloat (Pitfall 8), DPI mismatch (Pitfall 12) | Keep usetex=False, rasterize dense scatters, evaluate from saved PDF |
| Corner plots / posteriors | Style conflicts (Pitfall 7), memory leaks (Pitfall 6) | Pass explicit style kwargs, close figures after save |
| Batch figure generation | Memory leaks (Pitfall 6), rogue plt.show (Pitfall 9) | Use get_figure() exclusively, remove plt.show(), close all figures |

## Ordering Recommendations

Based on pitfall dependencies, the recommended approach for the visualization overhaul is:

1. **Test infrastructure first** -- Add smoke tests for all existing plot functions and a rcParams regression test BEFORE changing anything. This is the safety net. (Addresses Pitfalls 1, 4)
2. **Refactor shared infrastructure** -- Move `_fig_from_ax` to helpers, clean up `plt.show()`, standardize function signatures. No visual changes. (Addresses Pitfalls 9, 13, 15)
3. **Style system improvements** -- Modify the mplstyle file, evaluate usetex decision, add publication-quality settings. The test infrastructure from step 1 catches regressions. (Addresses Pitfalls 2, 10, 11, 12)
4. **Add new visualization capabilities** -- Corner plots, improved posterior visualization, etc. Each new library is evaluated for headless compatibility and style integration. (Addresses Pitfalls 3, 5, 7, 8, 14)
5. **Batch generation and thesis integration** -- Generate all figures with production data, verify PDF quality and size. (Addresses Pitfalls 6, 8)

The key principle: build the safety net before swinging on the trapeze.

## Sources

- [matplotlib Text rendering with LaTeX](https://matplotlib.org/stable/users/explain/text/usetex.html) -- LaTeX dependency requirements, font matching issues, PDF quality
- [matplotlib Fonts documentation](https://matplotlib.org/stable/users/explain/text/fonts.html) -- mathtext vs usetex, font configuration
- [Leo Stein: Fonts/sizes in matplotlib figures for LaTeX publications](https://duetosymmetry.com/code/latex-mpl-fig-tips/) -- practical tips for thesis-quality figures
- [pytest-mpl: image comparison plugin](https://github.com/matplotlib/pytest-mpl) -- regression testing for plot output
- [seaborn set_theme documentation](https://seaborn.pydata.org/generated/seaborn.set_theme.html) -- global rcParams mutation behavior
- [seaborn FAQ](https://seaborn.pydata.org/faq.html) -- relationship with matplotlib, style system
- [matplotlib Testing documentation](https://matplotlib.org/stable/devel/testing.html) -- image comparison decorators, check_figures_equal
- Direct codebase inspection of `plotting/` module, `conftest.py`, `pyproject.toml`, `emri_thesis.mplstyle`, all plot factory functions
