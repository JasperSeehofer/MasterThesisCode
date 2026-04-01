# Architecture Patterns: v1.3 Visualization Overhaul

**Domain:** Visualization overhaul for EMRI parameter estimation thesis
**Researched:** 2026-04-01

## Existing Architecture Summary

The current `plotting/` subpackage follows a clean, consistent pattern:

```
plotting/
  __init__.py          # Re-exports: apply_style, get_figure, save_figure, make_colorbar
  _style.py            # apply_style() -> Agg backend + emri_thesis.mplstyle
  _helpers.py          # get_figure(), save_figure(), make_colorbar()
  emri_thesis.mplstyle # rcParams: figsize, fonts, constrained_layout, viridis cmap
  bayesian_plots.py    # 5 factories: posteriors, redshift dist, host counts
  evaluation_plots.py  # 5 factories: CRB heatmap, violins, sky 3D, contour, gen time
  model_plots.py       # 4 factories: EMRI dist, rate, sampling, P_det grid
  catalog_plots.py     # 4 factories: BH mass, redshift, completeness, comoving vol
  simulation_plots.py  # 4 factories + PlottingCallback: GPU usage, PSD, noise, CRB coverage
  physical_relations_plots.py  # 1 factory: d_L(z)
```

**23 total factory functions** across 6 topic modules.

**Factory pattern:** Every plot function takes typed numpy arrays in, returns `(Figure, Axes)` out. No side effects (no `plt.show()`, no `plt.savefig()`). Optional `ax: Axes | None` parameter allows compositing onto existing axes. Caller decides where/how to save via `save_figure()`.

**Callback system:** `SimulationCallback` Protocol in `callbacks.py` decouples the simulation loop from visualization. `PlottingCallback` collects GPU stamps during simulation and produces plots in `on_simulation_end`.

**Style system:** Single `.mplstyle` file loaded once via `apply_style()`. Session-scoped `_plotting_style` fixture in `conftest.py` ensures tests use the same style. Current style: `text.usetex: False`, `figure.figsize: 6.4, 4.0`, `figure.constrained_layout.use: True`, font size 11.

## Recommended Architecture

### Principle: Extend, Do Not Replace

The existing factory pattern is sound and should be preserved. New visualization capabilities integrate as **new modules and new factory functions** within the existing structure, not as a replacement framework.

### Target Directory Structure

```
plotting/
  __init__.py              # MODIFY: add new re-exports
  _style.py                # MODIFY: extend for LaTeX toggle, thesis figure sizes
  _helpers.py              # MODIFY: add thesis column width constants
  _colors.py               # NEW: color palette, parameter-color mapping, colormap defs
  _data.py                 # NEW: covariance matrix reconstruction from CRB CSV columns
  emri_thesis.mplstyle     # MODIFY: upgrade rcParams for publication quality

  # --- Existing modules (minimal changes) ---
  bayesian_plots.py        # MODIFY: add uncertainty bands to posteriors
  evaluation_plots.py      # MODIFY: improve violin styling, replace 3D sky scatter
  model_plots.py           # MODIFY: styling improvements only
  catalog_plots.py         # MODIFY: styling improvements only
  simulation_plots.py      # MODIFY: enhance PSD plot with confusion noise overlay
  physical_relations_plots.py  # MODIFY: add cosmology comparison overlay

  # --- New modules ---
  fisher_plots.py          # NEW: Fisher ellipses, correlation matrix, condition number
  corner_plots.py          # NEW: corner plot wrappers using corner.py
  sky_plots.py             # NEW: Mollweide sky maps, localization ellipses
  psd_plots.py             # NEW: enhanced PSD with characteristic strain, waterfall
  convergence_plots.py     # NEW: H0 convergence with N_events, posterior width
  campaign_plots.py        # NEW: injection campaign summary dashboards
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `_style.py` | Style initialization, LaTeX toggle, thesis figure sizes | All plot modules (imported at style-load time) |
| `_helpers.py` | Figure creation, saving, colorbar creation | All plot modules |
| `_colors.py` | Color palette definitions, parameter-to-color mapping | All plot modules |
| `_data.py` | Covariance matrix reconstruction from CSV, CRB loading | `fisher_plots.py`, `corner_plots.py`, `sky_plots.py` |
| `fisher_plots.py` | Fisher matrix visualization (ellipses, heatmaps) | `_helpers.py`, `_colors.py`, `_data.py` |
| `corner_plots.py` | Thin wrapper around `corner.py` with project styling | `_style.py`, `_colors.py`; data from CRB CSV |
| `sky_plots.py` | Mollweide projections, sky localization | `_helpers.py`, `_colors.py`; data from CRB CSV (qS, phiS) |
| `psd_plots.py` | Enhanced PSD curves (characteristic strain, noise budget) | `_helpers.py`; data from `LisaTdiConfiguration` |
| `convergence_plots.py` | Statistical convergence of H0 posterior | `_helpers.py`; data from evaluation pipeline |
| `campaign_plots.py` | Multi-panel injection campaign summary | `_helpers.py`, other plot modules; aggregated campaign data |
| `bayesian_plots.py` | H0 posteriors with uncertainty bands | `_helpers.py`, `_colors.py` |
| `evaluation_plots.py` | Parameter uncertainty distributions, detection maps | `_helpers.py`, `_colors.py` |

### Data Flow

```
Simulation Pipeline                     Evaluation Pipeline
  |                                       |
  v                                       v
CRB CSV files (14 params +             H0 posterior arrays
  covariance entries + SNR +            (h_values, per-event posteriors,
  generation_time + host_idx)             combined posterior)
  |                                       |
  +-- _data.py (reconstruct cov) -----+   +-- loaded by BayesianStatistics --+
  |                                   |   |                                  |
  v                                   v   v                                  v
fisher_plots.py                    sky_plots.py    bayesian_plots.py    convergence_plots.py
corner_plots.py                    evaluation_plots.py                  campaign_plots.py
psd_plots.py (from LisaTdiConfiguration directly)
```

**Key data structures flowing into new plots:**

1. **CRB CSV columns:** `M, mu, a, p0, e0, x0, luminosity_distance, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0` (injected values) + `delta_X_delta_Y` (lower-triangle covariance entries) + `SNR, generation_time, host_galaxy_index, _simulation_index`
2. **Fisher/covariance matrix:** Reconstructed from `delta_X_delta_Y` columns in CRB CSV (14x14 symmetric matrix per event)
3. **H0 posterior:** `h_values` array + `posterior` array (from `BayesianStatistics.evaluate()`)
4. **PSD data:** Frequency array + PSD values from `LisaTdiConfiguration.power_spectral_density()`

## Patterns to Follow

### Pattern 1: Factory Function Signature (Preserve Existing)

**What:** Every plot function takes data arrays as positional args, optional `ax` for compositing, returns `(Figure, Axes)`.

**When:** Every new plot function.

**Example:**
```python
def plot_fisher_ellipse(
    covariance_2x2: npt.NDArray[np.float64],
    *,
    sigma_levels: tuple[float, ...] = (1.0, 2.0, 3.0),
    injected_values: tuple[float, float] | None = None,
    param_labels: tuple[str, str] = ("x", "y"),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = _fig_from_ax(ax)
    # ... draw ellipses from eigendecomposition of covariance_2x2 ...
    return fig, ax
```

### Pattern 2: Covariance Matrix Reconstruction Helper

**What:** A utility in `_data.py` that reconstructs the full 14x14 covariance matrix from the lower-triangle CSV columns.

**When:** Any plot that needs the Fisher/covariance matrix (ellipses, corner plots, correlation heatmaps).

**Example:**
```python
PARAMETER_ORDER: list[str] = [
    "M", "mu", "a", "p0", "e0", "x0", "luminosity_distance",
    "qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0",
]

def reconstruct_covariance_matrix(
    crb_row: dict[str, float],
) -> npt.NDArray[np.float64]:
    """Rebuild 14x14 covariance matrix from delta_X_delta_Y CSV columns."""
    n = len(PARAMETER_ORDER)
    cov = np.zeros((n, n))
    for i, pi in enumerate(PARAMETER_ORDER):
        for j in range(i + 1):
            pj = PARAMETER_ORDER[j]
            key = f"delta_{pi}_delta_{pj}"
            cov[i, j] = crb_row[key]
            cov[j, i] = crb_row[key]
    return cov

def extract_submatrix(
    cov: npt.NDArray[np.float64],
    param_indices: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """Extract 2x2 submatrix for a parameter pair."""
    i, j = param_indices
    return cov[np.ix_([i, j], [i, j])]
```

### Pattern 3: Thin Wrapper for External Libraries

**What:** Wrap `corner.py` to apply project styling and return `(Figure, Axes)` consistent with the factory pattern.

**When:** Integrating any external plotting library.

**Example:**
```python
import corner

def plot_parameter_corner(
    samples: npt.NDArray[np.float64],
    labels: list[str],
    truths: list[float] | None = None,
    *,
    quantiles: list[float] | None = None,
) -> tuple[Figure, Any]:
    """Corner plot of CRB parameter samples, styled for thesis."""
    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        quantiles=quantiles or [0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
    )
    return fig, fig.get_axes()
```

### Pattern 4: Multi-Panel Dashboard via Compositing

**What:** Use the existing `ax` parameter compositing to build multi-panel figures from individual factory functions.

**When:** Campaign summary plots, combined analysis dashboards.

**Example:**
```python
def plot_campaign_summary(
    campaign_data: CampaignData,
) -> tuple[Figure, Any]:
    fig, axes = plt.subplots(2, 3, figsize=THESIS_FULL_WIDTH_TALL)
    plot_detection_contour(data.z, data.M, ax=axes[0, 0])
    plot_combined_posterior(data.h, data.posterior, data.true_h, ax=axes[0, 1])
    plot_sky_mollweide(data.qS, data.phiS, data.sky_error, ax=axes[0, 2])
    # ... etc
    return fig, axes
```

### Pattern 5: LaTeX Toggle in Style System

**What:** `apply_style()` gains an optional `use_latex: bool = False` parameter. When True, enables `text.usetex` and sets font to match thesis document class.

**When:** Final thesis figure generation. Disabled by default because LaTeX rendering is slow and requires a TeX installation (not available on all machines or in CI).

**Example:**
```python
def apply_style(*, use_latex: bool = False) -> None:
    matplotlib.use("Agg")
    style_path = os.path.join(os.path.dirname(__file__), "emri_thesis.mplstyle")
    matplotlib.style.use(style_path)
    if use_latex:
        matplotlib.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Replacing the Factory Pattern with Class-Based Plotters

**What:** Creating `FisherPlotter`, `SkyMapPlotter` etc. classes that hold state and manage figure lifecycle.

**Why bad:** Breaks consistency with the existing 23 factory functions. Classes add indirection without benefit for stateless plot generation. The factory pattern is simpler and more composable.

**Instead:** Keep stateless factory functions. Use the `PlottingCallback` pattern only when data must be accumulated across simulation steps (which no new plot type requires).

### Anti-Pattern 2: Direct External Library Calls in Pipeline Code

**What:** Calling `corner.corner()` or `hp.mollview()` directly in `main.py` or `bayesian_statistics.py`.

**Why bad:** Couples pipeline code to specific visualization libraries. Makes it impossible to change visualization approach without modifying pipeline code.

**Instead:** All external library calls go through wrapper functions in `plotting/` modules. Pipeline code only imports from `master_thesis_code.plotting`.

### Anti-Pattern 3: Hardcoded Figure Sizes in Every Function

**What:** Every function specifying its own `figsize=(12, 8)` or similar (the current state for most functions).

**Why bad:** Thesis figures must match column widths. Scattered hardcoded sizes make consistency impossible and require global find-and-replace to change.

**Instead:** Define thesis-standard sizes in `_helpers.py`:
```python
THESIS_SINGLE_COLUMN = (3.5, 2.8)   # inches, ~88mm for single-column
THESIS_DOUBLE_COLUMN = (7.0, 4.5)   # inches, ~178mm for full-width
THESIS_SQUARE = (3.5, 3.5)          # for corner plots, heatmaps
THESIS_FULL_WIDTH_TALL = (7.0, 9.0) # for multi-panel dashboards
```

### Anti-Pattern 4: Heavy Dependencies for Marginal Benefit

**What:** Adding `arviz`, `seaborn`, `plotly`, `healpy` etc. when `matplotlib` + `corner` + `scipy` suffice.

**Why bad:** Each dependency adds installation complexity (especially on the cluster), potential version conflicts, and another API to learn. The thesis needs ~10 distinct new plot types, not a general-purpose visualization framework.

**Instead:** Use `matplotlib` (already installed) + `corner` (lightweight, GW community standard) + `scipy` (already installed, provides ellipse geometry via eigendecomposition). For sky maps, use matplotlib's built-in Mollweide projection (`projection='mollweide'`) which is sufficient for showing ~20-100 sky positions with error ellipses. Skip `healpy` -- it is designed for full HEALPix maps with millions of pixels, overkill for our ~20-100 detection scatter.

### Anti-Pattern 5: Using 3D Plots Where 2D Suffices

**What:** The existing `plot_sky_localization_3d` and `plot_cramer_rao_coverage` use 3D projections.

**Why bad:** 3D scatter plots in publications are hard to interpret, do not convey quantitative information well, and are not standard in the EMRI/LISA literature. 3D projections lose one degree of freedom to rotation.

**Instead:** Replace with 2D alternatives -- Mollweide for sky maps, 2D scatter/hexbin for parameter space coverage.

## Files: New vs Modified

### New Files (Create from Scratch)

| File | Purpose | New Dependencies |
|------|---------|-----------------|
| `plotting/_colors.py` | Centralized color palette, parameter-color mapping | None |
| `plotting/_data.py` | Covariance matrix reconstruction, CRB CSV loading helpers | None (uses pandas, already in deps) |
| `plotting/fisher_plots.py` | Fisher ellipses (2D pair plots), correlation heatmap, condition number | None (scipy already in deps for eigendecomposition) |
| `plotting/corner_plots.py` | Thin `corner.py` wrapper with thesis styling | `corner` (new dev dep) |
| `plotting/sky_plots.py` | Mollweide sky map, sky localization error visualization | None (matplotlib built-in `projection='mollweide'`) |
| `plotting/psd_plots.py` | Enhanced PSD: characteristic strain h_c(f), noise budget breakdown, EMRI signal overlay | None |
| `plotting/convergence_plots.py` | H0 convergence with N_events, posterior width vs N | None |
| `plotting/campaign_plots.py` | Multi-panel injection campaign summary dashboards | None (composes from other modules) |

### Modified Files (Existing, Need Changes)

| File | Changes | Risk |
|------|---------|------|
| `plotting/_style.py` | Add `use_latex` kwarg to `apply_style()`, keep backward-compatible default `False` | LOW |
| `plotting/_helpers.py` | Add thesis column width constants (`THESIS_SINGLE_COLUMN`, etc.) | LOW |
| `plotting/emri_thesis.mplstyle` | Upgrade: better tick direction (`in`), color cycle, legend styling, mathtext font | LOW -- additive |
| `plotting/__init__.py` | Re-export new public functions from new modules | LOW |
| `plotting/bayesian_plots.py` | Add optional `fill_between` uncertainty bands to `plot_combined_posterior` | LOW -- additive parameter |
| `plotting/evaluation_plots.py` | Deprecate `plot_sky_localization_3d` (keep for compat), add `plot_sky_localization_2d` delegating to `sky_plots.py` | MEDIUM -- new function, old unchanged |
| `plotting/simulation_plots.py` | Enhance `plot_lisa_psd` to accept optional confusion noise component for overlay | LOW -- additive parameter |
| `pyproject.toml` | Add `corner` to `[project.optional-dependencies.dev]` | LOW |
| `master_thesis_code_test/conftest.py` | Add fixture for LaTeX availability detection (skip tests needing TeX) | LOW |

### Files That Should NOT Change

| File | Reason |
|------|--------|
| `callbacks.py` | Callback protocol is stable; new plots do not need new callback hooks |
| `parameter_estimation/parameter_estimation.py` | Data producer, not consumer; no visualization changes |
| `bayesian_inference/bayesian_statistics.py` | Same -- produces data that flows to plots |
| `main.py` | Orchestration only; plot calls already isolated to callbacks/evaluation |
| `datamodels/parameter_space.py` | Data model; no visualization coupling |

## Scalability Considerations

| Concern | At 20 events | At 100 events | At 1000+ events |
|---------|--------------|---------------|-----------------|
| Corner plot render time | Instant | ~2s | Thin to every Nth event or use median + percentiles |
| Sky map density | Scatter points fine | Scatter points fine | Switch to KDE density or hexbin |
| CRB CSV loading | Trivial (<1 MB) | Trivial (~5 MB) | Load once via `_data.py`, pass to multiple plot functions |
| PDF export file size | Small (~100 KB) | Moderate (~500 KB) | Use PNG for dense scatters, PDF for line plots |
| Fisher ellipse overlay | ~20 ellipses | ~100 ellipses (use transparency) | Draw subset or alpha=0.1 + density coloring |
| Multi-panel dashboards | Fast | Fast | Memory concern with many subplots; generate panels separately if needed |

## Suggested Build Order

The build order minimizes risk by establishing infrastructure first, then building plot types in order of thesis importance.

### Phase 1: Style Infrastructure (No New Dependencies)

1. Create `_colors.py` (palette, parameter-color mapping)
2. Upgrade `emri_thesis.mplstyle` (tick direction, color cycle, mathtext)
3. Extend `_style.py` (LaTeX toggle with backward-compatible default)
4. Add thesis column width constants to `_helpers.py`
5. Update existing factory functions to use `_colors.py` palette

**Rationale:** Everything downstream depends on consistent styling. Get this right first. Zero risk -- additive changes only.

### Phase 2: Data Layer + Fisher Plots (No New Dependencies)

1. Create `_data.py` (covariance matrix reconstruction from CSV columns)
2. Create `fisher_plots.py` (2D Fisher ellipses for parameter pairs, correlation heatmap, condition number history)
3. Create `psd_plots.py` (characteristic strain h_c(f), noise budget breakdown with S_OMS, S_TM, S_conf components)

**Rationale:** Fisher ellipses are the most thesis-critical new plot type -- standard in EMRI PE papers. The `_data.py` layer is needed by multiple downstream modules. PSD plots are straightforward and use only existing dependencies.

### Phase 3: Sky Maps + Enhanced Existing Plots

1. Create `sky_plots.py` (Mollweide projection using `ax = fig.add_subplot(111, projection='mollweide')`)
2. Update `bayesian_plots.py` (add `credible_interval` parameter for shaded uncertainty bands)
3. Update `evaluation_plots.py` (add 2D sky localization delegating to `sky_plots`, improve violin colors)
4. Update `simulation_plots.py` (confusion noise overlay parameter on PSD plot)

**Rationale:** Sky maps depend on style infrastructure from Phase 1. Updating existing plots is low-risk.

### Phase 4: Corner Plots (New Dependency: corner)

1. Add `corner` to `pyproject.toml` dev dependencies
2. Create `corner_plots.py` (wrapper applying thesis style, returning `(Figure, Axes)`)
3. Add integration test producing a corner plot from synthetic CRB data

**Rationale:** Corner plots require a new dependency. Deferring to Phase 4 means the rest of the visualization stack works even if `corner` installation has issues. `corner` is lightweight (pure Python + matplotlib) so risk is low, but isolation is good practice.

### Phase 5: Convergence + Campaign Dashboards

1. Create `convergence_plots.py` (H0 posterior peak/width vs N_events, detection rate convergence)
2. Create `campaign_plots.py` (multi-panel summaries compositing earlier factory functions via `ax` parameter)
3. Update `__init__.py` with all new re-exports

**Rationale:** These depend on all previous phases and on having actual campaign data from the cluster runs. They are also the least standardized (no literature convention), so they benefit from having the other plot types working first as building blocks.

## Sources

- [corner.py documentation](https://corner.readthedocs.io/en/latest/) -- HIGH confidence, official docs
- [corner.py GitHub (dfm/corner.py)](https://github.com/dfm/corner.py) -- HIGH confidence
- [SciencePlots GitHub](https://github.com/garrettj403/SciencePlots) -- MEDIUM confidence (evaluated, decided against: custom mplstyle is simpler and avoids LaTeX requirement by default)
- [Publication-quality matplotlib for physics theses](https://jwalton.info/Embed-Publication-Matplotlib-Latex/) -- MEDIUM confidence
- [EMRI PE visualization standards (arXiv:2505.17814)](https://arxiv.org/html/2505.17814) -- HIGH confidence, recent LISA/EMRI paper showing Fisher ellipses, corner plots, sky maps as standard
- [Fisher matrix confidence ellipses (arXiv:0906.4123)](https://arxiv.org/pdf/0906.4123) -- HIGH confidence, standard reference for ellipse geometry from covariance matrices
- [healpy documentation](https://healpy.readthedocs.io/en/latest/) -- evaluated, decided against (HEALPix designed for full-sky surveys with millions of pixels; overkill for ~100 detection scatter)
- Existing codebase: all 6 `plotting/` modules, `_style.py`, `_helpers.py`, `callbacks.py`, `parameter_estimation.py` (CRB output format), `conftest.py`

---

*Architecture analysis: 2026-04-01 for v1.3 Visualization Overhaul*
