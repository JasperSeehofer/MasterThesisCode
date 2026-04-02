# Phase 19: Campaign Dashboards & Batch Generation - Research

**Researched:** 2026-04-02
**Domain:** Matplotlib composite layouts, batch figure generation, PDF size optimization
**Confidence:** HIGH

## Summary

Phase 19 wires the 20+ existing factory functions (from Phases 14-18) into a batch generation pipeline and adds a single composite dashboard figure. The technical scope is narrow: one new module (`dashboard_plots.py`), one implementation of an existing stub (`generate_figures()` in `main.py`), and targeted `rasterized=True` additions to scatter calls.

The key technical finding is that `plt.subplot_mosaic` with `per_subplot_kw` (available in matplotlib 3.10.8, confirmed installed) supports mixed projections in a single figure -- the Mollweide sky map subplot can coexist with standard rectilinear subplots. All four dashboard sub-plots (H0 posterior, SNR distribution, detection yield, sky localization) already accept an `ax` parameter, so they can render into pre-created mosaic axes.

The batch generation follows a straightforward manifest pattern: a list of (factory_function, data_loader, output_name) tuples iterated with try/except for graceful degradation. Data loading uses pandas `read_csv` for CRB files and JSON loading for posteriors, both patterns already established in the codebase.

**Primary recommendation:** Implement in two plans -- (1) dashboard composite factory + tests, (2) batch generation manifest + file size checks + tests.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Summary dashboard uses a 2x2 grid via `plt.subplot_mosaic` composing: H0 posterior (top-left), SNR distribution (top-right), detection yield (bottom-left), sky localization Mollweide (bottom-right).
- **D-02:** Dashboard is a factory function `plot_campaign_dashboard(...)` in a new `master_thesis_code/plotting/dashboard_plots.py` module, returning `(Figure, dict[str, Axes])` where keys match the mosaic labels.
- **D-03:** Dashboard uses `preset="double"` (7.0in width) for thesis double-column layout.
- **D-04:** Implement `generate_figures()` in `main.py` (stub already exists at line 608) to iterate a figure manifest and produce all thesis figures from a campaign working directory.
- **D-05:** Entry point is `--generate_figures <working_dir>` CLI flag (already wired in `arguments.py`). No separate script -- uses existing CLI infrastructure.
- **D-06:** Figure manifest is a Python list of `(factory_function, data_loader, output_name)` tuples defined in `generate_figures()`. Not a YAML config -- keep it simple and type-checked.
- **D-07:** Each figure is saved via `save_figure(fig, path, formats=("pdf",))` to `<working_dir>/figures/`. Directory created automatically by `save_figure`.
- **D-08:** Scatter plots with >1000 points use `rasterized=True` on the scatter call. This is set inside the factory functions where needed (sky_plots, evaluation_plots scatter calls).
- **D-09:** After saving each PDF, check `os.path.getsize()` and log a warning if >2MB. No auto-compression -- just a warning for manual review.
- **D-10:** Default output format is PDF only.
- **D-11:** Full figure manifest: H0 posteriors (combined + individual events), SNR distribution, detection yield, sky localization, Fisher ellipses (3 parameter pairs), corner plot, H0 convergence, detection efficiency, LISA PSD with noise decomposition, d_L(z) multi-H0, CRB coverage, uncertainty violins.
- **D-12:** Data loading uses existing CRB CSV files. Factory functions receive pre-loaded DataFrames/arrays.
- **D-13:** Figures that require missing data are skipped with a log warning (graceful degradation).

### Claude's Discretion
- Figure ordering in manifest (aesthetic choice)
- Exact subplot_mosaic layout string
- Whether to add a figure numbering scheme (e.g., `fig01_h0_posterior.pdf`)

### Deferred Ideas (OUT OF SCOPE)
None.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CAMP-01 | Multi-panel composite dashboard combining H0 posterior, SNR distribution, detection yield, sky map | `plt.subplot_mosaic` with `per_subplot_kw` enables mixed projections; all 4 factory functions accept `ax` parameter |
| CAMP-02 | Batch generation script produces all thesis figures from campaign working directory | Existing stub at `main.py:608`, CLI wiring at `arguments.py:73`, manifest pattern with graceful degradation |
| CAMP-03 | No single-figure PDF exceeds 2 MB; scatter plots with >1000 points use `rasterized=True` | 5 scatter calls identified across sky_plots.py, evaluation_plots.py, model_plots.py; `os.path.getsize()` for size check |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | 3.10.8 | `subplot_mosaic`, `per_subplot_kw`, all rendering | Already installed, confirmed version |
| pandas | (installed) | CRB CSV loading via `pd.read_csv` | Already used by `_data.py` |
| numpy | (installed) | Array operations for data preparation | Already used everywhere |

No new dependencies required. This phase uses only existing libraries.

## Architecture Patterns

### New File
```
master_thesis_code/plotting/
    dashboard_plots.py          # NEW: plot_campaign_dashboard() factory
```

### Modified Files
```
master_thesis_code/main.py      # Implement generate_figures() stub (line 608)
master_thesis_code/plotting/sky_plots.py        # Add rasterized=True to scatter
master_thesis_code/plotting/evaluation_plots.py # Add rasterized=True to scatter calls
```

### Pattern 1: subplot_mosaic with Mixed Projections

The dashboard needs a Mollweide projection for the sky map and standard rectilinear axes for the other three panels. `plt.subplot_mosaic` supports this via `per_subplot_kw`.

```python
fig, axd = plt.subplot_mosaic(
    [["posterior", "snr"],
     ["yield", "sky"]],
    figsize=(7.0, 7.0 / 1.618),  # preset="double" dimensions
    per_subplot_kw={"sky": {"projection": "mollweide"}},
)
```

Return type is `(Figure, dict[str, Axes])` matching the project convention (D-02). Note: `get_figure()` uses `plt.subplots` internally, so the dashboard function will call `plt.subplot_mosaic` directly using the preset dimensions from `_PRESETS["double"]`.

### Pattern 2: Factory Function ax Injection

All four dashboard sub-plot factory functions accept `ax: Axes | None = None`. The dashboard creates the mosaic, then calls each factory with the pre-created axes:

```python
plot_combined_posterior(h_values, posterior, true_h, ax=axd["posterior"])
plot_snr_distribution(snr_values, ax=axd["snr"])
plot_detection_yield(injected_z, detected_z, ax=axd["yield"])
plot_sky_localization_mollweide(theta, phi, snr, ax=axd["sky"])
```

Important: the factory functions create their own figure/axes when `ax=None`, but when given an existing ax, they draw into it and extract the figure via `_fig_from_ax(ax)`. The dashboard creates the figure once, so all four draw into the same figure.

### Pattern 3: Manifest-Based Batch Generation

```python
def generate_figures(output_dir: str) -> None:
    apply_style()
    figures_dir = os.path.join(output_dir, "figures")

    # Each entry: (name, callable_that_returns_fig_or_none)
    manifest = [
        ("h0_posterior_combined", lambda: _gen_h0_posterior(output_dir)),
        ("snr_distribution", lambda: _gen_snr_distribution(output_dir)),
        # ... etc
    ]

    for name, generator in manifest:
        try:
            result = generator()
            if result is None:
                _ROOT_LOGGER.warning(f"Skipping {name}: required data not found")
                continue
            fig, _ = result
            path = os.path.join(figures_dir, name)
            save_figure(fig, path, formats=("pdf",))
            _check_file_size(f"{path}.pdf", name)
        except Exception:
            _ROOT_LOGGER.warning(f"Failed to generate {name}", exc_info=True)
```

Each generator function handles its own data loading with `try/except` for missing files, returning `None` when data is unavailable.

### Pattern 4: Rasterized Scatter for PDF Size

```python
ax.scatter(x, y, rasterized=True)  # renders scatter as raster in vector PDF
```

This must be added to scatter calls in:
- `sky_plots.py:64` -- `ax.scatter(lon, lat, c=snr, ...)` -- sky localization
- `evaluation_plots.py:79` -- `ax.scatter(theta, phi, sky_error, ...)` -- 3D sky (less important)
- `evaluation_plots.py:208` -- `ax_main.scatter(inj, rec, ...)` -- injected vs recovered
- `evaluation_plots.py:232` -- `ax_resid.scatter(inj, residual, ...)` -- residuals

The `model_plots.py` scatter calls (lines 80, 90, 101) use small fixed datasets and are unlikely to exceed 1000 points, but can be rasterized for safety.

### Anti-Patterns to Avoid
- **Calling `plt.show()` or `plt.savefig()` inside factory functions:** Factory functions return `(fig, ax)` only. Saving is done by the caller (`generate_figures`).
- **Creating figures inside the manifest loop:** Each generator creates its own figure via the factory. The loop only handles saving and error handling.
- **Importing heavy modules at module level in `main.py`:** Use deferred imports inside `generate_figures()` (same pattern as `evaluate()` at line 600).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Mixed-projection multi-panel layout | Manual `fig.add_subplot` with GridSpec | `plt.subplot_mosaic` + `per_subplot_kw` | Handles geometry automatically, named axes dict |
| PDF file size check | Custom PDF parsing | `os.path.getsize()` | Simple, sufficient for 2MB threshold |
| Figure directory creation | Manual `os.makedirs` before each save | `save_figure()` already calls `os.makedirs` | Already implemented in `_helpers.py:80` |
| CRB covariance reconstruction | Manual column parsing | `_data.reconstruct_covariance()` | Already implemented, tested |

## Common Pitfalls

### Pitfall 1: Mollweide Axes Limitation
**What goes wrong:** Mollweide projection axes do not support `set_xlim`/`set_ylim` or standard tick formatting. Calling these methods raises errors or produces warnings.
**Why it happens:** Mollweide is a fixed full-sky projection. The longitude must be in [-pi, pi] and latitude in [-pi/2, pi/2].
**How to avoid:** The existing `plot_sky_localization_mollweide` already handles coordinate transforms correctly (colatitude to latitude, longitude wrapping). Do not add axis limit calls to the sky panel.
**Warning signs:** `ValueError: axis not convertible` or silent empty plot.

### Pitfall 2: Twin Axes in Dashboard Subpanels
**What goes wrong:** Both `plot_snr_distribution` and `plot_detection_yield` create twin y-axes via `ax.twinx()`. In a mosaic layout, the twin axes occupy the same grid cell but are not tracked by the mosaic dict.
**Why it happens:** `twinx()` creates a second Axes object layered over the original.
**How to avoid:** This works fine -- the twin axes are children of the original ax and render correctly in mosaic layouts. No special handling needed. Just be aware that `axd["snr"]` returns the primary axes; the secondary CDF/fraction axis is internal to the factory function.
**Warning signs:** None expected, but test visually.

### Pitfall 3: Data File Discovery
**What goes wrong:** `generate_figures()` cannot find CRB CSV files or posterior JSONs because the working directory structure varies between campaign types.
**Why it happens:** Different CLI modes (`--simulation_steps`, `--evaluate`, `--combine`) produce different output file structures.
**How to avoid:** Use `glob.glob()` with known patterns (e.g., `*.csv`, `posteriors_without_bh_mass/*.json`). Check existence before loading. Log clear warnings when files are missing (D-13).
**Warning signs:** All figures skipped with "data not found" warnings.

### Pitfall 4: Figure Aspect Ratio in Dashboard
**What goes wrong:** The 2x2 dashboard with golden-ratio aspect (7.0 x 4.327) gives each panel roughly 3.5 x 2.16 inches. The Mollweide projection may look squished.
**How to avoid:** Consider using a slightly taller figure for the dashboard (e.g., 7.0 x 5.5 or 7.0 x 6.0) with `height_ratios` to give the sky map more vertical space. This is Claude's discretion.
**Warning signs:** Sky map looks horizontally compressed.

### Pitfall 5: Memory Pressure from Batch Generation
**What goes wrong:** Generating 15+ figures without closing them accumulates memory.
**Why it happens:** Matplotlib figures consume significant memory, especially with raster elements.
**How to avoid:** `save_figure()` already calls `plt.close(fig)` by default (`close=True`). Each figure is created, saved, and closed in sequence. No accumulation.
**Warning signs:** None expected with current `save_figure` implementation.

## Code Examples

### Dashboard Factory Skeleton
```python
# Source: Verified against matplotlib 3.10.8 plt.subplot_mosaic signature
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from master_thesis_code.plotting._helpers import _PRESETS

def plot_campaign_dashboard(
    # ... data parameters for all 4 sub-plots
) -> tuple[Figure, dict[str, Axes]]:
    figsize = _PRESETS["double"]
    # Taller for Mollweide
    fig, axd = plt.subplot_mosaic(
        [["posterior", "snr"],
         ["yield", "sky"]],
        figsize=(figsize[0], figsize[0] * 0.75),
        per_subplot_kw={"sky": {"projection": "mollweide"}},
    )
    # Delegate to existing factories with ax= injection
    # ...
    fig.tight_layout()
    return fig, axd
```

### File Size Check Helper
```python
import os
import logging

_LOGGER = logging.getLogger(__name__)
_TWO_MB = 2 * 1024 * 1024

def _check_file_size(path: str, name: str) -> None:
    try:
        size = os.path.getsize(path)
        if size > _TWO_MB:
            _LOGGER.warning(
                "%s exceeds 2 MB (%d bytes) -- consider rasterizing dense elements",
                name, size,
            )
    except OSError:
        pass
```

### Rasterized Scatter Addition
```python
# In sky_plots.py line 64, add rasterized=True:
sc = ax.scatter(lon, lat, c=snr, cmap=CMAP, s=12, alpha=0.8, zorder=5, rasterized=True)
```

## Factory Function Signatures Reference

All functions the dashboard and batch need to call, with their required data parameters:

| Function | Module | Data Parameters | Notes |
|----------|--------|----------------|-------|
| `plot_combined_posterior` | bayesian_plots | `h_values, posterior, true_h` | Has `ax` param |
| `plot_event_posteriors` | bayesian_plots | `h_values, event_posteriors, true_h` | Has `ax` param |
| `plot_snr_distribution` | bayesian_plots | `snr_values` | Creates twinx internally |
| `plot_detection_yield` | simulation_plots | `injected_redshifts, detected_redshifts` | Creates twinx internally |
| `plot_sky_localization_mollweide` | sky_plots | `theta_s, phi_s, snr, covariances?` | Needs Mollweide projection |
| `plot_fisher_ellipses` | fisher_plots | `covariance, param_values, pairs?` | Returns array of axes |
| `plot_fisher_corner` | fisher_plots | `covariance, param_values, params?` | Creates own figure (corner lib) |
| `plot_h0_convergence` | convergence_plots | `h_values, event_posteriors` | Creates own 2-panel figure |
| `plot_detection_efficiency` | convergence_plots | `variable, detected` | Has `ax` param |
| `plot_lisa_psd` | simulation_plots | `frequencies, decompose=True` | Has `ax` param |
| `plot_distance_redshift` | physical_relations_plots | `redshifts, distances, h0_values?, distance_fn?` | Has `ax` param |
| `plot_cramer_rao_coverage` | simulation_plots | `M, qS, phiS` | Has `ax` param |
| `plot_uncertainty_violins` | evaluation_plots | `uncertainties: dict` | Has `ax` param |
| `plot_injected_vs_recovered` | evaluation_plots | `injected, recovered, uncertainties?` | Multi-panel |

## Data Loading Inventory

Files expected in a campaign working directory:

| File Pattern | Content | Used By |
|-------------|---------|---------|
| `cramer_rao_bounds_*.csv` | CRB data (14 params + 105 delta columns + SNR + metadata) | Most plots |
| `posteriors_without_bh_mass/*.json` | Per-event posterior JSONs | H0 posteriors, convergence |
| `posteriors_with_bh_mass/*.json` | Per-event posterior JSONs (with BH mass) | H0 posteriors (optional) |
| `combined_posterior_*.json` | Combined posterior result | H0 combined plot |
| `injection_campaign_*.csv` | Injection data for P_det | Detection efficiency |

Data loading pattern: `pd.read_csv()` for CSV, `json.load()` for posteriors. `load_posterior_jsons()` in `posterior_combination.py` already handles the JSON directory pattern.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (installed via uv dev extras) |
| Config file | `pyproject.toml` [tool.pytest] |
| Quick run command | `uv run pytest -m "not gpu and not slow" -x` |
| Full suite command | `uv run pytest -m "not gpu"` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CAMP-01 | Dashboard produces (Figure, dict[str, Axes]) with 4 named panels | unit | `uv run pytest master_thesis_code_test/plotting/test_dashboard_plots.py -x` | Wave 0 |
| CAMP-01 | Sky panel has Mollweide projection | unit | same as above | Wave 0 |
| CAMP-02 | generate_figures() produces PDFs in output_dir/figures/ | integration | `uv run pytest master_thesis_code_test/test_generate_figures.py -x` | Wave 0 |
| CAMP-02 | Missing data files produce log warnings, not crashes | unit | same as above | Wave 0 |
| CAMP-03 | scatter calls include rasterized=True | unit (grep/inspection) | `uv run pytest master_thesis_code_test/plotting/test_sky_plots.py -x` | Exists (partial) |
| CAMP-03 | PDFs over 2MB trigger warning log | unit | `uv run pytest master_thesis_code_test/test_generate_figures.py -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest -m "not gpu and not slow" -x`
- **Per wave merge:** `uv run pytest -m "not gpu"`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `master_thesis_code_test/plotting/test_dashboard_plots.py` -- covers CAMP-01 (dashboard smoke test)
- [ ] `master_thesis_code_test/test_generate_figures.py` -- covers CAMP-02 + CAMP-03 (batch generation + size check)

## Sources

### Primary (HIGH confidence)
- matplotlib 3.10.8 installed -- `plt.subplot_mosaic` signature confirmed via `help()` with `per_subplot_kw` support
- [matplotlib subplot_mosaic docs](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot_mosaic.html) -- per_subplot_kw projection support
- Project source code: all factory function signatures read directly from `master_thesis_code/plotting/*.py`
- `_helpers.py:_PRESETS["double"]` = (7.0, 4.327) confirmed

### Secondary (MEDIUM confidence)
- [matplotlib mosaic tutorial](https://matplotlib.org/stable/users/explain/axes/mosaic.html) -- mixed projection examples

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, all confirmed installed
- Architecture: HIGH -- all factory signatures verified, subplot_mosaic API confirmed
- Pitfalls: HIGH -- based on direct code inspection of twinx usage, projection constraints

**Research date:** 2026-04-02
**Valid until:** 2026-05-02 (stable domain, no fast-moving dependencies)
