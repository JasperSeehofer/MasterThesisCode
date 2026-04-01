# Technology Stack: Visualization Overhaul

**Project:** EMRI Parameter Estimation -- v1.3 Visualization Overhaul
**Researched:** 2026-04-01

## Executive Assessment

The existing matplotlib-only stack is **sufficient as the rendering engine** but needs one targeted addition: `corner` for the multi-parameter posterior plots that every GW paper includes. Beyond that, the existing matplotlib + scipy + numpy stack handles all other visualization needs (sky maps via built-in Mollweide projection, Fisher ellipses via eigendecomposition, uncertainty bands via `fill_between`).

**Verdict:** Keep matplotlib as the sole rendering backend. Add `corner` (one lightweight library) for corner plots. Enhance the existing `emri_thesis.mplstyle` for publication quality. That is the entire stack change.

## Recommended Stack

### Keep (Already Present)

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| matplotlib | (current in lockfile) | Core rendering engine, Mollweide projections, all plot types | Keep as sole backend |
| numpy | (current in lockfile) | Array operations for plot data | Already a dependency |
| scipy | (current in lockfile) | Eigendecomposition for Fisher ellipses, KDE, interpolation | Already a dependency |
| astropy | >=6.1.7 | Coordinate transforms (if needed for sky positions) | Already a dependency |

### Add: Corner Plots

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| corner | >=2.2.3 | Multi-parameter posterior visualization | De facto standard in GW community. Used in virtually every EMRI/LISA parameter estimation paper. Dan Foreman-Mackey's library, built on matplotlib, returns `(fig, ax)` -- drops directly into existing factory function pattern. Requires only matplotlib+numpy. Python >=3.9 so 3.13-safe. |

**Why corner over alternatives:**
- **ChainConsumer:** Appears discontinued (no PyPI release in 12+ months). Was popular in cosmology but corner has broader adoption in GW specifically.
- **ArviZ 1.0:** Requires Python >=3.12 (fine), but it is a large dependency designed for full Bayesian workflow visualization (diagnostics, trace plots, model comparison). Overkill -- this project uses Fisher matrix / Cramer-Rao bounds, not MCMC chains. ArviZ shines when you have emcee/PyMC/Stan chains with convergence diagnostics to check.
- **getdist:** Alternative from cosmology (Planck team). Good but less standard in EMRI/LISA literature than corner.

**Integration:** `corner.corner(samples, labels=param_names, fig=fig)` accepts and returns matplotlib figures. Works with `emri_thesis.mplstyle` automatically since it uses matplotlib's rcParams.

### Sky Maps: Use matplotlib Built-In Mollweide (No New Dependency)

For ~20-100 EMRI detections, matplotlib's built-in `projection='mollweide'` is sufficient. healpy is designed for full-sky HEALPix maps with millions of pixels (CMB, LIGO/Virgo alerts) -- overkill for plotting ~100 scatter points with localization ellipses.

**Approach:**
```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='mollweide')
ax.scatter(phi, theta, c=values, ...)  # scatter detections
# Add localization ellipses from Fisher matrix sky-position submatrix
```

**Why NOT healpy:**
- healpy's `mollview()` expects a full HEALPix map array (pixels covering the whole sky). Our data is ~100 scattered sky positions with error ellipses, not a pixelized probability map.
- healpy has a C extension build requirement (cfitsio). Adds build complexity for marginal benefit.
- healpy's `mollview()` uses `plt.gcf()` internally -- less clean than the project's OO API pattern.
- For the actual thesis use case (scatter + ellipses on Mollweide), matplotlib's built-in projection plus scipy eigendecomposition for ellipse geometry is simpler and dependency-free.

**If healpy is needed later:** If the thesis requires a true HEALPix density map (e.g., from a very large detection catalog), healpy v1.19.0 has Python 3.13 wheels on PyPI and can be added then. But start without it.

### Upgrade: Style System (No New Dependency)

| Change | File | Purpose | Why |
|--------|------|---------|-----|
| LaTeX toggle in apply_style() | `_style.py` | Optional `use_latex=True` for final thesis figures | Axis labels match thesis font when TeX is available |
| Set serif font family | `emri_thesis.mplstyle` | Match LaTeX document body | `font.family: serif` with Computer Modern |
| Match thesis textwidth | `_helpers.py` | Named figure size constants matching LaTeX column width | Avoid scaling artifacts in `\includegraphics` |
| Improve tick/legend styling | `emri_thesis.mplstyle` | Inward ticks, better legend frame | Standard physics publication style |
| Graceful usetex fallback | `_style.py` | CI and cluster without TeX | Auto-detect TeX availability, fall back to mathtext |

**Do NOT add SciencePlots.** The project already has `emri_thesis.mplstyle` with appropriate settings. SciencePlots provides generic journal styles (IEEE, Nature) -- not useful for a thesis with its own custom style. Enhance the existing stylesheet instead.

## Explicitly Rejected (Do NOT Add)

| Library | Why Tempting | Why Wrong |
|---------|-------------|-----------|
| **healpy** | "GW community standard for sky maps" | Designed for full-sky HEALPix maps (millions of pixels). This project has ~100 sky positions. matplotlib's built-in Mollweide projection + scipy for ellipses suffices. C extension adds build complexity. Can be added later if truly needed. |
| **plotly / bokeh** | "Interactive exploration" | Thesis output is PDF/LaTeX. JupyterLab (already a dev dep) handles interactive dev exploration with inline matplotlib. |
| **seaborn** | "Better statistical plots" | Physics-specific plots, not statistical distributions. Adds style conflicts with `emri_thesis.mplstyle`. Violin plots and histograms work fine in raw matplotlib. |
| **arviz** | "Bayesian visualization" | Designed for MCMC diagnostics (trace plots, R-hat, ESS). This pipeline is Fisher-matrix-based. ArviZ 1.0 pulls ~20 transitive dependencies for unused features. |
| **SciencePlots** | "Publication styles" | Custom mplstyle already exists. Adding SciencePlots creates competing style definitions. |
| **chainconsumer** | "Corner plots + LaTeX tables" | Appears unmaintained. corner covers the plotting use case; LaTeX tables via pandas + tabulate (already deps). |
| **ligo.skymap** | "GW sky map standard" | Heavy LIGO-specific dependency chain. Designed for CBC real-time alerts, not EMRI Fisher-matrix localization. |
| **getdist** | "Planck-style triangle plots" | Less standard in EMRI/LISA literature than corner. Heavier API. |

## Alternatives Considered (Summary)

| Category | Recommended | Alternative | Why Not Alternative |
|----------|-------------|-------------|---------------------|
| Corner plots | corner | ChainConsumer | Discontinued; corner is GW community standard |
| Corner plots | corner | ArviZ | Overkill for Fisher-matrix pipeline; large dep |
| Corner plots | corner | getdist | Less standard in EMRI/LISA; heavier API |
| Sky maps | matplotlib Mollweide | healpy | HEALPix overkill for ~100 points; C extension complexity |
| Sky maps | matplotlib Mollweide | ligo.skymap | Heavy deps; LIGO-specific |
| Style | enhance existing mplstyle | SciencePlots | Competing styles; custom sheet already exists |
| Interactive | JupyterLab (already present) | plotly/bokeh | Thesis output is PDF, not HTML |
| Fisher ellipses | scipy eigendecomposition | dedicated library | scipy already a dep; ellipses are 10 lines of code |

## Installation

```bash
# One new dependency
uv add corner
```

One dependency. That is the entire stack change.

## Integration Points with Existing Code

### How corner fits in

The existing `plotting/` module uses factory functions returning `(fig, ax)`. Corner plots follow the same pattern:

```python
# In a new plotting/corner_plots.py
def plot_parameter_corner(
    samples: npt.NDArray[np.float64],
    labels: list[str],
    truths: list[float] | None = None,
) -> tuple[Figure, Any]:
    """Corner plot of EMRI parameter posterior samples."""
    import corner
    fig = corner.corner(
        samples, labels=labels, truths=truths,
        quantiles=[0.16, 0.5, 0.84], show_titles=True,
        title_kwargs={"fontsize": mpl.rcParams["axes.titlesize"]},
    )
    return fig, fig.get_axes()
```

Returns a matplotlib Figure -- works with `save_figure()` unchanged. Import inside function to avoid forcing dependency on all plotting module users.

### How sky maps work (no new dependency)

Mollweide projections use matplotlib's built-in support. Fisher ellipses use scipy:

```python
# In a new plotting/sky_plots.py
def plot_sky_localization_mollweide(
    theta: npt.NDArray[np.float64],  # colatitude
    phi: npt.NDArray[np.float64],    # longitude
    values: npt.NDArray[np.float64], # e.g. sky localization area
) -> tuple[Figure, Axes]:
    """Mollweide projection sky map of EMRI detections."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    sc = ax.scatter(phi, theta, c=values, cmap='viridis', ...)
    fig.colorbar(sc, ax=ax)
    return fig, ax
```

Replaces the existing `plot_sky_localization_3d()` (non-standard 3D scatter).

### Style sheet enhancement

Keep `text.usetex: False` as default in the mplstyle (safe for CI and cluster). Add a `use_latex` parameter to `apply_style()`:

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

This keeps CI/cluster safe while allowing `apply_style(use_latex=True)` for final thesis figure generation on the dev machine where TeX is installed.

### mypy overrides

`corner` lacks type stubs. Add to `[[tool.mypy.overrides]]`:

```toml
[[tool.mypy.overrides]]
module = [
    # ... existing entries ...
    "corner.*",
]
ignore_missing_imports = true
```

Matches the existing pattern for cupy, few, scipy, etc.

## Confidence Assessment

| Decision | Confidence | Basis |
|----------|------------|-------|
| corner is the right choice | HIGH | Universal in GW literature; PyPI confirms 3.13 compat; matplotlib-native |
| matplotlib Mollweide over healpy | HIGH | ~100 detections = scatter plot, not HEALPix map. Built-in projection sufficient. Avoids C extension build dependency. |
| No ArviZ needed | HIGH | Fisher-matrix pipeline, not MCMC chains. ~20 transitive deps for unused features. |
| No interactive libs needed | HIGH | Thesis output is PDF. JupyterLab already available for dev exploration. |
| LaTeX as optional toggle | HIGH | Standard for physics theses. Must not break CI/cluster (where TeX may be absent). |
| No SciencePlots | HIGH | Custom mplstyle already exists. Competing style definitions. |
| No seaborn | HIGH | Physics-specific plots. Style pollution risk. |

## Sources

- [corner.py documentation](https://corner.readthedocs.io/en/latest/) -- v2.2.3, Python >=3.9
- [corner on PyPI](https://pypi.org/project/corner/)
- [healpy on PyPI](https://pypi.org/project/healpy/) -- v1.19.0, Python 3.13 wheels (evaluated, not recommended for this use case)
- [ArviZ on PyPI](https://pypi.org/project/arviz/) -- v1.0.0, requires Python >=3.12
- [ChainConsumer on PyPI](https://pypi.org/project/chainconsumer/) -- appears unmaintained
- [SciencePlots on PyPI](https://pypi.org/project/SciencePlots/) -- v2.2.1
- [IGWN sky map tutorial](https://emfollow.docs.ligo.org/userguide/tutorial/skymaps.html) -- healpy/Mollweide in GW context
- [gwnrtools corner plot tutorial](https://gwnrtools.github.io/gwnrtools/tutorials/MakingUsefulCornerPlots.html) -- corner.py in GW
- [Publication-quality matplotlib for LaTeX](https://jwalton.info/Embed-Publication-Matplotlib-Latex/)
- [matplotlib for papers](https://github.com/jbmouret/matplotlib_for_papers) -- figure sizing for LaTeX
- [matplotlib Mollweide projection](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/geo_demo.html) -- built-in support

---

*Stack analysis: 2026-04-01 -- v1.3 milestone (visualization overhaul)*
