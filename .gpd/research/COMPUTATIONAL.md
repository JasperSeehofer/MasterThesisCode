# Computational Methods: Interactive Scientific Visualization for GitHub Pages

**Physics Domain:** Interactive scientific visualization and GitHub Pages integration
**Researched:** 2026-04-07

### Scope Boundary

COMPUTATIONAL.md covers computational TOOLS, libraries, and infrastructure for adding interactive scientific figures (Plotly), Jupyter-based parameter exploration (JupyterLite), and notebook-in-Sphinx integration (MyST-NB) to the existing Sphinx + GitHub Pages deployment. It does NOT cover the physics content of the figures.

---

## Recommendation Summary

| Decision | Recommendation | Rationale |
|---|---|---|
| Interactive plotting library | **Plotly** (not Bokeh) | Better Sphinx integration, self-contained HTML, CDN option, WebGL for large data |
| Notebook-in-Sphinx extension | **MyST-NB** (not nbsphinx) | Caching via jupyter-cache, direct docutils AST, glue for embedding outputs, active maintenance |
| Serverless notebooks | **JupyterLite via jupyterlite-sphinx** (not Voila) | Static HTML deployment, no server needed, scipy/numpy available via Pyodide |
| Data reduction for 580MB JSON | **Pre-aggregate at build time** into ~1-5MB summary JSON | Client-side can handle ~200k points max; pre-compute KDE/binned summaries |
| Plotly HTML embedding | **CDN mode** (`include_plotlyjs='cdn'`) for Sphinx pages | Avoids 3MB per-figure overhead; single CDN load for all figures |

---

## Interactive Plotting: Plotly

### Why Plotly Over Bokeh

| Criterion | Plotly | Bokeh |
|---|---|---|
| Self-contained HTML export | `write_html()` -- single file, works offline | Requires `components()` or full HTML template |
| Sphinx integration | Sphinx-Gallery `_repr_html_` capture, raw HTML include, MyST-NB output capture | No maintained Sphinx directive |
| CDN option (reduce file size) | `include_plotlyjs='cdn'` reduces HTML from ~5MB to ~2KB + CDN load | No equivalent; must bundle or serve JS separately |
| WebGL for large datasets | `Scattergl`, `Heatmapgl` -- up to ~200k points client-side | Has WebGL support but less polished API |
| Scientific figure quality | Good LaTeX-like labels via MathJax, publication-style templates | Adequate but less common in scientific publishing |
| Mobile compatibility | Responsive by default, touch zoom/pan | Responsive but requires explicit configuration |
| Ecosystem maturity | Larger community, more documentation, Dash for apps | Smaller community, Panel for apps |

**Plotly is the clear winner** for this use case because the Sphinx integration story is stronger and the self-contained HTML export with CDN option solves the file-size problem elegantly.

### Plotly Version and Installation

```bash
uv add --optional dev "plotly>=6.0.0" "kaleido>=0.2.1"
```

- **plotly >= 6.0.0**: Current stable series (6.6.0 as of early 2026). Major API is stable.
- **kaleido**: For static image export (PNG/PDF fallback). Required for `fig.write_image()`.

### Plotly HTML File Sizes

| Mode | Size per figure | Use when |
|---|---|---|
| `include_plotlyjs=True` (default) | ~3-5 MB (embeds plotly.js) | Standalone offline files |
| `include_plotlyjs='cdn'` | ~2-50 KB (data only) | Sphinx/GitHub Pages (internet available) |
| `include_plotlyjs=False` | ~1-50 KB (data only, no JS) | Multiple figures on one page, load JS once |

**For Sphinx pages: Use `include_plotlyjs='cdn'`** for individual figures, or load plotly.js once in a custom template and use `include_plotlyjs=False` for all figures on the page.

### WebGL Performance Thresholds

| Data points | Renderer | Performance |
|---|---|---|
| < 1,000 | SVG (default) | Smooth, full interactivity |
| 1,000 - 200,000 | WebGL (`Scattergl`) | Smooth on modern GPU |
| 200,000 - 1,000,000 | WebGL + downsampling | Usable with `plotly-resampler` (LTTB algorithm) |
| > 1,000,000 | Pre-aggregate server-side | Client cannot handle; pre-bin/KDE |

### Known Issue: MathJax Conflict

Both Plotly and Sphinx load MathJax. Double-loading can cause rendering issues for LaTeX equations in interactive plots. The issue is documented at plotly/plotly.js#2403. Mitigation: test early with a single figure; if conflict appears, configure Plotly to use Sphinx's already-loaded MathJax instance rather than loading its own.

---

## Data Reduction Strategy for 580MB JSON Files

This is the critical engineering challenge. The per-galaxy likelihood JSON files are ~580MB each (7 h-values). Client-side JavaScript cannot load or render this. The solution is **build-time pre-aggregation**.

### Architecture: Build-Time Data Pipeline

```
[580MB JSON per h-value] --> [CI build script] --> [Pre-aggregated summary JSON, ~1-5MB]
                                                         |
                                                    [Plotly figures as HTML]
                                                         |
                                                    [Sphinx build] --> [GitHub Pages]
```

### Pre-Aggregation Strategies

| Strategy | Output size | Preserves | Loses | Best for |
|---|---|---|---|---|
| **Binned histogram** | ~10 KB | Distribution shape | Individual galaxy values | Posterior plots, H0 likelihood |
| **KDE on grid** | ~50-100 KB | Smooth density | Discrete structure | Sky maps, parameter distributions |
| **Top-N selection** | ~1-5 MB | Outliers, interesting events | Bulk distribution | Per-event detail views |
| **Decimation (every Nth)** | ~5-50 MB | Uniform sampling | Density information | Scatter plots of raw data |
| **LTTB downsampling** | ~100 KB | Visual shape of curves | High-frequency detail | 1D curves |

### Recommended Data Pipeline

Implement a **build-time reduction script** (`scripts/prepare_interactive_data.py`) that reads full 580MB JSONs and produces:

1. `combined_summary.json` (~50KB): binned posteriors per h-value on a 100-point h-grid
2. `event_highlights.json` (~200KB): top-50 highest-SNR events with full per-galaxy data
3. `sky_map_grid.json` (~500KB): HEALPix-binned sky distribution (nside=32, 12288 pixels)
4. `parameter_distributions.json` (~50KB): histograms of all 14 EMRI parameters (100 bins each)
5. `convergence_data.json` (~50KB): posterior convergence curves (already small)

**Total pre-aggregated data budget: ~1 MB.** Well within GitHub Pages limits.

### Specific Plot Types and Data Handling

| Plot | Raw data size | Reduction method | Target size | Interactivity |
|---|---|---|---|---|
| H0 posterior curve | ~580 MB (per-galaxy likelihoods) | Pre-compute combined posterior on h-grid (100 points) | ~5 KB | Hover: h-value, posterior value |
| Sky map of detections | ~50 MB (all event positions) | HEALPix binning (nside=32) | ~100 KB | Hover: pixel count, mean SNR |
| SNR vs distance scatter | ~10 MB (all events) | LTTB downsample to 5000 points + WebGL | ~200 KB | Hover: event ID, SNR, d_L |
| Fisher matrix heatmap | ~1 KB per event | Select representative event | ~5 KB | Hover: parameter names, values |
| Parameter distributions | ~10 MB | Histogram with 100 bins | ~10 KB per param | Hover: bin edges, counts |
| Convergence curves | ~1 MB | Already small; use directly | ~50 KB | Hover: iteration, metric value |

### Data Strategy for CI

The 580MB JSON files **cannot** be in the git repo or CI artifacts. Options:

| Strategy | Pros | Cons | Verdict |
|---|---|---|---|
| **A: Pre-aggregate locally, commit summary JSONs** | Simple, fast CI, ~1 MB in repo | Manual step after each campaign | **Use this** |
| B: Store full JSONs in GitHub Release assets | Reproducible, versioned | 4 GB per release; slow CI download | Archival only |
| C: Git LFS for full JSONs | Transparent git workflow | LFS bandwidth limits on free tier | Too expensive |
| D: External storage (S3/GCS) + CI fetch | Scalable | Infrastructure complexity | Overkill for thesis |

**Recommended approach (Strategy A):**

1. Run `scripts/prepare_interactive_data.py` locally after each production campaign
2. Commit the output to `docs/source/_static/data/*.json` (~1 MB total)
3. CI builds Sphinx + notebooks using these pre-aggregated files
4. No large data handling needed in CI

---

## Sphinx Integration: MyST-NB

### Why MyST-NB Over nbsphinx

| Feature | MyST-NB | nbsphinx |
|---|---|---|
| Execution caching | Built-in via jupyter-cache; skips unchanged notebooks | Re-executes every build |
| Markdown parser | MyST (direct to docutils AST) | Pandoc (Markdown -> reST -> docutils) |
| Glue mechanism | Embed notebook outputs in any doc page | Not available |
| Error reporting | Detailed per-cell error messages | Basic |
| Maintenance | Active (executablebooks project) | Active but smaller team |

**MyST-NB wins** because caching is essential (interactive notebooks may be slow to execute) and the glue mechanism lets you compute a Plotly figure in a notebook and embed it in a regular Sphinx RST page.

### Installation

```bash
uv add --optional dev "myst-nb>=1.2.0" "jupyter-cache>=1.0.0"
```

### Sphinx Configuration Changes

Add to `docs/source/conf.py`:

```python
extensions = [
    # ... existing extensions ...
    "myst_nb",
]

# MyST-NB configuration
nb_execution_mode = "cache"          # Cache outputs; re-execute only on change
nb_execution_timeout = 300           # 5 min timeout per cell
nb_execution_raise_on_error = True   # Fail build on notebook error
nb_render_image_options = {"width": "100%"}

# Suppress duplicate mathjax warning
suppress_warnings = ["mystnb.unknown_mime_type"]
```

### Embedding Plotly in Sphinx via MyST-NB

**Option A: Notebook with Plotly output (recommended for interactive exploration)**

Create `docs/source/interactive/posterior_explorer.ipynb` -- MyST-NB executes it during `sphinx-build`, captures the Plotly HTML output, and embeds it in the built page. The notebook loads pre-aggregated JSON from `_static/data/`, creates Plotly figures, and calls `fig.show()`.

**Option B: Raw HTML include for pre-built Plotly figures**

Generate Plotly HTML files via `scripts/prepare_interactive_data.py`, then include in RST pages:

```rst
.. raw:: html
   :file: _static/interactive/posterior_plot.html
```

Avoids notebook execution during Sphinx build but requires a separate generation step.

**Recommendation:** Use Option A for figures that benefit from notebook narrative (explanation + code + figure). Use Option B for figures that are just interactive versions of existing static plots (faster build, simpler).

---

## JupyterLite for Parameter Exploration

### Why JupyterLite Over Voila

| Feature | JupyterLite | Voila |
|---|---|---|
| Server requirement | **None** (runs in browser via WASM/Pyodide) | Requires running Jupyter server |
| GitHub Pages compatible | **Yes** (static files only) | **No** (needs server) |
| Scientific library support | NumPy, SciPy, Matplotlib, Pandas via Pyodide | Full CPython, all libraries |
| Startup time | 5-15 seconds (WASM initialization) | Instant (server already running) |
| Memory limit | Browser tab limit (~2-4 GB) | Server RAM |

**JupyterLite is the only option** for serverless GitHub Pages. Voila requires a running server.

### JupyterLite Limitations for This Project

| Limitation | Impact | Mitigation |
|---|---|---|
| No CuPy/GPU | Cannot run waveform generation | Pre-compute all heavy results; notebooks do lightweight analysis only |
| ~2-4 GB browser memory | Cannot load 580MB JSON | Load only pre-aggregated summaries (~1 MB) |
| Pyodide scipy version lag | Minor API differences possible | Pin to features in Pyodide's scipy; test in CI |
| No filesystem access | Cannot read local files | Serve data as static JSON via browser `fetch()` |
| 5-15s cold start | Users wait on first load | Show loading indicator; keep notebooks small |
| No `few` (EMRI waveforms) | Cannot generate waveforms in browser | Pre-compute waveform snapshots as JSON arrays |

### jupyterlite-sphinx Integration

```bash
uv add --optional dev "jupyterlite-sphinx>=0.16.0" "jupyterlite-pyodide-kernel>=0.4.0"
```

Add to `docs/source/conf.py`:

```python
extensions = [
    # ... existing ...
    "jupyterlite_sphinx",
]

jupyterlite_contents = ["interactive/notebooks/"]  # Notebooks to bundle
jupyterlite_dir = "interactive/lite"               # Build output directory
```

In a Sphinx RST page:

```rst
Interactive Parameter Exploration
=================================

Adjust the Hubble constant and SNR threshold to explore their effect on the posterior:

.. jupyterlite:: interactive/notebooks/h0_explorer.ipynb
   :width: 100%
   :height: 600px
```

### ipywidgets in JupyterLite

ipywidgets work in JupyterLite via the Pyodide kernel. For slider-based parameter exploration:

```python
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from js import fetch
import json

# Load pre-aggregated data
response = await fetch("../data/combined_summary.json")
data = json.loads(await response.text())

h_slider = widgets.FloatSlider(value=0.73, min=0.5, max=1.0, step=0.01,
                                description='h value:')
snr_slider = widgets.FloatSlider(value=20.0, min=5.0, max=100.0, step=1.0,
                                  description='SNR threshold:')

def update_plot(h, snr_threshold):
    # Filter and replot with new parameters using pre-aggregated data
    ...

widgets.interactive(update_plot, h=h_slider, snr_threshold=snr_slider)
```

**Note:** Start with manual cell re-execution (simpler). Add ipywidgets only if the user experience warrants the added complexity.

---

## CI Integration: Modified GitHub Actions Workflow

### Current Pipeline

```
check --> integration --> docs ------> pages
                |           |            |
          test-plots    docs-html    merge artifacts -> deploy
```

### Proposed Pipeline

```
check --> integration --> generate-interactive --> docs --> pages
                |              |                    |        |
          test-plots    interactive HTML +    docs-html   merge all -> deploy
                        summary JSON files
```

### New CI Job: `generate-interactive`

```yaml
  generate-interactive:
    runs-on: ubuntu-latest
    needs: check
    steps:
      - uses: actions/checkout@v6

      - name: Install uv
        uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true

      - name: Set up Python
        run: uv python install

      - name: Install dependencies
        run: uv sync --extra cpu --extra dev

      - name: Generate interactive figures
        run: uv run python -m scripts.generate_interactive_figures

      - name: Upload interactive artifacts
        uses: actions/upload-artifact@v7
        with:
          name: interactive-figures
          path: docs/build/interactive/
```

### Modified `pages` Job

```yaml
  pages:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    needs: [integration, docs, generate-interactive]
    permissions:
      pages: write
      id-token: write
    steps:
      - name: Download docs artifact
        uses: actions/download-artifact@v8
        with:
          name: docs-html
          path: _site/

      - name: Download test plots artifact
        uses: actions/download-artifact@v8
        with:
          name: integration-test-plots
          path: _site/test-plots/
        continue-on-error: true

      - name: Download interactive figures
        uses: actions/download-artifact@v8
        with:
          name: interactive-figures
          path: _site/interactive/

      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@v4
        with:
          path: _site/

      - name: Deploy to GitHub Pages
        uses: actions/deploy-pages@v4
```

**Estimated CI build time increase:** ~2-3 minutes for JupyterLite build, ~1 minute for notebook execution with caching. Total: ~3-4 minutes added.

---

## Plotly Style Integration with Existing Matplotlib Pipeline

### Dual-Output Pattern

Keep existing matplotlib plots for the thesis PDF. Add Plotly equivalents for the website. Do NOT replace matplotlib -- the thesis needs vector PDF output that Plotly cannot match.

```python
# Existing: matplotlib for thesis
def plot_posterior_matplotlib(data: PosteriorData) -> tuple[Figure, Axes]:
    """Static posterior plot for thesis PDF."""
    fig, ax = get_figure(preset="single")
    # ... matplotlib code ...
    return fig, ax

# New: Plotly for website
def plot_posterior_plotly(data: PosteriorData) -> go.Figure:
    """Interactive posterior plot for GitHub Pages."""
    fig = go.Figure()
    # ... plotly code with hover tooltips ...
    fig.update_layout(template="emri_thesis")
    return fig
```

### Plotly Template for Consistent Style

Create `master_thesis_code/plotting/_plotly_style.py` to match `emri_thesis.mplstyle`:

```python
"""Plotly template matching emri_thesis.mplstyle."""
import plotly.graph_objects as go
import plotly.io as pio

EMRI_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="serif", size=12),
        xaxis=dict(showgrid=True, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridcolor="lightgray"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        colorway=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"],
    )
)
pio.templates["emri_thesis"] = EMRI_TEMPLATE
pio.templates.default = "emri_thesis"
```

---

## Complete Dependency Additions

```bash
# Interactive plotting
uv add --optional dev "plotly>=6.0.0" "kaleido>=0.2.1"

# MyST-NB for notebook integration in Sphinx
uv add --optional dev "myst-nb>=1.2.0" "jupyter-cache>=1.0.0"

# JupyterLite for serverless notebooks on GitHub Pages
uv add --optional dev "jupyterlite-sphinx>=0.16.0" "jupyterlite-pyodide-kernel>=0.4.0"
```

No additional deps needed for data pre-aggregation (numpy/scipy/pandas already available).

---

## Open Questions

| Question | Why Open | Impact on Project | Approaches Being Tried |
|---|---|---|---|
| ipywidgets vs manual cell re-execution in JupyterLite? | Widgets add complexity; manual is simpler but less polished | UX quality of parameter exploration | Start with manual; add widgets if time permits |
| MathJax conflict between Plotly and Sphinx? | Both load MathJax; double-loading can cause rendering issues | Equation rendering in interactive plots | Known issue (plotly/plotly.js#2403); test early with one figure |
| MyST-NB vs raw HTML includes for Plotly? | MyST-NB is cleaner but adds execution overhead | Build time, complexity | Use MyST-NB for widget notebooks, raw HTML for static Plotly |
| How to version pre-aggregated data alongside code? | Data changes with each production campaign | Reproducibility of interactive figures | Strategy A (commit ~1MB summaries) handles this |

## Anti-Approaches

| Anti-Approach | Why Avoid | What to Do Instead |
|---|---|---|
| Bokeh for Sphinx integration | No maintained Sphinx directive; standalone HTML requires more boilerplate | Use Plotly with CDN mode |
| Voila for GitHub Pages | Requires running server; incompatible with static hosting | Use JupyterLite (WASM, serverless) |
| Loading 580MB JSON client-side | Browser will crash or hang for minutes | Pre-aggregate at build time to ~1 MB |
| Replacing matplotlib with Plotly for thesis | Plotly cannot produce publication-quality vector PDF for LaTeX | Keep dual pipeline: mpl for thesis, Plotly for web |
| nbsphinx for notebook rendering | Re-executes on every build; no caching; pandoc dependency | Use MyST-NB with jupyter-cache |
| Self-contained Plotly HTML (`include_plotlyjs=True`) | 3-5 MB per figure; 10 figures = 50 MB of duplicated plotly.js | Use CDN mode or load plotly.js once in template |
| Git LFS for full result JSONs | Bandwidth limits on free GitHub tier; overkill for thesis | Commit only pre-aggregated summaries (~1 MB) |
| sphinx-plotly-directive | Inactive maintenance; last release >12 months ago per Snyk | Use MyST-NB or raw HTML includes |

## Logical Dependencies

```
Pre-aggregated data (scripts/prepare_interactive_data.py)
  --> Plotly figures (plotting/_plotly_*.py or notebooks)
    --> Sphinx build with MyST-NB (docs/)
      --> GitHub Pages deploy (CI)

JupyterLite build (jupyterlite-sphinx)
  --> Depends on: pre-aggregated data in docs/source/_static/data/
  --> Depends on: notebook files in docs/source/interactive/notebooks/

MyST-NB execution
  --> Depends on: plotly, numpy, scipy in dev dependencies
  --> Depends on: pre-aggregated data accessible during Sphinx build
```

## Recommended Investigation Scope

Prioritize:
1. **Data pre-aggregation script** -- this unblocks everything else; without small data files, no interactive plots work
2. **Single Plotly figure in Sphinx** -- prove the integration works end-to-end (CDN mode, no MathJax conflict)
3. **MyST-NB with one notebook** -- confirm caching works, output renders in built docs
4. **JupyterLite embed** -- test that scipy/numpy work in Pyodide for the specific operations needed

Defer:
- ipywidgets in JupyterLite: requires results from steps 1-4 first
- Full Plotly versions of all matplotlib plots: incremental, do after proving the pipeline
- plotly-resampler: only needed if scatter plots have >200k points after pre-aggregation

## Key References

- [Plotly Python -- HTML export](https://plotly.com/python/interactive-html-export/)
- [Plotly performance guide -- WebGL](https://plotly.com/python/performance/)
- [MyST-NB documentation](https://github.com/executablebooks/MyST-NB)
- [MyST-NB vs nbsphinx discussion](https://github.com/pydata/pydata-sphinx-theme/issues/1474)
- [jupyterlite-sphinx](https://jupyterlite-sphinx.readthedocs.io/en/latest/)
- [JupyterLite deploy guide](https://jupyterlite.readthedocs.io/en/latest/quickstart/deploy.html)
- [Pyodide 0.28 release](https://blog.pyodide.org/posts/0.28-release/)
- [Plotly.js MathJax conflict](https://github.com/plotly/plotly.js/issues/2403)
- [Plotly vs Bokeh comparison](https://pauliacomi.com/2020/06/07/plotly-v-bokeh.html)
- [plotly-resampler for large datasets](https://ploomber.io/blog/plotly-large-dataset/)
- [Sphinx-Gallery Plotly support](https://sphinx-gallery.github.io/dev/auto_plotly_examples/plot_0_plotly.html)
