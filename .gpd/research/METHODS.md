# Methods: Publication-Quality Figure Design for EMRI Dark Siren Thesis

**Project:** EMRI Dark Siren H0 Inference -- Figure Refinement
**Domain:** Scientific visualization, matplotlib publication techniques
**Researched:** 2026-04-07

### Scope Boundary

This file covers analytical and practical METHODS for creating modern, publication-quality scientific figures in matplotlib. It does NOT cover the physics pipeline or computational infrastructure -- those belong in other research files. It answers: "What specific techniques, rcParams, palettes, and patterns produce figures matching the refined aesthetic of arXiv:2406.09228v1 (Inchauspe et al. 2025) for a REVTeX thesis?"

---

## Context and Current State

The current plotting pipeline uses `emri_thesis.mplstyle` with:
- `figure.figsize: 6.4, 4.0` (matplotlib default, not journal-matched)
- `font.size: 11` (too large for single-column REVTeX)
- `text.usetex: False` (mathtext renderer, not true LaTeX)
- `viridis` colormap (good for perceptual uniformity, wrong aesthetic for the target style)
- `tab10`-derived color cycle (not colorblind-optimized, not sequential emphasis)
- No spine removal, no tick direction control, default legend frames

The target aesthetic (arXiv:2406.09228v1) uses: clean minimal spines, LaTeX/Computer Modern typography, muted blue-dominant sequential palettes, filled contour regions for posteriors with smooth gradients, and a generally "less is more" design philosophy.

---

## Recommended Methods

### Primary Style Method: Custom mplstyle with LaTeX Typography

| Method | Purpose | Applicability | Limitations |
|---|---|---|---|
| Custom `.mplstyle` + `apply_style()` | Global rcParams for all figures | All plots | Cannot handle per-figure contour styling |
| `text.usetex: True` with Computer Modern | Journal-matched typography | All text, labels, legends | Requires TeX installation; slower rendering |
| Programmatic spine removal | Minimal L-frame axes | All 2D plots | Must be called per-axes or via mplstyle |
| `contourf` with custom `ListedColormap` | Smooth posterior fills | Posterior/density plots | Requires KDE preprocessing |

### Primary Palette Method: Curated Colorblind-Safe Palette

| Method | Purpose | Applicability | Limitations |
|---|---|---|---|
| Custom sequential blue palette (see below) | Posterior contour fills | Density/contour plots | Single-hue, not for categorical data |
| Okabe-Ito qualitative palette | Multi-line/categorical plots | Line plots, scatter, legends | Only 8 colors |
| Grayscale-safe accent color | Truth/reference lines | Overlay annotations | Must contrast with blue fills |

### Primary Smoothing Method: scipy.stats.gaussian_kde

| Method | Purpose | Convergence | Cost Scaling | Implementation |
|---|---|---|---|---|
| `scipy.stats.gaussian_kde` | Smooth 1D/2D posteriors from discrete samples | Bandwidth-dependent | O(N*M) for N samples, M grid points | scipy built-in |
| `scipy.ndimage.gaussian_filter` | Smooth 2D histograms on regular grids | sigma-dependent | O(grid_size) | scipy built-in |
| `corner.corner` with `smooth` param | Corner/triangle plots with credible regions | Tunable via `smooth` | O(N_params^2 * N_samples) | `corner` package |

---

## Method Details

### Method 1: REVTeX-Compatible mplstyle Configuration

**What:** A complete `.mplstyle` file that produces figures matching APS/REVTeX journal requirements and the modern minimal aesthetic.

**APS/REVTeX Figure Requirements (from APS Author Guidelines):**
- Single-column width: 3.375 inches (8.6 cm)
- Double-column width: 7.0 inches (17.8 cm)
- Minimum text height: 2 mm for capital letters and numerals
- Preferred formats: PDF (vector), EPS, PNG at 300+ dpi
- Fonts must be embedded in vector formats

**At 3.375" single-column width, the 2 mm minimum text constraint means:**
- 2 mm = 5.67 pt, so the absolute minimum font size is ~6 pt
- For readability, tick labels should be 7-8 pt, axis labels 8-9 pt
- This matches the `use_latex` branch in the current `apply_style()` which already sets font.size=10, but that is for the overall default. Axis-specific sizes need tuning.

**Recommended rcParams (complete mplstyle):**

```ini
# === Typography (LaTeX/Computer Modern) ===
text.usetex: True
font.family: serif
font.serif: Computer Modern Roman
mathtext.fontset: cm

# Font sizes calibrated for 3.375" single-column REVTeX
font.size: 8
axes.titlesize: 9
axes.labelsize: 8
xtick.labelsize: 7
ytick.labelsize: 7
legend.fontsize: 7

# === Figure dimensions ===
figure.figsize: 3.375, 2.53          # single-column, 4:3 aspect
figure.dpi: 150                       # screen preview
savefig.dpi: 300                      # publication output
savefig.bbox: tight
savefig.pad_inches: 0.02
figure.constrained_layout.use: True

# === Axes: minimal L-frame ===
axes.linewidth: 0.6
axes.edgecolor: 262626
axes.labelcolor: 262626
axes.spines.top: False
axes.spines.right: False
axes.grid: False
axes.axisbelow: True
axes.prop_cycle: cycler('color', ['0072B2', 'D55E00', '009E73', 'CC79A7', 'E69F00', '56B4E9', 'F0E442', '000000'])

# === Ticks: inward, subtle ===
xtick.direction: in
ytick.direction: in
xtick.major.width: 0.6
ytick.major.width: 0.6
xtick.minor.width: 0.4
ytick.minor.width: 0.4
xtick.major.size: 3.5
ytick.major.size: 3.5
xtick.minor.size: 2.0
ytick.minor.size: 2.0
xtick.minor.visible: True
ytick.minor.visible: True
xtick.top: False
ytick.right: False
xtick.color: 262626
ytick.color: 262626

# === Lines ===
lines.linewidth: 1.0
lines.markersize: 4

# === Legend: frameless ===
legend.frameon: False
legend.borderpad: 0.3
legend.handlelength: 1.5
legend.handletextpad: 0.4
legend.columnspacing: 1.0

# === Colormaps ===
image.cmap: Blues

# === Rendering ===
agg.path.chunksize: 10000
pdf.fonttype: 42
ps.fonttype: 42
```

**Key differences from current style:**
- `axes.spines.top: False` and `axes.spines.right: False` -- the single most impactful change for modern appearance
- `xtick.direction: in` -- inward ticks look cleaner and do not occlude data
- Font sizes reduced from 10-13 pt range to 7-9 pt range (correct for 3.375" width)
- `legend.frameon: False` -- removes the box around legends
- `pdf.fonttype: 42` and `ps.fonttype: 42` -- embeds fonts as TrueType in PDF/PS (required by many journals)
- Color cycle changed from tab10 to Okabe-Ito
- Near-black (`#262626`) instead of pure black for axes/ticks (softer, modern look)

**Double-column variant:** For double-column figures (7.0" wide), multiply font sizes by ~1.3x (so font.size: 10, labels: 10, ticks: 9). Use `figure.figsize: 7.0, 3.5` or similar.

**Implementation note for `apply_style()`:** The existing `use_latex=True` branch should be replaced entirely by this new mplstyle. The `use_latex=False` fallback (for CI) should use mathtext with `mathtext.fontset: cm` to approximate the look without requiring TeX.

---

### Method 2: Colorblind-Safe Palette System

**What:** A three-tier palette system: (1) sequential blue for density/contour fills, (2) qualitative Okabe-Ito for categorical/multi-line plots, (3) semantic accent colors for truth/mean/reference.

#### Tier 1: Sequential Blue Palette (Contour Fills)

For posterior density contours, use a custom sequential blue palette built from matplotlib's `Blues` colormap, sampled at specific points to create discrete credible-interval bands:

```python
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap

# Sample Blues colormap at 4 levels for 2-sigma, 1-sigma, inner fills
blues = get_cmap("Blues")
CONTOUR_COLORS = [
    blues(0.15),  # lightest: 3-sigma / background
    blues(0.30),  # 2-sigma band
    blues(0.50),  # 1-sigma band
    blues(0.75),  # peak / mode region
]

# For continuous contourf, use Blues directly or a truncated version:
from matplotlib.colors import LinearSegmentedColormap
blues_truncated = LinearSegmentedColormap.from_list(
    "blues_trunc", blues(np.linspace(0.1, 0.85, 256))
)
```

**Why Blues and not a custom palette:** The matplotlib `Blues` colormap is perceptually uniform (monotonically increasing lightness), prints correctly in grayscale (lighter values map to lighter gray), and is universally recognized in GW/cosmology literature. It is the standard for single-parameter posteriors.

**Grayscale safety:** `Blues` maps monotonically to grayscale because it varies only in lightness. This is not true of diverging or multi-hue colormaps.

#### Tier 2: Okabe-Ito Qualitative Palette (Multi-Line Plots)

The Okabe-Ito palette provides 8 colors distinguishable under all common forms of color vision deficiency:

| Index | Name | Hex | Use Case |
|---|---|---|---|
| 0 | Blue | `#0072B2` | Primary data line |
| 1 | Vermillion | `#D55E00` | Secondary data / contrast |
| 2 | Bluish Green | `#009E73` | Tertiary data |
| 3 | Reddish Purple | `#CC79A7` | Quaternary data |
| 4 | Orange | `#E69F00` | Fifth series |
| 5 | Sky Blue | `#56B4E9` | Light accent |
| 6 | Yellow | `#F0E442` | Rarely used (low contrast on white) |
| 7 | Black | `#000000` | Reference/baseline |

**Why Okabe-Ito over tab10:** Tab10 fails under deuteranopia (green-red confusion between colors 0/2 and 1/3). Okabe-Ito was specifically designed to remain distinguishable under protanopia, deuteranopia, and tritanopia. It is the recommended palette for scientific publications per Petroff (2021) and is now included in matplotlib 3.9+ as the `petroff10` style.

Source: Okabe & Ito (2002), "Color Universal Design (CUD) -- How to make figures and presentations that are friendly to Colorblind people," https://jfly.uni-koeln.de/color/

#### Tier 3: Semantic Accent Colors

| Role | Hex | Description | Rationale |
|---|---|---|---|
| TRUTH | `#D55E00` | Vermillion | High contrast against blue fills; Okabe-Ito vermillion |
| COMBINED | `#0072B2` | Deep blue | Matches contour palette; Okabe-Ito blue |
| INDIVIDUAL | `#56B4E9` | Sky blue | Lighter than combined; distinguishable from deep blue |
| REFERENCE | `#7F7F7F` | Medium gray | Neutral, does not compete with data |
| EDGE | `#262626` | Near-black | Softer than pure black |

**Change from current `_colors.py`:** Replace `TRUTH = "#2ca02c"` (green, from tab10) with `"#D55E00"` (vermillion, Okabe-Ito). Green truth lines are invisible to deuteranopes against blue contours.

---

### Method 3: Contour Smoothing for Posterior Distributions

**What:** Convert discrete posterior samples or histogram grids into smooth contour plots showing credible regions.

**Two distinct input scenarios in the codebase:**

**Scenario A: Discrete MCMC/posterior samples** (e.g., H0 posterior from `BayesianStatistics`)
Use `scipy.stats.gaussian_kde` for 1D, and `gaussian_kde` on stacked 2D data for corner plots.

```python
from scipy.stats import gaussian_kde

# 1D posterior smoothing
def smooth_posterior_1d(samples, x_grid, bw_method="scott"):
    """KDE-smooth a 1D posterior from discrete samples."""
    kde = gaussian_kde(samples, bw_method=bw_method)
    density = kde(x_grid)
    return density / np.trapz(density, x_grid)  # normalize

# 2D contour from samples
def contour_2d_from_samples(x_samples, y_samples, ax, levels=[0.68, 0.95]):
    """Plot 2D credible region contours from posterior samples."""
    data = np.vstack([x_samples, y_samples])
    kde = gaussian_kde(data)
    x_grid = np.linspace(x_samples.min(), x_samples.max(), 200)
    y_grid = np.linspace(y_samples.min(), y_samples.max(), 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    # Convert levels from credible fractions to density thresholds
    sorted_z = np.sort(Z.ravel())[::-1]
    cumsum = np.cumsum(sorted_z) / np.sum(sorted_z)
    thresholds = [sorted_z[np.searchsorted(cumsum, level)] for level in levels]

    ax.contourf(X, Y, Z, levels=[thresholds[1], thresholds[0], Z.max()],
                colors=[blues(0.25), blues(0.50)], alpha=0.8)
    ax.contour(X, Y, Z, levels=thresholds, colors=[blues(0.7)],
               linewidths=0.8)
```

**Scenario B: Pre-computed grids** (e.g., P_det on a regular (z, M) grid)
Use `scipy.ndimage.gaussian_filter` to smooth the grid, then `contourf`:

```python
from scipy.ndimage import gaussian_filter

def smooth_contour_from_grid(x_edges, y_edges, grid_values, ax, sigma=1.0):
    """Smooth a 2D histogram/grid and plot filled contours."""
    smoothed = gaussian_filter(grid_values, sigma=sigma)
    ax.contourf(x_edges, y_edges, smoothed.T, levels=20,
                cmap="Blues", alpha=0.9)
    ax.contour(x_edges, y_edges, smoothed.T, levels=5,
               colors=["#262626"], linewidths=0.4, alpha=0.5)
```

**Bandwidth/sigma selection:**
- For `gaussian_kde`: use `bw_method="scott"` (default). Scott's rule: `h = N^{-1/(d+4)}` where d is dimension. For 1D with ~1000 samples, this gives ~0.1 * std(data). Avoid over-smoothing by checking that known features (e.g., the posterior peak near H0=0.73) remain resolved.
- For `gaussian_filter`: `sigma=1.0` in grid units is a good starting point. Increase to 1.5-2.0 for very noisy grids. Never smooth beyond the grid resolution (sigma should be < half the number of bins in the narrowest dimension).

**Credible interval contour levels:** For a 2D Gaussian, the standard sigma-levels correspond to:

| Credible Level | Fraction of Probability | Contour Label |
|---|---|---|
| 1-sigma | 0.3935 | 39.3% |
| 2-sigma | 0.8647 | 86.5% |
| 3-sigma | 0.9889 | 98.9% |

The `corner` library uses levels `[0.118, 0.393, 0.675, 0.864]` by default (half-sigma spacing). For publication, the conventional choice is `[0.68, 0.95]` (1-sigma and 2-sigma in 1D, corresponding to 39.3% and 86.5% enclosed probability in 2D).

**References:**
- Scott (1992), "Multivariate Density Estimation," Wiley -- bandwidth selection theory
- `scipy.stats.gaussian_kde` documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
- `corner` library: https://corner.readthedocs.io/en/latest/api/ -- `smooth` parameter, `levels` parameter

---

### Method 4: Modern Aesthetic Techniques (What Makes Plots Look "Modern")

**What:** Specific techniques that distinguish the arXiv:2406.09228v1 style from traditional scientific plots.

The difference between "traditional" and "modern" scientific figures comes down to a handful of specific choices:

| Traditional | Modern | Implementation |
|---|---|---|
| Box frame (4 spines) | L-frame (2 spines) | `axes.spines.top/right: False` |
| Outward ticks | Inward ticks | `xtick.direction: in` |
| Boxed legend | Frameless legend | `legend.frameon: False` |
| Pure black axes | Near-black (`#262626`) | `axes.edgecolor: 262626` |
| Bold/large labels | Subtle/matched labels | Font sizes 7-9 pt at column width |
| Dense grid lines | No grid or faint grid | `axes.grid: False` |
| Rainbow/jet colormaps | Perceptually uniform, single-hue | `Blues`, Okabe-Ito |
| Thick lines (2.0+) | Thinner lines (0.8-1.2) | `lines.linewidth: 1.0` |
| Cluttered annotations | Clean whitespace | Fewer labels, tighter padding |
| Default padding | Tight layout | `savefig.bbox: tight` |

**Filled credible regions with gradient alpha:** For 1D posteriors, the "modern" look uses `fill_between` with transparency rather than plain histograms:

```python
def plot_posterior_1d(x, density, ax, color="#0072B2", label=None):
    """Plot a 1D posterior with filled credible bands."""
    ax.plot(x, density, color=color, linewidth=1.2, label=label)

    # 1-sigma fill
    peak = x[np.argmax(density)]
    cumulative = np.cumsum(density) * np.diff(x, prepend=x[0])
    cumulative /= cumulative[-1]
    lo_1s = x[np.searchsorted(cumulative, 0.16)]
    hi_1s = x[np.searchsorted(cumulative, 0.84)]

    mask_1s = (x >= lo_1s) & (x <= hi_1s)
    ax.fill_between(x, density, where=mask_1s, alpha=0.35, color=color)

    # 2-sigma fill (lighter)
    lo_2s = x[np.searchsorted(cumulative, 0.025)]
    hi_2s = x[np.searchsorted(cumulative, 0.975)]
    mask_2s = (x >= lo_2s) & (x <= hi_2s)
    ax.fill_between(x, density, where=mask_2s, alpha=0.15, color=color)
```

**Vertical truth lines:** Use dashed lines with controlled dash pattern:

```python
ax.axvline(h_true, color="#D55E00", linestyle="--",
           linewidth=0.8, dashes=(4, 3), label=r"$H_0^{\rm true}$")
```

**Clean axis labels with LaTeX:** Use `r"$H_0\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$"` not plain text. The `\mathrm{}` ensures units are upright, and thin spaces (`\,`) improve readability.

---

### Method 5: Corner Plots for Multi-Parameter Posteriors

**What:** Use the `corner` library (Dan Foreman-Mackey) for triangle/corner plots of multi-parameter posteriors, with style overrides to match the thesis aesthetic.

```python
import corner

fig = corner.corner(
    samples,                          # shape (N, ndim)
    labels=[r"$H_0$", r"$d_L$", ...],
    quantiles=[0.16, 0.5, 0.84],     # show median + 1-sigma on 1D
    levels=[0.68, 0.95],             # 1-sigma, 2-sigma in 2D
    smooth=1.2,                       # Gaussian smoothing sigma
    smooth1d=0.8,                     # 1D histogram smoothing
    color="#0072B2",                   # Okabe-Ito blue
    hist_kwargs={"linewidth": 1.0, "density": True},
    contourf_kwargs={"colors": [blues(0.25), blues(0.50)], "alpha": 0.85},
    contour_kwargs={"linewidths": 0.8, "colors": [blues(0.7)]},
    label_kwargs={"fontsize": 8},
    title_kwargs={"fontsize": 8},
    fill_contours=True,
    show_titles=True,
    title_fmt=".2f",
    plot_density=False,
    plot_datapoints=False,            # clean look, no scatter
    no_fill_contours=False,
    max_n_ticks=4,
)
```

**Key corner settings for modern look:**
- `plot_datapoints=False` -- removes scatter plot of individual samples (cleaner)
- `fill_contours=True` -- filled contours instead of just lines
- `smooth=1.2` -- light Gaussian smoothing for less jagged contours
- `max_n_ticks=4` -- fewer tick labels, less clutter
- Custom `contourf_kwargs` colors from the Blues palette

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|---|---|---|---|
| Style framework | Custom `.mplstyle` | SciencePlots package | SciencePlots adds top/right ticks, changes color cycle unnecessarily, increases execution time substantially, and we need more control than a preset provides |
| Qualitative palette | Okabe-Ito (8 colors) | Paul Tol Bright/Muted | Okabe-Ito is the most widely adopted in GW/physics community; now in matplotlib 3.9+ as petroff10 |
| Sequential palette | matplotlib `Blues` | Tol iridescent, custom LinearSegmentedColormap | `Blues` is universally recognized, perceptually uniform, grayscale-safe, zero dependencies |
| KDE library | `scipy.stats.gaussian_kde` | `sklearn.neighbors.KernelDensity`, `statsmodels` KDE | scipy is already a dependency, sufficient for 1D/2D, no need for additional packages |
| Corner plots | `corner` library | `getdist`, `chainconsumer`, manual subplots | `corner` is the de facto standard in GW astronomy, lightweight, highly configurable |
| Contour smoothing | `gaussian_filter` on grids | Cubic interpolation, RBF interpolation | Gaussian filter preserves total probability (it is a convolution), interpolation does not |
| Font rendering | `text.usetex: True` | `mathtext.fontset: cm` (mathtext) | `usetex` produces exact Computer Modern matching the thesis body; mathtext is a fallback for CI |

---

## Installation / Setup

```bash
# corner library for triangle plots (if not already installed)
uv add corner

# TeX installation for text.usetex: True (already needed for thesis)
# Arch: sudo pacman -S texlive-basic texlive-fontsrecommended texlive-latexrecommended
# Ubuntu: sudo apt install texlive-latex-base texlive-fonts-recommended cm-super dvipng

# tol-colors (optional, only if Paul Tol colormaps needed beyond Blues)
# uv add tol-colors
```

**No additional packages required** beyond `corner` for the recommended approach. `scipy`, `matplotlib`, and `numpy` are already dependencies.

---

## Validation Strategy

| Check | Expected Result | Tolerance | Reference |
|---|---|---|---|
| Font size at 3.375" width | Capital letters >= 2 mm tall | Measure in PDF viewer | APS figure guidelines |
| Grayscale rendering | Sequential fills remain distinguishable | Visual inspection in grayscale print | Matplotlib colormap docs |
| Colorblind simulation | All data series distinguishable under deuteranopia | Use Coblis or Color Oracle tool | Okabe & Ito (2002) |
| PDF font embedding | All fonts embedded (no Type 3) | `pdffonts output.pdf` shows no Type 3 | `pdf.fonttype: 42` setting |
| LaTeX rendering | No mathtext fallback warnings | Clean matplotlib output log | `text.usetex: True` |
| Credible intervals | 68% region contains ~68% of probability mass | Within 2% for N > 1000 samples | Statistical consistency |
| KDE bandwidth | Posterior peak location preserved vs raw histogram | Peak shift < 0.5 * bin width | Scott (1992) |
| Figure file size | PDF < 2 MB per figure (vector) | Check with `ls -lh` | Journal upload limits |

---

## Key Dimension Presets

For convenience in the plotting code, define standard figure sizes:

```python
# REVTeX two-column document widths
SINGLE_COL = 3.375    # inches
DOUBLE_COL = 7.0      # inches
MARGIN_NOTE = 1.0     # inches (if needed)

# Standard aspect ratios
GOLDEN = (1 + 5**0.5) / 2  # 1.618

# Common figure sizes
FIG_SINGLE = (SINGLE_COL, SINGLE_COL / GOLDEN)         # 3.375 x 2.086
FIG_SINGLE_SQUARE = (SINGLE_COL, SINGLE_COL)            # 3.375 x 3.375
FIG_DOUBLE = (DOUBLE_COL, DOUBLE_COL / GOLDEN)          # 7.0 x 4.326
FIG_DOUBLE_HALF = (DOUBLE_COL, DOUBLE_COL / (2*GOLDEN)) # 7.0 x 2.163
```

---

## Sources

- APS Physical Review Tips for Authors: https://journals.aps.org/prx/authors/tips-authors-physical-review-physical-review-letters -- Figure format, size, and font requirements
- Matplotlib customization docs: https://matplotlib.org/stable/users/explain/customizing.html -- Complete rcParams reference
- Matplotlib LaTeX rendering: https://matplotlib.org/stable/users/explain/text/usetex.html -- `text.usetex` configuration
- Matplotlib colormaps: https://matplotlib.org/stable/users/explain/colors/colormaps.html -- Sequential colormap perceptual properties
- Leo C. Stein, "Fonts/sizes in matplotlib figures for LaTeX publications": https://duetosymmetry.com/code/latex-mpl-fig-tips/ -- Font size calibration for column widths
- Okabe & Ito (2002), "Color Universal Design": https://jfly.uni-koeln.de/color/ -- Colorblind-safe palette design
- Okabe-Ito hex codes reference: https://conceptviz.app/blog/okabe-ito-palette-hex-codes-complete-reference
- Paul Tol color schemes: https://personal.sron.nl/~pault/ -- Additional colorblind-safe sequential/diverging schemes
- SciencePlots GitHub: https://github.com/garrettj403/SciencePlots -- Reference for journal-specific mplstyle patterns (not recommended for direct use)
- corner.py documentation: https://corner.readthedocs.io/en/latest/api/ -- Corner plot API, smoothing, levels
- scipy.stats.gaussian_kde: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html -- KDE implementation
- Inchauspe et al. (2025), "Measuring gravitational wave memory with LISA," Phys. Rev. D 111, 044044. arXiv:2406.09228 -- Reference aesthetic target
- J.A. Wilcox, "Python for Publication Quality Figures": https://www.jawilcox.com/blog/2024/python-for-publication-quality-figures/ -- Practical matplotlib tips
