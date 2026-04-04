# Phase 18: New Plot Modules - Research

**Researched:** 2026-04-02
**Domain:** Matplotlib visualization (Mollweide, corner plots, convergence diagnostics)
**Confidence:** HIGH

## Summary

Phase 18 adds four new factory-function plot types: a Mollweide sky localization map, Fisher-derived corner plots via the `corner` library, H0 convergence diagnostics, and a detection efficiency curve with Wilson score confidence intervals. All four build on the existing plotting infrastructure from Phases 15-17 (style sheet, color palette, label dictionary, data layer, figure helpers).

The key technical findings are: (1) `corner` 2.2.3 works with Python 3.13 and our stack but conflicts with `constrained_layout` -- must disable it via context manager; (2) matplotlib's built-in `projection='mollweide'` handles small `Ellipse` patches correctly without geodesic corrections; (3) astropy already provides `binom_conf_interval` with Wilson score method, avoiding a new dependency; (4) the H0 convergence plot is a straightforward cumulative-subset computation with no special library needs.

**Primary recommendation:** Add `corner` as a dev dependency. Create two new modules: `sky_plots.py` (Mollweide) and `convergence_plots.py` (H0 convergence + detection efficiency). Place the corner plot wrapper in `fisher_plots.py` alongside existing Fisher visualization code.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Replace `plot_sky_localization_3d` with Mollweide projection using matplotlib's built-in `mollweide` projection.
- **D-02:** Show detected EMRI sky positions as scatter points, with localization error ellipses from Fisher matrix sky angle uncertainties (delta_qS, delta_phiS, cross-term).
- **D-03:** Color points by SNR using viridis. Include colorbar.
- **D-04:** Input coordinates are ecliptic (qS = colatitude, phiS = longitude). Convert to Mollweide-compatible (latitude = pi/2 - qS, longitude = phiS - pi for [-pi, pi] range).
- **D-05:** Use `corner` library (dfm/corner) for triangle plot. Add via `uv add corner`.
- **D-06:** Input is 14x14 covariance matrix from CRB CSV (via `reconstruct_covariance`). Generate synthetic samples from N(param_values, covariance).
- **D-07:** Default 6-parameter subset: M, mu, a, d_L, qS, phiS (15 panels).
- **D-08:** Apply thesis styling: `_labels.py` LaTeX labels, `_colors.py` palette, match font sizes to mplstyle.
- **D-09:** Support both single-event and multi-event overlay modes.
- **D-10:** H0 convergence: combined H0 posterior narrows as N_events increases.
- **D-11:** Left panel: stacked posterior curves for subset sizes. Right panel or inset: 68% credible interval width vs N_events with 1/sqrt(N) expectation.
- **D-12:** Subsets randomly sampled from detection catalog. Accept `seed` parameter.
- **D-13:** 1D P_det as function of single variable (z or d_L) with CIs.
- **D-14:** Wilson score interval or bootstrap CI for binomial detection probability.
- **D-15:** Accept injection campaign data (injected + detected arrays).

### Claude's Discretion
- Number of synthetic samples for corner plot (1000-10000)
- Exact subset sizes for convergence plot
- Wilson vs bootstrap CI for detection efficiency
- Error ellipse scaling on Mollweide (may need angular size conversion)
- Whether convergence right-panel is a separate axes or inset
- Marker size and alpha for sky map scatter
- Corner plot diagonal: histogram or KDE

### Deferred Ideas (OUT OF SCOPE)
- Full 14-parameter corner plot (too dense for print)
- Animated convergence (cannot embed in thesis PDF)
- Multi-detector sky maps (LISA-only scope)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SKY-01 | Mollweide projection sky map with localization ellipses | Matplotlib `projection='mollweide'` verified working with Ellipse patches in data coordinates (radians). Coordinate transform documented. |
| FISH-03 | Corner plot of EMRI parameter subset from Fisher-derived Gaussian | `corner` 2.2.3 verified compatible. Styling override pattern documented. `constrained_layout` conflict identified and solved. |
| CONV-01 | H0 convergence plot showing posterior narrowing with N_events | Cumulative subset approach using existing `plot_subset_posteriors` pattern. Credible interval via `np.percentile`. |
| CONV-02 | Detection efficiency curve with confidence intervals | `astropy.stats.binom_conf_interval(method='wilson')` already available -- no new dependency. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| corner | 2.2.3 | Triangle/corner plots | De facto standard for parameter posterior visualization in astronomy. Only dependency: matplotlib>=2.1. |
| matplotlib | 3.10.8 | All plotting (Mollweide, convergence, efficiency) | Already in stack. Mollweide projection built-in. |
| numpy | 2.4.3 | Sample generation, statistics | Already in stack. `np.random.default_rng().multivariate_normal()` for Fisher samples. |
| astropy.stats | 7.x | Wilson score confidence intervals | Already in stack. `binom_conf_interval(method='wilson')` avoids hand-rolling or adding statsmodels. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.stats | 1.17.1 | Credible interval computation, optional KDE for corner diagonals | Already in stack. For `np.percentile`-based CI width in convergence plot. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `corner` | `chainconsumer`, `getdist` | `corner` is simpler, lighter (one dep), standard in GW community |
| `astropy.stats.binom_conf_interval` | Hand-rolled Wilson formula | Astropy is already a dependency, tested, handles edge cases (p=0, p=1) |
| `statsmodels.proportion_confint` | `astropy.stats.binom_conf_interval` | statsmodels not in stack; astropy already available |

**Installation:**
```bash
uv add corner
```

**Version verification:** `corner` 2.2.3 is current (checked 2026-04-02). Only runtime dependency is matplotlib>=2.1. Pure Python, py3-none-any wheel, no compilation needed.

## Architecture Patterns

### Recommended Module Structure
```
master_thesis_code/plotting/
    sky_plots.py              # NEW: plot_sky_localization_mollweide()
    convergence_plots.py      # NEW: plot_h0_convergence(), plot_detection_efficiency()
    fisher_plots.py           # EXTEND: add plot_fisher_corner()
    evaluation_plots.py       # EXISTING: deprecate plot_sky_localization_3d()
```

### Pattern 1: Factory Function with (fig, ax) Return
**What:** Every plot function takes data arguments and optional `ax`, returns `(Figure, Axes)`.
**When to use:** All four new functions.
**Example:**
```python
# Source: existing project convention, e.g. evaluation_plots.py
def plot_sky_localization_mollweide(
    theta_s: npt.NDArray[np.float64],
    phi_s: npt.NDArray[np.float64],
    snr: npt.NDArray[np.float64],
    *,
    covariances: list[npt.NDArray[np.float64]] | None = None,
    n_sigma: float = 1.0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = get_figure(preset="double", subplot_kw={"projection": "mollweide"})
    else:
        fig = _fig_from_ax(ax)
    # ... plot logic ...
    return fig, ax
```

### Pattern 2: Corner Plot with constrained_layout Context Manager
**What:** Temporarily disable `constrained_layout` when calling `corner.corner()` to avoid the `subplots_adjust` conflict.
**When to use:** `plot_fisher_corner()` only.
**Example:**
```python
import matplotlib
import corner

def plot_fisher_corner(
    covariance: npt.NDArray[np.float64],
    param_values: npt.NDArray[np.float64],
    # ...
) -> tuple[Figure, npt.NDArray[np.object_]]:
    # corner calls subplots_adjust internally, incompatible with constrained_layout
    with matplotlib.rc_context({"figure.constrained_layout.use": False}):
        fig = corner.corner(
            samples,
            labels=labels,
            truths=truths,
            truth_color=TRUTH,
            color=CYCLE[0],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            hist_kwargs={"edgecolor": EDGE},
        )
    axes = np.array(fig.axes).reshape(n_params, n_params)
    return fig, axes
```

### Pattern 3: Mollweide Coordinate Conversion
**What:** Convert ecliptic (qS, phiS) to Mollweide (latitude, longitude).
**When to use:** `plot_sky_localization_mollweide()`.
**Example:**
```python
# qS = colatitude [0, pi], phiS = longitude [0, 2*pi]
# Mollweide expects: latitude [-pi/2, pi/2], longitude [-pi, pi]
latitude = np.pi / 2 - theta_s      # colatitude -> latitude
longitude = phi_s - np.pi           # [0, 2pi] -> [-pi, pi]
# Wrap longitude to [-pi, pi]
longitude = (longitude + np.pi) % (2 * np.pi) - np.pi
```

### Pattern 4: Two-Panel Convergence Layout
**What:** Left panel for posterior curves, right panel for CI width vs N.
**When to use:** `plot_h0_convergence()`.
**Example:**
```python
fig, (ax_post, ax_ci) = get_figure(nrows=1, ncols=2, preset="double")
```

### Anti-Patterns to Avoid
- **Calling `plt.subplots()` directly in corner wrapper:** Corner creates its own figure. Do not create one beforehand. Pass styling via `rc_context`.
- **Using `constrained_layout` with corner:** Triggers UserWarning and breaks subplot spacing. Must disable.
- **Degrees on Mollweide axes:** Mollweide projection expects radians for all coordinates and angular sizes. Never pass degrees.
- **Large Ellipse patches on Mollweide:** For angular extents > ~30 deg, the flat-space Ellipse approximation breaks down. EMRI localization errors are typically < 1 deg, so this is safe.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Corner/triangle plots | Custom subplot grid with hist2d | `corner.corner()` | Handles subplot spacing, diagonal histograms, contour levels, label placement. 100+ edge cases. |
| Wilson score CI | Manual formula implementation | `astropy.stats.binom_conf_interval(method='wilson')` | Handles p=0, p=1, small n edge cases correctly. Already tested. |
| Mollweide projection | Custom coordinate transform + plot | `plt.subplots(subplot_kw={"projection": "mollweide"})` | Matplotlib's built-in handles grid lines, tick formatting, boundary. |
| Multivariate normal samples | Manual Cholesky decomposition + randn | `rng.multivariate_normal(mean, cov, size)` | NumPy handles near-singular covariance, uses efficient LAPACK routines. |

**Key insight:** All four visualizations have well-supported library solutions. The implementation work is primarily in wiring data pipelines and applying thesis styling, not in building visualization primitives.

## Common Pitfalls

### Pitfall 1: constrained_layout vs corner.corner()
**What goes wrong:** `corner` calls `fig.subplots_adjust()` internally. When `constrained_layout` is True (our mplstyle default), matplotlib emits a UserWarning and the layout manager fights with manual spacing, producing overlapping labels.
**Why it happens:** `constrained_layout` and `subplots_adjust` are mutually exclusive layout strategies in matplotlib.
**How to avoid:** Wrap all `corner.corner()` calls in `matplotlib.rc_context({"figure.constrained_layout.use": False})`.
**Warning signs:** UserWarning about "layout engine incompatible with subplots_adjust" in test output.

### Pitfall 2: Mollweide Coordinate Ranges
**What goes wrong:** Points outside longitude [-pi, pi] or latitude [-pi/2, pi/2] are silently clipped or produce RuntimeWarnings.
**Why it happens:** Mollweide projection has fixed domain. Input ecliptic coordinates may be [0, 2*pi] for longitude.
**How to avoid:** Always convert: `longitude = phiS - pi`, `latitude = pi/2 - qS`. Add explicit wrapping: `longitude = (longitude + pi) % (2*pi) - pi`.
**Warning signs:** Points missing from plot, RuntimeWarning about invalid values in Mollweide inverse.

### Pitfall 3: Near-Singular Covariance in Corner Samples
**What goes wrong:** `np.random.multivariate_normal()` raises `LinAlgError` or produces NaN samples for near-singular covariance matrices.
**Why it happens:** Fisher matrices from poorly constrained parameters can have very small eigenvalues. The 6-parameter submatrix may amplify conditioning issues.
**How to avoid:** Use `np.linalg.eigvalsh()` to check positive-definiteness before sampling. Add small ridge (`cov += epsilon * np.eye(n)`) if needed. Set `check_valid='warn'` in `multivariate_normal`.
**Warning signs:** NaN in samples, `LinAlgError: singular matrix`.

### Pitfall 4: Corner Plot Return Type
**What goes wrong:** `corner.corner()` returns a `Figure`, not `(Figure, Axes)`. Our convention expects `(Figure, Axes)` or `(Figure, ndarray[Axes])`.
**Why it happens:** Corner manages its own axes grid internally.
**How to avoid:** Extract axes from the returned figure: `axes = np.array(fig.axes).reshape(n_params, n_params)`. Return `(fig, axes)`.
**Warning signs:** Type errors in tests expecting tuple return.

### Pitfall 5: Empty Bins in Detection Efficiency
**What goes wrong:** Division by zero when computing P_det = detected/injected for bins with no injections.
**Why it happens:** Sparse injection campaigns at extreme redshifts can have empty bins.
**How to avoid:** Mask bins where `n_injected == 0`. `astropy.stats.binom_conf_interval` handles `k=0, n=0` gracefully (returns NaN).
**Warning signs:** RuntimeWarning about division by zero.

### Pitfall 6: Mollweide Axes Cannot Use set_xlim/set_ylim
**What goes wrong:** Calling `ax.set_xlim()` or `ax.set_ylim()` on a Mollweide axes raises an error or has no visible effect.
**Why it happens:** Geographic projections have fixed extent covering the full sky.
**How to avoid:** Do not set axis limits. The full sky is always shown. Zoom is not supported -- if needed, use a regular axes with manual projection.
**Warning signs:** Matplotlib error about geographic projections.

## Code Examples

### Mollweide Sky Map with Error Ellipses
```python
# Verified: matplotlib 3.10.8 Mollweide projection
import numpy as np
from matplotlib.patches import Ellipse
from master_thesis_code.plotting._helpers import get_figure, make_colorbar
from master_thesis_code.plotting._colors import CMAP, EDGE
from master_thesis_code.plotting._labels import LABELS
from master_thesis_code.plotting.fisher_plots import _ellipse_params

def plot_sky_localization_mollweide(
    theta_s: npt.NDArray[np.float64],   # colatitude [0, pi]
    phi_s: npt.NDArray[np.float64],     # longitude [0, 2*pi]
    snr: npt.NDArray[np.float64],
    *,
    covariances: list[npt.NDArray[np.float64]] | None = None,  # 2x2 sky sub-cov
    n_sigma: float = 1.0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = get_figure(preset="double", subplot_kw={"projection": "mollweide"})
    else:
        fig = _fig_from_ax(ax)

    # Coordinate transform: ecliptic -> Mollweide
    lat = np.pi / 2 - theta_s
    lon = (phi_s - np.pi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    sc = ax.scatter(lon, lat, c=snr, cmap=CMAP, s=12, alpha=0.8, zorder=5)
    make_colorbar(sc, fig, ax, label=LABELS["SNR"])

    if covariances is not None:
        for i, cov_2x2 in enumerate(covariances):
            w, h, angle = _ellipse_params(cov_2x2, n_sigma)
            ellipse = Ellipse(
                xy=(lon[i], lat[i]), width=w, height=h, angle=angle,
                facecolor="none", edgecolor=EDGE, linewidth=0.8,
                transform=ax.transData,
            )
            ax.add_patch(ellipse)

    ax.grid(True, alpha=0.3)
    return fig, ax
```

### Corner Plot with Thesis Styling
```python
# Verified: corner 2.2.3 with matplotlib 3.10.8
import matplotlib
import corner
import numpy as np

def plot_fisher_corner(
    covariance: npt.NDArray[np.float64],   # 14x14
    param_values: npt.NDArray[np.float64], # 14-element
    params: list[str] | None = None,
    *,
    n_samples: int = 5000,
    seed: int = 42,
) -> tuple[Figure, npt.NDArray[np.object_]]:
    if params is None:
        params = ["M", "mu", "a", "luminosity_distance", "qS", "phiS"]

    indices = [PARAMETER_NAMES.index(p) for p in params]
    sub_cov = covariance[np.ix_(indices, indices)]
    sub_mean = param_values[indices]

    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(sub_mean, sub_cov, size=n_samples)

    labels = [LABELS[label_key(p)] for p in params]
    truths = list(sub_mean)

    with matplotlib.rc_context({"figure.constrained_layout.use": False}):
        fig = corner.corner(
            samples,
            labels=labels,
            truths=truths,
            truth_color=TRUTH,
            color=CYCLE[0],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".3f",
            hist_kwargs={"edgecolor": EDGE},
        )

    n = len(params)
    axes = np.array(fig.axes, dtype=object).reshape(n, n)
    return fig, axes
```

### Wilson Score CI for Detection Efficiency
```python
# Verified: astropy.stats.binom_conf_interval available in project stack
from astropy.stats import binom_conf_interval
import numpy as np

def _compute_efficiency_with_ci(
    variable: npt.NDArray[np.float64],
    detected: npt.NDArray[np.bool_],
    bins: int = 20,
    confidence: float = 0.68,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
           npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Bin variable, compute detection efficiency and Wilson CI."""
    bin_edges = np.linspace(variable.min(), variable.max(), bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_inj = np.histogram(variable, bins=bin_edges)[0]
    n_det = np.histogram(variable[detected], bins=bin_edges)[0]

    # Mask empty bins
    mask = n_inj > 0
    efficiency = np.where(mask, n_det / n_inj, np.nan)

    ci = binom_conf_interval(n_det, n_inj, confidence_level=confidence, interval="wilson")
    ci_lo = ci[0]
    ci_hi = ci[1]

    return bin_centers, efficiency, ci_lo, ci_hi
```

### H0 Convergence with Cumulative Subsets
```python
# Standard numpy/scipy approach
import numpy as np

def _credible_interval_width(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    level: float = 0.68,
) -> float:
    """Compute credible interval width from a posterior evaluated on a grid."""
    # Normalize
    posterior = posterior / np.trapz(posterior, h_values)
    cdf = np.cumsum(posterior) * np.diff(h_values, prepend=h_values[0])
    cdf /= cdf[-1]
    lo = np.interp((1 - level) / 2, cdf, h_values)
    hi = np.interp((1 + level) / 2, cdf, h_values)
    return float(hi - lo)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 3D scatter for sky localization | Mollweide projection (standard in GW astronomy) | Always been standard | Print-friendly, readable, shows full sky context |
| Manual triangle grids | `corner` library | corner 1.0 (2016) | Community standard, handles all edge cases |
| Normal approximation CI | Wilson score CI | Well-established since Wilson 1927 | Better coverage for small n, proportions near 0/1 |

**Deprecated/outdated:**
- `plot_sky_localization_3d`: Non-standard, hard to read in print. Deprecate but keep for backward compat.

## Open Questions

1. **Multi-event corner overlay**
   - What we know: D-09 requests multi-event overlay in corner plot. `corner.corner()` supports `fig` parameter to overplot.
   - What's unclear: Whether overplotting multiple corner plots on the same figure produces readable output with more than 2-3 events.
   - Recommendation: Support up to 3-4 events with distinct colors from CYCLE. Beyond that, use individual corner plots. Test readability during implementation.

2. **Ellipse angular distortion on Mollweide**
   - What we know: Ellipse patches work in Mollweide data coordinates. For small angular errors (< 1 deg, typical for EMRI), the flat-space approximation is fine.
   - What's unclear: At high latitudes (near poles) or for events with large localization uncertainties, the ellipse may visually distort.
   - Recommendation: Accept the flat-space approximation. Document that it is valid for typical EMRI localization accuracy. If a future user needs geodesic ellipses, that is a separate feature.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest master_thesis_code_test/plotting/ -x -m "not gpu and not slow"` |
| Full suite command | `uv run pytest -m "not gpu and not slow"` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SKY-01 | Mollweide sky map returns (Figure, Axes) | unit | `uv run pytest master_thesis_code_test/plotting/test_sky_plots.py -x` | Wave 0 |
| SKY-01 | Ellipse patches added when covariances provided | unit | same as above | Wave 0 |
| FISH-03 | Corner plot returns (Figure, ndarray) | unit | `uv run pytest master_thesis_code_test/plotting/test_fisher_plots.py::test_plot_fisher_corner -x` | Wave 0 |
| FISH-03 | Multi-event overlay works | unit | same as above | Wave 0 |
| CONV-01 | Convergence plot returns (Figure, ndarray) | unit | `uv run pytest master_thesis_code_test/plotting/test_convergence_plots.py -x` | Wave 0 |
| CONV-02 | Detection efficiency returns (Figure, Axes) | unit | same as above | Wave 0 |
| CONV-02 | Wilson CI band present in output | unit | same as above | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest master_thesis_code_test/plotting/ -x -m "not gpu and not slow"`
- **Per wave merge:** `uv run pytest -m "not gpu and not slow"`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `master_thesis_code_test/plotting/test_sky_plots.py` -- covers SKY-01
- [ ] `master_thesis_code_test/plotting/test_convergence_plots.py` -- covers CONV-01, CONV-02
- [ ] Test fixtures: `sample_sky_coords`, `sample_sky_covariances`, `sample_injection_campaign` in conftest.py
- [ ] Existing `test_fisher_plots.py` needs new test for `plot_fisher_corner`
- [ ] Existing `test_evaluation_plots.py::test_plot_sky_localization_3d` stays (backward compat)

## Discretion Recommendations

Based on research findings, here are recommendations for areas left to Claude's discretion:

| Area | Recommendation | Rationale |
|------|---------------|-----------|
| N samples for corner | 5000 | Fast enough for smoke tests, dense enough for smooth contours. Parameter via kwarg. |
| Subset sizes for convergence | [1, 5, 10, 25, 50, 100] | Logarithmic-ish spacing shows 1/sqrt(N) scaling clearly. Capped at 100 or total detected. |
| Wilson vs bootstrap CI | Wilson score | Astropy already provides it. More principled for binomial data. No additional dependency. |
| Ellipse scaling on Mollweide | Use data-coordinate Ellipse (radians) directly | For EMRI-scale localization errors (<< 1 rad), flat-space approximation is accurate. |
| Convergence right panel | Separate axes (not inset) | Insets are harder to read in print at thesis column width. Two-panel layout via `get_figure(ncols=2)`. |
| Marker size/alpha for sky map | s=12, alpha=0.8 | Visible at thesis print resolution without overwhelming the projection grid. |
| Corner diagonal | Histogram (default) | KDE can produce artifacts with few samples. Histogram is corner's default and familiar. |

## Sources

### Primary (HIGH confidence)
- matplotlib 3.10.8 Mollweide projection -- tested locally, `projection='mollweide'` on `plt.subplots`
- corner 2.2.3 -- tested locally, `corner.corner()` signature from GitHub source
- astropy.stats.binom_conf_interval -- tested locally, `interval='wilson'` verified working

### Secondary (MEDIUM confidence)
- [corner.py API docs](https://corner.readthedocs.io/en/latest/api/) -- parameter descriptions
- [corner.py GitHub](https://github.com/dfm/corner.py/blob/main/src/corner/corner.py) -- source code for implementation details
- [Wilson score interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval) -- formula reference

### Tertiary (LOW confidence)
- None -- all findings verified locally

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries tested locally with exact project versions
- Architecture: HIGH -- follows established project patterns, all conventions verified in code
- Pitfalls: HIGH -- constrained_layout conflict reproduced and solved; coordinate ranges tested

**Research date:** 2026-04-02
**Valid until:** 2026-05-02 (stable domain, matplotlib/corner APIs unlikely to change)
