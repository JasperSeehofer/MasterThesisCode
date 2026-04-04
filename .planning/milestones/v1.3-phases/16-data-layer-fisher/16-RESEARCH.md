# Phase 16: Data Layer & Fisher Visualizations - Research

**Researched:** 2026-04-02
**Domain:** matplotlib plotting, covariance matrix reconstruction, Fisher information visualization
**Confidence:** HIGH

## Summary

Phase 16 creates a data loading layer (`_data.py`) that reconstructs 14x14 covariance matrices from CRB CSV files, plus three new plot factory functions: Fisher error ellipses, characteristic strain h_c(f), and parameter uncertainty distributions. All infrastructure needed is already in place from Phase 15 (style, helpers, colors, labels).

The CRB CSV format is well understood: 105 lower-triangle `delta_X_delta_Y` columns plus 14 parameter values and 5 metadata columns (T, dt, SNR, generation_time, host_galaxy_index). The reconstruction logic is straightforward linear algebra (fill symmetric matrix from lower triangle). The plotting factories follow the established `data in, (fig, ax) out` pattern used by all existing plot modules.

**Primary recommendation:** Implement `_data.py` first (it underpins the ellipse and uncertainty plots), then the three factory functions in a new `fisher_plots.py` module. Use `matplotlib.patches.Ellipse` for error ellipses (eigenvalue decomposition of 2x2 submatrix), direct PSD functions from `LISA_configuration.py` for the strain plot, and `ax.violinplot()` for uncertainty distributions.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Row-level API: `reconstruct_covariance(row: pd.Series) -> np.ndarray` returns a symmetric 14x14 matrix from the `delta_X_delta_Y` columns. Callers iterate over the DataFrame.
- **D-02:** `PARAMETER_NAMES: list[str]` -- ordered list `['M','mu','a','p0','e0','x0','luminosity_distance','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0']` as single source of truth for index-to-name mapping. Derived from `ParameterSpace._parameters_to_dict()` key order.
- **D-03:** Parameter grouping defined in `_data.py`:
  - `INTRINSIC = ['M','mu','a','p0','e0','x0']`
  - `EXTRINSIC = ['luminosity_distance','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0']`
- **D-04:** Three default parameter pairs: `(M, mu)`, `(luminosity_distance, qS)`, `(qS, phiS)`.
- **D-05:** Filled contours with alpha transparency for 1-sigma and 2-sigma ellipses.
- **D-06:** Factory supports single-event and multi-event overlay via optional parameter.
- **D-07:** Example EMRI waveform uses hardcoded reference parameters (precomputed h_c(f) curve).
- **D-08:** Three noise curves: instrument noise, galactic foreground, and total.
- **D-09:** Violin plot for multi-event mode (fractional uncertainties grouped by intrinsic/extrinsic).
- **D-10:** Single-event fallback: horizontal bar chart of fractional uncertainties.
- **D-11:** Factory supports both modes via optional parameter.

### Claude's Discretion
- Internal structure of `reconstruct_covariance()` (column parsing, symmetric fill)
- Exact default parameters for hardcoded reference EMRI in h_c(f) plot
- Ellipse computation method (eigenvalue decomposition of 2x2 submatrix)
- Aspect ratios and subplot layout for multi-panel ellipse figures
- Violin plot styling details (width, kernel bandwidth, etc.)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FISH-01 | `_data.py` module reconstructs 14x14 covariance matrices from CRB CSV `delta_X_delta_Y` columns | CRB CSV format verified: 105 lower-triangle columns, naming convention `delta_{row}_delta_{col}`. Reconstruction via symmetric fill of np.zeros((14,14)). |
| FISH-02 | 2D Fisher error ellipses (1-sigma, 2-sigma) for key EMRI parameter pairs | Standard eigenvalue decomposition of 2x2 covariance submatrix. `matplotlib.patches.Ellipse` for rendering. Three default pairs locked by D-04. |
| FISH-04 | Characteristic strain h_c(f) with example EMRI track overlaid on LISA sensitivity curve | `LisaTdiConfiguration` provides `power_spectral_density_a_channel()` (total) and `_confusion_noise()`. Instrument-only via `include_confusion_noise=False`. Characteristic strain = sqrt(f * S_n(f)). |
| FISH-05 | Parameter uncertainty distributions with intrinsic/extrinsic grouping and LaTeX labels | Fractional uncertainty = sqrt(diag(cov)) / param_value. Violin plot (multi-event) or bar chart (single-event). Grouping constants from D-03. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | (project pinned) | All plotting | Only visualization library in project; GW community standard |
| numpy | (project pinned) | Matrix operations, eigenvalue decomposition | Already used everywhere |
| pandas | (project pinned) | CRB CSV loading and row iteration | Already used for CRB data |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib.patches.Ellipse | (bundled) | Drawing 2D error ellipses | Fisher ellipse factory |
| numpy.linalg.eigh | (bundled) | Symmetric eigenvalue decomposition | Extracting ellipse orientation and axes |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib.patches.Ellipse | Parametric curve via np.cos/sin | Ellipse is cleaner, handles transforms automatically |
| Manual PSD decomposition | Separate LisaTdiConfiguration instances | Can use `include_confusion_noise` flag instead |

**No new dependencies required.** Everything is in the existing stack.

## Architecture Patterns

### Recommended Project Structure
```
master_thesis_code/plotting/
    _data.py              # NEW: CRB data loading, covariance reconstruction, constants
    fisher_plots.py       # NEW: error ellipses, strain plot, uncertainty distributions
    _colors.py            # existing
    _labels.py            # existing
    _helpers.py           # existing
    _style.py             # existing
    __init__.py           # update to export _data constants if needed
```

### Pattern 1: Factory Function Signature (established in codebase)
**What:** All plot factories take data as positional args, optional `ax: Axes | None = None`, return `tuple[Figure, Axes]`.
**When to use:** Every new plot function.
**Example:**
```python
# Source: master_thesis_code/plotting/bayesian_plots.py (existing pattern)
def plot_fisher_ellipses(
    covariance: npt.NDArray[np.float64],
    param_values: npt.NDArray[np.float64],
    param_names: list[str],
    pairs: list[tuple[str, str]] | None = None,
    *,
    sigma_levels: tuple[float, ...] = (1.0, 2.0),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)
    # ... plot ...
    return fig, ax
```

### Pattern 2: Covariance Reconstruction
**What:** Parse `delta_X_delta_Y` column names from a pandas Series, fill a symmetric 14x14 matrix.
**When to use:** `_data.py:reconstruct_covariance()`.
**Implementation approach:**
```python
def reconstruct_covariance(row: pd.Series) -> npt.NDArray[np.float64]:
    n = len(PARAMETER_NAMES)
    cov = np.zeros((n, n), dtype=np.float64)
    for i, row_name in enumerate(PARAMETER_NAMES):
        for j in range(i + 1):
            col_name = f"delta_{row_name}_delta_{PARAMETER_NAMES[j]}"
            cov[i, j] = row[col_name]
            cov[j, i] = cov[i, j]  # symmetric fill
    return cov
```

### Pattern 3: Eigenvalue-based Ellipse Parameters
**What:** Extract 2x2 submatrix from full covariance, compute eigenvalues/vectors for ellipse width, height, angle.
**When to use:** Fisher error ellipse plot.
**Implementation:**
```python
def _ellipse_params(cov_2x2: npt.NDArray[np.float64], n_sigma: float = 1.0) -> tuple[float, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2x2)
    # eigh returns sorted ascending; largest eigenvalue = semi-major axis
    width = 2 * n_sigma * np.sqrt(eigenvalues[1])
    height = 2 * n_sigma * np.sqrt(eigenvalues[0])
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    return width, height, angle
```

### Pattern 4: Characteristic Strain from PSD
**What:** Convert one-sided strain PSD S_n(f) to characteristic strain h_c(f) = sqrt(f * S_n(f)).
**When to use:** Strain sensitivity plot (FISH-04).
**Note:** For the EMRI signal track, h_c = 2f * |h(f)|, where |h(f)| is the Fourier amplitude. Since D-07 says precomputed, embed or load representative data.

### Anti-Patterns to Avoid
- **Importing `LisaTdiConfiguration` at module level in plot code:** This file has the unconditional cupy import issue (Known Bug 1). Use deferred import or create PSD with `include_confusion_noise=False` flag to get instrument-only curve.
- **Calling `_parameters_to_dict()` from plot code:** Use the `PARAMETER_NAMES` constant in `_data.py` instead -- no dependency on `ParameterSpace`.
- **Using bare `np.ndarray` type hints:** Project requires `npt.NDArray[np.float64]`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Error ellipse geometry | Manual cos/sin parametric curve | `matplotlib.patches.Ellipse` + `np.linalg.eigh` | Handles axis transforms, clipping, z-order automatically |
| Symmetric matrix eigendecomp | Custom eigenvalue solver | `np.linalg.eigh` (symmetric-optimized) | Guaranteed real eigenvalues, sorted, numerically stable |
| PSD computation | Re-derive PSD formula | `LisaTdiConfiguration.power_spectral_density_a_channel()` | Already implemented, tested, includes confusion noise toggle |
| LaTeX parameter labels | Hardcoded strings | `_labels.py:LABELS` dict | Single source of truth, already covers all 14 params |
| Figure sizing | Ad-hoc figsize tuples | `get_figure(preset="single"|"double")` | REVTeX-standard widths |

**Key insight:** This phase is 100% visualization code using existing math/physics infrastructure. No new physics formulas, no GPU code. All computational heavy lifting (PSD, covariance) is already implemented.

## Common Pitfalls

### Pitfall 1: Parameter Name Mismatch Between CSV and Labels
**What goes wrong:** CRB CSV uses `luminosity_distance` (from `_parameters_to_dict()`), but `_labels.py` uses `d_L`. Plotting code that naively looks up CSV column names in LABELS dict will get KeyError.
**Why it happens:** Two different naming conventions: physics shorthand in labels vs descriptive names in parameter space.
**How to avoid:** Add a `PARAM_TO_LABEL_KEY` mapping in `_data.py` that maps CSV names to LABELS keys. E.g., `{"luminosity_distance": "d_L", "x0": "Y0"}`. Also note `x0` in CSV maps to `Y0` in labels.
**Warning signs:** KeyError when trying to look up axis labels.

### Pitfall 2: Non-Positive-Definite Covariance Submatrices
**What goes wrong:** Negative eigenvalues from `np.linalg.eigh` on a 2x2 submatrix cause `np.sqrt` to return NaN for ellipse axes.
**Why it happens:** Numerical noise in Fisher matrix inversion can produce slightly non-PD covariance blocks.
**How to avoid:** Clamp eigenvalues to `max(eigenvalue, 0.0)` before taking sqrt. Log a warning if negative eigenvalues are encountered.
**Warning signs:** NaN ellipse dimensions, invisible or zero-size ellipses.

### Pitfall 3: Unconditional CuPy Import via LISA_configuration
**What goes wrong:** Importing `LisaTdiConfiguration` at module level in `fisher_plots.py` makes the module un-importable on CPU-only dev machines.
**Why it happens:** `LISA_configuration.py` has the Known Bug 1 (unconditional cupy import at top level, though there's a try/except guard now -- verify).
**How to avoid:** The current code has a `try/except` guard for cupy. But for the strain plot, create PSD data in a helper function that does the import locally, or pass precomputed PSD arrays to the plot factory.
**Warning signs:** ImportError on dev machine.

### Pitfall 4: Ellipse Aspect Ratio Distortion
**What goes wrong:** Ellipses appear circular or overly elongated because x/y axis scales differ by orders of magnitude (e.g., M ~ 1e6 vs mu ~ 10).
**Why it happens:** matplotlib draws in data coordinates; if scales differ greatly, visual aspect ratio is misleading.
**How to avoid:** Set `ax.set_aspect('auto')` (default) and let the ellipse render in data coords. Use `transforms` if equal-aspect rendering is needed. For the (M, mu) pair, the scale difference is ~5 orders of magnitude -- consider log scale or normalized parameters.
**Warning signs:** Ellipses that look like lines or circles when they shouldn't.

### Pitfall 5: Violin Plot with Too Few Points
**What goes wrong:** Violin plot KDE fails or looks misleading with very few events (< 5).
**Why it happens:** `ax.violinplot()` uses KDE internally; small samples produce unreliable density estimates.
**How to avoid:** Check event count; fall back to bar chart (D-10) if fewer than ~5 events.
**Warning signs:** Violin bodies that extend beyond physical bounds.

## Code Examples

### Covariance Reconstruction from CSV Row
```python
# _data.py
import numpy as np
import numpy.typing as npt
import pandas as pd

PARAMETER_NAMES: list[str] = [
    "M", "mu", "a", "p0", "e0", "x0",
    "luminosity_distance", "qS", "phiS", "qK", "phiK",
    "Phi_phi0", "Phi_theta0", "Phi_r0",
]

INTRINSIC: list[str] = ["M", "mu", "a", "p0", "e0", "x0"]
EXTRINSIC: list[str] = [
    "luminosity_distance", "qS", "phiS", "qK", "phiK",
    "Phi_phi0", "Phi_theta0", "Phi_r0",
]

# Maps CRB CSV parameter names to _labels.py LABELS keys
PARAM_TO_LABEL_KEY: dict[str, str] = {
    "luminosity_distance": "d_L",
    "x0": "Y0",
    # All others map to themselves
}

def label_key(param: str) -> str:
    """Map a CSV parameter name to its LABELS dict key."""
    return PARAM_TO_LABEL_KEY.get(param, param)


def reconstruct_covariance(row: pd.Series) -> npt.NDArray[np.float64]:
    """Reconstruct symmetric 14x14 covariance matrix from CRB CSV row."""
    n = len(PARAMETER_NAMES)
    cov = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            col = f"delta_{PARAMETER_NAMES[i]}_delta_{PARAMETER_NAMES[j]}"
            cov[i, j] = row[col]
            cov[j, i] = cov[i, j]
    return cov
```

### Fisher Error Ellipse Drawing
```python
# fisher_plots.py (partial)
from matplotlib.patches import Ellipse

def _ellipse_params(
    cov_2x2: npt.NDArray[np.float64],
    n_sigma: float,
) -> tuple[float, float, float]:
    """Compute (width, height, angle_deg) for an error ellipse."""
    vals, vecs = np.linalg.eigh(cov_2x2)
    vals = np.maximum(vals, 0.0)  # guard against numerical noise
    width = 2 * n_sigma * np.sqrt(vals[1])
    height = 2 * n_sigma * np.sqrt(vals[0])
    angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    return width, height, angle

# In the factory function, for each sigma level:
# w, h, ang = _ellipse_params(sub_cov, sigma)
# ellipse = Ellipse(xy=(center_x, center_y), width=w, height=h,
#                   angle=ang, alpha=0.3, facecolor=color, edgecolor=color)
# ax.add_patch(ellipse)
```

### Characteristic Strain Sensitivity Curve
```python
# fisher_plots.py (partial)
def plot_characteristic_strain(
    *,
    f_min: float = 1e-5,
    f_max: float = 1.0,
    n_points: int = 1000,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    from master_thesis_code.LISA_configuration import LisaTdiConfiguration

    freqs = np.geomspace(f_min, f_max, n_points)

    # Total PSD (instrument + confusion)
    lisa_total = LisaTdiConfiguration(include_confusion_noise=True)
    psd_total = lisa_total.power_spectral_density_a_channel(freqs)

    # Instrument only
    lisa_inst = LisaTdiConfiguration(include_confusion_noise=False)
    psd_inst = lisa_inst.power_spectral_density_a_channel(freqs)

    # Confusion noise only
    psd_confusion = lisa_total._confusion_noise(freqs)

    # Convert to characteristic strain: h_c = sqrt(f * S_n)
    h_total = np.sqrt(freqs * psd_total)
    h_inst = np.sqrt(freqs * psd_inst)
    h_conf = np.sqrt(freqs * psd_confusion)

    # Plot on log-log
    ax.loglog(freqs, h_total, label="Total", color=EDGE)
    ax.loglog(freqs, h_inst, label="Instrument", color=REFERENCE, ls="--")
    ax.loglog(freqs, h_conf, label="Galactic foreground", color=CYCLE[1], ls=":")
    # ... add EMRI track overlay ...
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No covariance visualization | Fisher error ellipses | This phase | Enables visual validation of parameter correlations |
| No strain sensitivity plot | h_c(f) with noise decomposition | This phase | Standard LISA sensitivity figure for thesis |
| No uncertainty overview | Violin/bar chart of fractional uncertainties | This phase | Quick assessment of estimation quality |

**Relevant prior work:**
- Phase 9 added galactic confusion noise to PSD -- strain plot benefits directly
- Phase 15 added style infrastructure (`_labels.py`, `_colors.py`, `_helpers.py`) -- all new factories use these
- Phase 14 established test patterns for plotting -- follow same conftest fixtures and smoke test approach

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via uv run) |
| Config file | pyproject.toml |
| Quick run command | `uv run pytest master_thesis_code_test/plotting/ -m "not gpu and not slow" -x` |
| Full suite command | `uv run pytest -m "not gpu and not slow"` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FISH-01 | `reconstruct_covariance` returns symmetric 14x14 from CSV row | unit | `uv run pytest master_thesis_code_test/plotting/test_data.py -x` | Wave 0 |
| FISH-01 | Diagonal elements are positive (valid covariance) | unit | `uv run pytest master_thesis_code_test/plotting/test_data.py::test_diagonal_positive -x` | Wave 0 |
| FISH-01 | Reconstructed matrix equals original (round-trip) | unit | `uv run pytest master_thesis_code_test/plotting/test_data.py::test_roundtrip -x` | Wave 0 |
| FISH-02 | Error ellipse factory returns (Figure, Axes) | smoke | `uv run pytest master_thesis_code_test/plotting/test_fisher_plots.py::test_plot_fisher_ellipses -x` | Wave 0 |
| FISH-02 | Multi-event overlay mode works | smoke | `uv run pytest master_thesis_code_test/plotting/test_fisher_plots.py::test_plot_fisher_ellipses_multi -x` | Wave 0 |
| FISH-04 | Characteristic strain factory returns (Figure, Axes) | smoke | `uv run pytest master_thesis_code_test/plotting/test_fisher_plots.py::test_plot_characteristic_strain -x` | Wave 0 |
| FISH-04 | Three noise curves present on axes | smoke | `uv run pytest master_thesis_code_test/plotting/test_fisher_plots.py::test_strain_three_curves -x` | Wave 0 |
| FISH-05 | Violin plot factory returns (Figure, Axes) for DataFrame input | smoke | `uv run pytest master_thesis_code_test/plotting/test_fisher_plots.py::test_plot_uncertainties_violin -x` | Wave 0 |
| FISH-05 | Bar chart factory returns (Figure, Axes) for Series input | smoke | `uv run pytest master_thesis_code_test/plotting/test_fisher_plots.py::test_plot_uncertainties_bar -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest master_thesis_code_test/plotting/ -m "not gpu and not slow" -x`
- **Per wave merge:** `uv run pytest -m "not gpu and not slow"`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `master_thesis_code_test/plotting/test_data.py` -- covers FISH-01 (covariance reconstruction, constants)
- [ ] `master_thesis_code_test/plotting/test_fisher_plots.py` -- covers FISH-02, FISH-04, FISH-05 (all three factory functions)
- [ ] Fixtures in `conftest.py`: `sample_crb_row` (pd.Series with 105 delta columns + 14 params + metadata), `sample_crb_dataframe` (5-10 rows)

## Project Constraints (from CLAUDE.md)

- **Typing:** All functions must have complete type annotations. Use `npt.NDArray[np.float64]`, `list[str]` not `List[str]`, `X | None` not `Optional[X]`.
- **Dataclass defaults:** Never use mutable default; use `field(default_factory=...)`.
- **File naming:** `snake_case.py`, prefix `_` for internal modules.
- **Docstrings:** NumPy-style for new code.
- **Pre-commit:** ruff + mypy run on every commit.
- **Tests:** Must run on CPU-only machine. `pytest -m "not gpu and not slow"` must pass.
- **No physics changes:** This phase is pure visualization -- no formula modifications, no GPU code, no physics-change protocol needed.
- **Plotting pattern:** Factory functions, `data in, (fig, ax) out`. No `plt.show()` or `plt.savefig()` in factories.
- **Style:** Use `_labels.py:LABELS`, `_colors.py` constants, `_helpers.py:get_figure()`.

## Open Questions

1. **EMRI signal track for h_c(f) plot**
   - What we know: D-07 says precomputed/hardcoded reference EMRI parameters. No waveform generation at plot time.
   - What's unclear: The exact h_c(f) data for a representative EMRI. Could embed a small array of (f, h_c) values computed offline, or compute from an analytic approximation.
   - Recommendation: Use a simple power-law approximation for the EMRI signal track (h_c ~ f^(-7/6) scaled to the right amplitude for a typical source), or extract one event's Fourier transform from saved data. Claude's discretion per CONTEXT.md.

2. **LisaTdiConfiguration import safety**
   - What we know: The file has a `try/except` guard for cupy at the top now.
   - What's unclear: Whether the current guard is sufficient for all CPU-only import paths.
   - Recommendation: Use deferred import inside the strain plot function (already shown in code example). This is safe regardless of the guard status.

## Sources

### Primary (HIGH confidence)
- `master_thesis_code/parameter_estimation/parameter_estimation.py:410-418` -- CRB CSV column construction verified
- `master_thesis_code/datamodels/parameter_space.py:150-166` -- 14 parameter names and order verified
- `evaluation/run_20260328_seed100_v3/simulations/cramer_rao_bounds.csv` -- actual CSV inspected: 105 delta columns, 19 non-delta columns confirmed
- `master_thesis_code/plotting/_helpers.py` -- factory pattern, get_figure, _fig_from_ax verified
- `master_thesis_code/plotting/_labels.py` -- LABELS dict verified, `luminosity_distance` mapping gap identified
- `master_thesis_code/plotting/_colors.py` -- CYCLE, TRUTH, REFERENCE, EDGE verified
- `master_thesis_code/LISA_configuration.py` -- PSD functions, confusion noise, include_confusion_noise flag verified
- `master_thesis_code_test/plotting/conftest.py` -- existing fixtures and _close_figures pattern verified

### Secondary (MEDIUM confidence)
- matplotlib.patches.Ellipse API -- standard matplotlib, well-documented
- np.linalg.eigh -- standard numpy, well-documented for symmetric matrices

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies, all in existing project
- Architecture: HIGH -- follows established patterns from Phase 14/15
- Pitfalls: HIGH -- based on direct code inspection and CSV format verification

**Research date:** 2026-04-02
**Valid until:** 2026-05-02 (stable domain, no external dependencies)
