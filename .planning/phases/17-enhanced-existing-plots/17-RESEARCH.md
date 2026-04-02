# Phase 17: Enhanced Existing Plots - Research

**Researched:** 2026-04-02
**Domain:** Matplotlib thesis-quality figure upgrades for EMRI plotting modules
**Confidence:** HIGH

## Summary

Phase 17 upgrades six existing plot modules (`bayesian_plots`, `evaluation_plots`, `model_plots`, `physical_relations_plots`, `simulation_plots`, `catalog_plots`) from basic functional plots to thesis-quality figures. The current codebase has a solid style infrastructure from Phases 15-16 (`_colors.py`, `_labels.py`, `_helpers.py`, `_data.py`, `_style.py`) that is almost entirely unused by the existing plot factories -- every module still uses hardcoded figsize, ad-hoc color strings (`"green"`, `"red"`, `"black"`), plain-text axis labels, and `plt.subplots()` instead of `get_figure(preset=...)`.

The work divides naturally into four groups: (1) Bayesian posterior plots (signature changes, new `color_by` parameter, credible interval shading, Planck/SH0ES reference bands -- highest complexity), (2) distribution plots (SNR histogram with CDF, detection yield dual-histogram, catalog histograms -- moderate complexity), (3) physics plots (LISA PSD decomposition, d_L(z) multi-H0 comparison -- low complexity since `fisher_plots.py` already implements the PSD decomposition pattern), and (4) recovery/evaluation plots (injected-vs-recovered scatter with residual sub-panels, detection probability heatmaps in two coordinate spaces -- moderate-high complexity). All 75 existing smoke tests pass and must remain green; several tests will need updated call signatures.

**Primary recommendation:** Split into 3 plans: Plan 1 (Bayesian posteriors + d_L), Plan 2 (distributions + PSD), Plan 3 (heatmaps + recovery scatter). This balances risk and natural dependency flow.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: 68%/95% credible intervals shown as shaded bands (darker for 68%, lighter for 95%) PLUS thin boundary lines at interval edges for precise reading.
- D-02: Planck (67.4 +/- 0.5) and SH0ES (73.0 +/- 1.0) shown as labeled vertical bands with 1-sigma shading. Text labels inline on the plot.
- D-03: Individual event posteriors use a `color_by` parameter accepting three modes: `'snr'`, `'redshift'`, `'dl_error'` (fractional luminosity distance error). Combined posterior always rendered as a thick black line on top.
- D-04: Single factory function `plot_event_posteriors(h_values, posteriors, color_by='snr', color_values=..., ...)` -- caller passes the color values array. Colorbar label updates automatically based on `color_by` selection.
- D-05: Posterior normalization: default peak-normalized (peak=1), optional `normalize='density'` parameter for proper probability density (integral=1).
- D-06: Histogram on left y-axis with cumulative fraction (CDF) as step function on right y-axis. SNR threshold as vertical dashed line with annotation showing fraction of events above threshold.
- D-07: Injected population as outline histogram, detected population as filled histogram inside it. Detection fraction curve on right y-axis. Shows both absolute counts and efficiency in one figure.
- D-08: Two separate factory functions for both coordinate spaces: P_det(z, M) and P_det(d_L, M). Both include colorbar spanning [0,1], detection contour lines at 0.5 and 0.9. Injected population overlay as scatter (detected=filled, missed=open circles).
- D-09: Three curves on log-log axes: S_inst(f) as dashed, S_gal(f) as dash-dot, total S_n(f) as solid thick line. Standard LISA frequency range. Matches Phase 9 confusion noise decomposition.
- D-10: Factory accepts a configurable list of H0 values. Default includes Planck (67.4), SH0ES (73.0), and simulation true value (73.0). Each curve labeled in legend with H0 value.
- D-11: Multi-panel grid (2x3 or 3x3) with configurable parameter list. Default key parameters: M, mu, d_L, a, e0, sky angles. Identity line + scatter points + 1-sigma CRB error bars on each panel.
- D-12: Residual sub-panels below each scatter (like HEP ratio plots). Shows (recovered - injected) to reveal systematic bias.

### Claude's Discretion
- Exact alpha levels for credible interval shading
- Colormap choice for event posterior color mapping (viridis or similar sequential)
- Histogram bin counts and edge styling
- Aspect ratios for multi-panel figures
- Residual sub-panel height ratio relative to main scatter
- Which specific sky angle parameters to include in default grid
- Line widths and dash patterns for PSD curves
- Error bar cap styling on recovery scatter

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CORE-01 | H0 posterior with shaded 68%/95% credible intervals and Planck/SH0ES reference lines | `plot_combined_posterior` in `bayesian_plots.py` -- needs credible interval computation, shaded bands, reference bands. scipy.stats for percentile computation. |
| CORE-02 | Individual event posteriors distinguishable by SNR/redshift via color mapping with combined posterior prominent | `plot_event_posteriors` in `bayesian_plots.py` -- needs signature change to accept `color_by`/`color_values`, ScalarMappable colorbar, thick combined line. |
| CORE-03 | SNR distribution histogram with cumulative distribution overlay and threshold annotation | `plot_detection_redshift_distribution` in `bayesian_plots.py` -- needs new factory or significant rework; add CDF on twinx, threshold vline with annotation. |
| CORE-04 | Detection yield vs redshift with injected/detected populations and detection fraction curve | New factory function needed (not in current codebase); dual histogram + efficiency on twinx. |
| CORE-05 | Detection probability heatmap with colorbar [0,1] and detection contours in (z, M) space | `plot_detection_probability_grid` in `model_plots.py` operates in (d_L, M); need new (z, M) variant per D-08. |
| CORE-06 | LISA PSD with galactic confusion noise as separate curve | `plot_lisa_psd` in `simulation_plots.py` -- needs PSD decomposition like `fisher_plots.py:plot_characteristic_strain`. |
| CORE-07 | d_L(z) with comparison curves for different H0 values | `plot_distance_redshift` in `physical_relations_plots.py` -- needs multi-curve support with configurable H0 list. |
| FISH-06 | Detection probability heatmap with injected population overlay | Extends D-08: scatter overlay on P_det heatmaps with detected/missed markers. |
| FISH-07 | Injected vs recovered scatter with identity lines and residual annotations | New multi-panel factory using `_data.py` for CRB reconstruction; HEP-style residual sub-panels per D-11/D-12. |
</phase_requirements>

## Current State Audit

### Module-by-Module Analysis

#### bayesian_plots.py (5 functions, HIGHEST upgrade effort)

| Function | Current State | Upgrade Required | Risk |
|----------|--------------|-----------------|------|
| `plot_combined_posterior` | Basic line + green vline, hardcoded figsize (10,6), no `_colors`/`_labels` | Add credible interval shading (D-01), Planck/SH0ES bands (D-02), normalization option (D-05), use `get_figure`/`_colors`/`_labels` | MEDIUM -- new optional params, backward-compatible |
| `plot_event_posteriors` | Dict[int, list[float]] input, uniform alpha, no color mapping | **Signature change** (D-03/D-04): add `color_by`, `color_values`, combined_posterior params; ScalarMappable colorbar | HIGH -- test must update |
| `plot_subset_posteriors` | Basic multi-line, hardcoded figsize | Style migration only (use `get_figure`, `_colors`, `_labels`) | LOW |
| `plot_detection_redshift_distribution` | Simple histogram | Add CDF on twinx for CORE-03 (D-06), threshold line | MEDIUM -- new optional params |
| `plot_number_of_possible_hosts` | Simple histogram | Style migration only | LOW |

**Key signature change:** `plot_event_posteriors` currently takes `posterior_data: dict[int, list[float]]`. D-04 specifies `posteriors` (array-like) + `color_by` + `color_values`. The test calls with a dict; this must change. Could keep backward compatibility by accepting both, but cleaner to update the signature and test.

#### evaluation_plots.py (5 functions, MODERATE effort)

| Function | Current State | Upgrade Required | Risk |
|----------|--------------|-----------------|------|
| `plot_mean_cramer_rao_bounds` | Basic imshow heatmap | Use `_labels`, `_colors.CMAP`, `make_colorbar`, `get_figure` | LOW |
| `plot_uncertainty_violins` | Basic violinplot | Use `_labels` for tick labels, `_colors` | LOW |
| `plot_sky_localization_3d` | 3D scatter (to be replaced in Phase 18) | Minimal -- will be replaced; just style migration | LOW |
| `plot_detection_contour` | hist2d in (z, M) space | Evolves into P_det heatmap (D-08); needs contour lines at 0.5/0.9, scatter overlay | HIGH |
| `plot_generation_time_histogram` | Basic histogram with mean line | Style migration, use `_colors.MEAN` | LOW |

#### model_plots.py (4 functions, MODERATE effort)

| Function | Current State | Upgrade Required | Risk |
|----------|--------------|-----------------|------|
| `plot_emri_distribution` | contourf in (z, M) | Style migration | LOW |
| `plot_emri_rate` | Basic log-log line | Style migration | LOW |
| `plot_emri_sampling` | hist2d sampling | Style migration | LOW |
| `plot_detection_probability_grid` | contourf in (d_L, M) | D-08: add contour lines at 0.5/0.9, scatter overlay (detected/missed), colorbar [0,1]. PLUS new (z, M) variant. | HIGH |

#### physical_relations_plots.py (1 function, LOW effort)

| Function | Current State | Upgrade Required | Risk |
|----------|--------------|-----------------|------|
| `plot_distance_redshift` | Single line plot | D-10: accept list of H0 values, plot multiple curves with legend | MEDIUM -- signature change adds `h0_values` param |

#### simulation_plots.py (4 functions + PlottingCallback, MODERATE effort)

| Function | Current State | Upgrade Required | Risk |
|----------|--------------|-----------------|------|
| `plot_gpu_usage` | Multi-line GPU memory | Style migration only | LOW |
| `plot_lisa_psd` | Generic PSD by channel name | D-09: PSD decomposition (S_inst, S_gal, S_total) with distinct line styles | MEDIUM -- either rework or add new dedicated function |
| `plot_lisa_noise_components` | S_OMS + S_TM only | Style migration | LOW |
| `plot_cramer_rao_coverage` | 3D scatter | Minimal (Phase 18 replaces) | LOW |

**Note:** `fisher_plots.py` already has `plot_characteristic_strain()` (lines 163-217) that implements the PSD decomposition pattern using `LisaTdiConfiguration(include_confusion_noise=True/False)`. D-09 asks for PSD (not characteristic strain), so the approach is the same but plots S_n(f) directly instead of h_c(f) = sqrt(f * S_n(f)).

#### catalog_plots.py (4 functions, LOW effort)

| Function | Current State | Upgrade Required | Risk |
|----------|--------------|-----------------|------|
| `plot_bh_mass_distribution` | Log-log histogram | Style migration | LOW |
| `plot_redshift_distribution` | Log-y histogram | Style migration | LOW |
| `plot_glade_completeness` | Basic line | Style migration | LOW |
| `plot_comoving_volume_sampling` | Basic density histogram | Style migration | LOW |

## Standard Stack

### Core (already in project)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | (project version) | All plotting | Core dependency |
| numpy | (project version) | Array operations, percentile computation | Core dependency |
| scipy.stats | (project version) | Percentile/HDI computation for credible intervals | Already a dependency |

### Supporting (already available)

| Library | Purpose | When to Use |
|---------|---------|-------------|
| `_colors.py` | `TRUTH`, `MEAN`, `EDGE`, `REFERENCE`, `CYCLE`, `CMAP` | Every plot -- replace all hardcoded color strings |
| `_labels.py` | `LABELS` dict with LaTeX labels | Every axis label |
| `_helpers.py` | `get_figure(preset=)`, `save_figure()`, `make_colorbar()` | Every figure creation |
| `_data.py` | `PARAMETER_NAMES`, `reconstruct_covariance()`, `label_key()` | Recovery scatter plots |
| `LisaTdiConfiguration` | PSD with/without confusion noise | PSD decomposition plot |

No new dependencies are needed.

## Architecture Patterns

### Pattern 1: Style Migration (every function)

Every factory function needs this baseline upgrade:

```python
# BEFORE (current state in all modules):
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("Redshift")
ax.plot(x, y, color="green", linestyle="dashed")

# AFTER:
from master_thesis_code.plotting._helpers import get_figure, _fig_from_ax
from master_thesis_code.plotting._colors import TRUTH, MEAN, EDGE, REFERENCE, CMAP
from master_thesis_code.plotting._labels import LABELS

if ax is None:
    fig, ax = get_figure(preset="single")  # or "double" for wide figures
else:
    fig = _fig_from_ax(ax)
ax.set_xlabel(LABELS["z"])
ax.plot(x, y, color=TRUTH, linestyle="dashed")
```

### Pattern 2: Dual-Axis (CDF overlay, detection fraction)

For D-06 and D-07 (CDF on right y-axis, detection fraction on right y-axis):

```python
def plot_snr_distribution(
    snr_values: npt.NDArray[np.float64],
    *,
    snr_threshold: float = 20.0,
    bins: int = 50,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = get_figure(preset="single")
    else:
        fig = _fig_from_ax(ax)

    # Left axis: histogram
    ax.hist(snr_values, bins=bins, edgecolor=EDGE, alpha=0.7)

    # Right axis: CDF
    ax2 = ax.twinx()
    sorted_snr = np.sort(snr_values)
    cdf = np.arange(1, len(sorted_snr) + 1) / len(sorted_snr)
    ax2.step(sorted_snr, cdf, color=MEAN, where="post")
    ax2.set_ylabel("Cumulative fraction")

    # Threshold annotation
    frac_above = np.mean(snr_values >= snr_threshold)
    ax.axvline(snr_threshold, color=REFERENCE, linestyle="--")
    ax.annotate(f"{frac_above:.0%} above threshold", ...)

    return fig, ax
```

**Return convention:** Always return the primary (left) axes. The caller gets the twinx via `ax.get_shared_x_axes()` if needed, but smoke tests only check `(Figure, Axes)`.

### Pattern 3: Color-Mapped Lines with Colorbar (D-03/D-04)

```python
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

norm = Normalize(vmin=color_values.min(), vmax=color_values.max())
cmap = plt.get_cmap(CMAP)

for i, posterior in enumerate(posteriors):
    ax.plot(h_values, posterior, color=cmap(norm(color_values[i])), alpha=0.4, linewidth=0.5)

# Colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
colorbar_label = {"snr": LABELS["SNR"], "redshift": LABELS["z"], "dl_error": r"$\sigma(d_L)/d_L$"}
fig.colorbar(sm, ax=ax, label=colorbar_label[color_by])
```

### Pattern 4: Credible Interval Shading (D-01)

```python
# Compute credible intervals from posterior
cumsum = np.cumsum(posterior) / np.sum(posterior)
h_lower_95 = h_values[np.searchsorted(cumsum, 0.025)]
h_upper_95 = h_values[np.searchsorted(cumsum, 0.975)]
h_lower_68 = h_values[np.searchsorted(cumsum, 0.16)]
h_upper_68 = h_values[np.searchsorted(cumsum, 0.84)]

ax.fill_between(h_values, 0, posterior, where=(h_values >= h_lower_95) & (h_values <= h_upper_95),
                alpha=0.15, color=CYCLE[0])  # 95% lighter
ax.fill_between(h_values, 0, posterior, where=(h_values >= h_lower_68) & (h_values <= h_upper_68),
                alpha=0.3, color=CYCLE[0])  # 68% darker
# Boundary lines
for boundary in [h_lower_68, h_upper_68, h_lower_95, h_upper_95]:
    ax.axvline(boundary, color=CYCLE[0], linewidth=0.5, alpha=0.5)
```

### Pattern 5: Reference Bands (D-02 -- Planck/SH0ES)

```python
# Planck: 67.4 +/- 0.5 (in dimensionless h units: 0.674 +/- 0.005)
planck_h, planck_sigma = 0.674, 0.005
ax.axvspan(planck_h - planck_sigma, planck_h + planck_sigma, alpha=0.15, color=CYCLE[0])
ax.axvline(planck_h, color=CYCLE[0], linewidth=0.8)
ax.text(planck_h, y_pos, "Planck", fontsize=8, ha="center")

# SH0ES: 73.0 +/- 1.0 (in dimensionless h: 0.73 +/- 0.01)
shoes_h, shoes_sigma = 0.73, 0.01
ax.axvspan(shoes_h - shoes_sigma, shoes_h + shoes_sigma, alpha=0.15, color=CYCLE[1])
ax.axvline(shoes_h, color=CYCLE[1], linewidth=0.8)
ax.text(shoes_h, y_pos, r"SH0ES", fontsize=8, ha="center")
```

**Note:** The h_values in the existing code range from 0.5 to 1.0 (dimensionless h), so Planck=0.674 and SH0ES=0.73 are the correct values. The D-02 text says "67.4" and "73.0" which are in km/s/Mpc units; need to use the correct unit system matching the x-axis.

### Pattern 6: Multi-Panel with Residual Sub-Panels (D-11/D-12)

```python
from matplotlib.gridspec import GridSpec

n_params = len(parameters)
ncols = 3
nrows = (n_params + ncols - 1) // ncols

fig = plt.figure(figsize=(7.0, 2.5 * nrows))  # or get_figure with custom size
gs = GridSpec(nrows * 2, ncols, height_ratios=[3, 1] * nrows, hspace=0.05)

for i, param in enumerate(parameters):
    row, col = divmod(i, ncols)
    ax_main = fig.add_subplot(gs[row * 2, col])
    ax_resid = fig.add_subplot(gs[row * 2 + 1, col], sharex=ax_main)

    # Main panel: identity line + scatter + error bars
    ax_main.plot([lo, hi], [lo, hi], color=REFERENCE, linestyle="--")
    ax_main.errorbar(injected, recovered, yerr=sigma_crb, fmt=".", color=CYCLE[0], capsize=2)

    # Residual panel
    residual = recovered - injected
    ax_resid.errorbar(injected, residual, yerr=sigma_crb, fmt=".", color=CYCLE[0], capsize=2)
    ax_resid.axhline(0, color=REFERENCE, linestyle="--")

    ax_main.set_ylabel(LABELS[label_key(param)])
    ax_resid.set_xlabel(LABELS[label_key(param)] + " (injected)")
    plt.setp(ax_main.get_xticklabels(), visible=False)
```

### Pattern 7: PSD Decomposition (D-09)

Already proven in `fisher_plots.py:plot_characteristic_strain()` (lines 188-206):

```python
from master_thesis_code.LISA_configuration import LisaTdiConfiguration

lisa_total = LisaTdiConfiguration(include_confusion_noise=True)
lisa_inst = LisaTdiConfiguration(include_confusion_noise=False)

freqs = np.geomspace(1e-5, 1e-1, 1000)
psd_total = lisa_total.power_spectral_density_a_channel(freqs)
psd_inst = lisa_inst.power_spectral_density_a_channel(freqs)
psd_confusion = psd_total - psd_inst

ax.loglog(freqs, psd_total, color=EDGE, linestyle="-", linewidth=2, label=r"$S_n(f)$ total")
ax.loglog(freqs, psd_inst, color=REFERENCE, linestyle="--", label=r"$S_\mathrm{inst}(f)$")
ax.loglog(freqs, np.maximum(psd_confusion, 0), color=CYCLE[1], linestyle="-.", label=r"$S_\mathrm{gal}(f)$")
```

**Import note:** `LisaTdiConfiguration` has a guarded `import cupy` at module top level. The guard works (`try/except ImportError`), so importing on CPU is safe. `fisher_plots.py` already does this import at function scope (deferred import pattern) -- follow the same pattern.

### Anti-Patterns to Avoid
- **Hardcoded figsize with plt.subplots:** Use `get_figure(preset=...)` instead
- **Hardcoded color strings:** Use `_colors.py` semantic names
- **Plain-text axis labels:** Use `_labels.py` LABELS dict
- **Direct plt.show()/plt.savefig():** Factory functions return (fig, ax); caller saves
- **Breaking return type:** All functions must return `tuple[Figure, Axes]` (or `tuple[Figure, Any]` for 3D)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PSD decomposition | Manual noise formula | `LisaTdiConfiguration(include_confusion_noise=True/False)` | Already implemented and tested; `fisher_plots.py` shows the pattern |
| Covariance reconstruction | Manual CSV parsing | `_data.py:reconstruct_covariance()` | Handles 14x14 symmetric matrix, column naming conventions |
| Parameter labels | Hardcoded strings | `_labels.py:LABELS` + `_data.py:label_key()` | LaTeX formatting, consistent across all plots |
| Figure sizing | Manual figsize tuples | `get_figure(preset="single"/"double")` | REVTeX column widths baked in |
| Colorbar creation | Manual `fig.colorbar()` | `make_colorbar()` from `_helpers.py` | Consistent styling |

## Common Pitfalls

### Pitfall 1: Unit Mismatch on H0 Axis
**What goes wrong:** D-02 mentions Planck "67.4" and SH0ES "73.0" in km/s/Mpc units, but the existing posterior code uses dimensionless h (0.5 to 1.0 range). Plotting reference bands at 67.4 on a 0.5-1.0 axis would be invisible.
**Why it happens:** Mixed units in the literature (H0 vs h).
**How to avoid:** Check the x-axis range. The existing `sample_h_values` fixture uses `np.linspace(0.5, 1.0, 50)`. Use h=0.674 (Planck) and h=0.73 (SH0ES) for dimensionless axis. If axis is in km/s/Mpc, use 67.4 and 73.0.
**Warning signs:** Reference lines not visible on the plot.

### Pitfall 2: Signature Changes Breaking Tests
**What goes wrong:** Changing `plot_event_posteriors` from `posterior_data: dict[int, list[float]]` to a new signature breaks the test that passes a dict.
**How to avoid:** Update tests in the same commit as the signature change. Keep all new parameters optional with sensible defaults so existing callers aren't broken.
**Warning signs:** Test failures after refactoring.

### Pitfall 3: twinx() Return Type Confusion
**What goes wrong:** Functions with dual y-axes (D-06, D-07) might return the secondary axes instead of the primary, breaking `isinstance(ax, Axes)` checks.
**How to avoid:** Always return the primary (left) axes from the factory function. Document that `ax.twinx()` exists but is internal.

### Pitfall 4: Negative Confusion PSD
**What goes wrong:** `psd_total - psd_inst` can produce tiny negative values from floating-point arithmetic at high frequencies where confusion noise is negligible.
**How to avoid:** Use `np.maximum(psd_confusion, 0.0)` before plotting (already done in `fisher_plots.py`).

### Pitfall 5: Colorbar with External Axes
**What goes wrong:** When caller passes their own `ax`, creating a colorbar via `fig.colorbar()` may steal space from unexpected axes in the caller's figure.
**How to avoid:** Use `make_colorbar(mappable, fig, ax)` which handles the `ax=` parameter correctly.

### Pitfall 6: GridSpec Figure vs get_figure
**What goes wrong:** Multi-panel recovery scatter (D-11/D-12) needs `GridSpec` with non-uniform height ratios. `get_figure()` creates uniform subplots.
**How to avoid:** For the recovery scatter, use `plt.figure()` + `GridSpec` directly. This is an acceptable exception to the `get_figure` pattern. Set the overall figsize to match "double" preset width (7.0 inches).

## Test Impact Analysis

### Tests That Need Signature Updates

| Test | Current Call | New Parameters | Action |
|------|-------------|---------------|--------|
| `test_plot_event_posteriors` | `posterior_data={0: [...], 1: [...]}` | Needs `color_values` array if `color_by` is specified | Update test: provide `color_values` or rely on default (no color mapping) |
| `test_plot_detection_contour` | `(redshifts, masses)` | May split into P_det variants | New tests for new factory functions |

### Tests That Just Need Assertions Tightened

All other tests call with positional/keyword args that won't change. They assert `isinstance(fig, Figure)` and `isinstance(ax, Axes)` -- these stay valid.

### New Tests Needed

| New Factory | Test File | What to Assert |
|-------------|-----------|----------------|
| `plot_snr_distribution` (D-06) | `test_bayesian_plots.py` or `test_simulation_plots.py` | Returns (Fig, Axes), twinx exists |
| `plot_detection_yield` (D-07) | `test_model_plots.py` or `test_bayesian_plots.py` | Returns (Fig, Axes) |
| `plot_detection_probability_zM` (D-08 variant) | `test_model_plots.py` | Returns (Fig, Axes) |
| `plot_injected_vs_recovered` (D-11/D-12) | `test_evaluation_plots.py` | Returns (Fig, Figure-level axes or array) |

## Plan Splitting Recommendation

### Plan 1: Bayesian Posteriors and Distance Relation (CORE-01, CORE-02, CORE-07)
**Files:** `bayesian_plots.py`, `physical_relations_plots.py`, their test files
**Scope:**
- Upgrade `plot_combined_posterior`: credible intervals (D-01), Planck/SH0ES bands (D-02), normalization (D-05), style migration
- Upgrade `plot_event_posteriors`: `color_by` parameter (D-03/D-04), colorbar, combined posterior line, style migration
- Style-migrate `plot_subset_posteriors`, `plot_number_of_possible_hosts`
- Upgrade `plot_distance_redshift`: multi-H0 curves (D-10), style migration
- Update affected tests
**Risk:** MEDIUM-HIGH (signature change on `plot_event_posteriors`)
**Estimated functions:** 5 upgraded + test updates

### Plan 2: Distribution Plots and PSD (CORE-03, CORE-06)
**Files:** `bayesian_plots.py` (SNR function), `simulation_plots.py`, `catalog_plots.py`, their test files
**Scope:**
- New `plot_snr_distribution` function or upgrade `plot_detection_redshift_distribution` (D-06): histogram + CDF + threshold annotation
- Upgrade `plot_lisa_psd` or add `plot_lisa_psd_decomposition` (D-09): three-curve PSD
- Style-migrate `plot_lisa_noise_components`, `plot_gpu_usage`, `plot_cramer_rao_coverage`
- Style-migrate all `catalog_plots.py` functions (4 functions, purely cosmetic)
- Update/add tests
**Risk:** LOW-MEDIUM (mostly additive)
**Estimated functions:** ~10 upgraded + new SNR function

### Plan 3: Heatmaps and Recovery Scatter (CORE-04, CORE-05, FISH-06, FISH-07)
**Files:** `model_plots.py`, `evaluation_plots.py`, their test files
**Scope:**
- Upgrade `plot_detection_probability_grid` with contours, scatter overlay (D-08)
- New `plot_detection_probability_zM` for (z, M) space (D-08)
- New `plot_detection_yield` function (D-07): dual histogram + efficiency curve
- New `plot_injected_vs_recovered` multi-panel with residuals (D-11/D-12)
- Style-migrate remaining `evaluation_plots.py` functions
- Update/add tests
**Risk:** MEDIUM-HIGH (new complex multi-panel factory, new factory functions)
**Estimated functions:** ~7 upgraded/new + test additions

### Dependency Order
Plan 1 and Plan 2 are independent -- can be done in any order.
Plan 3 depends loosely on style patterns established in Plans 1-2 but has no code dependency.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (project version) |
| Config file | `pyproject.toml` [tool.pytest] |
| Quick run command | `uv run pytest master_thesis_code_test/plotting/ -m "not gpu and not slow" -x -q` |
| Full suite command | `uv run pytest -m "not gpu and not slow"` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CORE-01 | Credible interval shading on combined posterior | smoke | `uv run pytest master_thesis_code_test/plotting/test_bayesian_plots.py::test_plot_combined_posterior -x` | Exists (needs update) |
| CORE-02 | Color-mapped event posteriors | smoke | `uv run pytest master_thesis_code_test/plotting/test_bayesian_plots.py::test_plot_event_posteriors -x` | Exists (needs update) |
| CORE-03 | SNR distribution with CDF | smoke | `uv run pytest master_thesis_code_test/plotting/test_bayesian_plots.py::test_plot_snr_distribution -x` | Wave 0 |
| CORE-04 | Detection yield dual histogram | smoke | `uv run pytest master_thesis_code_test/plotting/test_model_plots.py::test_plot_detection_yield -x` | Wave 0 |
| CORE-05 | P_det heatmap (z, M) | smoke | `uv run pytest master_thesis_code_test/plotting/test_model_plots.py::test_plot_detection_probability_zM -x` | Wave 0 |
| CORE-06 | PSD decomposition | smoke | `uv run pytest master_thesis_code_test/plotting/test_simulation_plots.py::test_plot_lisa_psd -x` | Exists (needs update) |
| CORE-07 | Multi-H0 d_L(z) | smoke | `uv run pytest master_thesis_code_test/plotting/test_physical_relations_plots.py::test_plot_distance_redshift -x` | Exists (update for new params) |
| FISH-06 | P_det with population overlay | smoke | Covered by CORE-05 test with scatter data | Wave 0 |
| FISH-07 | Injected vs recovered scatter | smoke | `uv run pytest master_thesis_code_test/plotting/test_evaluation_plots.py::test_plot_injected_vs_recovered -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest master_thesis_code_test/plotting/ -m "not gpu and not slow" -x -q`
- **Per wave merge:** `uv run pytest -m "not gpu and not slow"`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `test_bayesian_plots.py::test_plot_snr_distribution` -- covers CORE-03
- [ ] `test_model_plots.py::test_plot_detection_yield` -- covers CORE-04
- [ ] `test_model_plots.py::test_plot_detection_probability_zM` -- covers CORE-05/FISH-06
- [ ] `test_evaluation_plots.py::test_plot_injected_vs_recovered` -- covers FISH-07
- [ ] New fixtures in `conftest.py` for SNR arrays, injection/recovery data pairs

## Project Constraints (from CLAUDE.md)

- **Package manager:** uv (never manually edit pyproject.toml dependencies)
- **Testing:** `uv run pytest -m "not gpu and not slow"` must pass
- **Linting:** `uv run ruff check --fix` + `uv run ruff format` + `uv run mypy`
- **Type annotations:** Complete on all functions; `npt.NDArray[np.float64]` for arrays
- **Typing style:** `list[float]` not `List[float]`, `X | None` not `Optional[X]`
- **Pre-commit hooks:** Run automatically on commit (ruff + mypy)
- **Factory pattern:** All plot functions: data in, `(fig, ax)` out, no `plt.show()`/`plt.savefig()`
- **Physics changes:** Not applicable -- this phase is plotting only, no formula changes
- **Skill gates:** `/check` before commit, `/pre-commit-docs` before commit

## Sources

### Primary (HIGH confidence)
- Direct code reading of all 6 plot modules, 6 test files, 4 infrastructure files
- `fisher_plots.py` PSD decomposition pattern (lines 163-217)
- `LISA_configuration.py` confusion noise implementation (lines 86-137)
- `constants.py` PSD coefficients and physical constants

### Secondary (HIGH confidence)
- ROADMAP.md success criteria for Phase 17 (lines 119-134)
- 17-CONTEXT.md decisions D-01 through D-12
- Test run output: 75 passed, 0 failed

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new dependencies needed, all tools already in the codebase
- Architecture: HIGH -- patterns proven in `fisher_plots.py` (Phase 16), only need replication
- Pitfalls: HIGH -- identified from direct code reading and known matplotlib behaviors

**Research date:** 2026-04-02
**Valid until:** 2026-05-02 (stable -- matplotlib and project infrastructure unlikely to change)
