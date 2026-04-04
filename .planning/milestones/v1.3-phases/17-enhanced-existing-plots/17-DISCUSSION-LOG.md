# Phase 17: Enhanced Existing Plots - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 17-enhanced-existing-plots
**Areas discussed:** H0 posterior styling, Detection diagnostics, Physical relation plots, Parameter recovery scatter

---

## H0 Posterior Styling

### Credible Intervals

| Option | Description | Selected |
|--------|-------------|----------|
| Shaded bands | Fill between posterior curve with two alpha levels (darker 68%, lighter 95%) | |
| Vertical dashed lines | Mark interval boundaries with vertical lines only | |
| Both | Shaded bands plus thin boundary lines at interval edges | ✓ |

**User's choice:** Both — shaded bands + boundary lines
**Notes:** Standard GW literature style with added precision from boundary lines

### Reference Values

| Option | Description | Selected |
|--------|-------------|----------|
| Vertical bands with labels | Shaded vertical bands showing 1-sigma range with text labels | ✓ |
| Vertical lines only | Single vertical lines at central values with legend entries | |
| Vertical bands, no labels | Shaded bands with legend-only identification | |

**User's choice:** Vertical bands with labels (Recommended)

### Event Posterior Color Mapping

| Option | Description | Selected |
|--------|-------------|----------|
| Color-mapped by SNR | Sequential colormap by SNR, combined as thick black line | |
| Color-mapped by redshift | Sequential colormap by source redshift | |
| Uniform gray, combined highlighted | All individual posteriors in light gray | |

**User's choice:** All three modes via a `color_by` parameter — SNR, redshift, AND fractional d_L error
**Notes:** User wants all three as options in a single factory function, not just one

### API Design

| Option | Description | Selected |
|--------|-------------|----------|
| One function, color_by param | `plot_event_posteriors(color_by='snr', ...)` — DRY | ✓ |
| Three separate functions | Explicit names but duplicated logic | |

**User's choice:** One function, color_by param (Recommended)

### Normalization

| Option | Description | Selected |
|--------|-------------|----------|
| Peak-normalized | Posterior peak = 1, common in GW literature | |
| Probability density | Proper density, integral = 1 | |
| Both available | Default peak-normalized, optional `normalize='density'` | ✓ |

**User's choice:** Both available

---

## Detection Diagnostics

### SNR CDF

| Option | Description | Selected |
|--------|-------------|----------|
| Right y-axis CDF line | Histogram left axis, CDF right axis, threshold annotation | ✓ |
| Separate subplot | Two-panel: histogram + CDF | |
| Inset CDF | Small inset axes with CDF | |

**User's choice:** Right y-axis CDF line (Recommended)

### Detection Yield

| Option | Description | Selected |
|--------|-------------|----------|
| Stacked histograms + fraction | Outline injected, filled detected, fraction on right axis | ✓ |
| Side-by-side histograms | Adjacent bars per bin | |
| Two-panel: counts + fraction | Separate panels for counts and fraction | |

**User's choice:** Stacked histograms + fraction (Recommended)

### P_det Heatmap Coordinates

| Option | Description | Selected |
|--------|-------------|----------|
| P_det(z, M) with injected scatter | Heatmap in (z, M) space with scatter overlay | |
| P_det(d_L, M) with contours only | Heatmap in (d_L, M) space, contour lines only | |
| Both coordinate spaces | Two separate factory functions | ✓ |

**User's choice:** Both coordinate spaces

---

## Physical Relation Plots

### LISA PSD Decomposition

| Option | Description | Selected |
|--------|-------------|----------|
| Three curves: instrument, galactic, total | S_inst dashed, S_gal dash-dot, S_n solid thick | ✓ |
| Two curves: total + galactic only | Total solid, galactic as shaded fill | |
| Four curves: add characteristic strain | All noise + h_c(f) on second axis | |

**User's choice:** Three curves (Recommended)

### d_L(z) H0 Values

| Option | Description | Selected |
|--------|-------------|----------|
| Planck + SH0ES + simulation | Three fixed curves | |
| Five-value sweep | H0 = {65, 67.4, 70, 73.0, 75} gradient | |
| Configurable H0 list | Factory accepts list, defaults to Planck+SH0ES+sim | ✓ |

**User's choice:** Configurable H0 list

---

## Parameter Recovery Scatter

### Layout

| Option | Description | Selected |
|--------|-------------|----------|
| Multi-panel grid, key params | 2x3 or 3x3 grid, configurable parameter list | ✓ |
| Single configurable plot | One scatter at a time, caller loops | |
| Two panels: intrinsic + extrinsic | Grouped normalized axes | |

**User's choice:** Multi-panel grid (Recommended)

### Residuals

| Option | Description | Selected |
|--------|-------------|----------|
| Residual sub-panels | Narrow residual panel below each scatter (HEP-style) | ✓ |
| Color-coded residuals on scatter | Points colored by residual magnitude | |
| Summary statistics only | Text box annotations | |

**User's choice:** Residual sub-panels (Recommended)

---

## Claude's Discretion

- Exact alpha levels for credible interval shading
- Colormap choice for event posterior color mapping
- Histogram bin counts and edge styling
- Aspect ratios for multi-panel figures
- Residual sub-panel height ratio
- Default sky angle parameters in grid
- Line widths and dash patterns
- Error bar cap styling

## Deferred Ideas

None — discussion stayed within phase scope.
