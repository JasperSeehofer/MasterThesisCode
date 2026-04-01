# Phase 16: Data Layer & Fisher Visualizations - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-02
**Phase:** 16-data-layer-fisher
**Areas discussed:** Data layer design, Fisher ellipse plots, Characteristic strain, Uncertainty distributions

---

## Data Layer Design

### Data API

| Option | Description | Selected |
|--------|-------------|----------|
| Row-level function | `reconstruct_covariance(row) -> ndarray(14,14)`. Callers iterate. Simple, composable. | ✓ |
| Batch DataFrame function | `load_covariance_matrices(df) -> list[ndarray]`. Returns all at once. | |
| Both | Row-level primitive + batch wrapper. | |

**User's choice:** Row-level function
**Notes:** Matches factory pattern. Simple and composable.

### Parameter Grouping

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — define groups in _data.py | INTRINSIC and EXTRINSIC lists. Reused by multiple plot factories. | ✓ |
| No — let each plot module define own | Keep _data.py focused on CSV → matrix only. | |

**User's choice:** Yes — define groups in _data.py

### Parameter Names List

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — PARAMETER_NAMES list | Single source of truth for index ↔ name mapping. | ✓ |
| No — derive from CSV headers | Parse delta_X_delta_Y columns each time. | |

**User's choice:** Yes — PARAMETER_NAMES list

---

## Fisher Ellipse Plots

### Parameter Pairs

| Option | Description | Selected |
|--------|-------------|----------|
| M–μ, d_L–ι, sky angles | (M,mu), (d_L,qS), (qS,phiS). Mass, distance-inclination, sky. | ✓ |
| M–μ, M–a, d_L–z | Mass, spin-mass, cosmological. | |
| You decide | Claude picks most informative pairs. | |

**User's choice:** M–μ, d_L–ι, sky angles

### Ellipse Style

| Option | Description | Selected |
|--------|-------------|----------|
| Filled contours with alpha | Shaded 1σ/2σ. Standard GW literature style. | ✓ |
| Outline-only contours | Just boundaries. Cleaner for many events. | |
| You decide | Claude picks based on event count. | |

**User's choice:** Filled contours with alpha

### Overlay Mode

| Option | Description | Selected |
|--------|-------------|----------|
| Single event per plot | One covariance matrix. Subplot externally. | |
| Multi-event overlay | List input, overlays with color cycle. | |
| Both via optional parameter | Single default; optional list for overlay. | ✓ |

**User's choice:** Both via optional parameter

---

## Characteristic Strain

### Waveform Source

| Option | Description | Selected |
|--------|-------------|----------|
| Hardcoded reference EMRI | Representative event parameters as defaults. No generation at plot time. | ✓ |
| Generate on the fly from few | Fresh waveform via few. GPU/CPU dependency in plotting. | |
| Precomputed h_c from file | Saved array loaded at plot time. | |

**User's choice:** Hardcoded reference EMRI

### Noise Components

| Option | Description | Selected |
|--------|-------------|----------|
| Total PSD + galactic foreground | Two curves: S_n(f) and S_gal(f). | |
| Total PSD only | Just combined sensitivity. | |
| Total + instrument + galactic (3 curves) | Full decomposition: S_inst, S_gal, S_n. | ✓ |

**User's choice:** Total + instrument + galactic (3 curves)

---

## Uncertainty Distributions

### Visualization Type

| Option | Description | Selected |
|--------|-------------|----------|
| Bar chart with error bars | Horizontal bars of σ_i/x_i. Standard in PE papers. | |
| Violin plot | Distribution shape across events. Richer but busier. | ✓ |
| Box plot | Quartiles across events. | |
| You decide | Claude picks most appropriate. | |

**User's choice:** Violin plot

### Aggregation Mode

| Option | Description | Selected |
|--------|-------------|----------|
| Single event | One CRB row, 14 parameter bars. | |
| Multi-event aggregate | Full DataFrame, median + spread. | |
| Both via optional parameter | Single default; DataFrame for aggregate. | ✓ |

**User's choice:** Both via optional parameter

### Single-Event Fallback (follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Bar chart fallback | Single → bar chart; multiple → violin. Auto-switch. | ✓ |
| Always violin, thin line | Degenerate violin for one event. | |
| Only multi-event | Drop single-event support. | |

**User's choice:** Bar chart fallback

---

## Claude's Discretion

- Internal covariance reconstruction implementation
- Exact reference EMRI parameters for h_c(f) plot
- Ellipse computation method (eigenvalue decomposition)
- Subplot layouts and aspect ratios
- Violin plot styling details

## Deferred Ideas

None — discussion stayed within phase scope.
