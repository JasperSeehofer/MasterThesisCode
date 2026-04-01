# Requirements: EMRI Parameter Estimation v1.3

**Defined:** 2026-04-01
**Core Value:** The simulation pipeline runs reliably on the GPU cluster as SLURM array jobs, producing enough Cramer-Rao bounds for statistically meaningful Hubble constant posteriors.

## v1.3 Requirements

Requirements for visualization overhaul. Each maps to roadmap phases.

### Test Infrastructure

- [ ] **TEST-01**: Every existing plot factory function has a smoke test that verifies it returns (Figure, Axes) without error
- [ ] **TEST-02**: rcParams snapshot regression test detects unintended style mutations after `apply_style()`

### Style Infrastructure

- [ ] **STYLE-01**: All figures use standardized sizes matching thesis column widths (single ~3.5in, double ~7.0in)
- [ ] **STYLE-02**: All axis labels use LaTeX mathematical notation ($M_\bullet$, $d_L$, $\sigma$) via mathtext or optional `text.usetex`
- [ ] **STYLE-03**: `apply_style()` accepts `use_latex` toggle for opt-in LaTeX rendering with graceful fallback
- [ ] **STYLE-04**: Centralized color palette (`_colors.py`) replaces ad-hoc color strings across all plot modules
- [ ] **STYLE-05**: `_fig_from_ax` helper moved from `simulation_plots.py` to `_helpers.py` for shared use

### Core Result Figures (Existing Plot Upgrades)

- [ ] **CORE-01**: H0 posterior plot shows shaded 68%/95% credible intervals with Planck/SH0ES reference bands
- [ ] **CORE-02**: Individual event posteriors are color-coded by SNR or redshift with combined posterior highlighted
- [ ] **CORE-03**: SNR distribution includes cumulative overlay and threshold annotation line
- [ ] **CORE-04**: Detection yield vs redshift overlays injected vs detected populations with detection fraction curve
- [ ] **CORE-05**: Detection probability heatmap P_det(d_L, M) has clean colorbar with [0,1] probability range
- [ ] **CORE-06**: LISA PSD plot overlays galactic confusion noise component alongside instrument noise
- [ ] **CORE-07**: d_L(z) relation plot includes comparison curves for different H0 values

### Fisher & Parameter Estimation Plots (New)

- [ ] **FISH-01**: `_data.py` module reconstructs 14x14 covariance matrices from CRB CSV `delta_X_delta_Y` columns
- [ ] **FISH-02**: 2D Fisher error ellipses (1-sigma, 2-sigma) for key EMRI parameter pairs
- [ ] **FISH-03**: Corner plot of EMRI parameter subset using Fisher-derived Gaussian approximation (via `corner` library)
- [ ] **FISH-04**: Characteristic strain h_c(f) plot with example EMRI track overlaid on LISA sensitivity curve
- [ ] **FISH-05**: Parameter uncertainty distributions with intrinsic/extrinsic grouping and LaTeX labels
- [ ] **FISH-06**: Detection contour in (z, M) space with injected population overlay
- [ ] **FISH-07**: Injected vs recovered scatter showing measurement quality across parameter space

### Sky Localization (New)

- [ ] **SKY-01**: Mollweide projection sky localization map replaces existing non-standard 3D scatter plot

### Convergence & Diagnostics (New)

- [ ] **CONV-01**: H0 convergence plot showing posterior narrowing as N_events increases
- [ ] **CONV-02**: Detection efficiency curve (1D P_det slice with confidence intervals)

### Campaign & Batch (New)

- [ ] **CAMP-01**: Multi-panel summary composite figure combining key result plots
- [ ] **CAMP-02**: Batch figure generation script producing all thesis figures from campaign data
- [ ] **CAMP-03**: PDF size audit with `rasterized=True` fixes for scatter plots exceeding 2MB

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Dark Energy Model

- **wCDM-01**: w0, wa parameters used in distance calculations (currently silently ignored)

### Cosmological Parameters

- **COSMO-01**: Planck 2018 cosmological parameters replace WMAP-era values

### Observational Uncertainty

- **UNCERT-01**: Galaxy redshift uncertainty scaling corrected to standard (1+z) form

### Visualization Extras

- **VIZ-01**: Waveform strain time series plot for introduction chapter
- **VIZ-02**: Fisher condition number scatter for numerical stability analysis
- **VIZ-03**: Interactive Jupyter exploration notebooks (thesis output is PDF)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Interactive dashboards (Plotly, Bokeh) | Thesis output is PDF; interactive plots cannot be embedded |
| Full MCMC corner plots | Pipeline uses Fisher matrix, not MCMC; fake chains would be misleading |
| Animated waveform visualizations | Cannot embed in thesis PDF |
| 3D surface plots | Non-standard in EMRI/LISA literature; hard to read in print |
| Seaborn/Altair wrappers | Style pollution risk; matplotlib is GW community standard |
| healpy sky maps | Overkill for ~100 detections; built-in Mollweide sufficient |
| ArviZ integration | Designed for MCMC diagnostics; Fisher pipeline doesn't need it |
| Time-frequency spectrograms | Relevant to EMRI detection, not Fisher-matrix PE |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| — | — | (populated by roadmapper) |

**Coverage:**
- v1.3 requirements: 24 total
- Mapped to phases: 0
- Unmapped: 24

---
*Requirements defined: 2026-04-01*
*Last updated: 2026-04-01 after initial definition*
