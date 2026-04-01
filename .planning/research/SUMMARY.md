# Project Research Summary

**Project:** EMRI Parameter Estimation — v1.3 Visualization Overhaul
**Domain:** Scientific Python visualization for gravitational wave parameter estimation thesis
**Researched:** 2026-04-01
**Confidence:** HIGH

## Executive Summary

The v1.3 milestone is a focused visualization overhaul for a physics master thesis: produce publication-quality, static PDF-embeddable matplotlib figures from an existing EMRI parameter estimation pipeline. The existing `plotting/` subpackage already has a sound architecture — 23 factory functions across 6 topic modules returning `(Figure, Axes)` tuples with a centralized mplstyle. The research confirms this architecture is correct and should be extended, not replaced. The only new dependency warranted is `corner` (one lightweight, matplotlib-native library), which is the de-facto standard for multi-parameter posterior visualization in the gravitational-wave community.

The recommended approach is to work in five layers: (1) style infrastructure first, because everything downstream depends on consistent rcParams, figure sizes, and LaTeX handling; (2) shared data utilities for covariance matrix reconstruction from existing CRB CSV columns; (3) new plot types critical for thesis results (Fisher ellipses, enhanced posteriors, Mollweide sky maps, improved PSD plots); (4) the `corner` dependency integration; (5) composite campaign dashboards and convergence plots. This order is dictated by hard dependencies in the code and minimizes the risk of cross-module regressions during refactoring.

The dominant risk is silent regression: existing plot functions have no content-level tests, so style and infrastructure changes can break thesis-critical figures (H0 posterior, CRB heatmap) without any test failure. The mitigation is to establish smoke tests and rcParams regression checks before touching any existing function. A secondary risk is LaTeX rendering: `text.usetex` must remain `False` by default to keep CI and the cluster working; LaTeX should be an opt-in toggle for final figure generation on the dev machine only. The project correctly avoids interactive visualization libraries (plotly, bokeh) and heavy specialized dependencies (healpy, ArviZ, seaborn) that would add build complexity without benefiting a static PDF thesis.

## Key Findings

### Recommended Stack

The existing matplotlib + scipy + numpy stack handles all visualization requirements. The only warranted addition is `corner >= 2.2.3`, which is universal in EMRI/LISA parameter estimation papers, is pure Python built on matplotlib, accepts and returns matplotlib figures, and respects rcParams automatically. All seven alternatives evaluated (ChainConsumer, ArviZ, getdist, healpy, ligo.skymap, SciencePlots, seaborn) were rejected on clear technical grounds.

**Core technologies:**
- **matplotlib** (existing): Core rendering engine, Mollweide projections, all plot types — keep as sole backend; no replacements warranted
- **scipy** (existing): Eigendecomposition for Fisher ellipses, KDE — sufficient without any dedicated geometry library
- **corner >= 2.2.3** (new): Multi-parameter posterior corner plots — GW community standard, matplotlib-native, Python 3.13-safe
- **numpy / pandas** (existing): Array operations and CSV loading for plot data — already present, no changes needed

**Explicitly rejected:**
- `healpy`: Designed for full-sky HEALPix maps with millions of pixels; ~100 sky positions need only matplotlib's built-in `projection='mollweide'` plus scipy ellipse geometry
- `ArviZ`: MCMC diagnostics library; this pipeline uses Fisher matrices, not MCMC chains; ~20 transitive dependencies for unused features
- `seaborn`: Style pollution risk via `set_theme()`; physics plots do not need its statistical defaults

### Expected Features

The research identifies a clear three-tier feature set based on what thesis examiners and GW community reviewers expect.

**Must have (table stakes):**
- LISA sensitivity curve with all noise components including confusion noise overlay (every LISA paper)
- Characteristic strain plot with EMRI signal track overlaid on sensitivity curve
- SNR distribution histogram with cumulative overlay and threshold annotation
- Detection yield vs redshift with injected vs detected population overlay
- Detection probability heatmap P_det(d_L, M) with Wilson CI contours
- H0 posterior with shaded 68%/95% credible intervals and Planck/SH0ES reference bands
- Individual event posteriors (spaghetti plot) color-coded by SNR or redshift
- Parameter uncertainty distributions with intrinsic/extrinsic grouping and LaTeX labels
- Detection contour in (z, M) space with injected population overlay
- LaTeX mathematical notation in all axis labels ($M_\bullet$, $d_L$, $\sigma$)
- Consistent figure sizes matching thesis column widths (single: ~3.5 in, double: ~7.0 in)

**Should have (differentiators):**
- Fisher matrix error ellipses for key parameter pairs (M-mu, d_L-inclination, sky angles)
- Corner plot of EMRI parameter subset from Fisher-derived Gaussian approximation
- Mollweide sky localization map replacing the existing non-standard 3D scatter
- Injected vs recovered scatter showing measurement quality across parameter space
- H0 convergence plot showing posterior narrowing as N_events increases
- Detection efficiency curve (1D slice through P_det grid with Wilson CIs)
- Waveform strain time series for the introduction chapter

**Defer to v2+:**
- Multi-panel summary composite figure — build after individual plots are finalized
- Fisher condition number scatter — only if condition number becomes a dedicated thesis section
- Interactive exploration tooling — thesis output is PDF; JupyterLab already present for dev use

### Architecture Approach

The existing factory pattern (`data in, (Figure, Axes) out`, no side effects, optional `ax` for compositing) is architecturally sound and must be preserved across all 23 existing functions and all new additions. The overhaul adds 8 new modules (`_colors.py`, `_data.py`, `fisher_plots.py`, `corner_plots.py`, `sky_plots.py`, `psd_plots.py`, `convergence_plots.py`, `campaign_plots.py`) and modifies 8 existing files (style, helpers, 5 topic modules, `__init__.py`). The key architectural addition is a `_data.py` layer that reconstructs the 14x14 covariance matrix from the lower-triangle `delta_X_delta_Y` CSV columns — this is shared by Fisher ellipses, corner plots, and sky localization plots.

**Major components:**
1. `_colors.py` — Centralized color palette and parameter-to-color mapping; eliminates ad-hoc "green"/"red" scattered through modules
2. `_data.py` — Covariance matrix reconstruction from CRB CSV; shared utility for all Fisher-based visualizations
3. `fisher_plots.py` — 2D Fisher ellipses via scipy eigendecomposition, correlation heatmap, condition number history
4. `corner_plots.py` — Thin `corner.py` wrapper applying thesis styling, returning `(Figure, Axes)` consistently
5. `sky_plots.py` — Mollweide projection using matplotlib's built-in support, sky localization ellipses
6. `psd_plots.py` — Enhanced PSD with characteristic strain h_c(f) and noise budget components
7. `convergence_plots.py` — H0 posterior width vs N_events statistical convergence
8. `campaign_plots.py` — Multi-panel summaries compositing earlier factory functions via the `ax` compositing parameter

### Critical Pitfalls

1. **rcParams global state pollution** — Any library calling `set_theme()` (seaborn) or mutating `rcParams` at import time will silently override `emri_thesis.mplstyle`. Prevent by never adding seaborn globally; add a rcParams snapshot regression test before any refactoring begins.

2. **Silent regression of existing plot functions** — No content-level tests exist for the 23 current factory functions. Style and infrastructure changes can break thesis-critical figures with no test failure. Prevent by adding smoke tests for every existing function and `pytest-mpl` baselines for the 3-5 most critical figures before touching any existing code.

3. **LaTeX usetex breaks CI and cluster** — `text.usetex: True` requires a LaTeX installation absent on CI runners and cluster compute nodes. Prevent by keeping `False` as default in the mplstyle; expose `use_latex=True` as an opt-in parameter to `apply_style()` for final thesis figure generation on the dev machine only.

4. **matplotlib backend lock from eager pyplot imports** — `apply_style()` must call `matplotlib.use("Agg")` before any `import matplotlib.pyplot`. New libraries that import pyplot at module level in their `__init__.py` will lock the backend to TkAgg/Qt5Agg. Prevent by verifying headless compatibility of any new library; CI will catch this immediately on headless runners.

5. **PDF vector bloat from dense scatter plots** — Scatter plots with >1000 points produce multi-megabyte PDFs because each point is a separate vector object. Prevent by using `rasterized=True` on scatter layers while keeping axes/labels as vectors; use `hexbin`/`hist2d` for high-density data following the existing `plot_detection_contour` pattern.

## Implications for Roadmap

Based on combined research, the suggested phase structure follows strict dependency order: style infrastructure must precede all plots; data utilities must precede Fisher-based visualizations; existing plot improvements come before new modules that depend on them; the `corner` dependency is deferred until base infrastructure is stable.

### Phase 1: Test Infrastructure and Safety Net
**Rationale:** The PITFALLS research is unanimous — the project has no content-level tests for existing plot functions, making any refactoring hazardous. The safety net must exist before anything moves. This phase has zero visual risk and maximum leverage over all subsequent phases.
**Delivers:** Smoke tests for all 23 existing factory functions; rcParams snapshot regression test; `pytest-mpl` baselines for 5 critical figures; CI check banning `plt.show()` from production source.
**Addresses:** Pitfalls 1, 4, 9 (rcParams pollution, silent regression, rogue plt.show in `bayesian_inference_mwe.py`)
**Avoids:** Discovering broken thesis figures after the overhaul instead of before

### Phase 2: Style Infrastructure Upgrade
**Rationale:** Every downstream plot depends on consistent rcParams, figure sizes, and LaTeX handling. Infrastructure changes here are the highest-leverage, lowest-risk changes in the milestone. Must complete before any plot function changes.
**Delivers:** `_colors.py` centralized palette; upgraded `emri_thesis.mplstyle` (tick direction, color cycle, mathtext fonts); `apply_style(*, use_latex: bool = False)` with opt-in LaTeX toggle; thesis column width constants (`THESIS_SINGLE_COLUMN`, `THESIS_DOUBLE_COLUMN`) in `_helpers.py`; `_fig_from_ax` moved from `simulation_plots.py` to `_helpers.py`.
**Addresses:** Figure size inconsistency (HIGH severity), LaTeX rendering safety (Pitfalls 2, 10, 11, 12), cross-module coupling (Pitfall 13)
**Avoids:** rcParams pollution, constrained layout failures, DPI mismatches

### Phase 3: Data Layer and Fisher Visualizations
**Rationale:** `_data.py` is a shared dependency for three new modules (Fisher ellipses, corner plots, sky maps). Fisher ellipses are the most thesis-critical new plot type — standard in every EMRI PE paper (Barack & Cutler 2004, Babak et al. 2017). Building this before corner plots validates covariance reconstruction with simpler 2D ellipses first.
**Delivers:** `_data.py` with `reconstruct_covariance_matrix()` and `extract_submatrix()`; `fisher_plots.py` with 2D ellipses for key parameter pairs and correlation heatmap; `psd_plots.py` with characteristic strain h_c(f) and noise budget breakdown (S_OMS, S_TM, S_conf components).
**Uses:** scipy eigendecomposition (existing dep); existing CRB CSV covariance columns
**Implements:** `_data.py` and `fisher_plots.py` architecture components
**Avoids:** Pitfall 15 (signature inconsistency) by following the factory pattern from the start

### Phase 4: Enhanced Existing Plots
**Rationale:** Improvements to the 5 existing topic modules are low-risk additive changes producing immediate thesis-quality improvements without new dependencies. Each change is independently testable against the smoke tests from Phase 1. Modifying existing modules is higher risk than creating new ones; safer to do this after the data layer is proven in Phase 3.
**Delivers:** `bayesian_plots.py` — shaded 68%/95% credible intervals on H0 posterior; `evaluation_plots.py` — 2D sky localization delegating to `sky_plots.py` (replacing non-standard 3D scatter), improved violin colors; `simulation_plots.py` — confusion noise overlay parameter on PSD plot; `physical_relations_plots.py` — cosmology comparison overlay on d_L(z); consistent LaTeX axis labels across all modules.
**Addresses:** Missing credible intervals on H0 (HIGH severity), 3D plots non-standard (MEDIUM), missing confusion noise in PSD plot (MEDIUM), ad-hoc color usage (MEDIUM)

### Phase 5: New Plot Modules (Sky, Corner, Convergence)
**Rationale:** These depend on Phases 2-4 being complete. `sky_plots.py` and `corner_plots.py` depend on `_data.py` from Phase 3 and the style system from Phase 2. The `corner` dependency is deferred to this phase so the rest of the visualization stack works even if the new library has an installation issue.
**Delivers:** `sky_plots.py` Mollweide sky localization map; `corner_plots.py` thin wrapper with explicit thesis style kwargs; `convergence_plots.py` H0 posterior width vs N_events; `corner` added to `pyproject.toml`; mypy override for `corner.*` added to `pyproject.toml`.
**Uses:** corner >= 2.2.3 (new dependency); matplotlib built-in `projection='mollweide'`
**Addresses:** Pitfall 7 (corner style conflicts) via explicit `label_kwargs`/`title_kwargs` in wrapper function
**Avoids:** Pitfall 14 (mypy type stubs) by updating overrides immediately on dependency addition

### Phase 6: Campaign Dashboards and Batch Generation
**Rationale:** Multi-panel composite figures depend on all individual plot types from Phases 3-5 being finalized. Batch generation with production data stress-tests memory management and PDF output quality.
**Delivers:** `campaign_plots.py` multi-panel summaries compositing earlier factory functions; batch figure generation script for all thesis figures; PDF size audit with `rasterized=True` fixes for any single-figure PDF over 2 MB; full `__init__.py` re-export update.
**Addresses:** Pitfall 6 (memory leaks in batch generation), Pitfall 8 (PDF vector bloat)

### Phase Ordering Rationale

- **Phase 1 unconditionally first:** No content-level plot tests means any change could silently break thesis-critical figures; the safety net must precede all work.
- **Phase 2 before all plot work:** rcParams, figure sizes, and the color palette are global state shared by every factory function; downstream phases depend on this foundation.
- **Phase 3 before Phase 5:** `_data.py` covariance reconstruction is shared by Fisher ellipses, corner plots, and sky maps; validates the data layer with simpler 2D ellipses before the multi-panel corner plot.
- **Phase 4 after Phase 3:** Modifying existing modules is higher-risk than new modules; smoke tests from Phase 1 catch regressions, and the proven data layer from Phase 3 reduces uncertainty.
- **Phase 5 defers `corner` dependency:** If the library has any Python 3.13 compatibility issue at the exact installed version, it does not block the rest of the visualization stack.
- **Phase 6 last:** Composite figures require all component plots to be finalized; batch generation reveals memory and PDF size issues at scale that only manifest with production data.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (Fisher ellipses):** The ellipse geometry from covariance matrices has several conventions (1-sigma vs 2-sigma, chi-squared contours for 2 DOF). The exact `scipy.linalg.eigh` approach should reference arXiv:0906.4123 for the correct chi-squared threshold before coding.
- **Phase 5 (corner):** Confirm Python 3.13 wheel availability at the exact version selected; verify that `label_kwargs` and `title_kwargs` kwargs are sufficient to match thesis font sizes without triggering rcParams-mutating methods.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Standard pytest + pytest-mpl patterns; well-documented.
- **Phase 2:** matplotlib rcParams and mplstyle extension is well-documented; `apply_style()` extension is a straightforward additive change.
- **Phase 4:** Additive parameter additions to existing functions; well-understood risk profile.
- **Phase 6:** `plt.close(fig)` and `rasterized=True` are documented matplotlib patterns with no ambiguity.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official PyPI and docs verified; Python 3.13 compatibility confirmed for all recommended libraries; `corner` well-established in GW community with clear adoption in EMRI/LISA literature |
| Features | MEDIUM-HIGH | Based on survey of published EMRI/LISA papers (Barack & Cutler 2004, Babak et al. 2017, Gray et al. 2020, arXiv:2505.17814); community standards are clear for core figures; differentiator features are less strictly standardized |
| Architecture | HIGH | Based on direct codebase inspection of all 6 plotting modules, conftest.py, and existing factory functions; patterns are already established and consistent; new module boundaries are clear |
| Pitfalls | HIGH | Based on direct codebase inspection (confirmed rogue `plt.show()` in `bayesian_inference_mwe.py:177`, confirmed zero content-level plot tests, confirmed `_fig_from_ax` cross-module coupling); also cross-validated against matplotlib documentation |

**Overall confidence:** HIGH

### Gaps to Address

- **Full covariance matrix column names:** The research confirms that `delta_X_delta_Y` lower-triangle columns exist in the CRB CSV and can reconstruct a 14x14 matrix. The exact column naming convention for all 105 lower-triangle entries should be verified against an actual output CSV before coding `_data.py`. This is a one-minute check, not a research gap.
- **LaTeX availability on bwUniCluster:** The cluster compute nodes may or may not have a TeX distribution in the module system. The `use_latex=True` code path should include a preflight check using `matplotlib.checkdep_usetex(True)`. This is low-risk because `use_latex=False` is the default and CI/cluster never need `True`.
- **Production data scale:** All scalability estimates assume ~20-100 EMRI detections. If the Phase 12 production campaign produced significantly more, corner plot render times and PDF vector bloat thresholds need re-evaluation. The `rasterized=True` mitigation should be applied preemptively for all scatter plots as a defensive default.

## Sources

### Primary (HIGH confidence)
- [corner.readthedocs.io](https://corner.readthedocs.io/en/latest/) — v2.2.3 API, Python 3.13 compatibility, matplotlib integration
- [pypi.org/project/corner/](https://pypi.org/project/corner/) — version history, dependency requirements
- [matplotlib.org/stable/users/explain/text/usetex.html](https://matplotlib.org/stable/users/explain/text/usetex.html) — LaTeX rendering requirements and fallback behavior
- [arXiv:1803.01944 (Robson, Cornish & Liu 2019)](https://arxiv.org/abs/1803.01944) — LISA sensitivity curve convention (characteristic strain)
- [arXiv:1703.09722 (Babak et al. 2017)](https://arxiv.org/abs/1703.09722) — EMRI visualization standards (detection contours, sky maps, uncertainty distributions)
- [arXiv:0906.4123](https://arxiv.org/pdf/0906.4123) — Fisher matrix confidence ellipse geometry (chi-squared contours from covariance)
- [arXiv:2505.17814](https://arxiv.org/html/2505.17814) — Recent EMRI PE paper: Fisher ellipses, corner plots, sky maps as current community standard
- Direct codebase inspection: all 6 `plotting/` modules, `_style.py`, `_helpers.py`, `conftest.py`, `pyproject.toml`, `emri_thesis.mplstyle`, `bayesian_inference_mwe.py`

### Secondary (MEDIUM confidence)
- [arXiv:gr-qc/0310125 (Barack & Cutler 2004)](https://arxiv.org/abs/gr-qc/0310125) — Parameter uncertainty distribution conventions for EMRI PE
- [arXiv:1908.06050 (Gray et al. 2020)](https://arxiv.org/abs/1908.06050) — Dark siren H0 posterior visualization (credible intervals, spaghetti plots)
- [arXiv:2404.16092 (LIGO/Virgo/KAGRA O4a)](https://arxiv.org/abs/2404.16092) — Individual event posterior (spaghetti) plot conventions
- [jwalton.info/Embed-Publication-Matplotlib-Latex/](https://jwalton.info/Embed-Publication-Matplotlib-Latex/) — Thesis figure sizing in LaTeX (textwidth constants)
- [github.com/matplotlib/pytest-mpl](https://github.com/matplotlib/pytest-mpl) — Image comparison test infrastructure

### Tertiary (LOW confidence)
- [pypi.org/project/chainconsumer/](https://pypi.org/project/chainconsumer/) — Evaluated and rejected as apparently unmaintained; corner preferred
- [pypi.org/project/healpy/](https://pypi.org/project/healpy/) — Evaluated and rejected; built-in Mollweide projection sufficient for ~100 sky positions

---
*Research completed: 2026-04-01*
*Ready for roadmap: yes*
