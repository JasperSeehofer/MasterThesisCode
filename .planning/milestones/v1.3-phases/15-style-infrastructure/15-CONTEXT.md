# Phase 15: Style Infrastructure - Context

**Gathered:** 2026-04-01
**Status:** Ready for planning

<domain>
## Phase Boundary

A centralized style system that all downstream plot work (Phases 16-19) builds on: standardized figure sizing, LaTeX rendering toggle, shared color palette, and consolidated helpers. This phase creates infrastructure and conventions — bulk label migration happens in Phase 17.

</domain>

<decisions>
## Implementation Decisions

### Figure Sizing (STYLE-01)
- **D-01:** `get_figure()` in `_helpers.py` gains a `preset` parameter: `get_figure(preset="single")` / `get_figure(preset="double")`. Presets map to standard REVTeX two-column widths (~3.375in single, ~7.0in double).
- **D-02:** Default behavior (no preset, no figsize) keeps using whatever the mplstyle says. Existing code and tests remain untouched until plot modules are upgraded in Phase 17.
- **D-03:** Raw `figsize` parameter still accepted for one-off overrides.

### Color Palette (STYLE-04)
- **D-04:** `_colors.py` provides an ordered color cycle (for multi-line plots like individual posteriors) plus semantic names for special roles: truth/reference lines, mean/summary lines, histogram edges.
- **D-05:** Start with the ~4 distinct roles currently in the codebase (`"green"` truth, `"red"` mean, `"black"` hist edge, `"viridis"` cmap). Grow the palette in later phases as new plot types appear.

### LaTeX Label Convention (STYLE-02)
- **D-06:** Phase 15 establishes the convention and may provide label constants or a reference mapping. The actual bulk migration of `set_xlabel`/`set_ylabel` across all 6 plot modules happens in Phase 17 (Enhanced Existing Plots).
- **D-07:** Labels will be fully typeset — physics symbols AND units in mathtext: e.g., `$M_\bullet \, [M_\odot]$`, `$d_L \, [\mathrm{Mpc}]$`.

### LaTeX Toggle (STYLE-03)
- **D-08:** `apply_style(use_latex=True)` does three things: (1) sets `text.usetex: True`, (2) switches font family to serif/Computer Modern, (3) adjusts font sizes to match paper body text.
- **D-09:** Default `apply_style()` (no argument) keeps current behavior — mathtext rendering, no TeX dependency, works on headless CI.
- **D-10:** Target output is arXiv paper figures (not thesis). Figure size presets use REVTeX two-column defaults (~3.375in single column, ~7.0in double column).

### Helper Consolidation (STYLE-05)
- **D-11:** `_fig_from_ax` moves from `simulation_plots.py` to `_helpers.py`. Old location re-exports or is removed; all 6 plot modules update their imports.

### Claude's Discretion
- Exact height calculations for figure presets (aspect ratios)
- Internal structure of `_colors.py` (module-level constants, enum, or dict)
- Whether to provide a label constants module or just a docstring convention for Phase 17
- Font size values for the `use_latex=True` mode (match typical 10pt paper body)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Style infrastructure (source files to modify)
- `master_thesis_code/plotting/_style.py` — `apply_style()` implementation, must add `use_latex` parameter
- `master_thesis_code/plotting/_helpers.py` — `get_figure()`, `save_figure()`, `make_colorbar()`; add `preset` param and receive `_fig_from_ax`
- `master_thesis_code/plotting/emri_thesis.mplstyle` — 18 rcParams pinned by regression test; changes here must update the test

### Source of `_fig_from_ax` (to be moved)
- `master_thesis_code/plotting/simulation_plots.py:20` — current definition
- All importers: `bayesian_plots.py`, `catalog_plots.py`, `evaluation_plots.py`, `model_plots.py`, `physical_relations_plots.py`

### New file to create
- `master_thesis_code/plotting/_colors.py` — centralized color palette (does not exist yet)

### Tests (must stay green)
- `master_thesis_code_test/plotting/test_style.py` — 11 existing tests + rcParams regression test
- `master_thesis_code_test/plotting/conftest.py` — shared fixtures, session-scoped `apply_style()`

### Requirements
- `.planning/REQUIREMENTS.md` §Style Infrastructure — STYLE-01 through STYLE-05

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `get_figure()` in `_helpers.py` — already handles `nrows`, `ncols`, `figsize`, `**kwargs`; preset parameter is a natural extension
- `emri_thesis.mplstyle` — single source of truth for rcParams; LaTeX toggle overrides these at runtime
- rcParams regression test (Phase 14) pins 18 values — acts as safety net for any mplstyle changes

### Established Patterns
- All plot factories follow `data in, (fig, ax) out` pattern
- `_fig_from_ax` is used in every plot module (22 call sites) — moving it is high-impact but mechanical
- Ad-hoc colors: `"green"` (truth lines, 3 files), `"red"` (mean lines, 1 file), `"black"` (hist edges, 4 files), `"viridis"` (colormaps, 4 files)

### Integration Points
- `apply_style()` is called once at program entry and once via session fixture in tests
- `_colors.py` will be imported by all plot modules (replacing inline color strings in Phase 17)
- `_helpers.py` is already the shared utility module for plotting

</code_context>

<specifics>
## Specific Ideas

- Figures are for an arXiv paper, not a thesis document — sizing targets REVTeX two-column layout
- Font sizes in `use_latex=True` mode should match typical 10pt paper body text
- Color cycle should support overlaying multiple individual posteriors on one plot

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 15-style-infrastructure*
*Context gathered: 2026-04-01*
