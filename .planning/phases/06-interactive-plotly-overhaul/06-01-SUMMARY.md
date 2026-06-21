---
phase: 06-interactive-plotly-overhaul
plan: 01
subsystem: plotting (interactive web layer)
tags: [plotly, horizon, web, gh-pages, theming, offline]
requires:
  - master_thesis_code/plotting/_colors.py (HORIZON hex tokens, single source of truth)
  - master_thesis_code/plotting/_style.py (apply_style web theme intent)
provides:
  - master_thesis_code/plotting/_plotly_theme.py (HORIZON go.layout.Template + cividis colorscale)
  - self-contained offline interactive HTML (no CDN)
  - per-group trace toggling for dropdown figures
  - _STATIC_TWINS interactive->static-PDF map
affects:
  - GitHub Pages interactive/ deploy (now archival-robust, offline)
tech-stack:
  added: []           # no new dependency (plotly 6.6 already present)
  patterns:
    - "ONE shared go.layout.Template from _colors tokens (no hardcoded hex)"
    - "include_plotlyjs='directory' single-helper export"
    - "visibility vectors COMPUTED from trace group membership (legendgroup/meta)"
key-files:
  created:
    - master_thesis_code/plotting/_plotly_theme.py
  modified:
    - master_thesis_code/plotting/interactive.py
    - master_thesis_code/main.py
    - master_thesis_code_test/test_interactive.py
decisions:
  - "include_plotlyjs='directory' (one shared plotly.min.js) over 'cdn' (online-only) and True (per-file inline bloat)"
  - "Plotly web typography mirrors apply_style('web') 1.8x scale in the template (apply_style is matplotlib-only)"
  - "theme default 'web' for generate_all_interactive"
metrics:
  duration: ~1 session
  completed: 2026-06-21
  tasks: 4
  files: 4
  tests_added: 35   # 5 template + 5 offline + 9 per-group + 16 theme/twin (incl. parametrized)
---

# Phase 6 Plan 01: Interactive Plotly Overhaul Summary

Ported the settled static HORIZON design (Phases 1-5) to the Plotly interactive web
layer: all 8 factories now share ONE `go.layout.Template` built from `_colors` tokens
(navy/gold colorway + cividis colorscale + web-scaled typography), the GH-Pages
`interactive/` output is self-contained/offline (single shared `plotly.min.js`, no CDN),
the two dropdown figures toggle per-group trace sets instead of hand-maintained boolean
vectors, and the web theme + a machine-checkable `_STATIC_TWINS` map are wired into
generation.

## Tasks shipped

| # | Task | Commit |
|---|------|--------|
| 1 | Shared HORIZON `go.layout.Template` applied to all 8 factories | `837bfe6` |
| 2 | Self-contained offline HTML via `include_plotlyjs='directory'` | `7e6c5bb` |
| 3 | Per-group trace toggling for dropdown figures | `a104af0` |
| 4 | Web theme wiring + `_STATIC_TWINS` mapping for all 8 | `0a4153d` |

## VR-INT-01: shared HORIZON template

`master_thesis_code/plotting/_plotly_theme.py` exports `horizon_plotly_template()` +
the prebuilt `HORIZON_TEMPLATE` singleton + `CIVIDIS_COLORSCALE`. All hex tokens are
imported from `_colors` (`VARIANT_NO_MASS`, `VARIANT_WITH_MASS`, `REFERENCE`, `CYCLE`,
`CMAP`) — NO hex literal is defined in the theme module. The `colorway` leads with
navy `#1B2A4A` + gold `#E8A317`, then the Okabe-Ito cycle. The sequential colorscale is
cividis sampled from `matplotlib.colormaps[CMAP]` to 10 stops (`rgb(0,34,78)` →
`rgb(254,232,56)`), replacing the old `"Viridis"` literal. `template=HORIZON_TEMPLATE` is
applied to all 8 `interactive_*` factories; `interactive_sky_map`'s Scattergeo marker uses
`CIVIDIS_COLORSCALE` (markers don't inherit the template's sequential ramp); the combined
posterior series → navy, Planck band → `PLANCK`, SH0ES band → `SH0ES`, truth line → `TRUTH`.

## VR-INT-02: include_plotlyjs strategy

CHOSEN `'directory'` — one `plotly.min.js` is written into the output dir on the first
write and every HTML references it relatively, so the whole folder is offline /
archival-robust. Routed through ONE helper `_write_html_self_contained(fig, path)`; all 8
`write_html(..., include_plotlyjs='cdn')` call sites replaced. Rejected `'cdn'`
(online-only/fragile) and `True` (per-file ~3 MB inline bloat across 8 HTML).

Offline verify (real `simulations/` run): `simulations/interactive/plotly.min.js` (4.8 MB)
written once; the 3 freshly-generated HTML (sky_map, fisher_ellipses, closure_test)
reference `src="plotly.min.js"` locally and contain NO `cdn.plot.ly` / remote plotly URL.
(Three stale April HTML in that dir still contain CDN refs — they were NOT regenerated
because their source data is absent; the new pipeline never emits CDN. `interactive/` is
git-ignored so nothing was committed.)

## VR-INT-03: per-group trace toggling

Added `_visibility_for_group(fig, group, *, key='legendgroup')` + `_trace_group` helpers
that COMPUTE a full-length visibility vector from group membership (length always equals
`len(fig.data)` — self-correcting when a trace is added).
- `interactive_single_event_detail`: every trace tagged `legendgroup=f"event_{eid}"` +
  `meta`; dropdown buttons built via `_visibility_for_group`. Removed the
  `i*traces_per_event + k` arithmetic and the `traces_per_event=6` assumption (events with
  missing curves now add fewer traces without misaligning).
- `interactive_m_z_improvement`: metric blocks tagged `meta={"group": f"metric_{key}"}`,
  ref `'ref'`, panel-B `'panel_b'`, panel-C `'panel_c'`; panel-B frame indices derived from
  membership; dropdown visibility computed from groups (selected metric + always-on
  panel_b/panel_c + ref only for `hdi68_width`). Slider/frames + per-metric yaxis switch
  preserved.

## VR-INT-04: theme wiring + static twins

`generate_all_interactive(..., theme: Literal['talk','web']='web')` calls
`apply_style(theme)` once (matplotlib static twins) while the Plotly figures carry the
web-scaled template; `main.generate_interactive_figures(data_dir, *, theme='web')` threads
it through. `_STATIC_TWINS` (module-level `dict[str, str]`, `module:function` form) makes
the twin mapping machine-checkable; a parametrized test imports each twin and asserts it
callable.

The 8 interactive → static-PDF twin pairs (all present, none missing):

| # | Interactive factory | Static twin |
|---|---------------------|-------------|
| 1 | `interactive_combined_posterior` | `bayesian_plots:plot_combined_posterior` |
| 2 | `interactive_sky_map` | `sky_plots:plot_sky_localization_mollweide` |
| 3 | `interactive_fisher_ellipses` | `fisher_plots:plot_fisher_ellipses` |
| 4 | `interactive_h0_convergence` | `convergence_plots:plot_h0_convergence` |
| 5 | `interactive_m_z_improvement` | `convergence_analysis:plot_m_z_improvement_panels` |
| 6 | `interactive_single_event_detail` | `single_event_detail:plot_single_event_detail` |
| 7 | `interactive_closure_test_overlay` | `paper_figures:plot_closure_test_overlay` |
| 8 | `interactive_catalog_completeness` | `catalog_plots:plot_event_catalog_coverage` |

## Tests added

`master_thesis_code_test/test_interactive.py` — 53 tests pass total. New:
- `TestHorizonPlotlyTemplate` (5): colorway navy/gold, cividis endpoints (not Viridis),
  two-factory template application, singleton match.
- `TestGenerateAllInteractive` (+2): local `plotly*.js` lands in output_dir; each HTML
  references local plotly with no `cdn.plot.ly`/remote plotly URL; empty-data still `[]`.
- `TestPerGroupTraces` (9): synthetic single-event JSON fixture + synthetic
  `ImprovementBank`; legendgroup tagging, dropdown toggles exactly one group, initial
  one-group-visible, meta-group tagging, computed-membership vectors.
- `TestThemeAndStaticTwins` (16, incl. parametrized): web-theme font in force,
  `_STATIC_TWINS` covers exactly 8, each twin importable+callable, each `interactive_*`
  exists, smoke-generate raises nothing.

## Deviations from Plan

None — plan executed exactly as written. One presentation-only adjustment within scope:
`interactive_sky_map` uses an explicit `CIVIDIS_COLORSCALE` (exported from `_plotly_theme`)
rather than relying on template inheritance, because Scattergeo marker `colorscale` does
not inherit `template.layout.colorscale.sequential` automatically — this keeps the cividis
ramp single-sourced from `_colors.CMAP` (no hardcoded hex).

## Check gate

- `ruff check master_thesis_code/` — all checks passed
- `ruff format --check master_thesis_code/` — 62 files already formatted
- `mypy master_thesis_code/` — Success: no issues found in 62 source files
- `pytest -m "not gpu and not slow"` — 687 passed, 6 skipped, 15 deselected; coverage 67%

## --generate_interactive smoke

`uv run python -m master_thesis_code simulations --generate_interactive simulations` —
no traceback; 3 HTML written (sky_map, fisher_ellipses, closure_test); the other 5
self-skip when their data is absent (conditional-skip behavior preserved);
`plotly.min.js` written once; new HTML offline (no CDN).

## Taste flags for user review

- Template font family `"Helvetica Neue, Arial, sans-serif"` at `WEB_FONT_SIZE=15`
  (paper ~8pt × 1.8 web scale, rounded). Adjust in `_plotly_theme.py` if a different
  on-screen size/family is preferred.
- Background `#ffffff` paper+plot, gridlines `#e6e6e6`, axis/font color `#1a1a1a`.
- Colorway ordering: navy, gold, then Okabe-Ito CYCLE. Confirm navy/gold-first is the
  desired headline emphasis for incidental traces.
- cividis colorscale sampled to 10 stops — increase if banding is visible on dense
  heatmaps.

## Known Stubs

None — no hardcoded empty values, placeholders, or unwired data sources introduced.

## Self-Check: PASSED

- `master_thesis_code/plotting/_plotly_theme.py` — FOUND
- commits `837bfe6`, `7e6c5bb`, `a104af0`, `0a4153d` — FOUND on `viz/horizon-quick-wins`
- zero `include_plotlyjs='cdn'` / `Viridis` / hardcoded HORIZON hex in interactive.py — verified
- template colorway[:2] == (`#1B2A4A`, `#E8A317`) — verified
