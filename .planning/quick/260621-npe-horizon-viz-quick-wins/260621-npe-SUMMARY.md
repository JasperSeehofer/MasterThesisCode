---
phase: quick-260621-npe
plan: 01
subsystem: plotting
tags: [viz, colors, style, posteriors, hdi, theme]
requires: []
provides:
  - "HORIZON v2 semantic palette (navy/gold/vermillion + PLANCK/SH0ES band colors)"
  - "apply_style(theme='paper'|'talk'|'web') switch with byte-identical paper default"
  - "area-normalized PDF + nested 68/95% HDI + inline MAP in headline posteriors"
affects:
  - master_thesis_code/plotting/_colors.py
  - master_thesis_code/plotting/_style.py
  - master_thesis_code/plotting/bayesian_plots.py
  - master_thesis_code/plotting/paper_figures.py
tech-stack:
  added: []
  patterns:
    - "one base mplstyle + programmatic per-theme rcParams overrides (paper applies none)"
    - "nested HDI bands via shared compute_hdi_interval (one CI definition)"
    - "redundant encoding: color + linestyle + direct label for every comparison"
key-files:
  created: []
  modified:
    - master_thesis_code/plotting/_colors.py
    - master_thesis_code/plotting/_style.py
    - master_thesis_code/plotting/bayesian_plots.py
    - master_thesis_code/plotting/paper_figures.py
    - master_thesis_code_test/plotting/test_colors.py
    - master_thesis_code_test/plotting/test_style.py
    - master_thesis_code_test/plotting/test_bayesian_plots.py
    - master_thesis_code_test/plotting/test_paper_figures.py
decisions:
  - "apply_style: one base sheet + programmatic per-theme overrides, not 3 .mplstyle files"
  - "Task 4 (quadruplicate-posterior consolidation) DEFERRED to full viz-redesign milestone"
  - "CMAP stays viridis; cividis migration deferred (out of scope for quick slice)"
metrics:
  completed: 2026-06-21
  tasks-shipped: "3 of 4 + fig01 overlay polish (Task 4 deferred by design)"
  duration: "~1 session"
---

# Quick 260621-npe: HORIZON Viz Quick-Wins Slice Summary

HORIZON design-direction recolor + theme switch + headline-posterior treatment
in the plotting package only: killed the two-blues collision (navy vs gold),
added a paper/talk/web theme switch with a byte-identical paper default, and
adopted the field-standard area-normalized-PDF + nested-HDI + inline-MAP
convention for headline H0 posteriors. Pure software/design work — no physics
touched, no /physics-change triggered.

## Tasks Shipped

| Task | Subject | Commit |
|------|---------|--------|
| 1 | viz(colors): HORIZON v2 palette — kill two-blues, add band colors | `d12539a` |
| 2 | viz(style): add paper/talk/web theme switch to apply_style | `28a23e4` |
| 3 | viz(posteriors): area-normalized PDFs + nested HDI bands + inline MAP | `b95cd1f` |
| 4 | Consolidate quadruplicate H0-posterior paths | **DEFERRED** (see below) |
| follow-up | viz(fig01): clean two-variant overlay — single MAP/truth, no muddy bands | `1255107` |

## Orchestrator follow-up (commit `1255107`)

Rendering the figures on real (stale-archive) data surfaced a `fig01` defect:
the two-variant overlay double-drew — two "True h=0.73" legend entries and two
overlapping `MAP=` annotations (the secondary `With M_z` call did not suppress
them). Added a `show_truth` switch to `plot_combined_posterior` and wired
`fig01`'s secondary call (`main.py` `_gen_h0_posterior_combined`) to suppress
truth/MAP/references. Both variants now also skip HDI shading per §1.3 (never
shade a band under a many-variant peak-normalized overlay); the headline
area-norm + nested-HDI treatment remains on the single-posterior paper figures.
New test `test_plot_combined_posterior_show_truth_toggle`; full gate green
(mypy 119 files ✔, **pytest 589 passed**).

### Open design finding for the full milestone (needs the user + real data)
On the **stale v2.1 archive** (broad posterior, MAP≈0.86) the *paper* H0
comparison figures (`paper_h0_posterior`, `paper_h0_posterior_kde`) shade two
overlapping variant HDI bands that wash out the panel. On sharp production data
(seed400/seed500) the bands collapse to narrow slivers near 0.73 and read
cleanly — so this is largely a stale-data artifact, but the band-fill style for
**two overlapping variants** (outline-only? single-variant band? offset?) is a
genuine taste decision to revisit during the full milestone, ideally on the
fresh seed500 posterior once the cluster run lands.

## Final Palette Values (HORIZON v2, `_colors.py`)

| Name | Value | Role |
|------|-------|------|
| `VARIANT_NO_MASS` | `#1B2A4A` | observatory navy — Without M_z (headline) |
| `VARIANT_WITH_MASS` | `#E8A317` | signal gold — With M_z |
| `TRUTH` | `#C2451E` | warm vermillion — truth/injected rule ONLY |
| `REFERENCE` | `#4F4F4F` | scaffold gray — neutral secondary lines (was `#56B4E9`) |
| `PLANCK` | `#3E7CB1` | reserved band — Planck / early universe |
| `SH0ES` | `#9A6FB0` | reserved band — SH0ES / late universe |
| `MEAN` | `#D55E00` | unchanged |
| `EDGE` | `#1a1a1a` | unchanged |
| `ACCENT` | `#E69F00` | unchanged |
| `CYCLE` | Okabe-Ito 7 | unchanged |
| `CMAP` | `viridis` | unchanged (cividis migration deferred) |

Navy vs gold differ strongly in lightness (grayscale + deuteranopia safe).
`VARIANT_NO_MASS`, `VARIANT_WITH_MASS`, `REFERENCE` are pairwise distinct;
`PLANCK`/`SH0ES` are reserved band colors, never used for a data series.

## apply_style Design Decision

ONE base sheet (`emri_thesis.mplstyle`, unchanged) + programmatic per-theme
rcParams overrides, rather than three separate `.mplstyle` files.

- Rationale: themes are thin deltas (a font scale factor + two line weights);
  a small in-code dict is less duplication than three near-identical sheets,
  keeps one source of truth for the base, and avoids file-path plumbing —
  mirroring the existing `use_latex` in-code `rcParams.update` pattern.
- `theme="paper"` (default) applies NO override → byte-identical to today. This
  is the protected invariant: `test_apply_style_default_unchanged` and
  `test_rcparams_snapshot` pass UNCHANGED (verified, not weakened).
- `theme="talk"`/`"web"`: font sizes ×1.8, `lines.linewidth` 2.5, `axes.linewidth` 1.2.
- `web` affects matplotlib sizing only; CSS/Plotly export deferred to interactive milestone.
- `use_latex` layers last and intentionally wins on font sizes under any theme.

## Tests Updated for the New Normalization Convention

**`test_colors.py`** (Task 1): added `test_truth_is_horizon_vermillion`,
`test_variant_no_mass_is_horizon_navy`, `test_variant_with_mass_is_horizon_gold`,
`test_variant_and_reference_colors_are_pairwise_distinct`,
`test_planck_band_color_is_hex`, `test_sh0es_band_color_is_hex`,
`test_band_colors_distinct_from_data_series`. `test_cmap_is_viridis` and
`test_cycle_is_okabe_ito` kept passing unchanged.

**`test_style.py`** (Task 2): added `test_apply_style_paper_is_default_baseline`,
`test_apply_style_talk_scales_fonts` (with idempotent reset check),
`test_apply_style_accepts_web_theme`, `test_apply_style_talk_with_latex`.
`test_apply_style_default_unchanged` and `test_rcparams_snapshot` kept UNCHANGED.

**`test_bayesian_plots.py`** (Task 3, new normalization convention):
- `test_plot_combined_posterior_density_integrates_to_one` — area ≈ 1 under `normalize="density"`
- `test_plot_combined_posterior_hdi_bands_are_nested` — ≥2 nested PolyCollections
- `test_plot_combined_posterior_inline_map_annotation` — inline "MAP" text present
- `test_plot_combined_posterior_default_normalize_is_peak` — default peak (max ≈ 1), guards multi-variant overlays

**`test_paper_figures.py`** (Task 3, new `TestHeadlinePosteriorNormalization` class):
- `test_h0_comparison_is_area_normalized` — integrates ≈ 1
- `test_h0_comparison_ylabel_not_peak_normalized` — ylabel no longer "peak-normalized"
- `test_h0_comparison_has_hdi_bands` — ≥2 fill collections
- `test_h0_kde_is_area_normalized` — KDE curve integrates ≈ 1

No existing test pinned a single posterior peaking at 1.0 for the area-norm path,
and no test pinned old VARIANT/REFERENCE hex or the "peak-normalized" label, so no
existing assertions had to be loosened.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] matplotlib stubs reject bool ndarray in `fill_between(where=...)`**
- **Found during:** Task 3 (mypy gate)
- **Issue:** Passing a boolean `np.ndarray` mask to `fill_between(..., where=mask)`
  fails mypy (`expected Sequence[bool]`). The pre-existing committed code used the
  multi-line form and predated a typeshed/matplotlib-stub bump that now flags it.
- **Fix:** Cast each mask with `.tolist()` → `list[bool]` at both call sites in
  `bayesian_plots.py` and the `_shade_nested_hdi` helper in `paper_figures.py`.
  Also `np.asarray(ln.get_xdata())` in a test to satisfy `len()` typing.
- **Files modified:** `bayesian_plots.py`, `paper_figures.py`, `test_paper_figures.py`
- **Commit:** `b95cd1f`

**2. [Plan note] `plot_event_posteriors` left untouched (Task 3 Part B)**
- The many-variant spaghetti overlay already uses `CYCLE[0]`/`EDGE` (now the v2
  values) and is correctly peak-normalized with no HDI band — exactly the desired
  §1.3/§3.3 behavior. No edit was needed, as the plan anticipated.

## Task 4 Deferral — Four Duplicate H0-Posterior Code Paths

Task 4 (collapse the quadruplicate combined-H0-posterior code paths into one
canonical factory) is **DEFERRED to the full viz-redesign milestone**, per the
plan's explicit DEFERRED-OPTIONAL marking and the executor's judgment: it touches
4+ code paths across 3 files + `main.py` manifest wiring and carries the
golden-image regression risk the proposal calls out (§3.2 trade-offs). Tasks 1–3
deliver the user-visible quick wins atomically; Task 4 is the higher-effort
refactor and would balloon this slice.

The four (in practice five) duplicate paths to consolidate later:

1. **fig01 path** — `main.py:910` `_gen_h0_posterior_combined()` →
   `bayesian_plots.plot_combined_posterior` (overlays both variants on one ax;
   currently still peak-normalized at the call site — wire `normalize="density"`
   here during consolidation for the area-norm headline).
2. **`paper_figures.plot_h0_posterior_comparison`** (`paper_h0_posterior`) — now
   area-norm + nested-HDI + inline-MAP (Task 3); should delegate to the canonical factory.
3. **`paper_figures.plot_h0_posterior_kde`** (`paper_h0_posterior_kde`) — third copy,
   now consistent with the new convention (Task 3); should delegate.
4. **`convergence_analysis.plot_m_z_improvement_panels`** (`paper_m_z_improvement`)
   top posterior panel — `convergence_analysis.py:754-755` re-implements the
   peak-normalized two-variant posterior plot inline; fed by `compute_m_z_improvement_bank`.
   Route through the canonical factory.
5. **fig08-left convergence posterior panel** — verify whether
   `plot_posterior_convergence` (CI-width-vs-N, the figure's primary content)
   embeds a duplicate posterior panel; fold in only if one exists.

Consolidation design (for the milestone): extend `bayesian_plots.plot_combined_posterior`
to be the single canonical factory (it already has `normalize`, `show_credible`,
`show_references`, `annotate_map`, `color`, `ax` switches); have the paper_figures
wrappers and the main.py manifest delegate to it while preserving their
`(data_dir) -> (fig, ax)` signatures. The `test_canonical_map_consistency` suite is
the regression anchor. No data-plumbing change; canonical loaders untouched.

## Verification / Status

- **Check gate (run before every commit):** ruff check --fix ✔, ruff format ✔,
  mypy ✔ (no issues, 119 source files), pytest -m "not gpu and not slow" ✔
  (588 passed, 6 skipped, 15 deselected).
- **Figure regen** (`--generate_figures` against `simulations/_archive_v2_1_baseline`):
  21 generated / 2 skipped / 3 failed. The 3 failures
  (`fig09_detection_efficiency`, `paper_single_event`, `paper_convergence`) are
  PRE-EXISTING stale-data-availability errors ("n must be positive", "list index
  out of range", "no positive values") in the retired archive dir — NOT touched
  by this task and NOT code regressions. All figures this plan modified
  (`fig01_h0_posterior_combined`, `fig02_event_posteriors`, `paper_h0_posterior`,
  `paper_h0_posterior_kde`) regenerate cleanly with navy-vs-gold and the
  area-normalized + nested-HDI + inline-MAP headline treatment.
- **Invariant held:** `apply_style(theme="paper")` byte-identical to today;
  snapshot + default-unchanged tests pass unchanged.
- **No physics file touched.** Diff is confined to `_colors.py`, `_style.py`,
  `bayesian_plots.py`, `paper_figures.py` + their tests. /physics-change NOT triggered.

## Self-Check: PASSED
- `master_thesis_code/plotting/_colors.py` — FOUND
- `master_thesis_code/plotting/_style.py` — FOUND
- `master_thesis_code/plotting/bayesian_plots.py` — FOUND
- `master_thesis_code/plotting/paper_figures.py` — FOUND
- Commit `d12539a` — FOUND
- Commit `28a23e4` — FOUND
- Commit `b95cd1f` — FOUND
