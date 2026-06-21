---
phase: quick-260621-npe
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - master_thesis_code/plotting/_colors.py
  - master_thesis_code/plotting/_style.py
  - master_thesis_code/plotting/bayesian_plots.py
  - master_thesis_code/plotting/paper_figures.py
  - master_thesis_code/main.py
  - master_thesis_code_test/plotting/test_colors.py
  - master_thesis_code_test/plotting/test_style.py
  - master_thesis_code_test/plotting/test_bayesian_plots.py
  - master_thesis_code_test/plotting/test_paper_figures.py
autonomous: true
requirements: [VIZ-QW-1, VIZ-QW-2, VIZ-QW-3, VIZ-QW-4]
must_haves:
  truths:
    - "Without-M_z and With-M_z curves are visually distinct in color AND lightness (navy vs gold), not two blues"
    - "REFERENCE no longer collides with the With-M_z variant color"
    - "apply_style(theme='paper') with no other args produces output identical to today (all existing tests pass unchanged after the theme kwarg is added)"
    - "Headline single posteriors render as area-normalized PDFs with shaded nested 68/95% HDI and an inline MAP +/- 68% HDI annotation"
    - "Many-variant overlays keep peak-normalization and never shade an HDI band"
    - "All affected figures regenerate without error via --generate_figures against an existing data dir"
  artifacts:
    - path: "master_thesis_code/plotting/_colors.py"
      provides: "HORIZON v2 semantic palette (navy/gold/vermillion + PLANCK/SH0ES band colors)"
      contains: "VARIANT_NO_MASS"
    - path: "master_thesis_code/plotting/_style.py"
      provides: "apply_style(theme=...) switch with paper/talk/web layers"
      contains: "theme"
    - path: "master_thesis_code/plotting/bayesian_plots.py"
      provides: "area-normalized PDF + HDI bands + inline MAP in plot_combined_posterior"
      contains: "compute_hdi_interval"
  key_links:
    - from: "master_thesis_code/plotting/bayesian_plots.py"
      to: "master_thesis_code/plotting/_helpers.py"
      via: "import compute_hdi_interval"
      pattern: "compute_hdi_interval"
    - from: "master_thesis_code/plotting/paper_figures.py"
      to: "master_thesis_code/plotting/_colors.py"
      via: "VARIANT_NO_MASS / VARIANT_WITH_MASS imports"
      pattern: "VARIANT_NO_MASS"
---

<objective>
Implement the "quick-wins slice" of the visualization redesign: the HORIZON design
direction with Dark Siren Dispatch annotation discipline (user decision 2026-06-21,
docs/VIZ_REDESIGN_PROPOSAL.md §6). SOFTWARE/design work in the plotting package only.

This is a style + recolor + annotation pass on factory bodies and one signature
extension (apply_style theme switch). It is NOT a physics change (no formula,
constant, PSD coefficient, waveform parameter, or frequency limit is touched) and
therefore stays in GSD and must NOT trigger /physics-change.

Purpose: kill the two-blues collision (the central Without/With-M_z contrast of the
thesis currently relies on linestyle alone and collapses to indistinguishable grays
in print), adopt the field-standard area-normalized-PDF + shaded-HDI convention for
headline posteriors, add a paper/talk/web theme switch, and (optionally) consolidate
the quadruplicate combined-H0-posterior code paths.

Output: revised _colors.py (v2 palette), _style.py (theme switch), and headline
posterior factory bodies; updated tests; figures regenerated for visual confirmation.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@docs/VIZ_REDESIGN_PROPOSAL.md

# The design spec sections that govern this work:
#   §1 (9 headline changes), §3.2 (HORIZON direction + palette), §6 (implementation
#   sketch: items a/b/c/d). The chosen palette is HORIZON's: navy / gold / vermillion.

<constraints>
HARD constraints (from the proposal preamble — honor all):
  - REVTeX vector-PDF output preserved (pdf.fonttype 42, get_figure presets).
  - Colorblind- AND grayscale-safe encodings (navy #1B2A4A vs gold #E8A317 differ
    strongly in lightness; redundant color + linestyle + direct label).
  - Self-contained GH-Pages HTML (not touched here — interactive is out of scope).
  - NO data-plumbing rewrite: factory architecture (data in -> (fig, ax) out) and the
    canonical loaders (load_canonical_combined_posterior, compute_m_z_improvement_bank)
    are untouched. Only factory BODIES and apply_style change.

TYPING (CLAUDE.md): every public/private function gets complete annotations.
  Use list[...], X | None, npt.NDArray[np.float64]. NO `from __future__ import annotations`.
  Use Literal for the theme kwarg.

DO NOT touch: data pipeline, physics files, cluster scripts, interactive/Plotly code.
</constraints>

<interfaces>
<!-- Key contracts extracted from the codebase. Use these directly; no exploration needed. -->

Current _colors.py exported names (consumed by ~10 modules — MUST keep all of these
exported, do not rename or remove):
```python
CYCLE: list[str]          # Okabe-Ito 7-color cycle
TRUTH: str = "#009E73"
MEAN: str = "#D55E00"
EDGE: str = "#1a1a1a"
REFERENCE: str = "#56B4E9"   # <- currently COLLIDES with VARIANT_WITH_MASS
ACCENT: str = "#E69F00"
VARIANT_NO_MASS: str = "#0072B2"   # blue
VARIANT_WITH_MASS: str = "#56B4E9" # sky blue  <- the two-blues bug
SEQUENTIAL_BLUES: LinearSegmentedColormap
CMAP: str = "viridis"
```

Confirmed consumers of these names (grep results — verify nothing breaks):
  bayesian_plots.py, paper_figures.py, fisher_plots.py, sky_plots.py,
  simulation_plots.py, evaluation_plots.py, single_event_detail.py,
  catalog_plots.py, model_plots.py, physical_relations_plots.py,
  convergence_plots.py, convergence_analysis.py, main.py.

Current apply_style signature (_style.py):
```python
def apply_style(*, use_latex: bool = False) -> None: ...
```
It calls matplotlib.use("Agg"), loads emri_thesis.mplstyle, then (if use_latex)
overrides usetex/serif/font sizes via rcParams.update.

Existing HDI helper (USE THIS — already implemented, _helpers.py ~line 72):
```python
def compute_hdi_interval(
    h_values: npt.NDArray[np.float64],
    posterior: npt.NDArray[np.float64],
    level: float = 0.683,
) -> tuple[float, float]: ...
# Returns (lo, hi) highest-density interval; (nan, nan) if posterior integrates <= 0.
```

bayesian_plots._normalize_posterior already supports "peak" and "density" modes.

Headline posterior factories to touch:
  - bayesian_plots.plot_combined_posterior(...)  (the fig01 single-variant factory)
  - bayesian_plots.plot_event_posteriors(...)    (spaghetti + combined overlay)
  - paper_figures.plot_h0_posterior_comparison(data_dir)  (= "paper_h0_posterior";
        currently peak-norm, "o-"/"s--" marker-on-every-point, axvspan CI)
  - paper_figures.plot_h0_posterior_kde(data_dir)         (= "paper_h0_posterior_kde";
        THIRD copy, peak-norm + faded markers + KDE)
</interfaces>

<data_dir>
For figure-regeneration verification, a data dir with BOTH `posteriors/` and
`posteriors_with_bh_mass/` subdirs (each holding per-h `h_*.json` files plus a
`combined_posterior.json`) is required by --generate_figures.

CONFIRMED valid render target (data is stale/retired but the LAYOUT is correct, which
is all that matters for a render smoke test — we are not validating physics):
    simulations/_archive_v2_1_baseline/

It has: posteriors/ (38 per-h JSONs + combined_posterior.json),
posteriors_with_bh_mass/, and top-level combined_posterior.json /
combined_posterior_with_bh_mass.json (the legacy paths plot_h0_posterior_kde reads).

Regen command (run after each task on affected figures):
    uv run python -m master_thesis_code simulations/_archive_v2_1_baseline \
        --generate_figures simulations/_archive_v2_1_baseline

Then inspect the produced PDFs under simulations/_archive_v2_1_baseline/figures/
(fig01_h0_posterior_combined.pdf, fig02_event_posteriors.pdf, etc.). If a figure
returns None for missing data that is acceptable; a traceback is NOT.
NOTE: do not commit anything written under simulations/_archive_v2_1_baseline/.
</data_dir>

<check_gate>
The `check` quality gate (run before EACH commit per CLAUDE.md):
    uv run ruff check --fix master_thesis_code/ master_thesis_code_test/
    uv run ruff format master_thesis_code/ master_thesis_code_test/
    uv run mypy master_thesis_code/ master_thesis_code_test/
    uv run pytest -m "not gpu and not slow"
All four must pass. (.claude/skills/check is the same gate.)
</check_gate>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: COLOR SYSTEM v2 — HORIZON semantic palette (highest impact, lowest risk)</name>
  <files>master_thesis_code/plotting/_colors.py, master_thesis_code_test/plotting/test_colors.py</files>
  <behavior>
    Tests to update/add in test_colors.py FIRST (RED), then make them pass:
    - VARIANT_NO_MASS == "#1B2A4A" (navy) and is a valid 7-char hex.
    - VARIANT_WITH_MASS == "#E8A317" (gold) and is a valid 7-char hex.
    - VARIANT_NO_MASS != VARIANT_WITH_MASS AND neither equals REFERENCE
      (regression guard against the two-blues / reference collision).
    - TRUTH == "#C2451E" (vermillion truth rule).
    - New PLANCK == "#3E7CB1" and SH0ES == "#9A6FB0", both valid hex.
    - All previously-exported names still import: TRUTH, MEAN, EDGE, REFERENCE,
      ACCENT, CYCLE, CMAP, SEQUENTIAL_BLUES, VARIANT_NO_MASS, VARIANT_WITH_MASS.
    - Update the now-wrong test_truth_is... if it pins the old TRUTH value; update
      any assertion that pins REFERENCE == "#56B4E9".
    - test_cmap_is_viridis: KEEP CMAP == "viridis" unchanged (cividis is a
      separate, larger-scope item per §6a; do NOT change CMAP in this quick slice
      to avoid recoloring every heatmap — note this scoping in the docstring).
    - Keep test_cycle_is_okabe_ito passing (CYCLE is unchanged).
  </behavior>
  <action>
    Apply the HORIZON semantic palette to _colors.py (proposal §3.2 hex table, §6b):
    - VARIANT_NO_MASS = "#1B2A4A"  # HORIZON observatory navy — Without M_z (headline)
    - VARIANT_WITH_MASS = "#E8A317" # HORIZON signal gold — With M_z
    - TRUTH = "#C2451E"  # HORIZON warm vermillion — truth/injected rule ONLY
    - REFERENCE: reassign OFF "#56B4E9" so it no longer collides with the gold
      variant. Set REFERENCE = "#4F4F4F" (HORIZON scaffold gray — neutral secondary
      reference lines; this is the role REFERENCE plays in fisher_plots/catalog_plots/
      bayesian_plots SNR threshold). This keeps reference lines readable and never
      equal to a data-series color.
    - Add named cosmology-band reference colors:
        PLANCK = "#3E7CB1"  # mid cyan-blue — Planck / early-universe band
        SH0ES  = "#9A6FB0"  # muted purple — SH0ES / late-universe band
    - Leave CYCLE, MEAN, EDGE, ACCENT, SEQUENTIAL_BLUES, CMAP unchanged (CMAP stays
      "viridis" — cividis migration is out of scope for this quick slice).
    - Rewrite the module docstring: state it is the HORIZON v2 palette; that
      comparisons MUST be redundantly encoded (color + linestyle + direct label) for
      grayscale + deuteranopia safety; that navy vs gold are chosen for strong
      lightness separation; and note that CMAP/cividis migration is deferred to the
      full milestone. Add a one-line note that PLANCK/SH0ES are reserved band colors
      and must never be used for data series.
    All names keep complete type annotations (str). Do NOT add new imports beyond
    what exists. Per D (proposal §6b) this single edit propagates to every
    H0/per-event/convergence/single-event figure.
    Sanity-check that the two new band colors are distinct from VARIANT_* and TRUTH.
  </action>
  <verify>
    <automated>uv run pytest master_thesis_code_test/plotting/test_colors.py -x</automated>
    Also run the full check gate, then regenerate figures and confirm fig01/fig02 now
    show navy vs gold:
      uv run python -m master_thesis_code simulations/_archive_v2_1_baseline --generate_figures simulations/_archive_v2_1_baseline
  </verify>
  <done>
    test_colors.py passes; check gate green; VARIANT_NO_MASS/WITH_MASS/REFERENCE are
    pairwise distinct; PLANCK and SH0ES exported; fig01_h0_posterior_combined.pdf and
    fig02_event_posteriors.pdf regenerate without error showing navy (no-mass) vs gold
    (with-mass). Commit: "viz(colors): HORIZON v2 palette — kill two-blues, add band colors".
  </done>
</task>

<task type="auto">
  <name>Task 2: mplstyle v2 theme switch — apply_style(theme="paper"|"talk"|"web")</name>
  <files>master_thesis_code/plotting/_style.py, master_thesis_code_test/plotting/test_style.py</files>
  <action>
    Extend apply_style to add a theme switch (proposal §6c) WITHOUT changing the
    default output. New signature:
        def apply_style(
            *, theme: Literal["paper", "talk", "web"] = "paper", use_latex: bool = False
        ) -> None:
    Import Literal from typing.

    DESIGN DECISION (justify in the docstring): use ONE base mplstyle
    (emri_thesis.mplstyle, unchanged) + programmatic per-theme rcParams overrides in
    apply_style(), rather than 3 separate .mplstyle files. Rationale: the themes are
    thin deltas (font scale + line weight), keeping them as a small dict in code is
    less duplication than three near-identical sheets, keeps a single source of truth
    for the base, and avoids file-path plumbing — matching the existing use_latex
    pattern which already does in-code rcParams.update.

    Behavior:
    - Always: matplotlib.use("Agg"); load emri_thesis.mplstyle (as today).
    - theme="paper" (DEFAULT): apply NO extra base overrides — output must remain
      byte-for-byte identical to today's default. (use_latex still layers on top
      exactly as before.) This is the critical invariant: existing callers and
      test_apply_style_default_unchanged / test_rcparams_snapshot must pass UNCHANGED.
    - theme="talk": after loading the base, multiply font sizes by 1.8 and line
      weights heavier (lines.linewidth ~2.5, axes.linewidth ~1.2) for slides. Apply
      via rcParams.update reading the base values then scaling, e.g.
      font.size 8->~14.4, axes.titlesize 9->~16.2, axes.labelsize, xtick/ytick,
      legend.fontsize each x1.8.
    - theme="web": for matplotlib, apply the talk-like larger sizing (the proposal
      says web "matches interactive"); keep it simple — same scale-up as talk or a
      modest 1.5x. The CSS-custom-property export is interactive-layer work and is
      OUT OF SCOPE here; add a short docstring note that web theme currently only
      affects matplotlib sizing and the CSS/Plotly export is deferred to the
      interactive milestone.
    - use_latex: keep working under ALL themes — apply the existing usetex/serif
      override AFTER the theme override so usetex font sizes still win where set
      (preserve current behavior for paper; for talk/web the latex block currently
      hardcodes sizes — keep the existing latex sizes to avoid surprises, or scale
      them too; pick one and document it. Simplest: keep latex block as-is and note
      it overrides theme font scaling when both are set).

    Update the docstring fully (NumPy-style) documenting theme + use_latex and the
    one-base-plus-overrides decision. Complete type annotations; return type None.

    test_style.py updates:
    - Existing tests call apply_style() with no args — these MUST still pass because
      default theme="paper" == today. Verify (do NOT weaken them).
    - test_apply_style_latex_mode uses apply_style(use_latex=True) — still valid.
    - ADD: test_apply_style_paper_is_default_baseline — apply_style(theme="paper")
      gives font.size == 8.0 (same as default).
    - ADD: test_apply_style_talk_scales_fonts — apply_style(theme="talk") gives
      font.size > 8.0 (e.g. ~14.4) and lines.linewidth > 1.5; then apply_style()
      resets back to 8.0 (idempotent reset check).
    - ADD: test_apply_style_accepts_web_theme — apply_style(theme="web") runs and
      sets backend Agg.
    - Keep test_rcparams_snapshot pinned to the BASE sheet values (it calls
      apply_style() == paper default), unchanged.
  </action>
  <verify>
    <automated>uv run pytest master_thesis_code_test/plotting/test_style.py -x</automated>
    Run the full check gate. The default-unchanged + rcparams-snapshot tests passing
    is the proof the paper default is preserved.
  </verify>
  <done>
    apply_style(theme=...) exists with paper/talk/web; default (no args / theme="paper")
    output identical to today (snapshot + default-unchanged tests pass unchanged);
    talk theme scales fonts/lines; use_latex still works; new tests pass; check gate
    green. Commit: "viz(style): add paper/talk/web theme switch to apply_style".
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Area-normalized PDFs + 68/95% HDI bands + inline MAP in headline posteriors</name>
  <files>master_thesis_code/plotting/bayesian_plots.py, master_thesis_code/plotting/paper_figures.py, master_thesis_code_test/plotting/test_bayesian_plots.py, master_thesis_code_test/plotting/test_paper_figures.py</files>
  <behavior>
    Tests FIRST (RED), then implement:
    - bayesian_plots.plot_combined_posterior with normalize="density" produces a
      curve whose area integrates to ~1 (np.trapezoid(y, x) ≈ 1.0). Read the curve
      back from ax.get_lines()[0].get_ydata()/get_xdata().
    - With show_credible=True it shades TWO nested HDI regions (>= 2 PolyCollections
      in ax.collections) computed via compute_hdi_interval (68% darker, 95% lighter).
      The existing test_plot_combined_posterior_credible_intervals already asserts
      len(ax.collections) >= 2 — keep it passing under the new HDI implementation.
    - An inline MAP annotation text is present (ax.texts non-empty) reporting MAP and
      68% HDI when an annotation flag is on.
    - plot_combined_posterior default normalize MUST stay "peak" so existing
      single-variant overlay callers (main.py fig01 overlays two variants on one ax)
      are not silently changed; the HEADLINE area-norm path is selected explicitly.
      (i.e. do not flip the default; main.py will opt in — see below.)
    - paper_figures.plot_h0_posterior_comparison: assert the y-axis label no longer
      says "peak-normalized" and the curve is area-normalized (integrate ≈ 1). Update
      any existing test that pins peak-norm / the "Posterior (peak-normalized)" label.
    - Confirm NO test asserts a single posterior peaks at exactly 1.0 after this
      change; if one exists, update it to the area-norm convention.
  </behavior>
  <action>
    Apply the proposal's headline-posterior treatment (§1.3, §3.2, §6d) using the
    EXISTING compute_hdi_interval from _helpers.

    A. bayesian_plots.plot_combined_posterior:
       - Import compute_hdi_interval from _helpers.
       - Replace the cumsum-index CI machinery (the integer searchsorted block) with
         compute_hdi_interval at level=0.683 and level=0.954. Shade the 95% HDI
         region (alpha 0.15) and the nested 68% HDI region (alpha 0.30) under the
         curve via fill_between with a where-mask on (h_values >= lo) & (<= hi). Use
         the curve's `color` for both fills (single hue, graded alpha) — one CI
         definition, replacing boundary axvlines + the old cumsum CI.
       - Add an inline MAP annotation: compute MAP = h_values[argmax(normalized)] and
         the 68% HDI (lo68, hi68); annotate near the peak with
         "MAP = {map:.3f} +{hi68-map:.3f}/-{map-lo68:.3f}" (Dispatch number-on-the-
         curve discipline). Gate behind a new keyword `annotate_map: bool = True`
         (so multi-variant overlays can suppress it). Skip annotation if HDI is nan.
       - Drop the per-point marker style — keep a plain line (it already uses ax.plot
         without markers; ensure no "o-"/markers are introduced).
       - Update the reference-band block to use the new PLANCK and SH0ES color
         constants instead of CYCLE[6]/CYCLE[0] (import PLANCK, SH0ES). Keep the
         Planck/SH0ES band+label behavior; bands are full-height context, labeled at
         top — never reuse a data color.
       - KEEP normalize default = "peak" for backward compat; the area-norm headline
         is selected by passing normalize="density".
       - Complete type annotations on the new kwarg.

    B. bayesian_plots.plot_event_posteriors:
       - This is the many-variant spaghetti overlay: KEEP peak-normalization and do
         NOT shade any HDI band (per §1.3 / §3.3: never shade a band under a
         peak-normalized many-variant overlay). The only change here: ensure the
         de-emphasized individual curves and the combined hero line use the v2 colors
         from Task 1 (they already pull CYCLE[0]/EDGE — leave as-is unless a color is
         hardcoded). No HDI, no area-norm. (If nothing needs changing, leave the
         function untouched and note so in the summary.)

    C. paper_figures.plot_h0_posterior_comparison (the "paper_h0_posterior" copy):
       - Switch from peak-norm to area-normalized PDFs (divide each posterior by
         np.trapezoid(post, h)).
       - Remove the marker-on-every-point style ("o-"/"s--") -> plain solid line for
         Without M_z and dashed line for With M_z (redundant linestyle channel;
         navy solid vs gold dashed).
       - Replace the axvspan 68% CI with shaded nested 68/95% HDI under EACH variant
         curve via compute_hdi_interval (graded alpha 0.30/0.15, variant hue).
       - Add inline MAP +/- 68% HDI annotation for the headline (no-mass) variant.
       - Update y-axis label from "Posterior (peak-normalized)" to a PDF label, e.g.
         r"$p(h \mid \mathrm{data})$".
       - Drop the fig.tight_layout() call (constrained_layout is on) to avoid the
         documented double-layout conflict (§1.8) — only for THIS factory.

    D. paper_figures.plot_h0_posterior_kde (the "paper_h0_posterior_kde" copy):
       - Apply the same area-norm + nested-HDI + inline-MAP + drop-markers treatment
         so the THIRD copy matches. (Full consolidation into one factory is Task 4 /
         deferred; here just make this copy consistent with the new convention so the
         three copies don't visibly diverge.) Keep the KDE smoothing; area-normalize
         the KDE curve; shade HDI from the KDE curve via compute_hdi_interval. Drop
         the fig.tight_layout() call.

    All edits are factory BODIES only — signatures of paper_figures factories stay
    (data_dir) -> (fig, ax). compute_hdi_interval and the canonical loaders are
    untouched (no data-plumbing change).
  </action>
  <verify>
    <automated>uv run pytest master_thesis_code_test/plotting/test_bayesian_plots.py master_thesis_code_test/plotting/test_paper_figures.py -x</automated>
    Run the full check gate. Then regenerate and visually confirm headline posteriors
    are smooth area-normalized PDFs with two nested shaded bands + an inline MAP label:
      uv run python -m master_thesis_code simulations/_archive_v2_1_baseline --generate_figures simulations/_archive_v2_1_baseline
    Inspect fig01_h0_posterior_combined.pdf (note: main.py passes normalize default;
    if you want the headline area-norm in fig01 itself, pass normalize="density" at
    the call site in main.py — see Task 4 note; otherwise fig01 stays peak for now and
    paper_h0 / paper_h0_kde carry the area-norm headline).
  </verify>
  <done>
    Headline single posteriors render area-normalized with nested 68/95% HDI (via
    compute_hdi_interval) and an inline MAP +/- 68% HDI annotation; many-variant
    overlays remain peak-normalized with no shaded band; markers dropped on smooth
    curves; tight_layout removed from the two touched paper factories; tests updated
    to the new normalization convention and passing; check gate green; figures
    regenerate without error. Commit:
    "viz(posteriors): area-normalized PDFs + nested HDI bands + inline MAP".
  </done>
</task>

<task type="auto" gate="optional">
  <name>Task 4 (DEFERRED-OPTIONAL): Consolidate the quadruplicate combined-H0-posterior code paths</name>
  <files>master_thesis_code/plotting/paper_figures.py, master_thesis_code/plotting/bayesian_plots.py, master_thesis_code/main.py, master_thesis_code_test/plotting/test_paper_figures.py</files>
  <action>
    SCOPE DECISION: This is the higher-effort refactor and is EXPLICITLY MARKED
    DEFERRED-TO-FULL-MILESTONE. Tasks 1-3 land first as atomic commits and deliver the
    user-visible quick wins. Do Task 4 ONLY if Tasks 1-3 completed well within budget
    and the executor judges it will not balloon (it touches 4+ code paths across 3
    files + main.py manifest wiring + golden-image regression risk the proposal calls
    out in §3.2 trade-offs). If in doubt, STOP after Task 3 and record Task 4 as a
    planned-but-deferred item in the SUMMARY for the full viz-redesign milestone.

    If undertaken: collapse the four combined-H0-posterior code paths into ONE
    canonical factory:
      - fig01 path: main.py _gen_h0_posterior_combined -> plot_combined_posterior
      - paper_figures.plot_h0_posterior_comparison
      - paper_figures.plot_h0_posterior_kde
      - the paper_m_z_improvement top-panel posterior copy (top panel lives in the
        convergence/m_z dashboard; locate it — compute_m_z_improvement_bank feeds it —
        and route it through the canonical factory)
      - fig08-left convergence posterior panel (paper_figures.plot_posterior_convergence
        left panel, if it duplicates the posterior — verify; the convergence figure's
        primary content is CI-width-vs-N, so only fold in the posterior panel if one
        exists)
    Design: extend bayesian_plots.plot_combined_posterior to BE the canonical factory
    with the options it already needs plus theme/option switches
    (normalize="density"|"peak", show_hdi, show_references, annotate_map, theme passed
    through to styling). Have the paper_figures wrappers and main.py manifest delegate
    to it instead of re-implementing. Keep all factory signatures stable where they
    are public; where a paper wrapper becomes a thin delegate, preserve its
    (data_dir) -> (fig, ax) signature so tests and main.py keep working.
    Add/keep tests asserting all paths produce a consistent MAP (the existing
    test_canonical_map_consistency is the anchor — extend it if helpful).
    NO data-plumbing change; canonical loaders untouched.
    Wire any call-site normalization (e.g. main.py fig01 passing normalize="density"
    for the area-norm headline) here as part of consolidation.
  </action>
  <verify>
    <automated>uv run pytest master_thesis_code_test/plotting/ -x</automated>
    Full check gate + regenerate all H0-posterior figures and confirm they agree on
    the MAP and render without error:
      uv run python -m master_thesis_code simulations/_archive_v2_1_baseline --generate_figures simulations/_archive_v2_1_baseline
  </verify>
  <done>
    EITHER: one canonical posterior factory exists, all former duplicate paths delegate
    to it, main.py manifest wired, all paths agree on MAP, tests + check gate green,
    figures regenerate — committed as
    "viz(posteriors): consolidate quadruplicate H0-posterior paths into one factory";
    OR: deferred — SUMMARY records Task 4 as a planned item for the full viz-redesign
    milestone with the four code paths enumerated, and the plan is considered complete
    with Tasks 1-3 shipped.
  </done>
</task>

</tasks>

<verification>
- After Tasks 1-3 (and optionally 4), the full check gate passes:
  ruff check --fix + ruff format + mypy + pytest -m "not gpu and not slow", all green.
- `uv run python -m master_thesis_code simulations/_archive_v2_1_baseline
  --generate_figures simulations/_archive_v2_1_baseline` completes with no traceback;
  fig01/fig02 + paper H0 posteriors render with navy-vs-gold and (for headline single
  posteriors) area-normalized PDFs with nested HDI shading and inline MAP.
- No physics file touched (grep the diff: _colors.py, _style.py, bayesian_plots.py,
  paper_figures.py, main.py manifest + tests only). /physics-change NOT triggered.
- All previously-exported _colors names still import across the ~13 consumer modules
  (mypy + import smoke confirms).
</verification>

<success_criteria>
- VARIANT_NO_MASS (navy) and VARIANT_WITH_MASS (gold) are pairwise distinct and
  distinct from REFERENCE; PLANCK/SH0ES band colors exported; grayscale/CB-safe.
- apply_style(theme="paper") (default) is byte-identical to today (snapshot test
  unchanged); talk/web themes scale fonts/lines; use_latex still works.
- Headline single posteriors: area-normalized PDF + nested 68/95% HDI (via
  compute_hdi_interval) + inline MAP +/- 68% HDI; many-variant overlays stay
  peak-normalized with no band; markers dropped on smooth curves.
- Each of Tasks 1, 2, 3 is its own atomic commit; Task 4 either committed atomically
  or explicitly deferred in the SUMMARY.
- check gate green before every commit; affected figures regenerate cleanly.
</success_criteria>

<output>
After completion, create
`.planning/quick/260621-npe-horizon-viz-quick-wins/260621-npe-SUMMARY.md`
recording: which tasks shipped (1-3 required, 4 optional), the final palette values,
the apply_style decision (one-base + programmatic overrides) and rationale, the list
of tests updated for the new normalization convention, and — if Task 4 was deferred —
the four duplicate H0-posterior code paths enumerated for the full viz-redesign
milestone.
</output>
