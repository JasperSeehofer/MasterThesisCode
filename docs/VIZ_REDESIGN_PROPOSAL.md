# Visualization Redesign Proposal — EMRI Dark-Siren H₀ Inference

> Status: proposal / design document. No code changes are made by this file.
> Scope: software + design only. All proposals honor the existing
> factory-function + `apply_style()` architecture, REVTeX vector-PDF paper
> figures, colorblind safety, self-contained Plotly HTML interactives, and the
> single `--generate_figures` manifest path in `main.py`.

---

## 1. Context

### What the suite is today

The project ships **two figure families** driven from one entry point
(`main.generate_figures` in `darksiren_emri/main.py`, a list-of-tuples
manifest of `(name, generator)` callables):

- **Static figures** — 15 rendered today (`fig01`–`fig15`), 20 defined
  (`fig16`–`fig20` are data-gated), plus a `paper_*` set. Each is produced by a
  factory `plot_*(data) -> (fig, ax)` in a topic module
  (`bayesian_plots.py`, `fisher_plots.py`, `convergence_plots.py`,
  `sky_plots.py`, `simulation_plots.py`, `dashboard_plots.py`,
  `physical_relations_plots.py`). Callers (the manifest) own saving via
  `_helpers.save_figure` (PDF, dpi 300, fonttype 42).
- **Interactives** — 8 self-contained Plotly HTML pages from
  `plotting/interactive.py::generate_all_interactive`, each written with
  `fig.write_html(..., include_plotlyjs="cdn")` for static GitHub Pages hosting.

Style is centralized: `plotting/_style.py::apply_style()` forces the Agg backend
and loads `emri_thesis.mplstyle` (REVTeX 8 pt base, figsize 6.4×4.0, savefig.dpi
300, Type-42 fonts, frameless legends, `constrained_layout`, Okabe-Ito
`prop_cycle`, `image.cmap: viridis`). Colors live in `_colors.py` (Okabe-Ito +
semantic roles `TRUTH`/`MEAN`/`EDGE`/`REFERENCE`/`ACCENT` + the
`VARIANT_NO_MASS`/`VARIANT_WITH_MASS` pair). Labels live in `_labels.py`.
Width presets `single` (3.375 in) / `double` (7.0 in) live in `_helpers.py`.

This is a strong, well-factored baseline. The redesign **keeps every one of these
seams** and changes only what flows through them.

### Current weaknesses (from the inventory)

1. **Palette incoherence — two visual languages.** H₀/convergence figures
   (`fig01`, `fig02`, `fig08`, dashboard) use the blue `VARIANT_*` pair; CRB/Fisher
   figures (`fig06`, `fig07`, `fig12`, `fig14`, `fig04`, `fig09`) default to orange
   `CYCLE[0]`/sky-blue `CYCLE[1]`. `fisher_plots.plot_fisher_diagnostics`
   hardcodes `cmap='plasma'`, breaking the viridis convention. No de-facto
   method→color map (the field's bright=yellow / spectral=orange / dark=blue /
   combined=black) is encoded.
2. **The two variant blues are too close.** `#0072B2` petrol vs `#56B4E9` sky read
   as one color in `fig01`/`fig08`, and they are *not* separated by linestyle —
   a colorblind/greyscale failure despite the Okabe-Ito intent.
3. **No reference-band system shared across figure types.** Planck/SH0ES bands
   appear only on `fig01`/`fig15`, with colliding text labels; they are absent from
   convergence and from any cross-experiment comparison.
4. **No prior overlay.** GW H₀ posteriors are often prior-dominated; the flat H₀
   prior is never drawn, so the prior→posterior update is invisible.
5. **No validation/calibration figure.** Despite an active bias-investigation
   narrative (`docs/H0_BIAS_RESOLUTION.md`), there is no P–P plot, SBC rank plot,
   coverage plot, or forest plot tying the result to Planck/SH0ES.
6. **Size/aspect inconsistency.** `fig13`, `fig14` ignore the REVTeX presets
   (wide 6.4 in canvas, tiny fonts, huge PNGs); `fig09` is a single squeezed into a
   wide-short aspect.
7. **Redundancy.** `fig15` dashboard re-renders `fig01`+`fig03`+`fig04`+`fig05` at
   thumbnail scale with no new information; `fig01` ≈ `fig08`-left; `fig10` (PSD) and
   `fig13` (strain) are the same physics twice.
8. **Single-band uncertainty.** CI is shown as one shaded interval (deterministic
   construal error) rather than nested HDI levels.
9. **Data/labeling bugs surfaced as design debt.** `fig04` passes redshift as both
   injected and detected (meaningless ~1 fraction); `fig03`/`fig15` SNR annotation
   clips to a bare `100`; `fig11` `d_L` peaks at ~28 Mpc for z≤3 (unit error) with a
   `d_L(z)` underscore artifact; `fig06` axis offset notation collides with ticks;
   `fig14` 3D tick labels unreadable.

The redesign addresses 1–8 as design choices; 9 are bugs to fix in passing (the
design must not depend on the buggy behavior).

---

## 2. Design Directions

Four cohesive, named directions. Each is expressible as an
`emri_thesis.mplstyle` v2 + an extended `_colors.py` dictionary, with **zero**
changes to the factory-function contract. The ASCII mockups all show the
flagship **fig01 — combined H₀ posterior**.

---

### Direction A — "Observatory" (LVK-faithful)

**One-liner:** Adopt the de-facto LVK/gwcosmo visual grammar verbatim so the
thesis reads as native GW-cosmology literature.

- **Palette.** Lock the **method→color map**: bright siren = gold `#F0E442`,
  spectral = orange `#E69F00`, dark/catalog = blue `#0072B2`, **combined =
  black** `#1a1a1a`. The two pipeline variants become *the same blue* separated
  by **linestyle** (Without M_z = solid, With M_z = dashed) — matching the LVK
  "no-weight dashed / weighted solid" sensitivity convention and fixing the
  low-contrast two-blue problem. Tension anchors: **Planck = pink band**
  (`#CC79A7` at low alpha), **SH0ES = cyan/green band** (`#56B4E9` or `TRUTH`
  green), full-height behind curves, on *every* H₀ figure and the forest plot.
- **Typography.** REVTeX serif (Computer Modern via `usetex`), 8 pt base, 7 pt
  ticks. Direct end-of-line labels where a curve exits the axes; legend reserved
  for the band key only.
- **Layout philosophy.** Fixed H₀ x-range (left at h, secondary top axis in
  km/s/Mpc, ~50–100). Flat prior always drawn as a light dashed horizontal.
  68.3 % primary CI shaded under the curve, 90 % quoted in the title.

```
  P(h)                          fig01 — Observatory
  |                    Planck            SH0ES
  |                   ░░▒▒░░            ░░▒▒░░
 1.0|------------------░░▒▒-----╭─╮-----▒▒░░---------  ── prior (flat)
  |                   ░░▒▒    ╭╯   ╲   ▒▒░░
  |                   ░░▒▒   ╱      ╲  ▒▒░░          ── combined (black)
  |                   ░░▒▒  ╱ ▓68%▓  ╲ ▒▒░░          ┄┄ dark, With M_z (blue dash)
  |                   ░░▒▒ ╱▓▓▓▓▓▓▓▓╲ ▒▒░░           ── dark, Without M_z (blue)
  |                   ░░▒▒╱▓▓▓▓▓▓▓▓▓▓╲▒▒░░
  |__________________░░▒▒▔▔▔▔▔▔▔▔▔▔▔▔▔▒▒░░__________
   0.60   0.65    0.674│   0.70   0.73│   0.76   h
                    Planck         SH0ES/truth
   h = 0.728 +0.021 -0.019 (68%)  [90%: +0.034 -0.031]
```

**Best for:** the paper figures (`paper_*`). Maximally familiar to a GW-cosmology
referee; lowest novelty risk.

---

### Direction B — "Slate" (modern monochrome-accent)

**One-liner:** A restrained near-monochrome base (greys + one accent per figure)
that pushes data forward and makes the *result* curve unmistakable.

- **Palette.** Base everything in a 3-step neutral grey ramp (`#1a1a1a` /
  `#6e6e6e` / `#bdbdbd`) for context (priors, per-event lines, references); a
  **single accent** carries the headline series per figure — Crameri **batlow**
  endpoints used semantically (`#011959` deep / `#FACCFA` light) or Okabe-Ito
  blue for H₀. Variants separated by accent *shade* + linestyle. Reference bands
  become thin grey vertical rules with a small swatch label, not filled bands
  (avoids the muddy alpha-layering of `fig01`).
- **Typography.** Same REVTeX serif; slightly heavier headline line
  (`linewidth 1.6`) vs hairline context (`0.6`), so weight encodes importance.
- **Layout philosophy.** "Context recedes, result advances." Per-event spaghetti
  in pale grey; combined in full accent. Nested HDI (50/68/95 %) as three accent
  alphas instead of one band.

```
  P(h)                            fig01 — Slate
  |              Planck┊                 ┊SH0ES
 1.0|··············┊··········╭──╮········┊·············  ·· prior
  |               ┊        ╭─╯    ╲       ┊          ▓ 50% HDI
  |               ┊       ╱ ▓▓▓▓▓▓ ╲      ┊          ▒ 68% HDI
  |               ┊      ╱ ▒▒▓▓▓▓▒▒ ╲     ┊          ░ 95% HDI
  |               ┊     ╱░▒▒▒▓▓▓▓▒▒▒░╲    ┊      ━━ combined (accent)
  |     (faint grey per-event lines)      ┊      ── per-event (grey)
  |_______________┊___░░▒▒▓▓▓▓▒▒░░___ ____┊___________
   0.60         0.674      0.70   0.73  0.76        h
            ┊Planck                  ┊SH0ES
```

**Best for:** thesis body chapters where many figures sit close together — the
shared neutral base makes the suite read as one document.

---

### Direction C — "Atlas" (scientific-colormap forward)

**One-liner:** Lean into Crameri scientific colormaps as a *system*: one
sequential, one diverging, one cyclic, applied by data type across every field
plot, with categorical Okabe-Ito only for discrete series.

- **Palette.** `image.cmap` → **batlow** (sequential: SNR, density, P_det,
  posterior height) replacing viridis; **vik** for any signed quantity
  (residuals, pulls, MAP bias, the new P–P/coverage deviations); **romaO/twilight**
  for phase/angle (sky, inclination). Discrete series keep Okabe-Ito. The diverging
  map unlocks a whole class of *new* validation figures (bias maps, pull
  distributions) that have no good encoding today.
- **Typography.** REVTeX serif; colorbars get explicit "[unit], level" captions;
  every continuous figure passes the greyscale + deuteranopia check by
  construction (batlow/vik/cividis are CVD-safe and monotonic-lightness).
- **Layout philosophy.** Field plots (sky, P_det surface, coverage) become the
  visual centerpieces; the 1D H₀ posterior borrows the diverging map only for its
  *residual-from-truth* inset.

```
  P(h)                            fig01 — Atlas
  |   [batlow colorbar: per-event SNR ρ ]  ▏20 ──► 80
  |                Planck            SH0ES
 1.0|---------------▓▓▓---╭──╮---------▓▓▓--------------  ── prior
  |                ▓▓▓  ╱     ╲        ▓▓▓
  |                ▓▓▓ ╱ event ╲       ▓▓▓   per-event lines
  |                ▓▓▓╱ lines   ╲      ▓▓▓   colored by ρ
  |                ▓▓│ colored   │     ▓▓▓   (batlow)
  |                ▓▓│ by SNR    │     ▓▓▓   ━━ combined (black)
  |________________▓▓▓___________▓▓▓________________
   0.60          0.674   0.73  0.76               h
  ┌── inset: (h - h_true) residual, vik diverging ──┐
  │  −0.05 ◄═══ blue │ white │ red ═══► +0.05        │
  └──────────────────────────────────────────────────┘
```

**Best for:** the field-heavy figures (`fig05` sky, `fig20` P_det surface,
completeness maps) and the new validation set. Highest scientific-rigor signal.

---

### Direction D — "Storyboard" (narrative / interactive-first)

**One-liner:** Treat the suite as a guided narrative — a fixed
*event → host → likelihood → posterior → combined → tension* ordering (the
Hitchhiker's-Guide methods sequence), with the interactives as scrollytelling
explorers and the static figures as the print stills of that story.

- **Palette.** Inherits Direction A's method→color map (so static and interactive
  agree pixel-for-pixel), plus a per-step accent so each narrative beat has an
  identity reused in both media.
- **Typography.** Static: REVTeX serif. Interactive: matching CSS stack, with
  step captions; Plotly `_strip_latex` already bridges labels.
- **Layout philosophy.** Static figures gain a tiny "step N/6" breadcrumb in the
  corner; interactives become Scrollama sticky-graphic explorers (single pinned
  Plotly figure swapped per scroll step), Tangle-scrubber for the H₀-tension
  beat. Everything stays self-contained HTML (`include_plotlyjs`, no server).

```
  fig01 — Storyboard   [ step 6/6 · combine ]
  P(h)            Planck            SH0ES
 1.0|·············░▒░·······╭──╮······░▒░···········  ·· prior
  |              ░▒░      ╱      ╲     ░▒░
  |   ← step 5   ░▒░     ╱ ▓68%▓  ╲    ░▒░   step 6 →
  |  per-event   ░▒░    ╱▓▓▓▓▓▓▓▓▓╲   ░▒░   combined
  |  posteriors  ░▒░   ╱▓▓▓▓▓▓▓▓▓▓▓╲  ░▒░   (black)
  |____________░▒░▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔░▒░___________
   0.60       0.674      0.73   0.76          h
  ▸ "Stacking 30 events sharpens h from ±0.06 to ±0.02."
```

**Best for:** the public-facing GitHub Pages gallery and the thesis-intro
intuition figures; pairs with, rather than replaces, A for the paper.

---

## 3. Per-Figure Current → Proposed

Legend: **Keep** (encoding fine, restyle only) · **Restyle** · **Rework**
(new encoding) · **Merge** · **Retire** · **Fix** (bug, not design).

| # | Figure | Current | Proposed | Direction lever |
|---|--------|---------|----------|-----------------|
| 01 | h0_posterior_combined | 2 near-identical blues, single CI band, Planck/SH0ES filled bands with colliding labels, truth `True h` twice in legend | **Rework.** Variants = one blue + linestyle (solid/dashed). Nested 50/68/95 % HDI. Add **flat prior** overlay. Pink Planck + green/cyan SH0ES bands with swatch labels (no duplicate truth entry). Secondary top axis in km/s/Mpc. Title quotes 68 %(90 %). | A bands+prior+method-color; B nested HDI |
| 02 | event_posteriors | faint per-event curves dwarfed by bold combined (scale mismatch) | **Rework.** Peak-normalize per-event so shapes are visible; color per-event lines by SNR or #hosts (batlow), combined as black headline. Optional facet by SNR tercile. | C sequential-by-SNR; A combined=black |
| 03 | snr_distribution | hist + twin CDF, threshold line, clipped `100` annotation | **Restyle + Fix.** Fix the `…% above threshold` clip. Grey histogram, accent CDF, threshold as labeled vertical rule. Keep twin-axis. | B neutral hist + one accent |
| 04 | detection_yield | redshift passed as injected **and** detected → meaningless ~1 fraction with spurious dip | **Rework + Fix.** Requires real injected-vs-detected arrays (gate on injection CSV; show "data missing" otherwise). Open vs filled hist + true detection fraction. Otherwise **Merge** into the selection-function explainer (new). | C; new-figure NF-4 |
| 05 | sky_localization | Mollweide scatter, viridis SNR | **Restyle.** Swap viridis→**batlow** (Atlas system); rasterize markers; degree graticule labels lightened. | C colormap system |
| 06 | fisher_ellipses | 1σ/2σ orange ellipses, offset notation collides with ticks | **Restyle + Fix.** Use `ScalarFormatter` with `useOffset` placed in axis label (not over ticks); truth crosshair in green; 1σ/2σ in two accent alphas. | B two-alpha accent |
| 07 | corner_plot | corner, green truths, orange contours, KDE-smoothed | **Restyle.** **Disable KDE smoothing** (Fisher Gaussian → analytic contours, no artificial broadening); 1- and 2-σ; truths from Planck15 with `np.nan` suppression for params lacking a truth; `[0.16,0.5,0.84]` titles. | C/A convention fix |
| 08 | h0_convergence | left ≈ fig01; right CI-width vs N with band + 1/√N | **Merge + Rework.** Drop the left posterior panel (redundant with fig01); keep/expand the CI-width-vs-N convergence panel, add Planck/SH0ES band as horizontal target-width reference and the 1/√N guide. Variant = linestyle. | A bands; merge redundancy |
| 09 | detection_efficiency | step P_det vs z, ~1 with notches, wide-short aspect | **Rework.** Fold into **selection-function explainer** (NF-4): smooth p_det vs z with a *family* of curves for σ_dL/dL, showing threshold softening. Single-column aspect. | C/A; new-figure NF-4 |
| 10 | lisa_psd | log-log S_n decomposition | **Keep (restyle).** Decompose total/instrument/confusion with linestyle redundancy; consistent single-column size. | B linestyle redundancy |
| 11 | distance_redshift | per-H₀ d_L(z); **unit bug** (~28 Mpc max), `d_L(z)` artifact | **Fix + Restyle.** Fix the Mpc unit scale; clean legend label; direct-label the h-curves at their right endpoints instead of a legend. | sci-viz direct labeling |
| 12 | uncertainty_violins | violins of σ/|x|, intrinsic orange / extrinsic blue | **Restyle.** Keep the 2-group split but encode group by position + a single hue pair from the locked palette; add nested quantile markers; log y labeled. | B/A grouping |
| 13 | characteristic_strain | log-log h_c, wide default canvas, big PNG | **Merge.** Redundant physics with fig10 (PSD). Keep **one** sensitivity figure (prefer h_c with an example EMRI track) at REVTeX size; retire the other or make it a two-panel small-multiple (S_n | h_c) sharing the x-axis. | retire redundancy |
| 14 | crb_coverage | 3D scatter, unreadable ticks, big PNG | **Rework.** Replace mplot3d with a 2D small-multiple of pairwise coverage (M–qS, M–phiS, qS–phiS) or a 2D hexbin density; offset-formatted M axis. 3D → interactive only. | sci-viz small multiples |
| 15 | campaign_dashboard | 2×2 thumbnail re-render of 01/03/04/05, label collisions | **Retire (static).** No new information; collisions. Replace its role with the **Storyboard** interactive landing page (D) or a single methods-sequence strip figure. | merge/retire |
| — | paper_* set | parallel REVTeX figures | **Adopt Direction A** as the canonical paper look; share the locked palette + prior + bands so paper and gallery agree. | A |

### Interactives (HTML, stay self-contained `write_html`/`include_plotlyjs`)

| Interactive | Current | Proposed |
|-------------|---------|----------|
| combined_posterior.html | 1D posterior + CI + Planck/SH0ES lines | Add flat-prior trace + nested HDI shading; method→color map; optional **HOPs** Play button cycling bootstrap posterior draws (existing `go.Frame`+`updatemenus` machinery). |
| sky_map.html | Scattergeo by SNR | Swap to batlow colorscale; keep hover (z, d_L). |
| fisher_ellipses.html | per-event ellipses | Disable smoothing analog (exact ellipses); consistent σ-level legend. |
| h0_convergence.html | 2-panel posterior-vs-N + CI-vs-N | Add Planck/SH0ES target-width band; keep slider. |
| m_z_improvement.html | metric dropdown + per-N posterior | Keep; align metric colors to locked palette; HOPs option. |
| single_event_detail.html | per-event L(h), host weights | **Storyboard** scrollytelling candidate (event→host→posterior). |
| closure_test.html | per-h_true overlay | Keep; add MAP-vs-truth diagonal inset (a coverage teaser). |
| catalog_completeness.html | host counts + coverage-vs-d_L | Pair with the **selection-function explainer** explorable (Scrollama sticky graphic + Tangle H₀ scrubber); model on GaiaUnlimited. |
| **NEW** consider switching all to `include_plotlyjs="directory"` | per-file CDN | one shared local `plotly.min.js` → offline-robust Pages, smaller per file. (Still self-contained, still static.) |

---

## 4. New Figure Ideas

All implemented as **new factories** `plot_*(data) -> (fig, ax)` + new manifest
entries — no architectural change. Numbering continues the static set.

- **NF-1 · H₀-in-context forest plot** (`plot_h0_forest`).
  Di-Valentino-style horizontal whisker plot: one categorical row per
  measurement (Planck, SH0ES, GWTC dark sirens, this work), point = central H₀,
  caps = 68 % CL (asymmetric), grouped early/indirect vs late/direct, with the
  **same Planck-pink + SH0ES-cyan vertical bands** as fig01 so the two figures
  read as one visual system. Data schema mirrors `H0TensionRealm/dataset.csv`
  (a small committed CSV). *This is the single highest-impact addition.*

- **NF-2 · P–P / coverage plot** (`plot_pp_coverage`).
  Per-parameter ECDF of the true-value percentile rank vs the diagonal, grey
  1/2/3-σ confidence band, per-parameter + combined KS p-value in the legend
  (bilby/Ashton convention). Directly demonstrates the Fisher/CRB pipeline is
  unbiased — the missing companion to `docs/H0_BIAS_RESOLUTION.md`.

- **NF-3 · SBC rank / centered-ECDF-difference** (`plot_sbc_rank`).
  Modern upgrade of NF-2 for the *whole H₀ pipeline*: rank of true h among
  posterior draws over many simulated catalogs; plot rank-ECDF **minus uniform**
  (flat null at zero) with **simultaneous** (gamma-adjusted) bands. ∪-shape =
  overconfident, ∩ = underconfident. Uses Direction C's vik diverging map for the
  deviation.

- **NF-4 · Selection-function explainer** (`plot_selection_function`).
  Smooth p_det vs true redshift (or d_L) with a **family** of curves for several
  σ_dL/dL, making the SNR=20 threshold-softening explicit (Hitchhiker Fig. 1 /
  Chen–Fishbach). Absorbs and fixes the broken `fig04`/`fig09`. Same axis the
  likelihood uses.

- **NF-5 · Line-of-sight redshift-prior figure** (`plot_los_redshift_prior`).
  Per sky direction: blue histogram of catalog galaxy z, orange reconstructed
  p_cat(z), black-dashed uniform-in-comoving-volume fallback (Hitchhiker Fig. 2).
  Communicates the completeness-correction story directly.

- **NF-6 · MAP-bias vs truth summary** (`plot_map_bias`).
  Residual `(h_MAP − h_true)/h_true` across closure runs / event subsets with a
  vik diverging encoding and a zero line — the static counterpart to the closure
  interactive, closing the bias narrative.

- **NF-7 · Methods-sequence strip** (`plot_methods_sequence`, optional).
  A single horizontal small-multiple strip
  (localization → LOS prior → per-event likelihood → per-event posterior →
  combined → H₀-over-bands) replacing the retired dashboard `fig15` with an
  *informative* composite instead of a redundant one.

- **NF-8 (interactive) · H₀-in-context tension explorer.**
  Scrollama sticky graphic + Tangle scrubber over the Planck/SH0ES bands already
  hardcoded in `interactive.py` (`_PLANCK_H_RANGE`, `_SHOES_H_RANGE`); greenfield,
  self-contained HTML.

---

## 5. Recommended Default + `emri_thesis.mplstyle` v2 Sketch

### Recommendation

**Adopt Direction A ("Observatory") as the default for the paper (`paper_*`) and
all H₀ figures, layered with Direction C ("Atlas") for the continuous-field and
validation figures (sky, P_det, P–P/SBC, coverage).**

Why this pairing:

1. **A is the lowest-risk, highest-recognition choice for a GW-cosmology
   referee.** The method→color map, Planck/SH0ES bands, flat-prior overlay, and
   68 %(90 %) reporting are exactly what GWTC-4/5 and gwcosmo use; matching them
   makes the result legible *and* fixes the two worst current defects (the
   indistinguishable twin blues → blue+linestyle; the missing prior).
2. **C is non-negotiable for the field plots and the new validation set.**
   Swapping `image.cmap` viridis→batlow and adopting vik for signed quantities is
   a one-line style change that makes the sky map, P_det surface, and the new
   P–P/SBC/bias figures CVD-safe and greyscale-robust *by construction* — and it
   removes the rogue `plasma`.
3. **They share one palette dictionary and one mplstyle**, so the suite stays
   coherent; B and D are reframings of the same tokens (B = weight-encoded subset
   of A's palette; D = A + narrative scaffolding for the interactives). Picking
   A+C does not preclude later promoting D for the public gallery, because D
   inherits A's colors.
4. **It maps cleanly onto the existing seams.** Everything below is a change to
   `emri_thesis.mplstyle` and `_colors.py` plus new factories — the
   `apply_style()` / `(fig, ax)` / manifest contracts are untouched.

### `_colors.py` v2 additions (sketch — design only)

```python
# Method -> color map (de-facto LVK standard); reuse everywhere.
METHOD: dict[str, str] = {
    "bright":   "#F0E442",  # gold  — bright siren (EM counterpart)
    "spectral": "#E69F00",  # orange — spectral siren (mass spectrum)
    "dark":     "#0072B2",  # blue  — dark / galaxy-catalog siren
    "combined": "#1a1a1a",  # black — combined / fiducial headline
}

# Variants share ONE hue; distinguish by linestyle (CVD/greyscale safe).
VARIANT_STYLE: dict[str, tuple[str, str]] = {
    "no_mass":   (METHOD["dark"], "-"),    # Without M_z  (solid)
    "with_mass": (METHOD["dark"], "--"),   # With M_z     (dashed)
}

# Tension anchors (full-height bands) — SAME on posterior AND forest plot.
PLANCK_BAND: str = "#CC79A7"   # pink, low alpha
SHOES_BAND:  str = "#56B4E9"   # cyan/green (or reuse TRUTH for SH0ES≈truth)
PRIOR:       str = "#9e9e9e"   # neutral grey, dashed — flat H0 prior

# Scientific colormaps (Atlas) — requires `cmcrameri` (optional dep).
#   SEQUENTIAL -> batlow   (SNR, density, P_det, posterior height)
#   DIVERGING  -> vik      (residuals, pulls, MAP bias, P-P/SBC deviation)
#   CYCLIC     -> romaO    (phase / sky angle / inclination)
# Fallbacks (no new dep): cividis (seq), RdBu (div), twilight (cyclic).
SEQUENTIAL_CMAP = "batlow"   # was "viridis"
DIVERGING_CMAP  = "vik"      # NEW
CYCLIC_CMAP     = "romaO"    # NEW
GREY_BAD: str = "#bdbdbd"    # reserved neutral for out-of-range/reference
```

### `emri_thesis.mplstyle` v2 sketch (REVTeX-true, unchanged contract)

```ini
# --- Sizing: figures sized to FINAL printed width; NO LaTeX scaling ---
figure.figsize: 3.375, 2.086     # single column = golden ratio (was 6.4x4.0)
figure.dpi: 150
savefig.dpi: 600                 # 600 for line art (was 300); photos still 300
savefig.bbox: tight
savefig.pad_inches: 0.02

# --- Typography: serif/CM to match REVTeX body; >=7pt APS minimum ---
font.family:     serif           # apply_style(use_latex=True) -> Computer Modern
font.size:       9               # body-matched (was 8)
axes.titlesize:  9
axes.labelsize:  9
xtick.labelsize: 8
ytick.labelsize: 8
legend.fontsize: 8
pdf.fonttype: 42                 # keep: no Type-3 (unchanged)
ps.fonttype:  42

# --- Lines: weight encodes importance (Slate-compatible) ---
lines.linewidth: 1.0             # headline factories may bump to 1.6; context 0.6

# --- Axes / ticks / legend: unchanged publication style ---
axes.linewidth: 0.8
axes.spines.top:   False
axes.spines.right: False
xtick.direction: in
ytick.direction: in
legend.frameon:  False

# --- Color: Okabe-Ito prop_cycle kept for categorical lines ---
axes.prop_cycle: cycler(color=["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7"])

# --- Default colormap: batlow (Atlas). If cmcrameri absent, set cividis ---
image.cmap: cividis              # was viridis; batlow set in code when available

figure.constrained_layout.use: True
agg.path.chunksize: 10000
```

Notes on honoring the hard constraints:

- **Factory contract preserved.** All of the above is consumed *inside* existing
  `plot_*` factories via the `_colors.py` dict + the style sheet. No factory
  signature changes; callers still save.
- **REVTeX vector-PDF + CVD-safe.** Serif/CM, ≥7 pt, Type-42, batlow/vik/cividis
  (all CVD-safe, monotonic-lightness, greyscale-robust); add a CI helper to run
  each new factory through a deuteranopia + desaturation check in tests.
- **Width discipline.** Default figsize → single-column 3.375 in; `double`
  preset unchanged; `fig13`/`fig14` brought onto presets (fixing the rogue wide
  canvas). No `\includegraphics` rescaling — figures sized at final width.
- **Interactives stay static.** Plotly only, `write_html`; the only change is
  `include_plotlyjs="cdn"` → `"directory"` (still self-contained for Pages) and
  reuse of the existing `go.Frame`/`updatemenus` machinery for HOPs — no server,
  no new runtime.
- **Single entry path.** Every new figure (NF-1…NF-7) and every restyle ships as
  a factory + a `manifest.append(...)` entry in `main.generate_figures`; the
  `cmcrameri` colormap dependency is optional with a cividis/RdBu/twilight
  fallback so the no-dep path still renders.
```
