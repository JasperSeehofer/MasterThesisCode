# Visualization Redesign Proposal — EMRI Dark-Siren H₀ Inference

**Status:** Decision-ready proposal (no implementation in this document)
**✅ DECISION (2026-06-21, Jasper):** Chosen direction = **HORIZON, with Dark Siren Dispatch's annotation discipline folded in** (the recommended default, §6). Implementation to be scoped as a GSD milestone.
**Scope:** All ~26 static (paper + diagnostic) figures and 8 interactive Plotly figures
**Hard constraints honored throughout:** REVTeX vector-PDF output, colorblind- AND grayscale-safe encodings, self-contained GitHub-Pages HTML, and the existing factory-function architecture (`data in → (fig, ax) out`) is preserved — this is a style + recolor + annotation + consolidation pass, **not** a data-plumbing rewrite.
**Implementation note:** the work described here would be scoped as a GSD milestone/phase later. This document drives that plan; it does not execute it.

---

## 1. Executive Summary

### The problem

The current figure set is *competent-classic*: clean factory architecture, an Okabe-Ito cycle, REVTeX-aware sizing, and graceful runtime fallbacks. But it reads as 2015-era matplotlib rather than a 2024 top-journal / LVK-collaboration result. The audit and field research converge on a consistent set of structural weaknesses:

1. **The two-blues collision.** `VARIANT_NO_MASS = #0072B2` and `VARIANT_WITH_MASS = #56B4E9` are *both blue*, and `REFERENCE` is *also* `#56B4E9` (identical to with-mass). Every Without/With-M_z comparison — the central scientific contrast of the thesis — relies on linestyle alone and collapses to indistinguishable grays in print. The `results.tex` caption already says "blue solid / red dashed," so the code contradicts its own intended design.
2. **Quadruplicate central result.** The combined H₀ posterior is rendered by *at least four* independent code paths (`fig01`, `paper_h0_posterior`, `paper_h0_posterior_kde`, the `(0,1)` panel of `paper_m_z_improvement` / left panel of `fig08`) with three different normalizations and two different CI definitions. They can and do drift.
3. **Non-standard normalization.** Peak-normalization to 1.0 is the house style for paper figures but strips probability interpretation and makes the two variants' areas non-comparable. The field convention is area-normalized PDFs with shaded HDI.
4. **Marker-on-every-grid-point** clutter hides the shape of sharply-peaked posteriors.
5. **Inconsistent credible intervals.** Three CI machineries coexist; `compute_hdi_interval` (the LIGO/Virgo convention, already in `_helpers`) is used by *none* of the headline figures.
6. **Wasted colormap dynamic range.** Every 2D map uses raw `viridis` with linear norm; SNR and P_det pile up at one end, so most events render the same hue. The failure is *normalization*, not just colormap choice.
7. **Missing field-expected figures.** Grep confirms there is **no** H₀ forest/tension plot, **no** PP-plot/coverage figure, **no** dedicated selection-function/detection-horizon explainer, and **no** "where do the constraints come from" population view. These are the single highest-value additions.
8. **Layout-engine conflict.** The stylesheet sets `constrained_layout.use: True`, yet five factories also call `fig.tight_layout()` — a documented double-layout warning/conflict.
9. **Interactive set is incoherent.** Eight standalone Plotly dumps with `include_plotlyjs='cdn'` (CDN-dependent, not truly self-contained), no shared layout template, the same two-blues problem, a hand-authored `index.html` with a drift-prone hardcoded footer, and fragile visibility-vector bookkeeping.

### The 5–8 highest-impact cross-cutting changes

These cut across every design direction below. They are the spine of the redesign.

1. **Fix the variant palette once, in `_colors.py` v2.** Reassign the With-M_z variant to a non-blue, opposite-lightness hue (gold/orange) and give `REFERENCE` its own color. One edit propagates to every H₀ / per-event / convergence / single-event / interactive figure. Encode every comparison with **color + linestyle + direct label** (redundant channels) so it survives grayscale and deuteranopia.
2. **Consolidate the central result into ONE canonical H₀-posterior factory** with a theme/option switch (`peak_vs_pdf`, `show_hdi`, `show_context_bands`, `theme`). Retire the peak-norm + marker-on-every-point + KDE twins. Likewise fold the two convergence figures into ONE log-log factory with the fitted slope annotated inline.
3. **Adopt area-normalized PDFs + shaded nested 68%/95% HDI** for all single/headline posteriors, using the existing `compute_hdi_interval`. One CI definition project-wide. Peak-normalize *only* for explicit many-variants-overlaid comparisons, and never shade a band under a peak-normalized curve.
4. **Switch the default sequential colormap to `cividis`** (print-first, near-linear luminance, CVD-robust) and give every heatmap an explicit `LogNorm`/robust-clip norm, a `set_bad` no-data color, and a semantic contour overlay (P_det = 0.5 horizon, 68/95% credible). Use a diverging map (`cm.vik`, `TwoSlopeNorm(vcenter=0.73)`) for bias/residual maps only.
5. **Install an annotation layer.** Active declarative titles state the finding; the MAP ± HDI is annotated inline on the curve; reference values are named at their rule; scaling laws are labeled on the curve; legends are replaced by direct end-of-line color-matched labels. This is the single biggest "looks like a communication, not a plot" lever.
6. **Add the four missing field-expected figures** as new factories on existing data plumbing: the **H₀ forest/tension plot**, a **bilby-style PP-plot** (grey 1-2-3σ binomial bands + per-parameter KS p-values), a **p_det(d_L) survival + (M, d_L) horizon-contour selection explainer**, and a **"where do constraints come from" population view**.
7. **Build a `theme='paper'|'talk'|'web'` switch layered on `apply_style()`** — one base design language, three thin override layers, with the web theme exporting the *same hex tokens* as CSS custom properties / a Plotly template so print and web read as one document.
8. **Unify and harden the interactive set:** one shared `go.layout.Template`, `include_plotlyjs='directory'` (or partial bundle) for self-contained-at-site-level output, and an Observable Framework shell for coherence + archival robustness, fed by precomputed Python JSON snapshots (the `phase45` h-sweep cube already exists).

---

## 2. Current-State Assessment

### Static figures

| Figure | What it shows | Current chart type | Key weakness |
|---|---|---|---|
| `fig01_h0_posterior_combined` | Combined H₀ posterior, both variants, 68/95% bands, Planck/SH0ES context, truth | 1D line PDF + `fill_between` + `axvspan` + `axvline` | Both variants blue; muddy boundary `axvlines`; CI via integer-index cumsum (inconsistent with rest); no HDI option |
| `fig02_event_posteriors` | Per-event spaghetti + combined overlay, optional metadata color | Overlaid 1D lines + thick overlay + optional colorbar | `color_by` never enabled by `main.py` → uninformative monochrome haze; combined line doesn't pop; no HDI band |
| `fig08_h0_convergence` | Posterior (dup) + CI-width vs N | 2-panel: posterior lines + linear `o-/s--` scatter w/ 1/√N ref | Linear axes (paper twin is log-log); left panel duplicates fig01; literal `\%` label bug; two-blue |
| `paper_h0_posterior` | H₀ posterior comparison, peak-norm, 68% bands | 1D peak-norm line + marker on every point + `axvspan` | Peak-norm (non-standard); marker clutter; two-blue; no context bands; duplicate of fig01 |
| `paper_single_event` | 4 representative single-event likelihoods × 2 variants | 4×2 grid of peak-norm curves + markers | Marker clutter; widths not annotated; dead whitespace; raw event ids in labels; two-blue |
| `paper_convergence` | 68% CI width vs N, both variants, 1/√N ref | log-log errorbar scatter + power-law ref line | 16/84 as caps not band; exponent not annotated; ref color ≈ with-mass hue; two-blue |
| `paper_snr_distribution` | SNR histogram + SNR-vs-d_L scatter colored by z | 2-panel hist + colored scatter (text fallback) | viridis dynamic-range waste; linear-count hist hides power-law tail; cluster paths hardcoded in fallback |
| `paper_h0_posterior_kde` | KDE-smoothed posterior comparison | Faded markers + KDE lines + `axvspan` | KDE broadens near-delta peak (warns about MAP drift); peak-norm; THIRD copy of central result |
| `paper_m_z_improvement` | 2×3 M_z improvement dashboard | log-log curves + posterior + 2 line plots + violin + text box | Busy montage; FOURTH posterior copy; 8pt monospace text dump; 10.5in non-REVTeX width |
| `fig06_fisher_ellipses` | 1-/2-σ Fisher ellipses, 3 param pairs | Row of axes w/ stacked `Ellipse` patches | No σ legend; no truth/center marker; muddy overlap; `tight_layout` over constrained; no equal aspect |
| `fig07_corner_plot` | Joint Fisher posterior, 6 params | `corner.corner` from 5000 MVN samples | Dated thin-line aesthetic; truths = sample mean (no info); MC noise on analytic ellipse; ignores house style |
| `fig13_characteristic_strain` | LISA h_c(f), noise components + toy EMRI | 4-line log-log | Toy power-law EMRI (not real); confusion subtraction can go negative; no shaded floor; corner legend |
| `fig12_uncertainty_violins` | Fractional CRB σ over 14 params | violin (≥10 rows) OR horizontal bar | Hidden dispatch threshold; no reference lines (1%/10%); rotated mathtext ticks; no precision ordering |
| `fisher_quality_diagnostic` | Fisher conditioning / degeneracy audit | grouped log-y bars + plasma scatter | Breaks factory convention (calls `save_figure`, returns None); plasma cmap; "slot N" debug labels |
| `fig09_detection_efficiency` | Detection efficiency vs z + Wilson band | step curve + `fill_between` CI | Single color line+band; no 50% guide; no SNR-threshold annotation; empty bins break curve |
| `fig10_lisa_psd` | LISA S_n(f), decomposed | multi-line log-log | Legacy-mode f-string labels; total near-black dominates; no shaded floor; duplicates fig13 content |
| `fig04_detection_yield` | Injected vs detected z + fraction | step+fill hist + twinx line | `main.py` wires detected as both → fraction trivially 1.0; same color injected/detected; twin-axis clash |
| `fig14_crb_coverage` | Param-space coverage (M, qS, phiS) | 3D `scatter3D` | 3D occlusion anti-pattern; 16×9 hardcoded; bare ASCII labels; no density encoding |
| `fig19_info_monotonicity` | Per-event 1D vs 2D HDI width | scatter + identity line | Violators not color-distinguished; overplot blob; faint identity line; no marginals; linear wastes range |
| `fig20_pdet_surface` | Empirical P_det(d_L, M) | `imshow` heatmap + colorbar | Fake-index y-axis w/ hand-formatted ticks; NaN bins invisible; viridis waste; no horizon contour; Mpc/Gpc unit bug |
| `fig05_sky_localization` | EMRI sky positions colored by SNR | Mollweide scatter + optional ellipses | Linear viridis SNR waste; `transData` ellipses geometrically wrong on Mollweide; dead lon arithmetic; unstyled graticule |
| `fig11_distance_redshift` | d_L(z) at several H₀ | multi-line, linear axes | Primary curve same color as first comparison; raw ASCII label; near-degenerate curves (no residual panel) |
| `fig15_campaign_dashboard` | 2×2 campaign summary | `subplot_mosaic` (posterior/snr/yield/sky) | Pasted-together panels; Mollweide breaks grid alignment; no (a)-(d) labels; SNR shown twice, two scales |
| `fig16_catalog_completeness` | GLADE completeness + host coverage | 2 line plots + median±band + twinx + text box | Schematic sigmoid risks reading as real data; two-blue; dual legends + twin-axis + monospace box overload |
| `fig17_single_event_detail` | Per-event host weighting + L(h) | 2×3 grid (2 bars + scatter + 3 lines) | Two-blue; peak-norm L(h); hardcoded 11×5.6 figsize; independently-sorted bars misalign; monospace debug boxes |
| EMRI population / model / catalog raw dists | EMRI density (z,M), sampling, rate; BH-mass/z/volume histograms | contourf / hist2d / log-log line / hist | Inconsistent/unlabeled colorbars; raw ASCII labels; model & draws not on shared scale; 30-level banded contourf; viridis |
| Supporting eval diagnostics | CRB heatmap, uncertainty violins (dup), inj-vs-rec, detection contour, sky 3D | imshow / violin / GridSpec / hist2d / 3D scatter | Duplicate violin impl; 3D sky anti-pattern; raw covariance (not correlation); detection_contour vs pdet_surface overlap |

### Interactive figures

| Figure | What it shows | Current chart type | Key weakness |
|---|---|---|---|
| `interactive_combined_posterior` | 1D combined posterior + bands + truth | `go.Scatter` fill + `add_vrect`/`add_vline` | Two reference markers both blue-family; opaque overlapping vrects; PDF-norm here vs peak-norm in static fig01 |
| `interactive_sky_map` | Detected events on Mollweide, colored by SNR | `go.Scattergeo` (mollweide) | viridis dynamic-range waste; 1473 markers overplot (~318 KB); unstyled graticule; no coordinate-frame label |
| `interactive_fisher_ellipses` | Fisher 1-/2-σ ellipses, 3 pairs × ≤10 events | `make_subplots` row, `fill='toself'` | Raw param-key subplot titles; 60 traces, only 7 colors → ambiguity; no center/truth markers; ~255 KB |
| `interactive_h0_convergence` | Posterior sharpening + CI-width vs N | 2-panel: lines + line+ref | Linear-linear right panel (undersells √N); orange reused cross-panel; no bootstrap band |
| `interactive_m_z_improvement` | M_z tightening dashboard, dropdown + slider | `make_subplots` 2×2 + updatemenu + frames | Two-blue; fake text-subplot via `annotations[:3]` slicing (fragile); brittle visibility-vector bookkeeping; ~500 lines |
| `interactive_single_event_detail` | Per-event host weighting anatomy | 2×3 + event-picker dropdown | Two-blue; bare-rank bar x-axis; bars sorted differently per panel; redundant 3rd L(h); fixed 6-trace stride fragility |
| `interactive_closure_test_overlay` | Posteriors at different h_true | overlaid peak-norm lines + truth vlines | Off-brand Plotly default colors; truth vlines not color-tied to curves; hardcoded `[0.55,0.85]` axis range |
| `interactive_catalog_completeness` | Median hosts vs d_L + coverage bars | 2-panel: lines + bar | `REFERENCE` collides with with-cut line color; coverage ignores M_z cut; empty bins plot as misleading zeros |

---

## 3. Design Directions

Four directions are presented. All four share the cross-cutting spine of §1; they differ in voice, palette, and how far they push the interactive-first / editorial dimensions. Each direction's ASCII mockups are preserved verbatim from the design exploration.

---

### 3.1 Direction A — **Quiet Index**

> *Maximal data-ink economy with a quiet, semantic palette — the obviously-a-top-journal-2024 look, ported to dark sirens.*

**Philosophy.** Quiet Index treats every figure as a sentence: a single declarative finding, stated in the title, proven by an uncluttered data layer, and annotated in place so the caption is optional. The visual language is restrained to the point of looking effortless — smooth curves with shaded HDI bands instead of marker-dappled lines, direct end-of-line labels instead of boxed legends, near-black ink on white, and exactly one saturated accent per figure to carry the eye. Modernity comes from discipline, not decoration: a strict 2pt typographic ladder, a CB- and grayscale-safe two-variant pair that survives B/W print, perceptually-uniform colormaps with locked norms, and one consistent semantic role for every recurring quantity across paper, talk, and web.

**Palette (hex table).**

| Hex | Role |
|---|---|
| `#1A1A20` | Ink — axes, primary text, spines (near-black, not `#000`) |
| `#5A5A66` | Muted ink — tick labels, secondary annotations, reference-curve labels |
| `#9AA0AA` | Whisper gray — gridlines (when essential), no-data hatch, de-emphasized spaghetti |
| `#004488` | Variant NO_MASS / primary — Tol high-contrast blue, solid |
| `#DDAA33` | Variant WITH_MASS / secondary — Tol high-contrast gold, dashed |
| `#BB5566` | ACCENT / this-measurement highlight — Tol high-contrast rose |
| `#117733` | TRUTH / injected value — Tol muted green, thin rule |
| `#882255` | Reference (Planck) — muted wine vertical band |
| `#999933` | Reference (SH0ES) — muted olive vertical band |
| `#F3F3F5` | Panel/inset wash + HDI-95% band base |

**Sequential colormap.** `cividis` locked as default (most print-and-CVD-robust). `magma` for high-dynamic-range posterior-density surfaces. `cm.vik` with `TwoSlopeNorm(vcenter=0.73)` for the one diverging case (MAP−truth bias map) with redundant 0-level contour lines. Every heatmap sets `set_bad('#F3F3F5')`.

**Typography.** Paper: `usetex` Latin Modern Roman serif (body-match), true LaTeX math; CI fallback `mathtext.fontset: cm`. Strict 2pt ladder at final size (panel letter 9pt bold, axis 8pt, tick 7pt `#5A5A66`, data label 7-8pt semibold in series color). Figures sized to exact `\columnwidth` so LaTeX never rescales. Web: Source Sans / Open Sans mirrored as CSS custom properties; ONS px ladder (title 20px/700, axis 14px/400). KaTeX for equations. Talk: sans always, every size ×1.8, line weights 2.5-3pt.

**Layout.** `constrained_layout` only (every `tight_layout()` stripped). Two widths via `get_figure` presets (3.404in single, 7.055in double). Small-multiples use `sharex/sharey`, interior ticks removed, one centered axis label per shared edge, one shared colorbar, panels sorted by quantity of interest. Bold 9pt `(a)(b)` top-left at fixed offset. Annotation strategy is load-bearing: active titles, inline MAP±HDI, named reference rules, slope triangles on convergence curves, direct end-of-line labels. One accent per figure maximum.

**Treatment principles.** No per-point markers on smooth curves (one faint marker at MAP only). Area-normalize all single posteriors; nested 68%/95% HDI fills (alpha 0.30/0.15). Two-variant pair blue solid vs gold dashed (redundant hue + lightness + linestyle). Reference context as dedicated muted-band style labeled at top axis. Every 2D map `cividis` + shared log/robust norm + semantic contour. Corner/Fisher contours filled 68%+95%, drawn analytically from covariance, single neutral truth crosshair. All text through one LABELS provider; one unit per quantity. One canonical factory per quantity with `theme=` switch.

**Paper vs interactive.** Paper: v2 base mplstyle + `paper.mplstyle` override layered through `apply_style(theme='paper')`. Ruthlessly consolidated set — one area-normalized H₀ factory, one log-log convergence factory, 3D scatters → 2D marginal/corner, m_z dashboard → two focused figures. Every figure vector PDF, ≥7pt, passes a `colorspacious` deuteranopia + grayscale gate. Interactive: keep Plotly factories but apply one shared `go.layout.Template` exporting the same palette as CSS custom properties; `include_plotlyjs='directory'` + cartesian partial bundle; Observable Framework shell with precomputed JSON snapshots, KaTeX equations, insight-bearing interactions only (h-slider, estimator toggle, SNR/z brushing). Every interactive degrades to the static PDF twin.

**Signature new figure.** The H₀ forest / tension plot (Di Valentino whisker): ~10-12 horizontal point+68%CI rows grouped early-universe (blue) vs late-universe (rose), full-height Planck (wine) and SH0ES (olive) bands, THIS EMRI result the bottom row in bold rose at heavier weight. The single most-expected cosmology figure the pipeline lacks; doubles as the hero interactive (Concept C) with hover-for-citation and an accumulation slider.

**Trade-offs.** (1) Restraint is unforgiving — minimalism only reads as elevated if annotation + typography are executed precisely; the annotation work is non-optional. (2) `usetex` Latin Modern needs a working TeX in CI; the `mathtext.fontset: cm` fallback introduces a paper-vs-CI path to keep in sync. (3) Consolidating four posterior paths + two convergence figures is real refactoring with regression risk — needs golden-image tests. (4) Peak-norm → area-norm changes how the central result *looks*; harmless numerically but coordinate with `results.tex` captions. (5) Observable Framework adds a JS toolchain. (6) Conservative-safe is also the ceiling: correct and current, but not visually daring.

**ASCII mockups (verbatim).**

```
--- Headline H0 posterior (canonical, area-normalized + nested HDI + context bands + inline MAP) ---
EMRI dark sirens recover H0 to +2.6/-2.4%  (68% HDI)
 p(h | data)                                                  
   |                          .-^-.                            
   |                        .'     '.        <- With M_z (gold,dash)
   |              .--._    .'  ###### '.                        
   |            .'     '-.'  ########## '.   With M_z           
   |           /        ::############::: \                    
   |  Planck  /        :::############::: \   SH0ES            
   | [wine]  /       :::: h=0.736 ::::::: \  [olive]           
   |  band  /      ::::: +0.019/-0.018 :::: \  band             
   |       /  ____::::::::::::::::::::::::::__\___  Without M_z 
   |______/__/    `--::::: 68% HDI :::::--'    \__\___ (blue)   
   |     :        :  95% HDI (lighter)  :          :           
   +-----:--------+----------|----------+----------:-------- h  
        0.66    0.69   truth 0.73     0.76       0.80          
  Shaded = 68/95% HDI under curve. Thin green rule = injected 0.73.
```

```
--- NEW H0 forest / tension plot (this EMRI result in context, grouped early/late) ---
H0 [km/s/Mpc]   Early-universe vs late-universe + this work       
           60        65        70        75        80           
            |Planck   |         |    SH0ES|                       
            | band    |         |    band |                       
 Planck 2018|--o--|   :         :         :     CMB   (blue grp)  
 DESI BAO   | |--o--| :         :         :     BAO              
 ----------- - - - - -:- - - - -:- - - - -:- - - - - - - - - - -  
 SH0ES Ceph :         :         :  |--o--|      ladder(rose grp) 
 TRGB CCHP  :         :     |--o--|     :       ladder           
 GW170817   :     |------o------|       :       siren            
 LVK dark   :  |----------o----------|  :       dark siren       
 ===========:=========:=========:=========:===================== 
 THIS WORK  :         :     |--O--|     :   EMRI  [ROSE, bold]   
            point=MAP, bar=68% CI; vertical bands=Planck/SH0ES
```

---

### 3.2 Direction B — **HORIZON**

> *Observatory-grade, data-forward figures that read as a confident result across print, slide, and a dark web variant.*

**Philosophy.** HORIZON treats every figure as a press-ready result from a collaboration, not a lab notebook. The data layer is bold and high-contrast; everything else (grid, spines, reference context) recedes to a calm gray scaffold so the measurement is the loudest thing on the page. It commits to one semantic color law applied identically across paper, talk, and web — H₀ always navy, the mass channel always gold, truth always a warm reference rule — and it ships a genuine dark variant so the same figure works in a journal column and a dark-room slide without recoloring. The aesthetic borrows directly from LVK/observatory result figures: heavier headline strokes, saturated-but-CB-safe accents, full-height reference bands, and an annotation layer that states the number on the plot.

**Palette (hex table).**

| Hex | Role |
|---|---|
| `#1B2A4A` | Ink / data series 1 (Without M_z, headline) — observatory navy; on dark variant → `#EAF0FB` foreground |
| `#E8A317` | Signal gold / series 2 (With M_z) — high-lightness accent, the bold-data hero color |
| `#C2451E` | Truth / injected H₀ — warm vermillion rule, ONLY for the known-truth line |
| `#3E7CB1` | Reference family A (Planck / early-universe) — mid cyan-blue band |
| `#9A6FB0` | Reference family B (SH0ES / late-universe) — muted purple band |
| `#4F4F4F` | Scaffold gray — axes, ticks, secondary text, grid |
| `#222222` | Near-black text — titles, panel labels, primary annotations |
| `#F2EFE9` | Warm paper tint (talk/web cards; paper PDF stays pure white); dark variant bg `#0E1726` |

**Sequential colormap.** `cividis` for heatmaps with explicit `vmin/vmax` (robust-clip) or `LogNorm`; `magma` is the high-drama escape hatch but is less B/W-monotone, so heatmaps lean `cividis`. Mark empty bins with explicit gray `set_bad`; overlay the 0.5/0.9 horizon contour. True log axes via `pcolormesh + set_yscale('log')`.

**Typography.** Paper: `usetex` Latin Modern Roman (body-match), `mathtext.fontset: cm` fallback. Strict 2pt ladder (panel 9pt bold, axis 8pt, tick 7pt `#4F4F4F`, data label 8pt semibold in series color). Bold reserved for one thing per figure. Talk: sans, every size ×1.8, strokes ×2. Web: Source Sans / system, px ladder, palette exported as CSS custom properties (`--horizon-navy:#1B2A4A …`). One convention locked: sans figure labels on a serif body (the Nature-modern signal).

**Layout.** One base mplstyle (`emri_horizon.mplstyle` v2) + three thin overrides routed through `apply_style(theme=...)`. Column-exact sizing via true `\columnwidth`. Small-multiples `sharex/sharey`, outer-edge labels only, one shared colorbar. Bold `(a)(b)` top-left at fixed offset. Heavy reference scaffold: a single full-height band style for Planck/SH0ES reused everywhere. Annotation strategy is the spine — active titles, inline MAP±HDI, named reference rules, direct color-matched end-of-line labels. `constrained_layout` only.

**Treatment principles.** Area-normalized PDFs everywhere for single/headline posteriors; peak-norm only for many-variants overlays. Nested 68/95% HDI shaded under curve, one hue, graded alpha (replaces boundary axvlines and full-height axvspans). No per-point markers (one filled diamond at MAP). Variant law: navy solid vs gold dashed (opposite hue AND lightness). Reference law: Planck `#3E7CB1`, SH0ES `#9A6FB0` full-height bands labeled at top; truth `#C2451E` rule; reference colors never reused for data. Headline series gets the heaviest stroke (1.8pt) and highest z-order; scaffold 0.6-0.8pt gray. Confidence through weight hierarchy. CVD + grayscale gate before shipping.

**Paper vs interactive.** Paper: v2 artifacts only — `emri_horizon.mplstyle` + `paper.mplstyle` override + `_colors` v2 re-exporting the VARIANT/REFERENCE/TRUTH law, routed through existing `apply_style()` and factories (no data plumbing touched). Consolidate the 4 posterior paths and 2 convergence figures; split dashboards; strip `tight_layout`; route every label through LABELS; reconcile Mpc/Gpc. Add the 4 missing figures. Interactive: one shared `go.layout.Template` in all 8 factories with the exact same hex law from `_colors` v2 as CSS custom properties; `include_plotlyjs='directory'`; Observable Framework shell, KaTeX, Python JSON loaders. Replace fragile visibility-vector bookkeeping with per-group traces. A dark variant ships for slides/embeds. Every interactive degrades to the static PDF.

**Signature new figure.** The H₀ FOREST / TENSION plot — ~10-14 horizontal rows grouped early (navy) vs late (gold), Planck (`#3E7CB1`) and SH0ES (`#9A6FB0`) full-height bands, THIS WORK's EMRI result the bottom row in bold (navy fill, gold edge, lw2, tight CI) with its number annotated inline. The figure a referee and a defense committee both look for and the set entirely lacks. Natural interactive twin: hover-for-citation + accumulation animation.

**Trade-offs.** Bold/saturated + heavier strokes risk reading as "marketing" in a conservative PRD column — keep the scaffold genuinely quiet and bold ONLY data + headline number; the dark variant must never be the print default. Two themes × `usetex` doubles the rendering-path test surface (CI needs TeX; `mathtext.fontset: cm` fallback isn't pixel-identical). Forest literature values must be sourced and dated carefully. Consolidating 4 posterior factories into 1 is the riskiest refactor. Observable Framework adds a JS toolchain (fallback: stay on Plotly with `include_plotlyjs='directory'`). Cividis is slightly less "punchy" than the bold brand wants for hero heatmaps.

**ASCII mockups (verbatim).**

```
--- H0 posterior (headline, paper theme) — area-normalized PDF, nested 68/95% HDI shaded under curve, full-height reference bands, inline MAP annotation, direct labels, no legend box ---
  EMRI dark sirens recover H0 to +2.7/-2.5 km/s/Mpc (68% HDI)
  p(H0 | data)                                                  
   |        Planck:           SH0ES:                            
   |        [3E7CB1]          [9A6FB0]                           
   |          ::               ||                               
   |          ::    /\  <- H0 = 73.0 +2.7/-2.5   ___ Without Mz 
   |          ::   /::\ navy, lw1.8                   (navy)     
   |          ::  /::::\                                         
   |          :: /######\ <-68% HDI (dark fill)   - - With Mz   
   |          ::/########\                            (gold)    
   |         _//#########%\__ <-95% HDI (light)                 
   |     ___/:: ##########%%\___                                
   |  __/    ::               %%%\____                          
   +--+------++------+------+------+------+--- H0 [km/s/Mpc]     
     60      67|    70     73|R    76     83                     
            Planck         truth (C2451E, dashed)               
```

```
--- NEW H0 forest / tension plot (signature figure) — measurements as point + 68% CI rows grouped early/late, Planck+SH0ES full-height bands, this EMRI result bold at bottom ---
  The Hubble tension, with this work                          
        Planck band[3E7CB1]    SH0ES band[9A6FB0]              
              :::                  |||                          
  EARLY  Planck 2018    |-o-|      :::          67.4+/-0.5     
  (navy) DESI+BBN      |--o--|     :::          68.5+/-0.6     
         ----------------:::-------|||---------------------    
  LATE   SH0ES Cepheid       :::  |--o--|       73.0+/-1.0     
  (gold) TDCOSMO lens        ::: |---o---|      74.0+/-1.8     
         GW170817 siren  |------- o -------|    70  +12/-8     
         LVK dark+cat   |--------o--------|     76.6 +13/-9    
         ----------------:::-------|||---------------------    
  >> THIS WORK (EMRI)        :::|=O=|   <bold>  73.0 +2.7/-2.5 
         navy fill, lw2, gold edge -- the result row           
         +---------+----:::---+---|||--+---------+--- H0        
        60        67   :::   70    73  ||   80              140 
```

```
--- 2x2 campaign dashboard (talk/web theme) — unified panel-label system, shared SNR colorbar, one accent law, no debug-montage clash ---
  EMRI H0 CAMPAIGN  -- 990 events, h_true=0.73          [dark variant ok]
  +-----------------------------+-----------------------------+
  |(a) H0 POSTERIOR             |(b) SNR DISTRIBUTION         |
  |        /\  navy             | ||                          |
  |   ___ /##\ ___  73.0        | ||||___       log-y tail    |
  |  /    ####    \             | ||||||||___                 |
  |_/  68/95 HDI   \__          | |||||||||||||___ rho>=20    |
  +-----------------------------+-----------------------------+
  |(c) DETECTION YIELD vs z     |(d) SKY MAP (cividis, log)   |
  |  injected ---- (gray)       |   . :  cividis SNR, shared  |
  |  detected ####(navy fill)   |  : .:: . colorbar w/ (b)    |
  |  ___                        | .::.. : .:  ecliptic graticule|
  | /   \___ fraction (gold)    |  : .  ::. .                 |
  +-----------------------------+--------------[SNR 20###45]--+
   shared accent law: navy=detected, gray=injected, gold=fraction
```

---

### 3.3 Direction C — **Dark Siren Dispatch**

> *Every figure is a headline: one sentence, one number, one finding — read it without the caption.*

**Philosophy.** Dark Siren Dispatch treats each figure as visual journalism in the Pudding/Distill/NYT tradition: the conclusion lives in an active title, the key number is annotated directly on the data, and the reader never decodes a legend. Generous whitespace, direct-labeled series, and a restrained annotation layer turn diagnostic plots into explanations. The result reads modern because the chart stops being "here is the data, go figure it out" and becomes "here is what we found, here is the evidence" — accessibility-first, with the paper figures as the disciplined, annotation-trimmed subset of the richer thesis-and-web story.

**Palette (hex table).**

| Hex | Role |
|---|---|
| `#0072B2` | PRIMARY / Without M_z — Okabe-Ito blue, solid |
| `#E69F00` | VARIANT / With M_z — Okabe-Ito orange, dashed (replaces broken second blue `#56B4E9`) |
| `#222222` | HERO / combined-posterior line + body text — near-black |
| `#009E73` | TRUTH / injected H₀ = 0.73 — Okabe-Ito green rule, labeled inline |
| `#BB5566` | TENSION ACCENT / SH0ES late-universe + this-measurement highlight — Tol high-contrast red |
| `#004488` | EARLY-UNIVERSE / Planck CMB family + reference bands — Tol deep blue |
| `#707071` | CONTEXT / reference bands, secondary annotations, scaling-law guides, de-emphasized spaghetti |
| `#414042` | AXIS / tick labels and axis lines — warm dark gray |
| `#EFE9DD` | PAPER GROUND (web/talk only) — warm editorial canvas; paper PDF stays pure white |

**Sequential colormap.** `cividis` single project default (print-first, near-linear luminance, perceptually uniform). Fixes wasted dynamic range when paired with `LogNorm` or robust-percentile clipping. Residual/bias maps use `cm.vik` with `TwoSlopeNorm(vcenter=0.73)`. Always overlay the semantic contour (P_det = 0.5 horizon) as a redundant grayscale-safe channel. `set_bad('#DDDDDD')` marks empty bins.

**Typography.** Paper: keep `pdf.fonttype 42`; switch `usetex` path to Latin Modern Roman; `mathtext.fontset: cm` fallback. Strict 2pt ladder (panel 9pt bold `#222222`, axis 8pt, tick 7pt `#414042`, data label 7-8pt semibold in series color, annotation 7pt `#707071`). Web: humanist sans `'Source Sans 3','Open Sans',Helvetica,Arial,sans-serif`; ONS px ladder (active title 22px/700, dek 16px/500 gray, axis 14px/400). KaTeX for math. Talk: web sans, every size ×1.8, 2.5px hero line.

**Layout.** 12-column editorial grid mental model. Single-column figs sized to exact `\columnwidth` (3.40in); double to `\textwidth` (7.06in). Every chart reserves a fixed TOP STRIP for active title + dek and a RIGHT GUTTER (~18%) for direct labels and the inline result number — annotations live in deliberate whitespace, never over data. Small-multiples are the default for sweeps/per-event/per-seed: shared x AND y, outer-edge tick labels only, panels sorted by quantity of interest. Bold `(a)(b)` top-left, identical offset. `constrained_layout` everywhere. Annotation = restrained pointers only (thin leader line, single dot, faint highlight band — never arrow+box+shadow).

**Treatment principles.** ACTIVE TITLE STATES THE FINDING. NUMBER-ON-THE-CURVE near the peak. Area-normalized PDF + shaded HDI for the ONE headline figure; peak-norm only for many-curve overlays (never shade CIs under them). DIRECT-LABEL, KILL LEGENDS. REDUNDANT ENCODING ALWAYS (color + linestyle + direct label). No per-point markers on smooth/peaked curves. FIX THE TWO-BLUES SYSTEMICALLY in `_colors` v2 (orange variant, distinct `#707071` reference). ONE PERCEPTUAL SEQUENTIAL MAP (`cividis`) with `set_bad` + LogNorm. STANDARDIZE ON HDI; one shading style; annotate the interval in words. REFERENCE VALUES ARE NAMED CONTEXT, NOT DATA. ANNOTATE SCALING LAWS ON THE CURVE. SPLIT DASHBOARDS, DON'T SHRINK THEM.

**Paper vs interactive.** Paper is the DISCIPLINED SUBSET of the editorial language: same palette, direct-labeling, number-on-the-curve, HDI shading — but the annotation layer is trimmed to journal restraint and the active title moves into the REVTeX caption's first sentence (PRD wants neutral figure interiors). Deliverable = v2 `emri_thesis.mplstyle` + v2 `_colors.py` + revised factory bodies; vector PDF, ≥7pt, CB+grayscale verified via `colorspacious`. Consolidate the 4 posterior paths; split the dashboards; unify the convergence figures. Add the four missing factories following the existing contract. Interactive: GH-Pages becomes an EXPLORABLE EXPLAINER built with Observable Framework (Plot/Vega-Lite/D3/KaTeX under one theme mirroring the mplstyle hex-for-hex). Python loaders precompute at build time. Insight-bearing interactions only: h-sweep slider, estimator toggle, SNR/z brush cross-filter, HOPs uncertainty animation. Retire per-figure CDN dumps.

**Signature new figure.** The H₀-in-context FOREST / TENSION plot with an interactive companion. ~10-12 rows (Planck/DESI early `#004488`; SH0ES/TRGB/H0LiCOW late `#BB5566`; GW170817; LVK dark+GLADE+), full-height Planck/SH0ES bands, THIS EMRI result the bold highlighted bottom row. Answers "so what?" in one glance. Web version adds accumulation animation + hover-for-citation — the literal embodiment of the editorial thesis.

**Trade-offs.** (1) The editorial annotation layer is LABOR, hand-tuned per figure, can collide at REVTeX width — mitigated by reserved title-strip + right-gutter whitespace. (2) Finding-as-title risks editorializing in a physics paper — mitigated by keeping paper interiors neutral and moving the claim into the caption lead (paper and web diverge slightly in voice by design). (3) Forest plot requires curated literature values. (4) Observable Framework is a new dependency vs the `fig.write_html` one-liner. (5) Latin Modern + `usetex` needs TeX in CI. (6) HDI on a near-delta posterior can look thin (mitigated by inset zoom or stated grid resolution). None touches data plumbing.

**ASCII mockups (verbatim).**

```
--- Headline H0 posterior (canonical, paper theme) — area-normalized PDF, shaded nested HDI, number-on-the-curve, named reference rules, direct labels, no legend box ---
EMRI dark sirens recover H0 to within 4.2% (68% HDI)             
 Combined posterior over 990 detected events, injected h = 0.73     
                                                                    
 p(h | data)        Planck      truth      SH0ES                    
   |                  :           |          :                      
   |                  :          .#.         :                      
   |                  :         .###.        :   H0 = 0.738         
   |                  :        .#####.       :    +0.031/-0.028     
   |       (with M_z, orange dashed)         :   ___ With M_z       
   |                  :      .#########.     :  /                   
   |             ░░░░░░░░░░▓▓▓███████▓▓▓░░░░░░░░  68% HDI (dark)    
   |          ░░░░░░░░░░░░▓▓▓█████████▓▓▓░░░░░░░░ 95% HDI (light)   
   |________░░░____:____________________:___░░░____ Without M_z    
     0.62   0.66       0.70   0.74   0.78      0.82                 
                      H0 / 100  [km/s/Mpc]                          
```

```
--- NEW H0 forest / tension plot (signature) — this measurement vs the literature, grouped early/late, Planck+SH0ES vertical bands, EMRI result bold at the bottom ---
Where this EMRI measurement lands in the Hubble tension          
 Point + 68% CI; bands = Planck (blue) & SH0ES (red) 1-sigma       
            Planck band ░         SH0ES band ▒                     
 EARLY  Planck 2018      ░├─●─┤░                                   
        DESI BAO         ░├──●──┤░                                 
        ----------------------------------------------            
 LATE   SH0ES Cepheids        ▒    ├──●──┤▒                        
        TRGB                ▒  ├───●───┤  ▒                        
        H0LiCOW lensing     ▒    ├────●────┤▒                      
 SIRENS GW170817 bright   ├──────────●──────────┤                 
        LVK dark+GLADE+      ├────────●────────┤                  
        ════════════════════════════════════════════            
 ►THIS   EMRI dark siren      ░  ▒  ┣━━●━━┫  (bold, #BB5566)       
        ----------------------------------------------            
        66    68    70    72    74    76    78   H0 [km/s/Mpc]    
```

```
--- PP-plot / coverage (NEW, referee-expected) — bilby-style, nested grey 1-2-3σ bands, per-parameter lines, KS p in direct labels ---
Posterior is well-calibrated: every parameter tracks the diagonal
 Fraction of injections in C.I. vs credible level; grey = 1-2-3sigma
  1.0|                                       ╱▒▒                   
     |                                  ╱░▒▒▒  · H0  p=0.62        
     |                             ╱░░▒▒▒/                         
 0.6 |                        ╱░░░▒▒/                              
     |                   ╱░░░▒▒▒/      (lines inside band = good)  
     |              ╱░░▒▒▒/                                        
 0.2 |         ╱░░▒▒/                                              
     |    ╱░░▒▒/                                                   
  0.0|_▒▒╱___________________________________                     
     0.0       0.2     0.4     0.6     0.8    1.0                  
              credible interval (combined KS p = 0.41)            
```

---

### 3.4 Direction D — **SIREN ATLAS**

> *The explorable drives the page; the PDF is a still frame of it. One data cube, one design-token set, two renderers.*

**Philosophy.** SIREN ATLAS treats the GitHub-Pages explorable as the primary artifact and the REVTeX PDF as a deterministic "still" projected down from the same design tokens and the same precomputed snapshots. The signature interaction is brushing-and-linking: brush an SNR or redshift range and watch the sky, the per-event spaghetti, and the stacked H₀ posterior re-weight together, so "where does the constraint come from" becomes a gesture rather than a sentence. The visual language is calm and annotation-led, so a reader who never touches a slider still gets the conclusion from a static frame. Modernity comes from the architecture (Python loaders → JSON snapshots → Observable Framework shell with Plot/Vega-Lite/D3 per figure) and from a single token file exported both as an mplstyle v2 and as CSS custom properties.

**Palette (hex table).**

| Hex | Role |
|---|---|
| `#0072B2` | Variant A / Without M_z — Okabe-Ito blue, solid |
| `#E69F00` | Variant B / With M_z — Okabe-Ito orange, dashed (the two-blues fix) |
| `#1a1a1a` | Headline / combined posterior — heavier near-black |
| `#009E73` | TRUTH — injected H₀ = 0.73 rule; reserved, never data |
| `#7A6FAE` | Planck 2018 reference band — desaturated violet, context-only |
| `#B0AFAE` | SH0ES reference band — neutral warm gray, context-only |
| `#CC79A7` | ACCENT / 3rd series or "this work" highlight — Okabe-Ito reddish purple |
| `#222222` | Text / axis labels — near-black |
| `#6E6E6E` | Tick labels, secondary annotation, leader lines — mid gray |
| `#D9D9D9` | HDI/credible fill base + no-data bins (`set_bad`) |

**Sequential colormap.** `cividis` default, `magma` for high-dynamic-range posterior surfaces, diverging (`cm.vik` or `coolwarm` + `TwoSlopeNorm(vcenter=0.73)`) for bias/residual maps. The documented SNR/p_det failures are NORMALIZATION failures, not colormap-choice failures — the fix is `LogNorm` / robust percentile clipping / explicit `vmin-vmax`. `cividis` + correct norm + a 0.5/0.9 contour layer is the field-standard look.

**Typography.** Paper: `usetex` Latin Modern Roman + true LaTeX math; `mathtext.fontset=cm` fallback. Strict 2pt ladder at final size. Figures sized to exact `\columnwidth`. Web: Source Sans / Open Sans + KaTeX (math matches the paper character-for-character); px ladder title 20px/700, subtitle 16px/600, axis 14px/400, floor 12px. Talk: sans, ~1.8× the print ladder. Same hex tokens drive all three; only fonts/sizes/weights differ per override.

**Layout.** 12-column responsive web grid (max ~1100px), each explorable a self-contained scene card; controls docked top-right in a consistent control rail, never over data. Linked-view scenes use a fixed L-shape: driver panel (histogram or sky) top-left, stacked posterior the dominant right panel, per-event detail bottom strip. Paper projection: factories unchanged; composition via `get_figure` REVTeX presets. Small-multiples `sharex/sharey`, interior ticks stripped, one shared colorbar, panels sorted by quantity of interest, bold `(a)(b)`. `constrained_layout` everywhere. Annotation = active titles, inline MAP±HDI, named reference rules, direct end-of-line labels.

**Treatment principles.** No per-point markers (one faint MAP marker). Area-normalized PDFs by default; peak-norm only for explicit overlays (never shade CIs under them). Nested 68/95% HDI fills of one hue (alpha 0.30/0.15), boundary line as redundant channel; one CI definition via `compute_hdi_interval`. Two-series always color + linestyle + direct label (blue solid vs orange dashed); REFERENCE reassigned off `#56B4E9`. Reference values reserved muted context style. `cividis` + explicit norm + `set_bad('#D9D9D9')`. Heatmaps use `pcolormesh` + true log axes + 0.5/0.9 horizon contour. Direct-label, drop legend boxes. Dashboards become single-story paper figures; the composite survives only as a web/talk variant. CVD + grayscale gate. **One data snapshot feeds both renderers** — matplotlib renders the PDF still, Plot/Vega/D3 render the web from the identical numbers (no recomputation, no drift). Interactions answer a real thesis question only.

**Paper vs interactive.** Paper (Theme A) is a thin override on the shared token base: a v2 `emri_thesis.mplstyle` generated FROM the token file, plus `apply_style(theme='paper')` enabling `usetex` + Latin Modern + column-exact figsize. Factory architecture and canonical data plumbing untouched. Consolidate the quadruplicate posterior paths and the two convergence figures. Every paper figure is the deterministic static still of its web scene, rendered from the same JSON snapshot. Interactive: replace standalone Plotly dumps with an Observable Framework site (GH-Pages via Actions, no server, no CDN dependence). Python loaders run at build time → JSON/Arrow snapshots (the phase45 h-sweep cube is exactly this shape). Per-figure engine by need: Observable Plot for routine scenes; Vega-Lite for linked/brushed scenes (declarative selection algebra); D3 + Aladin Lite for the 1-2 hero explorables. One Framework CSS theme imports the same hex tokens; KaTeX for equations. The hand-authored `index.html` and drift-prone footer replaced by file-based routing with provenance from the snapshot manifest. Residual Plotly (3D only) uses `include_plotlyjs='directory'`. Highest-ROI first: an H₀ slider over the already-precomputed h-sweep.

**Signature new figure.** "WHERE DOES THE CONSTRAINT COME FROM" — a linked-view explorable (web) that doubles as a 3-panel static figure (paper). Driver: an SNR×redshift histogram you brush. Linked: (1) the Mollweide sky map (`cividis`, `LogNorm` SNR, styled graticule + ecliptic line) showing only brushed events; (2) the per-event spaghetti de-emphasized to faint gray with the brushed subset highlighted; (3) the stacked H₀ posterior rebuilt LIVE from only the brushed events, with its HDI shrinking/shifting as you widen the brush. Finally surfaces fig02's latent `color_by` population story (which `main.py` never enabled). Companion: the H₀ FOREST/TENSION plot.

**Trade-offs.** (1) Observable Framework is a genuinely new web stack; the 1-2 D3 hero explorables are high-effort bespoke code — mitigation: ship the Plot/Vega-Lite scenes first (the h-sweep slider is nearly free given the precomputed cube), treat D3 heroes as stretch goals. (2) Interactive-first means more upfront work at the data-loader/snapshot boundary; a loader schema change forces revalidating both renderers (but this is also the safeguard against drift). (3) Building the PDF as a projection of the web design constrains the paper to what both media express. (4) Forest plot requires curated literature values. (5) Risk that the explorable eclipses the paper figures — mitigated by the hard rule that every scene degrades to a self-sufficient static still. (6) Latin Modern + `usetex` needs TeX in CI; the fallback yields a slightly different (still good) look.

**ASCII mockups (verbatim).**

```
--- H0 posterior (headline) — area-normalized PDF, shaded nested HDI, named reference rules, inline annotation, direct labels (no legend box). Web version adds a 'stack N events' slider; this still is the N=all frame. ---
  EMRI dark sirens recover H0 to +2.8/-2.4% (68% HDI)        [scene: posterior]
  p(H0 | data)                                  Planck  truth  SH0ES
    |                                              :       ¦      :
    |                        .-=*#%@%#*=-.         :       ¦      :
    |                     .=#@@ Combined @@#=.     :       ¦      :  <- black, heavy
    |                   =#@@@##############@@#=    :       ¦      :
    |        Without Mz / @@@:::::68% HDI:::::@@ \ With Mz  ¦      :
    |   #0072B2 solid _.-'@@@:::::::::::::::::::@@'-._ #E69F00 dash :
    |   _____....----''  @@:::::::::::::::::::::::@@ ``----...._____ :
    |  ::::::::::95% HDI:::::::::::::::::::::::::::::::95% HDI::::::::
    +----+--------+--------+--------+--------+--------+--------+----->
      0.64     0.67     0.70    [0.73]    0.76     0.79     H0/100
   note: H0 = 0.731 +0.020/-0.018  | curves area-normalized | 68/95% HDI shaded
   web: [<==slider: stack 1..990 events==>]  watch the peak sharpen + lock to truth
```

```
--- H0-in-context FOREST / tension plot (the SIGNATURE new figure) — early-universe vs late-universe color blocks, Planck+SH0ES full-height bands, THIS WORK bolded at the bottom. Web version: hover a row for method/citation, brush a tension region to highlight. ---
  H0 in context                          Planck band ::   :: SH0ES band
  EARLY (CMB/BAO) #7A6FAE   LATE (ladder) #B0AFAE   THIS WORK #CC79A7
  Planck 2018 (CMB)          ::|-•-|::        |          :        67.4
  DESI BAO+BBN               ::|--•--|:       |          :        68.5
  TDCOSMO lensing            :        |    |---•---|     :        74.0
  TRGB (CCHP)                :        |  |---•---|       :        72.0
  SH0ES Cepheid              :        |       |  |-•-|   :        73.0
  GW170817 bright siren      :     |---------•---------| :        70  (wide)
  LVK GWTC-4 dark+catalog    :   |------------•----------|:       76.6
  ----------------------------------------------------------------------
  >> THIS WORK (EMRI, 2D) << :        |     |•|          :  0.731  <- bold #CC79A7
  ----------------------------------------------------------------------
    65        68        71        74        77        80   H0 [km/s/Mpc]
   web: hover row -> method+dataset+arXiv ; brush [70..74] -> rows inside light up
```

---

## 4. Per-Figure Redesign Recommendations

These are largely direction-independent improvements — the palette / typography differ per direction, but the chart-type and encoding changes hold across all four.

| Figure | Current | Proposed chart type / encoding | Why |
|---|---|---|---|
| H₀ posterior (fig01, paper_h0, paper_h0_kde, m_z panel, fig08-left) | 4 code paths: peak-norm / marker-on-every-point / KDE / axvspan | **ONE canonical factory:** area-normalized PDF, shaded nested 68/95% HDI under curve, inline MAP±HDI, named reference bands, direct labels, `theme=` + options | Kills the quadruplicate-drift hazard; adopts the field-standard normalized-PDF + shaded-HDI convention; one CI definition |
| `fig02_event_posteriors` | Monochrome spaghetti haze (color_by never enabled) | De-emphasize curves to faint gray, default `color_by='snr'`, hero combined line with shaded HDI, top SNR/z rug | Makes the "where does the constraint come from" story visible; doubles as the population view |
| Convergence (`fig08` + `paper_convergence`) | Linear vs log-log, two N-grids, `\%` bug | **ONE log-log factory:** shaded 16/84 band, markers at medians only, inline-annotated fitted slope, unified N-grid | Same scaling law must look the same; reveals N^-1/2 as a straight line; fixes the label bug |
| `paper_single_event` | 4×2 peak-norm grid + markers | Overlay both variants per event (4 panels), filled likelihoods, annotated 68% width per panel, shared x + single legend | Reclaims dead whitespace; makes the M_z narrowing quantitative; removes marker clutter |
| `paper_snr_distribution` | Linear hist + viridis scatter; cluster-path fallback | Log-y (or log-log) hist (power-law tail), hexbin / robust-clipped `cividis` scatter, explicit horizon line; warn-not-draw on missing data | Reveals the SNR tail; recovers dynamic range; removes environment-specific artifacts from a publication factory |
| `fig06_fisher_ellipses` | Patches, no σ key, no center | Filled 68%+95% contours from covariance, truth crosshair + center marker, σ/ρ annotated, equal aspect, σ legend | Makes the degeneracy story legible; analytic = noise-free; unifies with corner aesthetic |
| `fig07_corner_plot` | corner.py, MC samples, truths=mean | Styled triangle (analytic ellipses), filled 68%+95% (stated in caption), real injected truths, house typography, lower triangle only | Removes MC noise + seed dependence; modern filled-contour look; informative crosshairs |
| `fig13` + `fig10` (strain + PSD) | Two near-duplicate noise figures, divergent styling | Merge into one 2-panel noise figure (PSD + h_c), shaded floor, real/labeled-schematic EMRI track, direct line labels, region annotations | Cuts duplication; pedagogical noise decomposition; honest EMRI track |
| `fig12_uncertainty_violins` | violin/bar dispatch, no reference | One chart type (horizontal violins), 1%/10%/100% reference bands, family color + label, sorted by precision | Removes the hidden dispatch; adds interpretive anchors; ranks best/worst-measured params |
| `fisher_quality_diagnostic` | Calls save_figure, returns None, plasma | Return `(fig, axes)`; `cividis`; named axis labels; reframe panel 2 as a degeneracy-population view | Restores the factory contract; house cmap; reader-facing labels |
| `fig09_detection_efficiency` | Step + Wilson band, no anchors | Add 50% guide + SNR-threshold annotation, adaptive/quantile bins, counts rug, fold into selection explainer | Anchors the selection threshold; tames noisy high-z bins |
| `fig04_detection_yield` | twinx, detected==injected | Fix call site; distinct injected/detected colors; ratio panel stacked above hist with Wilson band | Shows real yield; removes twin-axis clash; harmonizes with efficiency figure |
| `fig14_crb_coverage` | 3D scatter | 2D marginal / corner pair-grid with density shading, `get_figure` preset + LABELS | 3D is occlusion-prone and unreadable in vector PDF; feeds the population view |
| `fig19_info_monotonicity` | scatter + identity, linear | Color by sign of (w_with − w_no), log-log + hexbin, marginal histograms, shaded improvement region, annotated median | Highlights violators; handles the concentrated cluster; connects to where constraints originate |
| `fig20_pdet_surface` | imshow fake-index, viridis | `pcolormesh` + true log y-axis, robust/`cividis` norm, 0.5/0.9 horizon contours, `set_bad`, injected-population overlay, Mpc units | Physically faithful mass spacing; explicit horizon; doubles as selection explainer; fixes unit bug |
| `fig05_sky_localization` | linear-viridis Mollweide, transData ellipses | `LogNorm`/robust SNR color, hexbin density mode, projected localization contours (or flat inset), styled labeled graticule + ecliptic line | Recovers SNR dynamic range; fixes the geometrically-wrong ellipses; declares the coordinate frame |
| `fig11_distance_redshift` | overplotted near-degenerate curves | Add lower residual panel (d_L(h)−d_L(0.73))/d_L(0.73), emphasized fiducial, sequential ramp for off-fiducial h, direct labels | Makes the H₀ sensitivity that drives the inference actually visible |
| `fig15_campaign_dashboard` | 4 pasted panels, no labels | Unified theme: (a)-(d) labels, shared SNR colorbar across hist + sky, one accent law, rebalanced layout (full-width sky strip) | Turns a debug montage into a designed talk/web asset |
| `fig16_catalog_completeness` | schematic sigmoid + dual legend + twinx + box | Mark reference as schematic (gray) or cite empirical; two-row small-multiple (counts / coverage sharing x); variant color+texture; box → caption | Removes the misleading-as-real risk; kills twin-axis ambiguity; standardizes units |
| `fig17_single_event_detail` | 2×3, two-blue, peak-norm, hardcoded size | Paired/dumbbell chart keyed on galaxy_id; normalized L(h) + shaded HDI; REVTeX presets; constrained_layout; count boxes → annotations | Honest per-host comparison; removes redundant scatter; matches the rest of the document |
| EMRI population (dist / sampling / rate) | Inconsistent colorbars, raw labels | Route colorbars through `make_colorbar` w/ density label; share norm between model contour + sampled hist (or overlay); LABELS for rate; lighter contourf + 50/90% mass-fraction lines | Lets "does the sampler reproduce the model" be judged; basis for the population view |
| Catalog raw distributions | hist, no expected curve | Overlay theoretical mass-function / dV/dz reference; neutral catalog color; step histograms; verify volume label; combine into a small-multiple | Turns descriptive histograms into checks; lighter vector output |
| CRB heatmap (eval diagnostics) | raw covariance, viridis | Normalize to correlation matrix, diverging cmap centered at 0, route through `get_figure` | Structure visible regardless of parameter scale |
| Uncertainty violins (eval, duplicate) | second divergent impl | Consolidate into the single styled house factory | Removes the two-impl divergence |
| Sky-localization 3D | 3D scatter | Mollweide / 2D localization-ellipse view consistent with fig05 | 3D anti-pattern; consistency |
| detection_contour vs pdet_surface | overlapping (z/d_L, M) views, inconsistent axes | Unify into one axis convention with optional contour overlays | Removes overlapping content |
| All interactive (8) | per-figure styling, two-blues, cdn | Shared `go.layout.Template`, v2 palette as CSS custom properties, `include_plotlyjs='directory'`, per-group traces, KaTeX titles | One web design language; self-contained; archival-robust; kills fragile bookkeeping |
| `interactive_sky_map` | viridis, 1473 markers ~318 KB | `LogNorm`/quantile color, density layer toggle, second encoding (size=d_L), styled graticule, downsample/bin | Recovers dynamic range; cuts payload; surfaces depth-SNR relationship |
| `interactive_h0_convergence` | linear-linear, no band | log-log right panel + inline slope, bootstrap band, sequential N ramp tied across panels, anchor ref to largest-N | Reads the √N law instantly; shows sampling scatter |
| `interactive_m_z_improvement` | fake text-subplot, brittle visibility | Layout-annotation block / HTML caption; cleaner trace model; single headline annotation; named threshold constant | Removes the `annotations[:3]` fragility and manual offset bookkeeping |
| `interactive_single_event_detail` | bare-rank bars sorted differently | Dumbbell/paired chart keyed on galaxy_id; per-event trace groups; consistent L(h) scale; galaxy-ID ticks | Honest comparison; hardens against partial-load desync |
| `interactive_closure_test_overlay` | off-brand defaults, hardcoded range | CYCLE/sequential-by-h_true colors, truth vline color-matched to curve, (MAP−h_true) annotation, data-driven axis range | On-brand; pairs peaks to injections; quantitative closure |
| `interactive_catalog_completeness` | REFERENCE collides w/ with-cut line | Distinct coverage color, plot both with/without coverage, IQR band, gap empty bins, explicit fallback note | Completes the host-pruning story; removes false color relationship |

---

## 5. Creative NEW / Reframed Figures

All five are buildable on the existing data plumbing — `compute_hdi_interval` already yields the band, the phase45 h-sweep cube already exists, per-event SNR/z and the p_det surface are already computed. None requires a new data pipeline; each is a new factory following the `data in → (fig, ax) out` contract.

### 5.1 H₀ forest / tension plot (the signature figure — currently absent)
- **What it shows.** ~10-12 horizontal point + 68% CI rows — Planck 2018, DESI BAO, SH0ES Cepheids, TRGB, TDCOSMO/H0LiCOW lensing, GW170817 bright siren, LVK GWTC-4/5 dark+combined — grouped early-universe vs late-universe with a separator rule, full-height Planck and SH0ES 1σ bands behind everything, and THIS EMRI dark-siren result as the bold highlighted bottom row.
- **Why it matters.** The single most-expected cosmology figure the field demands and the pipeline entirely lacks; it positions the whole thesis result at a glance and answers "so what?". A referee and a defense committee both look for it.
- **Static form.** REVTeX single or double column; literature table hardcoded with dated citations; `compute_hdi_interval` for the EMRI band; grouped color blocks; full-height reference bands.
- **Interactive form.** Hover any row for method/dataset/arXiv citation; brush a tension region to highlight rows inside; accumulation slider showing the EMRI CI tightening as N events stack.

### 5.2 PP-plot / coverage (referee-mandatory for an injection-recovery thesis)
- **What it shows.** One cumulative line per parameter (empirical CDF of the credible level at which each injected truth falls) against the diagonal, with nested grey 1-2-3σ binomial bands and per-parameter KS p-values in direct labels plus a combined p-value.
- **Why it matters.** For any simulation-based-inference thesis, the PP-plot is effectively the proof of calibration referees treat as non-negotiable. The pipeline has no inference-coverage figure (existing "coverage" hits are catalog coverage / a 3D CRB scatter).
- **Static form.** Square axes [0,1]², bilby-style nested grey bands, colored per-parameter lines, KS p in legend/labels.
- **Interactive form.** Hover a line for the parameter's per-credible-level deviation; toggle parameters on/off; optionally a rank-histogram (SBC) companion view.

### 5.3 Selection-function / detection-horizon explainer
- **What it shows.** A p_det(d_L) survival curve (1→0, optionally multiple SNR-threshold/mass curves with a population-spread band) paired with a (M, d_L) p_det heatmap carrying the 0.5 (and 10/90%) horizon contour, with the injected population overlaid as faint scatter; optionally a true-vs-detected redshift histogram for the Malmquist story.
- **Why it matters.** The selection function is the heart of the bias story (the validated 1D 0.76→0.75, 2D 0.747→0.7375 result); the field expects a dedicated horizon figure and the pipeline only has it latently (pdet_surface + injected scatter is ~80% there).
- **Static form.** `pcolormesh` + true log axes + `cividis` robust norm + horizon contour + `set_bad`.
- **Interactive form.** H₀ (and SNR-threshold) slider driving the surface + implied horizon + Malmquist bias side-by-side; p_det-survival on/off toggle as a live before/after.

### 5.4 "Where do the constraints come from" population view
- **What it shows.** A driver SNR×redshift (or (z, M)) view; linked to the sky map, the per-event spaghetti (de-emphasized except the highlighted subset), and the stacked H₀ posterior rebuilt from the selected events — answering which events dominate the constraint.
- **Why it matters.** The question every dark-siren referee asks; currently exists only latently as fig02's never-enabled `color_by` option and the m_z host-count violin.
- **Static form.** 3-panel figure: SNR/z scatter (or the host-reduction violin) + sky map + stacked posterior, with the dominant-information region annotated.
- **Interactive form.** Brush the SNR/z histogram → cross-filter sky + spaghetti + stacked posterior live (Vega-Lite declarative selection). The hero linked-view explorable.

### 5.5 Interactive event → host → posterior explorable (hero scene)
- **What it shows.** Left: a sky patch with the GW localization contour and GLADE+ candidate hosts sized/colored by weight. Center: drag the event or scrub an h-slider; candidates re-weight live (galaxy z → predicted d_L at current h → likelihood vs the event's d_L posterior). Right: the single-event H₀ posterior rebuilt as the sum of per-galaxy contributions; hovering a galaxy highlights its contribution. Toggle: completeness correction on/off.
- **Why it matters.** Makes the entire dark-siren likelihood — host marginalization + completeness — legible in one manipulable scene; the figure that makes the method *understandable*.
- **Static form.** A 3-panel still (sky + weight bars/dumbbell + per-event L(h)) — essentially the redesigned `fig17` content arranged as the explorable's still frame.
- **Interactive form.** D3 (+ Aladin Lite for the sky) under Observable Framework; data precomputed in Python; the stretch-goal hero, shipped after the lower-effort Plot/Vega-Lite scenes.

---

## 6. Recommended Default Direction + Implementation Sketch

### Recommended default: **HORIZON, with Dark Siren Dispatch's annotation discipline folded in.**

HORIZON is the recommended default because it best fits a physics thesis aimed at a PRD/PRX-style submission *and* a defense + GH-Pages audience: it is observatory-grade and confident without editorializing the figure interiors, it ships a genuine dark variant for slides, and its semantic color law (navy H₀ / gold mass-channel / warm truth rule / cyan-Planck / purple-SH0ES) is the most directly LVK-legible. Fold in Dispatch's annotation-as-design layer (active titles in captions, number-on-the-curve, named reference rules, direct labels) for communication clarity. SIREN ATLAS's single-data-snapshot / two-renderer architecture is the right *long-term* web foundation and should be adopted for the interactive layer specifically, but the safe incremental path keeps the existing Plotly factories with a shared template first and migrates to Observable Framework as a follow-on. Quiet Index is the fallback if TeX-in-CI or the dark variant proves too costly — it is the most conservative, lowest-risk subset of the same spine.

### Implementation sketch (honoring the factory architecture — NO data-plumbing rewrite)

**a. `emri_thesis_v2.mplstyle` changes (deltas from the current sheet):**
- `image.cmap: cividis` (was `viridis`).
- `axes.prop_cycle`: keep Okabe-Ito *as the categorical cycle*, but the variant pair is governed by `_colors.py`, not the cycle.
- `mathtext.fontset: cm` (kills the DejaVu-math tell on the headless/CI path).
- Text/axis colors to near-black `#222222` (not pure `#000`); tick/secondary to scaffold gray.
- Keep `constrained_layout.use: True`, `legend.frameon: False`, `pdf.fonttype 42`, spines top/right off.
- Keep `figure.figsize` defaults but route real sizing through `get_figure` presets, retuned to exact REVTeX widths (single 246pt → 3.404in, double 510pt → 7.055in) so LaTeX never rescales.

**b. `_colors.py` v2 (the single highest-ROI edit):**
- `VARIANT_WITH_MASS = "#E8A317"` (gold) — or Dispatch's `#E69F00` — replacing `#56B4E9`.
- `VARIANT_NO_MASS = "#1B2A4A"` (navy) — or keep `#0072B2` for the Quiet/Dispatch palettes.
- `REFERENCE` reassigned off `#56B4E9` so it no longer collides with the variant; add reserved `PLANCK` / `SH0ES` band colors and a `TRUTH` rule color.
- Add `CMAP = "cividis"`, a `SEQUENTIAL` (magma escape hatch), and a `DIVERGING` (`cm.vik` / `coolwarm` with helper for `TwoSlopeNorm(vcenter=0.73)`).
- Add `set_bad` no-data color constant.
- Because every module imports from `_colors`, this propagates to all H₀ / per-event / convergence / single-event / interactive figures in one edit.

**c. Theme switch layered on `apply_style()`:**
- Extend the existing `apply_style(*, use_latex=False)` to `apply_style(*, theme='paper'|'talk'|'web', use_latex=...)`.
- Base = `emri_thesis_v2.mplstyle`. `theme='paper'` layers `paper.mplstyle` (turn on `usetex` + Latin Modern, column-exact figsize). `theme='talk'` layers `talk.mplstyle` (sans, sizes ×1.8, strokes ×2). `theme='web'` is informational for matplotlib but primarily exports the palette as CSS custom properties + a Plotly template.
- Switch the `use_latex` serif from `Computer Modern Roman` → `Latin Modern Roman` (+ `\usepackage{lmodern}` in the REVTeX preamble).

**d. Revised factory touch-points (bodies only, signatures unchanged):**
- Consolidate the four combined-posterior code paths into ONE canonical factory (area-normalize, shade nested HDI via `compute_hdi_interval`, inline MAP, named reference bands, `theme=` + `peak_vs_pdf`/`show_kde` options).
- Consolidate the two convergence figures into ONE log-log factory (shaded 16/84 band, inline slope; fixes the `\%` label bug).
- Strip every `fig.tight_layout()` call (5 factories) — rely on `constrained_layout`.
- Drop per-point markers; switch heatmaps to `pcolormesh` + true log axes + explicit norm + `set_bad` + horizon contour; route all axis/legend text through the LABELS provider; reconcile Mpc/Gpc.
- Bring the convention-violating factories (`fisher_quality_diagnostic`, hardcoded-figsize 3D scatters) back to the `(fig, ax)`-return + `get_figure`-preset contract.
- Add the five new factories (forest, PP-plot, selection explainer, population view, event→host→posterior still).
- Guard against regression with golden-image tests before/after the consolidation.

**e. Interactive stack choice:**
- **Phase 1 (low risk, immediate):** add one shared `go.layout.Template` applied in all 8 Plotly factories, exporting the v2 palette as CSS custom properties; switch `include_plotlyjs='cdn'` → `'directory'` (or cartesian partial bundle) for self-contained-at-site-level, archival-robust output; replace fragile visibility-vector bookkeeping with per-group traces.
- **Phase 2 (follow-on):** adopt Observable Framework as the static-site shell (GH-Pages via GitHub Actions). Python data loaders emit compact JSON snapshots (the phase45 h-sweep cube already has this shape); the browser only loads + interacts; KaTeX renders equations matching the paper; one CSS theme imports the same hex tokens. Per-figure engine by need: Observable Plot (routine scenes), Vega-Lite (linked/brushed scenes), D3 + Aladin Lite (the 1-2 hero explorables). Replace the hand-authored `index.html` + drift-prone footer with file-based routing + snapshot-manifest provenance.
- Every interactive figure degrades to the static vector PDF twin rendered from the same data — the interactive is augmentation, never the only record.

**f. Quality gate (every figure, before shipping):** a `colorspacious` deuteranopia + luminance-grayscale simulation pass; color is never the only channel (always + linestyle / marker / direct label).

**Scope note.** This entire effort is a style + recolor + annotation + consolidation pass on factory *bodies*, plus new factories on existing data plumbing. The canonical-posterior data plumbing (`load_canonical_combined_posterior`, `compute_m_z_improvement_bank`) and the `data in → (fig, ax) out` contract are untouched. It would be scoped as a GSD milestone (software work — not a GPD/physics milestone) with phases roughly: (1) `_colors` v2 + `emri_thesis_v2.mplstyle` + theme switch + golden-image baseline; (2) factory consolidation + annotation + heatmap fixes; (3) the five new figures; (4) interactive Phase 1 (template + bundling); (5) interactive Phase 2 (Observable Framework).

---

## 7. Open Questions for the User

1. **Default direction.** Confirm HORIZON (with Dispatch's annotation discipline) as the default, or prefer the more conservative Quiet Index, the fully-editorial Dark Siren Dispatch, or the interactive-first SIREN ATLAS? The palette and voice diverge meaningfully.
2. **TeX in CI.** Is a working TeX install acceptable in the CI/pages build for the `usetex` Latin Modern paper path, or should the project commit to the `mathtext.fontset: cm` fallback as the primary renderer (simpler, one path, slightly less body-match)?
3. **Variant hue.** Gold (`#E8A317` / `#DDAA33`) vs orange (`#E69F00`) for With-M_z — and navy (`#1B2A4A`) vs the existing blue (`#0072B2`) for Without-M_z? All four pairs are CB- and grayscale-safe; this is the most visible single decision.
4. **Normalization switch.** Confirm moving the house default from peak-normalization to area-normalized PDFs for single/headline posteriors. This changes how "the central result" *looks* (harmless numerically) and must be coordinated with the `results.tex` captions.
5. **Active-title placement.** For the paper, keep figure interiors neutral and move the finding into the caption's first sentence (PRD convention), or allow active declarative titles inside paper figures too?
6. **Observable Framework adoption.** Commit to the Observable Framework migration now (best coherence/archival/load-weight, new JS toolchain), or stay on Plotly with a shared template + `include_plotlyjs='directory'` and defer Framework?
7. **Forest-plot literature set.** Which measurements and exactly which values/citations should populate the forest (Planck 2018, DESI BAO, SH0ES, TRGB/CCHP, TDCOSMO/H0LiCOW, GW170817, LVK GWTC-4/5 dark+combined)? These must be sourced, dated, and kept current.
8. **PP-plot scope.** Should the PP-plot/coverage figure cover all 14 EMRI parameters, the inference target (H₀) only, or a curated subset? And PP-plot, rank-histogram (SBC), or both?
9. **Dark variant.** Is a genuine dark web/slide variant wanted (HORIZON ships one), or is light-only sufficient across all media?
10. **Golden-image testing.** Acceptable to add image-comparison (golden-image) tests to guard the factory consolidation against silent visual regressions, given the added CI artifacts and maintenance?
