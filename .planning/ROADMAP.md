# Roadmap: v2.3 Visualization Redesign (HORIZON)

**Milestone:** v2.3 Visualization Redesign (HORIZON)
**Defined:** 2026-06-21
**Source spec:** `docs/VIZ_REDESIGN_PROPOSAL.md` (HORIZON direction + Dark Siren Dispatch annotation discipline)
**Granularity:** standard (7 phases)
**Coverage:** 24/24 not-yet-shipped requirements mapped (4 of the 28 VR-* total were shipped in the foundation quick task)

## Scope Note

This roadmap covers **milestone v2.3 only** — the figure/visualization redesign in the
chosen **HORIZON** direction. It is **all GSD** (software/design: plotting package +
interactive HTML). It contains **NO physics**: no formula, constant, PSD coefficient,
waveform parameter, or frequency-limit change — so it **never** routes to GPD
`/physics-change`. The colormap/norm/axis-encoding changes are presentation, not
computed-value changes.

- **v2.2 Pipeline Correctness is PAUSED** (not shipped). Its remaining bias/correctness
  work is gated on a fresh trusted production re-sim. Snapshots are preserved at
  `.planning/milestones/v2.2-ROADMAP.md`, `.planning/milestones/v2.2-REQUIREMENTS.md`,
  and `.planning/milestones/v2.2-phases/`. Resume by restoring those and re-pointing STATE.md.
- **Prior milestone roadmaps** (v1.0–v2.1, cumulative) are archived under `.planning/milestones/`.
- **Phase numbering resets to 1.** The historical global 35–50 sequence is tangled (v2.1
  PubFigs / v2.2 Phase-35 collisions documented in STATE.md); this is a clean ledger.
  `.planning/phases/` was empty at milestone start.

## Foundation (already shipped — do NOT re-plan)

Quick task `260621-npe` (branch `viz/horizon-quick-wins`) landed the HORIZON
foundation that every phase below builds on:

- **VR-F1** — `_colors.py` v2 HORIZON palette: navy `#1B2A4A` (Without M_z) / gold
  `#E8A317` (With M_z) / vermillion `#C2451E` truth + `PLANCK`/`SH0ES` band colors; the
  two-blues collision is removed; CB- and grayscale-safe.
- **VR-F2** — `apply_style(theme="paper"|"talk"|"web")` switch; `paper` default byte-identical.
- **VR-F3** — headline posteriors render area-normalized + nested 68/95% HDI + inline `MAP ±`.
- **VR-F4** — fig01 two-variant overlay deduplicated.

Note: `_colors.CMAP` was deliberately left at `viridis` in the foundation slice (see the
scope note in `_colors.py`); the `cividis` migration is **Phase 2** here.

## Overview

The redesign proceeds from low-risk, no-data-gate consolidation outward to higher-risk
new figures and the interactive overhaul. First we **consolidate** the quadruplicate
H₀-posterior code into one canonical factory (the riskiest refactor, but it makes every
later annotation/recolor edit land in one place). Then **colormaps & heatmaps** are
modernized package-wide (cividis + explicit norms + true axes). Next come the two
**non-data-gated new static figures** (selection-function explainer, population view),
followed by the two **data-gated new figures** (H₀ forest, PP-plot) — scaffolded now on
literature/synthetic placeholders, finalized when the production/seed500 posterior lands.
With the static design settled, the **annotation & decluttering rollout** sweeps the whole
set, and the **interactive Plotly overhaul** ports the settled HORIZON language to the web.
A final **stretch** phase holds the Observable Framework migration and the hero
event→host explorable, explicitly deferred unless prior phases land early.

Every phase honors the factory-function architecture (`data in → (fig, ax) out`) +
`apply_style()`, adds no data-plumbing rewrite (canonical loaders untouched), carries
complete type annotations, and passes the per-commit check gate (ruff + mypy + pytest
`"not gpu and not slow"`). Render-verify each figure against an existing data dir:
`simulations/_archive_v2_1_baseline/posteriors/` for posterior figures,
`results/figures_seed200/` for the others.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Posterior Factory Consolidation** [GSD] — Collapse the 4–5 duplicate combined-H₀-posterior paths into ONE canonical factory; regression-anchor on MAP agreement ✅ 2026-06-21 (b3ee7dc..1a43709)
- [x] **Phase 2: Colormap & Heatmap Modernization** [GSD] — cividis default + explicit norms + `set_bad` + true `pcolormesh` axes + diverging bias maps + semantic contours ✅ 2026-06-21 (6192594..512f66f)
- [x] **Phase 3: New Static Figures — Selection & Population** [GSD] — Detection-horizon explainer + "where the constraints come from" population view (no data gate) ✅ 2026-06-21 (13d200b..30bce20)
- [ ] **Phase 4: New Figures (Data-Gated) — H₀ Forest & PP-Plot** [GSD] — Scaffold the forest/tension + bilby-style PP-plot factories on placeholder/synthetic data; flag final numbers as data-gated
- [ ] **Phase 5: Annotation & Decluttering Rollout** [GSD] — Active titles + direct labels + named rules + scaling-law labels; strip `tight_layout`; restore factory contracts; 3D→2D; unify `LABELS`; Mpc/Gpc; modern corner plot
- [ ] **Phase 6: Interactive Plotly Overhaul** [GSD] — One shared `go.layout.Template`; self-contained `include_plotlyjs` strategy; per-group traces; talk/web theme + static PDF twin parity
- [ ] **Phase 7: Interactive Stretch — SIREN ATLAS & Event→Host Explorable** [GSD] [STRETCH/DEFERRED] — Observable Framework two-renderer shell + hero event→host→posterior explorable

## Phase Details

### Phase 1: Posterior Factory Consolidation
**Goal**: Every rendering of the central H₀ result flows through one canonical factory, so the quadruplicate-drift hazard is gone and later recolor/annotation edits land in exactly one place.
**Routing**: [GSD] — software refactor of factory bodies; no physics, never `/physics-change`. Canonical data loaders (`load_canonical_combined_posterior`) untouched.
**Depends on**: Nothing (first phase; builds on shipped VR-F1..F4 foundation)
**Requirements**: VR-CONS-01, VR-CONS-02
**Success Criteria** (what must be TRUE):
  1. The five identified paths — fig01 (`main.py:_gen_h0_posterior_combined`), `paper_figures.plot_h0_posterior_comparison`, `plot_h0_posterior_kde`, `convergence_analysis.plot_m_z_improvement_panels` top panel, fig08-left — all delegate to ONE canonical factory while preserving their `(data_dir) -> (fig, ax)` signatures.
  2. `test_canonical_map_consistency` is extended to cover all consolidated paths and passes — every path reports the same MAP to tolerance.
  3. fig01 is wired to the area-normalized headline treatment (VR-F3) with the `theme=` switch passing through; regenerating it via `--generate_figures` against `simulations/_archive_v2_1_baseline/posteriors/` produces a PDF with no traceback.
  4. The check gate (ruff + mypy + pytest `"not gpu and not slow"`) is green; the regenerated paper PDF stays vector and ≥7pt.
**Plans**: 1 plan
Plans:
- [ ] 01-01-PLAN.md — Make plot_combined_posterior the single canonical H₀-posterior factory; delegate paper_figures + convergence panels + fig01/fig08; extend the cross-path rendered-MAP regression; green gate + archive regen
**UI hint**: yes

### Phase 2: Colormap & Heatmap Modernization
**Goal**: Every 2D map uses a perceptually-uniform, CB- and grayscale-safe colormap with an explicit norm and physically-faithful axes, so dynamic range is no longer washed out and no-data bins read as no-data.
**Routing**: [GSD] — colormap/norm/axis-encoding changes only; presentation, not physics (no computed value changes). Never `/physics-change`.
**Depends on**: Phase 1 (the consolidated 2D-posterior path hosts the 68/95% credible-contour overlay in VR-CMAP-05)
**Requirements**: VR-CMAP-01, VR-CMAP-02, VR-CMAP-03, VR-CMAP-04, VR-CMAP-05
**Success Criteria** (what must be TRUE):
  1. `_colors.CMAP` is `cividis` and an audit of all `image.cmap` / explicit-cmap call sites confirms every sequential heatmap uses it (diverging maps excepted); a call-site sweep is clean.
  2. Every heatmap sets an explicit norm (`LogNorm` or robust percentile clip) and a `set_bad` no-data color — verified by regenerating fig05 (sky), fig20 (pdet surface), and the CRB/eval heatmaps with visible dynamic range and visible no-data bins.
  3. fig05 and fig20 render via `pcolormesh` with true (log where appropriate) axes — no fake-index `imshow`, and fig20's Mpc/Gpc tick labels are physically correct.
  4. Bias/residual maps use a diverging map (`cmcrameri vik`) with `TwoSlopeNorm(vcenter=0.73)`; sequential maps stay sequential.
  5. Semantic contour overlays render where they aid reading: `P_det = 0.5` horizon contour on pdet maps, 68/95% credible contours on 2D posteriors; check gate green.
**Plans**: 1 plan
Plans:
- [ ] 02-01-PLAN.md — cividis default + NO_DATA/DIVERGING constants + diverging_norm/make_heatmap_norm/credible-contour helpers; explicit norm + set_bad on every heatmap; fig20 pcolormesh true-log axes + Mpc/Gpc fix + 0.5/0.9 horizon contour; fig05 SNR norm fix; clean call-site sweep + green gate
**UI hint**: yes

### Phase 3: New Static Figures — Selection & Population
**Goal**: The two highest-value field-expected static figures that need NO new data exist as factories: a selection-function / detection-horizon explainer and a "where do the constraints come from" population view.
**Routing**: [GSD] — new factories on existing data plumbing (p_det surface, per-event SNR/z, fig02's latent `color_by`); no physics. Never `/physics-change`.
**Depends on**: Phase 2 (both reuse the modernized heatmap/contour + sky-map encodings)
**Requirements**: VR-NEW-03, VR-NEW-04
**Success Criteria** (what must be TRUE):
  1. A selection-function explainer factory renders the `p_det(d_L)` survival curve + `(M, d_L)` p_det heatmap with 0.5/0.9 horizon contours and injected-population overlay, via `--generate_figures` against `results/figures_seed200/` with no traceback.
  2. A population-view factory renders the driver SNR×z histogram linked to the sky map + de-emphasized per-event spaghetti + a stacked posterior rebuilt from the selected events — surfacing fig02's never-enabled `color_by`.
  3. Both follow the `data in → (fig, ax) out` contract + `apply_style()` + `get_figure` presets (no hardcoded figsize), carry complete type annotations, and produce vector REVTeX-width PDFs.
  4. Both pass a CB + grayscale safety read; check gate green.
**Plans**: 1 plan
Plans:
- [ ] 03-01-PLAN.md — selection_plots.py (survival curve + fig20 pdet heatmap horizon-contour composite) + population_plots.py (SNR×z driver + Mollweide sky + de-emphasized color_by spaghetti + canonical stacked posterior) + fig21/fig22 manifest wiring + fig02 surfaces color_by; tests per factory; green gate
**UI hint**: yes

### Phase 4: New Figures (Data-Gated) — H₀ Forest & PP-Plot
**Goal**: The two referee-/defense-expected figures the pipeline lacks — the H₀ forest/tension plot and a bilby-style PP-plot/coverage figure — exist as fully-styled factories with passing tests, scaffolded on literature + synthetic data NOW, with the final this-work numbers explicitly flagged as DATA-GATED on the production/seed500 posterior.
**Routing**: [GSD] — new factories + tests; literature values are curated context, not computed physics. Never `/physics-change`.
**Depends on**: Phase 1 (forest reuses the `compute_hdi_interval` band machinery exposed by the canonical posterior factory); Phase 2 (PP-plot grey-band shading + diverging accents)
**Requirements**: VR-NEW-01, VR-NEW-02
**Success Criteria** (what must be TRUE):
  1. The H₀ forest factory renders early-vs-late grouped point+68%CI rows with full-height Planck/SH0ES bands and a bold this-work row, sourced from a hardcoded dated-citation literature table + a PLACEHOLDER this-work value; the final number is marked DATA-GATED via a single constant/loader hook to swap when the production posterior lands.
  2. The PP-plot/coverage factory renders bilby-style nested grey 1/2/3σ binomial bands + per-parameter cumulative lines + KS p-values, validated on SYNTHETIC injection-recovery ranks; a unit test asserts well-calibrated synthetic ranks fall inside the bands.
  3. Both factories follow the `(fig, ax)` contract + `apply_style()`, carry complete type annotations, and produce vector REVTeX-width PDFs that pass CB + grayscale reads.
  4. The roadmap/STATE records the data-gate: finalize the this-work forest number + real PP-plot ranks once the cluster posterior is trusted; check gate green on the scaffolded versions.
**Plans**: 1 plan
Plans:
- [ ] 04-01-PLAN.md — Scaffold forest_plot.py (literature table + bold this-work row + Planck/SH0ES bands + THIS_WORK_H0 data-gate) + pp_plot.py (bilby nested binomial bands + per-param cumulative lines + KS p + synthetic-rank data-gate); fig23/fig24 manifest wiring; tests per factory; green gate
**UI hint**: yes

### Phase 5: Annotation & Decluttering Rollout
**Goal**: The full static figure set carries the HORIZON + Dispatch annotation discipline and is decluttered — active titles, direct labels, named reference rules, scaling-law labels — and every layout/convention defect is fixed across the package.
**Routing**: [GSD] — annotation, layout, and contract fixes to factory bodies; no computed-value changes. Never `/physics-change`.
**Depends on**: Phase 1 (rollout is far cheaper once the posterior paths are consolidated); Phases 2–4 (annotate the final figure set, including the new figures, once the design is settled)
**Requirements**: VR-ANNO-01, VR-ANNO-02, VR-ANNO-03, VR-ANNO-04, VR-ANNO-05, VR-ANNO-06, VR-ANNO-07
**Success Criteria** (what must be TRUE):
  1. The annotation layer is applied across the set: active declarative titles, direct end-of-line color-matched labels replacing legend boxes where it reads better, named reference rules, and scaling-law labels (e.g. the `N^-1/2` convergence law) — verified by regenerating the affected figures with no traceback.
  2. The 5 `fig.tight_layout()` calls that fight `constrained_layout` are removed and a grep gate confirms zero remain; per-point markers are dropped on smooth/near-delta curves.
  3. Convention-violating factories (`fisher_quality_diagnostic`, hardcoded-figsize factories) return `(fig, ax)` and use `get_figure` presets; the two 3D-scatter anti-patterns are replaced with 2D marginal/corner/Mollweide views.
  4. All axis/legend text routes through the single `LABELS` provider, the Mpc/Gpc unit inconsistency is reconciled, and the corner plot is modernized with filled/gradient contours (ChainConsumer/arviz-style) replacing the thin-orange/green-crosshair look.
  5. Check gate green; regenerated paper PDFs stay vector and ≥7pt.
**Plans**: TBD
**UI hint**: yes

### Phase 6: Interactive Plotly Overhaul
**Goal**: All 8 Plotly interactive figures share one HORIZON design language, are self-contained at site level, use a robust trace model, and each degrades to its static PDF twin.
**Routing**: [GSD] — interactive HTML/Plotly templating only; no physics. Never `/physics-change`.
**Depends on**: Phase 5 (the interactive template should mirror the settled static HORIZON design); Phase 2 (cividis/norm law exported to the Plotly template); Phase 3 (population view's interactive twin)
**Requirements**: VR-INT-01, VR-INT-02, VR-INT-03, VR-INT-04
**Success Criteria** (what must be TRUE):
  1. One shared `go.layout.Template` is applied across all 8 Plotly factories, matching the HORIZON palette/typography (exported as the same hex tokens used by the static figures).
  2. An `include_plotlyjs` strategy (`'directory'` or partial bundle) is chosen and applied so the GH-Pages output is self-contained at site level and archival-robust — verified by opening the generated HTML offline.
  3. Per-group traces replace the fragile visibility-vector bookkeeping in the dropdown/slider figures (`interactive_m_z_improvement`, `interactive_single_event_detail`).
  4. talk/web theme is wired into interactive generation and every interactive figure has a verified static PDF twin; `--generate_interactive` runs with no traceback against an existing data dir; check gate green.
**Plans**: TBD
**UI hint**: yes

### Phase 7: Interactive Stretch — SIREN ATLAS & Event→Host Explorable
**Goal**: (Stretch / explicitly deferred) The long-term web foundation — an Observable Framework two-renderer (single snapshot → static still + web scene) architecture — and the hero event→host→posterior explorable.
**Routing**: [GSD] — web stack + new explorable; no physics. Never `/physics-change`.
**Depends on**: Phase 6 (only attempted if the Plotly overhaul lands with budget to spare)
**Requirements**: VR-INT-05, VR-NEW-05
**Success Criteria** (what must be TRUE):
  1. (If undertaken) The Observable Framework single-snapshot → two-renderer shell (SIREN ATLAS) builds via GitHub Actions with no server / CDN dependence, importing the same hex tokens as the static figures.
  2. (If undertaken) The event→host→posterior explorable renders a sky patch with GLADE+ candidate hosts re-weighting under an h-slider and a single-event posterior rebuilt as a sum of per-galaxy contributions; it degrades to a 3-panel static still (redesigned fig17 content).
  3. This phase is OPTIONAL: if budget is not available after Phase 6, VR-INT-05 and VR-NEW-05 are formally deferred to a follow-on milestone with no impact on v2.3 ship.
**Plans**: TBD
**Note**: STRETCH/DEFERRED — both requirements are marked stretch/follow-on in REQUIREMENTS.md. Default expectation is deferral unless Phases 1–6 land with margin.
**UI hint**: yes

## Progress

| Phase | Routing | Plans Complete | Status | Completed |
|-------|---------|----------------|--------|-----------|
| 1. Posterior Factory Consolidation | GSD | 1/1 | Complete ✅ | 2026-06-21 |
| 2. Colormap & Heatmap Modernization | GSD | 0/1 | Not started | - |
| 3. New Static Figures — Selection & Population | GSD | 1/1 | Complete ✅ | 2026-06-21 |
| 4. New Figures (Data-Gated) — H₀ Forest & PP-Plot | GSD | 0/1 | Not started | - |
| 5. Annotation & Decluttering Rollout | GSD | 0/TBD | Not started | - |
| 6. Interactive Plotly Overhaul | GSD | 0/TBD | Not started | - |
| 7. Interactive Stretch — SIREN ATLAS & Event→Host (DEFERRED) | GSD | 0/TBD | Not started | - |

## Coverage

All not-yet-shipped v2.3 requirements map to exactly one phase. The VR-F* foundation
requirements were shipped in quick task `260621-npe` and are not re-planned here.

| Phase | REQ-IDs | Count |
|-------|---------|-------|
| Foundation (shipped) | VR-F1, VR-F2, VR-F3, VR-F4 | 4 |
| 1 | VR-CONS-01, VR-CONS-02 | 2 |
| 2 | VR-CMAP-01, VR-CMAP-02, VR-CMAP-03, VR-CMAP-04, VR-CMAP-05 | 5 |
| 3 | VR-NEW-03, VR-NEW-04 | 2 |
| 4 (data-gated) | VR-NEW-01, VR-NEW-02 | 2 |
| 5 | VR-ANNO-01, VR-ANNO-02, VR-ANNO-03, VR-ANNO-04, VR-ANNO-05, VR-ANNO-06, VR-ANNO-07 | 7 |
| 6 | VR-INT-01, VR-INT-02, VR-INT-03, VR-INT-04 | 4 |
| 7 (stretch/deferred) | VR-INT-05, VR-NEW-05 | 2 |

**Not-yet-shipped mapped:** 24/24 ✓ — 100% coverage, no orphans, no duplicates.
Of these, 4 are flagged: VR-NEW-01 / VR-NEW-02 are DATA-GATED (scaffold now, finalize on
the production posterior); VR-INT-05 / VR-NEW-05 are STRETCH/DEFERRED.

## Hard Constraints (every phase)

- **All GSD, no physics.** No formula/constant/PSD/waveform/frequency-limit change.
  Colormap/norm/axis changes are presentation, not computed values. Never `/physics-change`.
- **Factory architecture preserved.** `data in → (fig, ax) out`; `apply_style()` always;
  no data-plumbing rewrite (canonical loaders untouched); no throwaway analysis scripts.
- **Per-commit check gate.** ruff + mypy + pytest `"not gpu and not slow"` green before commit.
- **Output contracts.** REVTeX vector-PDF preserved (≥7pt); colorblind- AND grayscale-safe
  (color never the only channel — always + linestyle / marker / direct label); self-contained
  GH-Pages HTML; complete type annotations.
- **Render-verify** against an existing data dir each phase:
  `simulations/_archive_v2_1_baseline/posteriors/` (posteriors) or `results/figures_seed200/` (others).

## Links

- **Spec artifact:** `docs/VIZ_REDESIGN_PROPOSAL.md`
- **Requirements:** `.planning/REQUIREMENTS.md`
- **State:** `.planning/STATE.md`
- **Paused v2.2 snapshots:** `.planning/milestones/v2.2-ROADMAP.md`, `.planning/milestones/v2.2-REQUIREMENTS.md`, `.planning/milestones/v2.2-phases/`
- **Prior shipped milestones:** `.planning/milestones/v{1.0,1.1,1.2,1.3,1.4,2.1-biasres,2.1-cumulative}-ROADMAP.md`

---
*Created: 2026-06-21 — v2.3 Visualization Redesign (HORIZON) roadmap from docs/VIZ_REDESIGN_PROPOSAL.md*
