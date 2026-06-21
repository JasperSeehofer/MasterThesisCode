# Requirements — v2.3 Visualization Redesign (HORIZON)

**Defined:** 2026-06-21
**Core Value:** Measure H₀ from simulated EMRI dark siren events with galaxy catalog completeness correction, producing publication-ready results.

## v2.3 Requirements

Implement the full figure/visualization redesign (`docs/VIZ_REDESIGN_PROPOSAL.md`) in the chosen **HORIZON** design direction with **Dark Siren Dispatch** annotation discipline — modern, field-aware, publication- and web-ready figures. SOFTWARE/design work in the plotting package + interactive HTML only: no physics formula/constant/waveform change, GSD never `/physics-change`. Honor the factory-function architecture + `apply_style()`, REVTeX vector-PDF, colorblind- AND grayscale-safe, self-contained GH-Pages HTML, complete type annotations, and a per-commit check gate (ruff+mypy+pytest "not gpu and not slow").

### Foundation (shipped — quick task `260621-npe`, branch `viz/horizon-quick-wins`)

- [x] **VR-F1**: `_colors.py` v2 HORIZON palette — navy `#1B2A4A` (Without M_z) / gold `#E8A317` (With M_z) / vermillion `#C2451E` truth / `PLANCK`+`SH0ES` band colors; two-blues collision removed; CB- and grayscale-safe
- [x] **VR-F2**: `apply_style(theme="paper"|"talk"|"web")` switch; paper default byte-identical to prior output
- [x] **VR-F3**: Headline posteriors render as area-normalized PDFs with shaded nested 68/95% HDI (via `compute_hdi_interval`) + inline `MAP ±` annotation
- [x] **VR-F4**: fig01 two-variant overlay deduplicated (single MAP/truth, no muddy bands per §1.3)

### Posterior Factory Consolidation

- [ ] **VR-CONS-01**: The four/five duplicate combined-H0-posterior code paths collapse into ONE canonical factory; `paper_figures` wrappers + `main.py` manifest delegate to it while preserving `(data_dir) -> (fig, ax)` signatures. Paths: fig01 (`main.py:_gen_h0_posterior_combined`), `paper_figures.plot_h0_posterior_comparison`, `plot_h0_posterior_kde`, `convergence_analysis.plot_m_z_improvement_panels` top panel, fig08-left (verify). No data-plumbing rewrite; canonical loaders untouched
- [ ] **VR-CONS-02**: All consolidated paths agree on the MAP (regression anchor `test_canonical_map_consistency` extended); fig01 wired to the area-norm headline treatment; theme passes through

### Colormap & Heatmap Modernization

- [ ] **VR-CMAP-01**: `_colors.CMAP` default switched to `cividis` (perceptually-uniform, CB-safe); audit all `image.cmap`/explicit-cmap call sites
- [ ] **VR-CMAP-02**: Every heatmap sets an explicit norm (`LogNorm` or robust percentile clip) and a `set_bad` no-data color; no silent autoscale washing out dynamic range
- [ ] **VR-CMAP-03**: Sky map (fig05) and pdet surface (fig20) switch from fake-index `imshow` to `pcolormesh` with true (log where appropriate) axes — fixes the wasted-dynamic-range viridis sky scatter
- [ ] **VR-CMAP-04**: Bias/residual maps use a diverging map (`cmcrameri vik`) with `TwoSlopeNorm(vcenter=0.73)`; sequential maps stay sequential
- [ ] **VR-CMAP-05**: Semantic contour overlays added where they aid reading: `P_det = 0.5` detection-horizon contour on pdet maps; 68/95% credible contours on 2D posteriors

### New Field-Expected Figures (on existing data plumbing — no new pipeline)

- [ ] **VR-NEW-01**: H0 forest / tension plot — this EMRI result vs Planck/SH0ES/DESI/TRGB/lensing/GW170817/LVK dark+combined; grouped early-vs-late, full-height reference bands, bold result row. *Final this-work number is DATA-GATED on the production posterior; scaffold with a placeholder + literature values now.*
- [ ] **VR-NEW-02**: PP-plot / coverage — bilby-style nested grey 1/2/3σ binomial bands, per-parameter cumulative lines, KS p-values. Referee-mandatory calibration proof. *DATA-GATED: needs injection-recovery rank data; scaffold the factory + test on synthetic ranks now.*
- [ ] **VR-NEW-03**: Selection-function / detection-horizon explainer — `p_det(d_L)` survival curve + `(M, d_L)` p_det heatmap with 0.5/0.9 horizon contour + injected-population overlay
- [ ] **VR-NEW-04**: "Where do the constraints come from" population view — driver SNR×z histogram linked to the sky map + de-emphasized per-event spaghetti + stacked posterior rebuilt from the selected events (surfaces `fig02`'s never-enabled `color_by`)
- [ ] **VR-NEW-05**: Event → host → posterior explorable (interactive hero scene; sky patch with GLADE+ candidate hosts re-weighting under an h-slider, single-event posterior rebuilt as a sum of per-galaxy contributions). *Stretch / may be a follow-on.*

### Annotation & Decluttering Rollout

- [ ] **VR-ANNO-01**: Annotation layer across the figure set — active declarative titles stating the finding, direct end-of-line color-matched labels replacing legend boxes where it reads better, named reference rules, scaling-law labels (e.g. the `N^-1/2` convergence law)
- [ ] **VR-ANNO-02**: Per-point markers dropped on smooth/near-delta curves across the package
- [ ] **VR-ANNO-03**: The 5 `fig.tight_layout()` calls that fight the stylesheet's `constrained_layout` removed
- [ ] **VR-ANNO-04**: Convention-violating factories restored to the `(fig, ax)`-return + `get_figure`-preset contract (e.g. `fisher_quality_diagnostic`, hardcoded-figsize factories)
- [ ] **VR-ANNO-05**: The two 3D-scatter anti-patterns replaced with 2D marginal/corner/Mollweide views
- [ ] **VR-ANNO-06**: All axis/legend text routed through the single `LABELS` provider; Mpc/Gpc unit inconsistency reconciled
- [ ] **VR-ANNO-07**: Corner plot modernized — filled/gradient contours (ChainConsumer/arviz-style) replacing the dated thin-orange/green-crosshair look

### Interactive Overhaul (Plotly → GH Pages)

- [ ] **VR-INT-01**: One shared `go.layout.Template` applied across all 8 Plotly factories, matching the HORIZON palette/typography
- [ ] **VR-INT-02**: `include_plotlyjs` strategy chosen for self-contained-at-site-level + archival-robust output (e.g. `'directory'` or partial bundle)
- [ ] **VR-INT-03**: Per-group traces replace the fragile visibility-vector bookkeeping
- [ ] **VR-INT-04**: talk/web theme wired into interactive generation; every interactive degrades to its static PDF twin
- [ ] **VR-INT-05**: Observable Framework single-snapshot → two-renderer architecture (SIREN ATLAS, the long-term web foundation). *Follow-on / stretch — out of scope unless prior phases land early.*

## Out of Scope (this milestone)

- Any physics formula/constant/waveform/PSD change (would route to GPD `/physics-change`)
- Data-pipeline / simulation / cluster changes
- Resuming v2.2 Pipeline Correctness (paused; gated on a fresh trusted production re-sim)

## Traceability

Roadmap: `.planning/ROADMAP.md` (created 2026-06-21, 7 phases, all [GSD] — no physics).
Every not-yet-shipped requirement maps to exactly one phase. The VR-F* foundation was
shipped in quick task `260621-npe` and is not re-planned.

| Requirement | Phase | Routing | Status |
|-------------|-------|---------|--------|
| VR-F1 | Foundation (260621-npe) | GSD | Done |
| VR-F2 | Foundation (260621-npe) | GSD | Done |
| VR-F3 | Foundation (260621-npe) | GSD | Done |
| VR-F4 | Foundation (260621-npe) | GSD | Done |
| VR-CONS-01 | Phase 1 — Posterior Factory Consolidation | GSD | Pending |
| VR-CONS-02 | Phase 1 — Posterior Factory Consolidation | GSD | Pending |
| VR-CMAP-01 | Phase 2 — Colormap & Heatmap Modernization | GSD | Pending |
| VR-CMAP-02 | Phase 2 — Colormap & Heatmap Modernization | GSD | Pending |
| VR-CMAP-03 | Phase 2 — Colormap & Heatmap Modernization | GSD | Pending |
| VR-CMAP-04 | Phase 2 — Colormap & Heatmap Modernization | GSD | Pending |
| VR-CMAP-05 | Phase 2 — Colormap & Heatmap Modernization | GSD | Pending |
| VR-NEW-03 | Phase 3 — New Static Figures (Selection & Population) | GSD | Pending |
| VR-NEW-04 | Phase 3 — New Static Figures (Selection & Population) | GSD | Pending |
| VR-NEW-01 | Phase 4 — New Figures (Data-Gated) | GSD | Pending (DATA-GATED) |
| VR-NEW-02 | Phase 4 — New Figures (Data-Gated) | GSD | Pending (DATA-GATED) |
| VR-ANNO-01 | Phase 5 — Annotation & Decluttering Rollout | GSD | Pending |
| VR-ANNO-02 | Phase 5 — Annotation & Decluttering Rollout | GSD | Pending |
| VR-ANNO-03 | Phase 5 — Annotation & Decluttering Rollout | GSD | Pending |
| VR-ANNO-04 | Phase 5 — Annotation & Decluttering Rollout | GSD | Pending |
| VR-ANNO-05 | Phase 5 — Annotation & Decluttering Rollout | GSD | Pending |
| VR-ANNO-06 | Phase 5 — Annotation & Decluttering Rollout | GSD | Pending |
| VR-ANNO-07 | Phase 5 — Annotation & Decluttering Rollout | GSD | Pending |
| VR-INT-01 | Phase 6 — Interactive Plotly Overhaul | GSD | Pending |
| VR-INT-02 | Phase 6 — Interactive Plotly Overhaul | GSD | Pending |
| VR-INT-03 | Phase 6 — Interactive Plotly Overhaul | GSD | Pending |
| VR-INT-04 | Phase 6 — Interactive Plotly Overhaul | GSD | Pending |
| VR-INT-05 | Phase 7 — Interactive Stretch | GSD | Pending (STRETCH/DEFERRED) |
| VR-NEW-05 | Phase 7 — Interactive Stretch | GSD | Pending (STRETCH/DEFERRED) |

**Coverage:** 24/24 not-yet-shipped requirements mapped to exactly one phase. No orphans,
no duplicates. 4 shipped (VR-F1..F4). 2 data-gated (VR-NEW-01/02). 2 stretch/deferred
(VR-INT-05, VR-NEW-05).
