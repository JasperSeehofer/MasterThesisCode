# Milestones

## Prior Work (GSD-tracked)

- **v1.0** EMRI HPC Integration (Shipped: 2026-03-27) — 5 phases
- **v1.1** Clean Simulation Campaign (Shipped: 2026-03-29) — 3 phases
- **v1.2** Production Campaign & Physics Corrections (In Progress) — Phases 9-13, last completed: Phase 10

## GPD-tracked

- **v1.2.1** "With BH Mass" Likelihood Bias Audit (On Hold: 2026-03-31) — Phases 14-16, last completed: Phase 15
  - /(1+z) fix applied but insufficient; awaiting injection P_det data for Phase 16
- **v1.2.2** Injection Campaign Physics Analysis (Completed: 2026-04-01) — Phases 17-20, 8 plans, 16 tasks
  - Verified injection/simulation parameter consistency (14 parameters, d_L round-trip to 2e-13)
  - Detection yield 0.22-0.81% across 7 h-values; >99% GPU time on sub-threshold events
  - IS estimator with exact backward compat; VRF 11.8-24.9x in boundary bins
  - Full validation: VALD-01 PASS (916 bins), VALD-02 PASS (alpha_grid = alpha_MC exactly)
  - See `.gpd/milestones/v1.2.2-ROADMAP.md` for full archive

- **v2.1** Publication Figures (In Progress: 2026-04-07, GSD-tracked) — Phases 29-33
  - Unified figure manifest, publication-refined style, galaxy-level visualizations, interactive GitHub Pages
  - Research survey complete (4 scouts: prior work, methods, computational, pitfalls)
  - 23 requirements defined, 5 phases planned
  - Implementation tracked in GSD (`.planning/`)

See `.planning/MILESTONES.md` for full history.

---
