# Roadmap: EMRI Parameter Estimation

## Milestones

- ✅ **v1.0 EMRI HPC Integration** — Phases 1-5 (shipped 2026-03-27)
- ✅ **v1.1 Clean Simulation Campaign** — Phases 6-8 (shipped 2026-03-29)
- ✅ **v1.2 Production Campaign & Physics Corrections** — Phases 9-13 (shipped 2026-04-01)
- ✅ **v1.3 Visualization Overhaul** — Phases 14-19 (shipped 2026-04-02)
- ✅ **v1.4 Posterior Numerical Stability** — Phases 21-23 (shipped 2026-04-02)

## Phases

<details>
<summary>✅ v1.0 EMRI HPC Integration (Phases 1-5) — SHIPPED 2026-03-27</summary>

- [x] Phase 1: Code Hardening (2/2 plans) — CPU-safe imports, --use_gpu/--num_workers CLI flags
- [x] Phase 2: Batch Compatibility (1/1 plan) — Non-interactive merge/prepare scripts with entry points
- [x] Phase 3: Cluster Environment (1/1 plan) — modules.sh + setup.sh for bwUniCluster 3.0
- [x] Phase 4: SLURM Job Infrastructure (3/3 plans) — simulate/merge/evaluate pipeline with dependency chaining
- [x] Phase 5: Documentation (2/2 plans) — cluster/README.md quickstart, CLAUDE.md/README.md sections

Full details: `.planning/milestones/v1.0-ROADMAP.md`

</details>

<details>
<summary>✅ v1.1 Clean Simulation Campaign (Phases 6-8) — SHIPPED 2026-03-29</summary>

- [x] Phase 6: Data Cleanup (1/1 plan) — Removed stale artifacts, verified .gitignore
- [x] Phase 7: Cluster Access (1/1 plan) — SSH ControlMaster, environment preflight
- [x] Phase 8: Simulation Campaign (2/2 plans) — Smoke-test pipeline, validation, H0 posterior

Full details: `.planning/milestones/v1.1-ROADMAP.md`

</details>

<details>
<summary>✅ v1.2 Production Campaign & Physics Corrections (Phases 9-13) — SHIPPED 2026-04-01</summary>

- [x] Phase 9: Galactic Confusion Noise (1/1 plan) — Added galactic foreground to LISA PSD
- [x] Phase 10: Five-Point Stencil Derivatives (1/1 plan) — O(epsilon^4) Fisher matrix derivatives
- [x] Phase 11: Validation Campaign (2/2 plans) — Corrected physics validated at small scale
- [x] Phase 11.1: Simulation-Based Detection Probability (5/5 plans) — Replaced KDE P_det with simulation-based
- [x] Phase 12: Production Campaign (1/1 plan) — 100+ task campaign on bwUniCluster
- [x] Phase 13: H0 Posterior Sweep (1/1 plan) — Full H0 posterior over [0.6, 0.9]

Full details: `.planning/milestones/v1.2-ROADMAP.md`

</details>

<details>
<summary>✅ v1.3 Visualization Overhaul (Phases 14-19) — SHIPPED 2026-04-02</summary>

- [x] Phase 14: Test Infrastructure & Safety Net (2/2 plans) — 23 smoke tests + rcParams regression test
- [x] Phase 15: Style Infrastructure (1/1 plan) — _colors.py, _labels.py, REVTeX presets, LaTeX toggle
- [x] Phase 16: Data Layer & Fisher Visualizations (2/2 plans) — _data.py CRB loader, fisher_plots.py factory functions
- [x] Phase 17: Enhanced Existing Plots (3/3 plans) — credible intervals, heatmaps with contours, injected-vs-recovered scatter
- [x] Phase 18: New Plot Modules (2/2 plans) — Mollweide sky map, corner plots, H0 convergence, efficiency curve
- [x] Phase 19: Campaign Dashboards & Batch Generation (2/2 plans) — 2×2 dashboard mosaic, 15-figure manifest pipeline

Full details: `.planning/milestones/v1.3-ROADMAP.md`

</details>

<details>
<summary>✅ v1.4 Posterior Numerical Stability (Phases 21-23) — SHIPPED 2026-04-02</summary>

- [x] Phase 21: Analysis & Post-Processing (2/2 plans) — Log-space combination script, 4 strategies, diagnostics, comparison table
- [x] Phase 22: Likelihood Floor & Overflow Fix (1/1 plan) — Physics-motivated floor in single_host_likelihood, underflow detection
- [x] Phase 23: Deploy & Validate (2/2 plans) — Deployed to cluster at 5793f70, validation PASS (|Δ MAP|=0.00 < 0.05)

Full details: `.planning/milestones/v1.4-ROADMAP.md`

</details>

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Code Hardening | v1.0 | 2/2 | Complete | 2026-03-26 |
| 2. Batch Compatibility | v1.0 | 1/1 | Complete | 2026-03-26 |
| 3. Cluster Environment | v1.0 | 1/1 | Complete | 2026-03-27 |
| 4. SLURM Job Infrastructure | v1.0 | 3/3 | Complete | 2026-03-27 |
| 5. Documentation | v1.0 | 2/2 | Complete | 2026-03-27 |
| 6. Data Cleanup | v1.1 | 1/1 | Complete | 2026-03-27 |
| 7. Cluster Access | v1.1 | 1/1 | Complete | 2026-03-28 |
| 8. Simulation Campaign | v1.1 | 2/2 | Complete | 2026-03-29 |
| 9. Galactic Confusion Noise | v1.2 | 1/1 | Complete | 2026-03-29 |
| 10. Five-Point Stencil | v1.2 | 1/1 | Complete | 2026-03-29 |
| 11. Validation Campaign | v1.2 | 2/2 | Complete | 2026-04-01 |
| 11.1 Simulation-Based P_det | v1.2 | 5/5 | Complete | 2026-04-01 |
| 12. Production Campaign | v1.2 | 1/1 | Complete | 2026-04-01 |
| 13. H0 Posterior Sweep | v1.2 | 1/1 | Complete | 2026-04-01 |
| 14. Test Infrastructure | v1.3 | 2/2 | Complete | 2026-04-01 |
| 15. Style Infrastructure | v1.3 | 1/1 | Complete | 2026-04-01 |
| 16. Data Layer & Fisher | v1.3 | 2/2 | Complete | 2026-04-02 |
| 17. Enhanced Existing Plots | v1.3 | 3/3 | Complete | 2026-04-02 |
| 18. New Plot Modules | v1.3 | 2/2 | Complete | 2026-04-02 |
| 19. Campaign Dashboards | v1.3 | 2/2 | Complete | 2026-04-02 |
| 21. Analysis & Post-Processing | v1.4 | 2/2 | Complete | 2026-04-02 |
| 22. Likelihood Floor & Overflow Fix | v1.4 | 1/1 | Complete | 2026-04-02 |
| 23. Deploy & Validate | v1.4 | 2/2 | Complete | 2026-04-02 |
