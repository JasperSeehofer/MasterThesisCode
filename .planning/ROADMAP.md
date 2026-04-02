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

### Phase 21: Analysis & Post-Processing
**Goal**: The zero-likelihood problem is fully documented, all combination methods are compared quantitatively, and a robust post-processing script combines per-event posteriors in log-space
**Depends on**: Phase 13 (v1.2 campaign data exists)
**Requirements**: ANAL-01, ANAL-02, POST-01, NFIX-01
**Success Criteria** (what must be TRUE):
  1. A diagnostic report identifies which events produce zero likelihoods at which h-bins, with root causes documented (no hosts in error volume, catalog coverage gaps, redshift mismatch)
  2. A comparison table shows MAP estimates and posterior shapes for all four combination methods (naive, Option 1 exclude-zeros, Option 2 per-event-floor, Option 3 physics-floor) across both BH mass variants
  3. A standalone combination script loads per-event posterior JSONs and produces the joint H0 posterior using `np.sum(np.log(...))` with a log-shift-exp trick to avoid underflow
  4. The combination script accepts a CLI flag or config option to select zero-handling strategy (Option 1, 2, or 3)
  5. Running the combination script on existing campaign data reproduces the known naive MAP values (0.72 with BH mass, 0.86 without) as a sanity check
**Plans:** 2/2 plans complete
Plans:
- [x] 21-01-PLAN.md — Core posterior combination module with all strategies, log-space accumulation, diagnostics, comparison table + unit tests
- [x] 21-02-PLAN.md — CLI wiring (--combine, --strategy flags) + integration tests against real campaign data

### Phase 22: Likelihood Floor & Overflow Fix
**Goal**: The evaluate pipeline computes physically grounded likelihoods for all events (no zeros from catalog gaps) and detects numerical underflow correctly
**Depends on**: Phase 21
**Requirements**: NFIX-02, NFIX-03
**Success Criteria** (what must be TRUE):
  1. `single_host_likelihood` in `bayesian_statistics.py` returns a physically motivated floor value (not zero) when no host galaxy produces nonzero likelihood, following `/physics-change` protocol with documented derivation
  2. `check_overflow` (or its replacement) detects product-to-zero underflow in addition to overflow-to-inf, logging a warning when the posterior product would collapse to zero
  3. Running the evaluate pipeline on a subset of campaign data produces no zero-valued posterior bins (all h-bins have nonzero likelihood for every event)
**Plans:** 1/1 plans complete
Plans:
- [x] 22-01-PLAN.md — Implement per-event-min physics floor in posterior_combination.py + remove check_overflow dead code

### Phase 23: Deploy & Validate
**Goal**: Updated code is running on the cluster and validated against existing baselines before the pending evaluate jobs execute
**Depends on**: Phase 22
**Requirements**: DEPL-01, DEPL-02
**Success Criteria** (what must be TRUE):
  1. The updated codebase (with log-space accumulation, physics floor, and overflow fix) is deployed to `~/MasterThesisCode` on bwUniCluster before the evaluate SLURM jobs start
  2. A validation run produces H0 posteriors that are compared against existing baselines: naive (MAP=0.72/0.86), Option 1 (MAP=0.68/0.66), confirming the new method produces physically reasonable results
  3. The validation results and baseline comparison are documented (saved to working directory or committed)
**Plans:** 2/2 plans complete
Plans:
- [x] 23-01-PLAN.md — Local validation: three-strategy comparison + v1.4-validation.md
- [x] 23-02-PLAN.md — Deploy to bwUniCluster via git merge/push/pull

## Progress

**Execution Order:**
Phases execute in numeric order: 21 -> 22 -> 23

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
| 9. Galactic Confusion Noise | v1.2 | 1/1 | Complete | - |
| 10. Five-Point Stencil | v1.2 | 1/1 | Complete | - |
| 11. Validation Campaign | v1.2 | 2/2 | Complete | - |
| 11.1 Simulation-Based P_det | v1.2 | 5/5 | Complete | 2026-04-01 |
| 12. Production Campaign | v1.2 | 1/1 | Complete | - |
| 13. H0 Posterior Sweep | v1.2 | 1/1 | Complete | - |
| 14. Test Infrastructure | v1.3 | 2/2 | Complete | 2026-04-01 |
| 15. Style Infrastructure | v1.3 | 1/1 | Complete | 2026-04-01 |
| 16. Data Layer & Fisher | v1.3 | 2/2 | Complete | 2026-04-02 |
| 17. Enhanced Existing Plots | v1.3 | 3/3 | Complete | 2026-04-02 |
| 18. New Plot Modules | v1.3 | 2/2 | Complete | 2026-04-02 |
| 19. Campaign Dashboards | v1.3 | 2/2 | Complete | 2026-04-02 |
| 21. Analysis & Post-Processing | v1.4 | 2/2 | Complete    | 2026-04-02 |
| 22. Likelihood Floor & Overflow Fix | v1.4 | 1/1 | Complete    | 2026-04-02 |
| 23. Deploy & Validate | v1.4 | 2/2 | Complete    | 2026-04-02 |
