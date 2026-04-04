# Roadmap: EMRI Parameter Estimation

## Milestones

- ✅ **v1.0 EMRI HPC Integration** — Phases 1-5 (shipped 2026-03-27)
- ✅ **v1.1 Clean Simulation Campaign** — Phases 6-8 (shipped 2026-03-29)
- ✅ **v1.2 Production Campaign & Physics Corrections** — Phases 9-13 (shipped 2026-04-01)
- ✅ **v1.3 Visualization Overhaul** — Phases 14-19 (shipped 2026-04-02)
- ✅ **v1.4 Posterior Numerical Stability** — Phases 21-23 (shipped 2026-04-02)
- 🔄 **v1.5 Galaxy Catalog Completeness Correction** — Phases 24-27 (in progress)

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
- [x] Phase 19: Campaign Dashboards & Batch Generation (2/2 plans) — 2x2 dashboard mosaic, 15-figure manifest pipeline

Full details: `.planning/milestones/v1.3-ROADMAP.md`

</details>

<details>
<summary>✅ v1.4 Posterior Numerical Stability (Phases 21-23) — SHIPPED 2026-04-02</summary>

- [x] Phase 21: Analysis & Post-Processing (2/2 plans) — Log-space combination script, 4 strategies, diagnostics, comparison table
- [x] Phase 22: Likelihood Floor & Overflow Fix (1/1 plan) — Physics-motivated floor in single_host_likelihood, underflow detection
- [x] Phase 23: Deploy & Validate (2/2 plans) — Deployed to cluster at 5793f70, validation PASS (|delta MAP|=0.00 < 0.05)

Full details: `.planning/milestones/v1.4-ROADMAP.md`

</details>

### v1.5 Galaxy Catalog Completeness Correction

- [ ] **Phase 24: Completeness Estimation** — Compute GLADE+ f(z) from B-band luminosity comparison
- [ ] **Phase 25: Likelihood Correction** — Add completion term + comoving volume element (Gray et al. 2020 physics changes)
- [ ] **Phase 26: Verification** — Limiting case checks + bias reduction on 534-detection dataset
- [ ] **Phase 27: Cluster Deployment** — Deploy corrected code and run production evaluation

## Phase Details

### Phase 24: Completeness Estimation
**Goal**: GLADE+ completeness fraction f(z) is computed from the actual catalog data and available as an interpolatable function
**Depends on**: Nothing (self-contained analysis of existing catalog)
**Requirements**: COMP-01, COMP-02
**Success Criteria** (what must be TRUE):
  1. f(z) curve is computed from B-band luminosity comparison against Schechter function and matches the Dalya et al. (2022) figures: ~90% at z=0.029, declining to <<50% at z>0.11
  2. f(z) returns values in [0, 1] for all z in the EMRI detection range (z=0.03–0.20) with no extrapolation failures
  3. f(z) is available as an interpolating function callable from bayesian_statistics.py without loading the full catalog each time
**Plans**: TBD

### Phase 25: Likelihood Correction
**Goal**: The dark siren likelihood combines catalog and completion terms weighted by f(z), implementing Gray et al. (2020) Eq. 9
**Depends on**: Phase 24
**Requirements**: LIKE-01, LIKE-02, LIKE-03, LIKE-04
**Success Criteria** (what must be TRUE):
  1. dV_c/dz/dOmega (comoving volume element) is implemented in physical_relations.py with a reference comment to Gray et al. (2020) Appendix A.2.3
  2. p_Di() computes L_comp^i(h) (completion term integral over comoving volume prior) and combines it with the existing catalog term as: f_i * L_cat^i + (1 - f_i) * L_comp^i
  3. The completeness function f(z) threads through evaluate() -> p_D() -> p_Di() without any hardcoded values
  4. All physics changes carry Physics Change Protocol sign-offs and reference comments (arXiv:1908.06050 equation numbers)
**Plans**: TBD
**UI hint**: no

### Phase 26: Verification
**Goal**: The corrected likelihood passes all limiting-case checks and reduces the H0 bias on the 534-detection dataset
**Depends on**: Phase 25
**Requirements**: VER-01, VER-02, VER-03, VER-04
**Success Criteria** (what must be TRUE):
  1. Setting f=1 everywhere produces a posterior identical (to numerical precision) to the pre-correction code on the same dataset
  2. Setting f=0 everywhere produces a broad posterior with peak within 5% of h=0.73 (statistical siren result, no catalog information)
  3. Running the corrected likelihood on the 534-detection dataset produces a MAP estimate visibly shifted toward h=0.73 compared to the biased MAP=0.66 baseline
  4. Perturbing f(z) by +/-20% shifts the posterior peak by less than 20% of the peak shift from bias correction, confirming the result is not dominated by completeness uncertainty
**Plans**: TBD

### Phase 27: Cluster Deployment
**Goal**: The corrected code runs on bwUniCluster and produces a validated production H0 posterior with real P_det
**Depends on**: Phase 26
**Requirements**: DEPL-01
**Success Criteria** (what must be TRUE):
  1. Corrected code is deployed to bwUniCluster and the evaluate job completes without error
  2. Production posterior (with real P_det) shows MAP closer to h=0.73 than the pre-correction baseline (MAP=0.66)
  3. Deployment commit is tagged and the cluster run is recorded in STATE.md with commit hash
**Plans**: TBD

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
| 24. Completeness Estimation | v1.5 | 0/? | Not started | - |
| 25. Likelihood Correction | v1.5 | 0/? | Not started | - |
| 26. Verification | v1.5 | 0/? | Not started | - |
| 27. Cluster Deployment | v1.5 | 0/? | Not started | - |
