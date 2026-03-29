# Roadmap: EMRI Parameter Estimation

## Milestones

- ✅ **v1.0 EMRI HPC Integration** — Phases 1-5 (shipped 2026-03-27)
- ✅ **v1.1 Clean Simulation Campaign** — Phases 6-8 (shipped 2026-03-29)
- 🚧 **v1.2 Production Campaign & Physics Corrections** — Phases 9-13 (in progress)

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

### 🚧 v1.2 Production Campaign & Physics Corrections (In Progress)

**Milestone Goal:** Fix known physics bugs in the Fisher matrix and PSD, then run a production-scale simulation campaign with full H0 posterior sweep.

- [ ] **Phase 9: Galactic Confusion Noise** - Add galactic foreground confusion noise to LISA PSD
- [ ] **Phase 10: Five-Point Stencil Derivatives** - Upgrade Fisher matrix from forward-difference to O(epsilon^4) stencil
- [ ] **Phase 11: Validation Campaign** - Verify corrected physics with small test run and calibrate thresholds
- [ ] **Phase 12: Production Campaign** - Run 100+ task simulation on bwUniCluster with corrected physics
- [ ] **Phase 13: H0 Posterior Sweep** - Evaluate full Hubble constant posterior over [0.6, 0.9]

## Phase Details

### Phase 9: Galactic Confusion Noise
**Goal**: LISA PSD includes the galactic confusion foreground, producing physically correct SNR estimates
**Depends on**: Phase 8 (v1.1 cluster pipeline operational)
**Requirements**: PHYS-02
**Success Criteria** (what must be TRUE):
  1. `power_spectral_density_a_channel()` returns a PSD that includes the galactic confusion noise term from Babak et al. (2023) Eq. 17
  2. PSD at 1 mHz with confusion noise is measurably larger than PSD without confusion noise
  3. Existing CPU tests pass with the updated PSD (no regression)
  4. A reference comment citing Babak et al. (2023) arXiv:2303.15929 appears above the confusion noise implementation
**Plans**: TBD

### Phase 10: Five-Point Stencil Derivatives
**Goal**: Fisher matrix uses O(epsilon^4) five-point stencil derivatives, producing accurate Cramer-Rao bounds
**Depends on**: Phase 9
**Requirements**: PHYS-01, PHYS-03
**Success Criteria** (what must be TRUE):
  1. `compute_fisher_information_matrix()` calls `five_point_stencil_derivative()` instead of `finite_difference_derivative()`
  2. CRB computation timeout is increased to at least 120 seconds to accommodate the ~4x increase in waveform evaluations
  3. Fisher matrix condition numbers are logged for each event, enabling detection of ill-conditioned matrices
  4. A reference comment citing Vallisneri (2008) arXiv:gr-qc/0703086 appears at the call site
**Plans**: TBD

### Phase 11: Validation Campaign
**Goal**: Corrected physics produces valid results at small scale, with calibrated timeouts and thresholds for the production run
**Depends on**: Phase 10
**Requirements**: SIM-01, SIM-03
**Success Criteria** (what must be TRUE):
  1. A 3-5 task validation campaign completes on bwUniCluster with the corrected PSD and Fisher derivatives
  2. Detection rates and d_L error distributions are compared against v1.1 smoke test results to confirm physics changes behave as expected
  3. d_L fractional error threshold is recalibrated based on observed stencil accuracy (tightened from 10% if data supports it)
  4. Per-task wall time fits within SLURM limits, confirming timeout and job parameters are ready for production scale
**Plans**: TBD

### Phase 12: Production Campaign
**Goal**: A statistically sufficient catalog of Cramer-Rao bounds exists for meaningful H0 inference
**Depends on**: Phase 11
**Requirements**: SIM-02
**Success Criteria** (what must be TRUE):
  1. A 100+ task simulation campaign completes on bwUniCluster, producing merged Cramer-Rao bounds CSV
  2. The detection catalog contains enough events (order 1000+) for statistically meaningful Bayesian inference
  3. `run_metadata.json` records the git commit, seed, and SLURM job IDs for full reproducibility
**Plans**: TBD

### Phase 13: H0 Posterior Sweep
**Goal**: A full Hubble constant posterior with credible intervals is computed from the production detection catalog
**Depends on**: Phase 12
**Requirements**: H0-01, H0-02
**Success Criteria** (what must be TRUE):
  1. The evaluation pipeline runs over a grid of h-values spanning [0.6, 0.9], producing per-h likelihood values
  2. A combined posterior plot shows the normalized H0 posterior with MAP estimate and 68%/95% credible intervals
  3. All posterior results are saved to files for inclusion in the thesis
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 9 -> 10 -> 11 -> 12 -> 13

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
| 9. Galactic Confusion Noise | v1.2 | 0/0 | Not started | - |
| 10. Five-Point Stencil Derivatives | v1.2 | 0/0 | Not started | - |
| 11. Validation Campaign | v1.2 | 0/0 | Not started | - |
| 12. Production Campaign | v1.2 | 0/0 | Not started | - |
| 13. H0 Posterior Sweep | v1.2 | 0/0 | Not started | - |
