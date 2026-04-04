# Roadmap: EMRI Parameter Estimation -- Dark Siren H0 Inference

## Milestones

- Done: **v1.0** EMRI HPC Integration (Shipped: 2026-03-27) -- 5 phases (GSD-tracked)
- Done: **v1.1** Clean Simulation Campaign (Shipped: 2026-03-29) -- 3 phases (GSD-tracked)
- Active: **v1.2** Production Campaign & Physics Corrections -- Phases 9-13 (GSD-tracked, last completed: Phase 10)
- On Hold: **v1.2.1** "With BH Mass" Likelihood Bias Audit -- Phases 14-16 (last completed: Phase 15, awaiting P_det data for Phase 16)
- Done: **v1.2.2** Injection Campaign Physics Analysis (Completed: 2026-04-01) -- Phases 17-20

## Phases

<details>
<summary>On Hold: v1.2.1 "With BH Mass" Likelihood Bias Audit (Phases 14-16)</summary>

### Phase 14: First-Principles Derivation

**Goal:** The correct "with BH mass" dark siren likelihood is derived from first principles
**Status:** Complete (2026-03-31)
**Plans:** 2/2 complete

### Phase 15: Code Audit & Fix

**Goal:** Every term in bayesian_statistics.py matches the Phase 14 derivation
**Status:** Complete (2026-03-31)
**Plans:** 2/2 complete
**Result:** /(1+z) fix applied but insufficient -- posterior still monotonically decreasing

### Phase 16: Validation

**Goal:** Numerical evidence confirms both likelihood channels produce consistent H0 posteriors
**Status:** Not started (blocked on injection P_det data)

</details>

<details>
<summary>Done: v1.2.2 Injection Campaign Physics Analysis (Phases 17-20) -- COMPLETED 2026-04-01</summary>

- [x] Phase 17: Injection Physics Audit (2/2 plans) -- completed 2026-03-31
- [x] Phase 18: Detection Yield & Grid Quality (2/2 plans) -- completed 2026-04-01
- [x] Phase 19: Enhanced Sampling Design (2/2 plans) -- completed 2026-04-01
- [x] Phase 20: Validation (2/2 plans) -- completed 2026-04-01

See `.gpd/milestones/v1.2.2-ROADMAP.md` for full archive.

</details>

### Phase 24: Completeness Estimation

**Goal:** GLADE+ completeness fraction f(z) is computed from the actual catalog data and available as an interpolatable function
**Status:** Complete (2026-04-04)
**Plans:** 1/1 complete

Plans:
- [x] 24-01-PLAN.md -- Refactor GLADE+ completeness module with f(z, h) interface and comoving volume element
**Result:** f(z,h) interface delivered with 23 tests passing. Verification: 4/5 claims confirmed; data provenance (number vs luminosity completeness) deferred to Phase 25 review.

### Phase 25: Likelihood Correction

**Goal:** The dark siren likelihood combines catalog and completion terms weighted by f(z), implementing Gray et al. (2020) Eq. 9
**Status:** Complete (2026-04-04)
**Plans:** 1/1 complete

Plans:
- [x] 25-01-PLAN.md -- Implement completeness-corrected likelihood (completion term + combination formula + tests)
**Result:** Gray et al. (2020) Eq. 9 combination formula implemented: p_i = f_i * L_cat + (1-f_i) * L_comp. Completion term via fixed_quad. 11 tests, 4/4 contract claims verified (HIGH confidence).
