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

## v1.5 Completeness Correction (Phases 24-25) -- COMPLETED 2026-04-04

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

## v2.0 Paper (Phases 26-28) -- Active

### Phase 26: Paper Draft

**Goal:** First complete draft of the PRD paper "Constraints on the Hubble constant from EMRI dark sirens with LISA using the massive black hole mass"
**Status:** Complete (2026-04-05)
**Plans:** 1/1 complete
**Result:** All sections drafted (Introduction, Method, Results, Discussion, Conclusions, Appendix A). 11-page PDF builds with REVTeX4-2. 21 references. 25 RESULT PENDING markers awaiting production run.

### Phase 27: Production Run & Figures

**Goal:** Extract numerical results from existing completeness-corrected cluster data, generate publication figures, fill all 25 \pending{} and 4 \todo{} markers in the paper
**Status:** Planned (4 plans)
**Depends on:** Phase 25 (completeness code), Phase 26 (paper structure)
**Plans:** 4 plans

Plans:
- [ ] 27-01-PLAN.md -- Fix pipeline bugs (directory name, corrupted JSON) and validate data integrity
- [ ] 27-02-PLAN.md -- Extract MAP, CI, precision, bias; generate posterior comparison and single-event figures
- [ ] 27-03-PLAN.md -- Generate convergence and SNR distribution figures
- [ ] 27-04-PLAN.md -- Fill all \pending{} markers and include figures in paper LaTeX

### Phase 28: Review & Submission

**Goal:** Internal peer review, resolve all TODO markers, finalize co-authors, submit to PRD + arXiv
**Status:** Not started
**Depends on:** Phase 27 (final results and figures)

## v2.2 Pipeline Correctness — Active

### Phase 43: Posterior Calibration Fix

**Goal:** Diagnose and fix the SC-3 MAP=0.860 failure from Phase 40. Determine whether root cause is (H1) D(h) denominator missing from `--combine`/`extract_baseline` code path, (H2) CRBs on disk store equatorial sky angles while v2.2 catalog now expects ecliptic coordinates, or both. Fix confirmed root causes. Re-run `--evaluate` to confirm MAP ≈ 0.73 ± 0.01.
**Status:** Planning (plans created 2026-04-27)
**Routing:** GSD+GPD
**Depends on:** Phase 40 (GAPS_FOUND — SC-3 MAP=0.860; user Q1=insert fix phase)
**Plans:** 3 plans

Plans:
- [ ] 43-01-PLAN.md -- Diagnostic --evaluate run: measure true v2.2 MAP; determine BRANCH-A (H1-only) or BRANCH-B (H1+H2)
- [ ] 43-02-PLAN.md -- Apply fixes: H2 CRB equatorial→ecliptic migration (BRANCH-B); H1 extract_baseline D(h) deprecation/correction
- [ ] 43-03-PLAN.md -- Post-fix --evaluate verification (MAP ≈ 0.73 ± 0.01); VERIFY-04 re-assessment; Phase 42 decision

---

### Phase 32: Completion Term Fix

**Goal:** Fix the systematic low-h bias in the H0 posterior caused by the completion term L_comp. Extend the L_comp denominator to the full detectable volume per Gray et al. (2020) Eq. A.19, precompute D(h) table, and validate that MAP shifts toward h=0.73 with bias shrinking as ~1/sqrt(N).
**Status:** Complete (2026-04-08)
**Depends on:** Phase 25 (completeness code), Phase 27 (production baseline)
**Plans:** 2/2 complete
**Result:** Full-volume D(h) denominator eliminates H0 posterior bias: MAP 0.60→0.73 (both channels), bias -17.8%→0.0% at 59 events (SNR≥20). Validated locally; cluster production run pending.

Plans:
- [x] 32-01-PLAN.md -- Implement D(h) full-volume denominator precomputation and fix L_comp
- [x] 32-02-PLAN.md -- Validate: MAP comparison, bias-vs-N convergence, per-event L_comp decomposition
