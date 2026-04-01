# Roadmap: EMRI Parameter Estimation -- Dark Siren H0 Inference

## Milestones

- Done: **v1.0** EMRI HPC Integration (Shipped: 2026-03-27) -- 5 phases (GSD-tracked)
- Done: **v1.1** Clean Simulation Campaign (Shipped: 2026-03-29) -- 3 phases (GSD-tracked)
- Active: **v1.2** Production Campaign & Physics Corrections -- Phases 9-13 (GSD-tracked, last completed: Phase 10)
- On Hold: **v1.2.1** "With BH Mass" Likelihood Bias Audit -- Phases 14-16 (last completed: Phase 15, awaiting P_det data for Phase 16)
- Active: **v1.2.2** Injection Campaign Physics Analysis -- Phases 17-20

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

### Active: v1.2.2 Injection Campaign Physics Analysis (Phases 17-20)

**Milestone Goal:** Audit injection campaign physics, analyze detection yield, and design enhanced sampling strategies to improve P_det grid quality with fewer GPU hours.

## Contract Overview

| Contract Item | Advanced By Phase(s) | Status |
| --- | --- | --- |
| Injection physics consistency audit | Phase 17 | Planned |
| Detection yield report with quantified waste | Phase 18 | Planned |
| P_det grid quality assessment with per-bin CIs | Phase 18 | Planned |
| Enhanced sampling design with expected improvement factor | Phase 19 | Planned |
| Validation: enhanced sampling produces unbiased P_det | Phase 20 | Planned |
| Acceptance signal: >2x GPU time reduction for equivalent grid quality | Phase 19, 20 | Planned |

## Phase Dependencies

| Phase | Depends On | Enables | Critical Path? |
|-------|-----------|---------|:-:|
| 17 - Injection Physics Audit | -- | 18 | Yes |
| 18 - Detection Yield & Grid Quality | 17 | 19 | Yes |
| 19 - Enhanced Sampling Design | 18 | 20 | Yes |
| 20 - Validation | 19 | -- | Yes |

**Critical path:** 17 -> 18 -> 19 -> 20 (strictly sequential)

**Data dependency note:** Phase 17 is primarily code-audit and can proceed without injection CSVs. Phase 18 requires the rsynced injection data from bwUniCluster. If data is unavailable, Phase 17 completes fully while Phase 18 blocks on data arrival.

**Late research finding:** LVK does NOT grid P_det — they compute the selection integral alpha(h) as a direct MC sum over all injections (Mandel, Farr & Gair 2019). The grid-based approach works but is non-standard. Phase 18 should compare grid-based vs direct MC alpha(h). Phase 19 should consider whether the direct MC integral makes grid improvements unnecessary.

## Phase Details

### Phase 17: Injection Physics Audit

**Goal:** The injection campaign parameter distributions and cosmological model are verified consistent with the simulation pipeline, and waveform failure patterns are characterized
**Depends on:** Nothing (entry point; primarily code-reading, minimal data dependency)
**Requirements:** AUDT-01, AUDT-02, AUDT-03
**Contract Coverage:**
- Advances: Injection physics consistency audit
- Deliverables: Parameter consistency report (injection_campaign() vs data_simulation()); cosmological model consistency check (dist() usage in both pipelines); waveform failure characterization by parameter region
- Anchor coverage: Phase 11.1 design decisions D-01 through D-09; injection_campaign() in main.py:396-572; Model1CrossCheck EMRI sampling; ParameterSpace parameter bounds
- Forbidden proxies: "Looks similar" without line-by-line code comparison; failure rate estimate without parameter-region breakdown

**Success Criteria** (what must be TRUE when this phase completes):

1. injection_campaign() and data_simulation() draw from identical parameter distributions (M, z, spin, sky angles, eccentricity) -- verified by line-by-line comparison of Model1CrossCheck and ParameterSpace usage in both code paths
2. The cosmological model (dist() function) is called with identical parameters (Omega_m, h) in both injection and evaluation pipelines -- any discrepancy documented with impact assessment
3. The d_L-to-z round-trip inversion is consistent to within 1e-4 relative error across the z range [0, 0.5] for all 7 h-values
4. Waveform failure modes are categorized (timeout, parameter bounds, solver divergence) and their correlation with (z, M) regions is assessed from SLURM logs or injection CSV status columns
5. The z > 0.5 importance sampling cutoff is evaluated: confirmed safe for h in [0.60, 0.90] by SNR scaling argument (SNR ~ M^{5/6}/d_L, and d_L(z=0.5, h=0.60) is documented)

**Plans:** 2/2 complete

Plans:
- [x] 17-01-PLAN.md -- Parameter distribution comparison (AUDT-01) + cosmological model consistency and d_L round-trip test (AUDT-02)
- [x] 17-02-PLAN.md -- Waveform failure characterization by exception type and parameter region (AUDT-03)

### Phase 18: Detection Yield & Grid Quality

**Goal:** The detection yield is quantified per h-value, compute waste is broken down by cause, and the P_det grid quality is assessed with per-bin confidence intervals
**Depends on:** Phase 17 (parameter consistency verified before analyzing data)
**Requirements:** YELD-01, YELD-02, YELD-03, GRID-01, GRID-02, GRID-03
**Contract Coverage:**
- Advances: Detection yield report with quantified waste; P_det grid quality assessment with per-bin CIs
- Deliverables: Detection fraction table per h-value (3 significant figures); compute waste breakdown pie chart (failures / undetectable / detected); per-bin Wilson CI heatmap for 30x20 grid; 30x20 vs 15x10 grid comparison; quality flag implementation in SimulationDetectionProbability
- Anchor coverage: Injection CSV data from cluster; SimulationDetectionProbability class; Farr (2019) N_eff > 4*N_det criterion; Brown, Cai, DasGupta (2001) for Wilson CI recommendation
- Forbidden proxies: Estimated yield without actual data; average CI without per-bin breakdown; grid comparison without interpolation error metric

**Success Criteria** (what must be TRUE when this phase completes):

1. Detection fraction (N_det / N_total) is computed per h-value to 3 significant figures, and the overall waste fraction (undetectable + failed) is quantified
2. GPU compute waste is decomposed into three categories with percentages: (a) waveform generation failures, (b) sub-threshold events (SNR < 20), (c) successful detections
3. The z > 0.5 cutoff is validated: zero detections above z = 0.5 confirmed for all 7 h-values, or exceptions documented with their (z, M, h) coordinates
4. Per-bin Wilson 95% confidence intervals are computed for the 30x20 grid, with bins having fewer than 10 injections flagged as unreliable
5. The 30x20 vs 15x10 grid comparison shows whether the coarser grid achieves CI half-widths below 0.15 in the detection boundary region (where 0.05 < P_det < 0.95)

**Status:** Complete (2026-04-01)
**Plans:** 2/2 complete

Plans:
- [x] 18-01-PLAN.md -- Detection yield per h-value, compute waste breakdown, z>0.5 cutoff validation
- [x] 18-02-PLAN.md -- Per-bin Wilson CIs for 30x20 grid, 30x20 vs 15x10 comparison, quality flag implementation
**Result:** 663 detections at SNR>=15 (f_det 0.22-0.81%); 15x10 grid recommended (3.2x better CIs); z>0.5 cutoff safe; quality flags added to SimulationDetectionProbability

### Phase 19: Enhanced Sampling Design

**Goal:** An importance-weighted histogram estimator and stratified sampling strategy are designed that provably reduce GPU time by >2x for equivalent P_det grid quality
**Depends on:** Phase 18 (grid quality diagnostics identify where improvements are needed)
**Requirements:** SMPL-01, SMPL-02, SMPL-03
**Contract Coverage:**
- Advances: Enhanced sampling design with expected improvement factor; acceptance signal (>2x GPU time reduction)
- Deliverables: Self-normalized IS estimator implementation compatible with SimulationDetectionProbability; stratified sampling allocation algorithm with Neyman-optimal bin budgets; two-stage pilot design (30% uniform + 70% targeted) with combined importance weights
- Anchor coverage: Tiwari (2018) arXiv:1712.00482 for IS estimator; Owen (2013) for stratified sampling theory; Phase 18 boundary identification and quality metrics; Kish (1965) for effective sample size
- Forbidden proxies: Sampling strategy that introduces bias in P_det estimates; proposal distribution without importance weight correction; improvement factor claimed without variance reduction calculation

**Success Criteria** (what must be TRUE when this phase completes):

1. The self-normalized IS estimator is implemented and recovers the standard histogram estimator when importance weights are uniform (w_i = 1 for all i) -- verified numerically to machine precision
2. The stratified allocation algorithm concentrates injections on detection boundary bins (0.05 < P_det < 0.95) using Neyman-optimal allocation based on Phase 18 pilot variance estimates
3. The two-stage design (30% pilot + 70% targeted) is specified with: (a) pilot sample size per h-value, (b) boundary identification algorithm, (c) targeted proposal distribution, (d) combined weight formula
4. Expected variance reduction factor is computed from Phase 18 data: the boundary-region CI half-width reduction is >2x compared to uniform allocation with the same total injection count
5. The proposal distribution q(theta) has full support over the prior p(theta) -- no region has q = 0 where p > 0 (guaranteed by the defensive mixture: alpha*uniform + (1-alpha)*targeted)

**Plans:** 2 plans

Plans:
- [ ] 19-01-PLAN.md -- IS-weighted histogram estimator implementation with uniform-weight recovery test (SMPL-01)
- [ ] 19-02-PLAN.md -- Neyman-optimal allocation, VRF computation, two-stage design with defensive mixture (SMPL-02, SMPL-03)

### Phase 20: Validation

**Goal:** The enhanced sampling design is verified to produce unbiased P_det estimates consistent with the uniform baseline
**Depends on:** Phase 19 (enhanced sampling implemented)
**Requirements:** VALD-01
**Contract Coverage:**
- Advances: Validation that enhanced sampling produces unbiased P_det
- Deliverables: Side-by-side comparison of uniform vs enhanced P_det grids; per-bin statistical test results; P_det monotonicity verification; round-trip consistency check
- Anchor coverage: Uniform baseline P_det grid (Phase 18); enhanced P_det grid (Phase 19); Farr (2019) N_eff criterion; physical boundary conditions (P_det -> 1 at low z, P_det -> 0 at high z)
- Forbidden proxies: Qualitative "looks right" without statistical test; comparison on globally-averaged P_det without per-bin breakdown

**Success Criteria** (what must be TRUE when this phase completes):

1. Per-bin P_det from enhanced sampling agrees with uniform baseline within 2-sigma Wilson CIs for all bins with sufficient statistics (N_total >= 10 in both methods)
2. P_det is monotonically non-increasing in z at fixed M (within statistical noise) -- any non-monotonic structure flagged as potential waveform failure artifact
3. Physical boundary conditions satisfied: P_det -> 1 at (low z, high M) and P_det -> 0 at (high z, low M) for all h-values
4. The Farr (2019) criterion N_eff > 4 * N_det is satisfied globally for the enhanced sampling, with per-bin N_eff reported

**Plans:** TBD

## Risk Register

| Phase | Top Risk | Probability | Impact | Mitigation |
|-------|---------|:-:|:-:|-----------|
| 17 | Parameter distribution mismatch between injection and simulation | LOW | HIGH | Line-by-line code comparison; any mismatch immediately documented and escalated |
| 18 | Injection data not yet rsynced from cluster | HIGH | HIGH | Phase 17 (code audit) proceeds independently; Phase 18 blocks cleanly on data |
| 18 | Waveform failures dominate compute waste (>60%) | MEDIUM | MEDIUM | Separate failure analysis from sampling design; enhanced sampling cannot fix failures |
| 19 | Importance weights have high variance (low N_eff) | MEDIUM | HIGH | Defensive mixture proposal caps weight ratio; Kish N_eff diagnostic catches this |
| 20 | Enhanced and uniform grids disagree beyond 2-sigma | LOW | MEDIUM | Debug: check weight normalization, proposal support, bin boundary alignment |

## Backtracking Triggers

- Phase 17: If injection and simulation pipelines use DIFFERENT cosmological parameters or parameter distributions, STOP and escalate -- existing P_det grids may be invalid
- Phase 18: If detection yield is >50%, enhanced sampling provides diminishing returns -- reassess Phase 19 scope (contract stop condition)
- Phase 19: If the defensive mixture proposal yields N_eff < 10 in boundary bins, increase the uniform fraction alpha or increase total injection count
- Phase 20: If per-bin disagreement exceeds 2-sigma in >10% of bins, return to Phase 19 and debug importance weight computation

## Progress

**Execution Order:** 17 -> 18 -> 19 -> 20

| Phase | Plans Complete | Status | Completed |
| --- | --- | --- | --- |
| 17. Injection Physics Audit | 2/2 | Complete | 2026-03-31 |
| 18. Detection Yield & Grid Quality | 2/2 | Complete | 2026-04-01 |
| 19. Enhanced Sampling Design | 0/2 | In progress | - |
| 20. Validation | TBD | Not started | - |
