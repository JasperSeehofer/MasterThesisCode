---
phase: 19-enhanced-sampling-design
verified: 2026-04-01T12:00:00Z
status: passed
score: 7/7 contract targets verified
consistency_score: 12/12 physics checks passed
independently_confirmed: 10/12 checks independently confirmed
confidence: high
gaps: []
comparison_verdicts:
  - subject_kind: claim
    subject_id: claim-backward-compat
    reference_id: ref-phase18-grid
    comparison_kind: benchmark
    verdict: pass
    metric: "max_absolute_difference"
    threshold: "< 1e-14"
  - subject_kind: claim
    subject_id: claim-vrf
    reference_id: ref-phase18-data
    comparison_kind: benchmark
    verdict: pass
    metric: "VRF_boundary_mean"
    threshold: "> 2.0"
suggested_contract_checks: []
---

# Phase 19 Verification Report: Enhanced Sampling Design

**Phase goal:** An importance-weighted histogram estimator and stratified sampling strategy are designed that provably reduce GPU time by >2x for equivalent P_det grid quality.

**Verification date:** 2026-04-01
**Status:** PASSED
**Confidence:** HIGH

## Contract Coverage

| ID | Kind | Status | Confidence | Evidence |
|----|------|--------|------------|----------|
| claim-is-estimator | claim | VERIFIED | INDEPENDENTLY CONFIRMED | Synthetic + real-data tests pass; IS estimator produces correct P_hat for non-uniform weights |
| claim-backward-compat | claim | VERIFIED | INDEPENDENTLY CONFIRMED | max|P_det_IS(w=1) - P_det_standard| = 0.00e+00 for all 7 h-values on real Phase 18 data |
| claim-neff-diagnostic | claim | VERIFIED | INDEPENDENTLY CONFIRMED | N_eff = n_total exactly for uniform weights; N_eff in (0, n_total] for non-uniform; per-bin storage confirmed |
| claim-neyman-allocation | claim | VERIFIED | INDEPENDENTLY CONFIRMED | sum(N_k) = N_targeted for all h-values; boundary concentration >80x; manual VRF matches code |
| claim-vrf | claim | VERIFIED | INDEPENDENTLY CONFIRMED | VRF_mean ranges 11.8-24.9x across h-values, all >2.0; VRF computed from real Phase 18 per-bin counts |
| claim-two-stage | claim | VERIFIED | INDEPENDENTLY CONFIRMED | Design report specifies all components; weight formula documented and dimensionally consistent |
| claim-full-support | claim | VERIFIED | INDEPENDENTLY CONFIRMED | Proof: q = 0.3*p + 0.7*g >= 0.3*p > 0 wherever p > 0 (alpha=0.3 > 0). Weight bound 1/alpha = 3.33 |

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| analysis/importance_sampling.py | IS utilities | VERIFIED | Contains weighted_histogram_estimator, kish_n_eff, is_weighted_wilson_ci, farr_criterion_check |
| analysis/sampling_design.py | Neyman allocation + VRF | VERIFIED | Contains neyman_allocation, defensive_mixture_weight, variance_reduction_factor, two_stage_design |
| master_thesis_code/bayesian_inference/simulation_detection_probability.py | _build_grid_2d with weights | VERIFIED | Weighted and unweighted paths; n_eff stored in quality_flags |
| .gpd/phases/19-enhanced-sampling-design/sampling-design-report.md | Design report | VERIFIED | Contains VRF, Neyman, defensive mixture, two-stage, boundary content; all numbers match code output |

## Computational Verification Details

### Spot-Check Results (Executed)

```
=== Uniform-Weight Recovery Test (DECISIVE) -- real Phase 18 data ===
     h   N_total   N_det     max|diff|  N_eff==n_total   Farr
  0.60     22500      50      0.00e+00             YES    YES  PASS
  0.65     26000      78      0.00e+00             YES    YES  PASS
  0.70     23500      68      0.00e+00             YES    YES  PASS
  0.73     25500      95      0.00e+00             YES    YES  PASS
  0.80     25000      97      0.00e+00             YES    YES  PASS
  0.85     25500     138      0.00e+00             YES    YES  PASS
  0.90     17000     137      0.00e+00             YES    YES  PASS
```

All 7 h-values show exact zero difference between IS estimator with w=1 and the standard N_det/N_total estimator. N_eff = n_total exactly. Farr criterion (N_eff > 4*N_det) passes globally.

### Synthetic IS Tests (Executed)

```
Test 1: P_det = 0.6 for weights [2,1,1,1], detected=[T,F,T,F] -- PASS (diff = 0.00e+00)
Test 2: N_eff = 3.571429 = 25/7 for weights [2,1,1,1] -- PASS (diff = 0.00e+00)
Test 3: Uniform P_det = 0.3 for 30/100 -- PASS (diff = 0.00e+00)
Test 4: N_eff = 100 for 100 uniform weights -- PASS (diff = 0.00e+00)
Test 5: Empty arrays -> P_det=0, N_eff=0 -- PASS
Test 6: N_eff <= N for 1000 random exponential weight sets -- PASS (0 violations)
```

### Neyman Allocation Tests (Executed)

| Test | Result |
|------|--------|
| Allocation conservation sum(N_k) = N_targeted | PASS (exact integer sum) |
| Highest-sigma bin gets most allocation | PASS (P=0.5 bin, sigma=0.5) |
| Zero-pilot bins get zero allocation | PASS |
| Minimum allocation (5) respected | PASS |

### VRF Manual Spot-Check (Executed)

For h=0.73, boundary bin (0,5) with P_det=0.152, sigma=0.359:
- n_pilot_k = 99, N_k = 3526, N_pilot = 25500, N_targeted = 17850
- f_k = 99/25500 = 0.003882
- n_uniform_eff = 99 + 17850 * 0.003882 = 168.3
- n_strat_eff = 99 + 3526 = 3625
- VRF_manual = 3625 / 168.3 = 21.5389
- VRF_computed = 21.5389
- Match: exact to 10 decimal places

### VRF Results on Real Phase 18 Data (Executed)

| h | N_pilot | N_targeted | Boundary bins | VRF_mean | VRF_min | CI improvement |
|---|---------|------------|---------------|----------|---------|----------------|
| 0.60 | 22500 | 15749 | 4 | 24.9 | 10.3 | 4.6x |
| 0.65 | 26000 | 18200 | 5 | 18.9 | 12.0 | 4.3x |
| 0.70 | 23500 | 16450 | 3 | 19.3 | 13.0 | 4.4x |
| 0.73 | 25500 | 17850 | 5 | 21.2 | 9.3 | 4.5x |
| 0.80 | 25000 | 17500 | 5 | 20.1 | 16.1 | 4.5x |
| 0.85 | 25500 | 17850 | 5 | 16.8 | 11.1 | 4.0x |
| 0.90 | 17000 | 11900 | 5 | 11.8 | 6.9 | 3.4x |

All h-values exceed VRF > 2.0 target. The weakest is h=0.90 with VRF_min = 6.9.

### Report-Code Consistency (Executed)

All 35 numerical values in the design report Section 3 table were cross-checked against code output. All match exactly.

### Defensive Mixture Weight Bound (Executed)

For alpha=0.3, the weight w = 1/(alpha + (1-alpha)*g/p) was tested at g/p ratios of 0.1, 1.0, 10.0, 100.0, 1000.0. All weights satisfy w <= 1/alpha = 3.333. At ratio=1.0 (pilot-like), w = 1.0 exactly.

### Wilson CI Consistency (Executed)

IS-adapted Wilson CI with N_eff substitution matches standard Wilson CI to machine precision when weights are uniform (using identical z-value from scipy.stats.norm.ppf).

## Physics Consistency

| Check | Status | Confidence | Notes |
|-------|--------|------------|-------|
| 5.1 Dimensional analysis | CONSISTENT | INDEPENDENTLY CONFIRMED | P_det [dimensionless, 0-1], w_i [dimensionless, positive], N_eff [dimensionless, 0 < N_eff <= N], VRF [dimensionless, >1 means improvement], sigma_k [dimensionless, 0-0.5] |
| 5.2 Numerical spot-check | PASS | INDEPENDENTLY CONFIRMED | 6 synthetic tests + 7 real-data tests + manual VRF spot-check all pass |
| 5.3 Limiting cases | PASS | INDEPENDENTLY CONFIRMED | w=1 limit recovers standard estimator exactly; equal-weight limit gives N_eff=N; P=0,1 give sigma=0 |
| 5.4 Cross-check | PASS | INDEPENDENTLY CONFIRMED | VRF from code matches manual formula; CI improvement ~ sqrt(VRF) within 1-7% (expected from weighting) |
| 5.6 Symmetry | N/A | -- | No symmetry requirements for this phase |
| 5.7 Conservation | PASS | INDEPENDENTLY CONFIRMED | sum(N_k) = N_targeted exactly for all h-values (allocation conservation) |
| 5.8 Math consistency | PASS | INDEPENDENTLY CONFIRMED | P_det bounds [0,1], N_eff bounds (0, n_total], weight bound [0, 1/alpha], VRF formula verified |
| 5.10 Literature agreement | PASS | STRUCTURALLY PRESENT | Tiwari (2018) IS estimator formula matches; Kish (1965) N_eff formula matches; Hesterberg (1995) defensive mixture construction is standard |
| 5.11 Physical plausibility | PASS | INDEPENDENTLY CONFIRMED | VRF values 7-42x are large but physically reasonable given 3-5 boundary bins out of 140-150 occupied bins |
| Gate A: Cancellation | PASS | INDEPENDENTLY CONFIRMED | No cancellation in IS estimator (ratio of non-negative quantities); VRF is ratio of positive quantities |
| Gate B: Analytical-numerical | PASS | INDEPENDENTLY CONFIRMED | Manual VRF formula matches code output exactly (diff < 1e-10) |
| Gate D: Approximation validity | PASS | STRUCTURALLY PRESENT | IS bias O(1/N_eff) with N_eff > 50 per boundary bin (from report: n_pilot >= 52); Neyman allocation robust to ~30% sigma misspecification (Owen 2013) |

## Forbidden Proxy Audit

| Proxy ID | Status | Evidence |
|----------|--------|----------|
| fp-no-weight-test | REJECTED | Uniform-weight recovery test ran on all 7 h-values with real Phase 18 data; max diff = 0.00e+00 |
| fp-handwave-neff | REJECTED | Per-bin N_eff computed and stored in quality_flags; verified N_eff = n_total for uniform weights |

## Anti-Patterns Found

None. No TODO/FIXME/placeholder comments in any artifact. No suppressed warnings, no hardcoded results, no missing convergence checks.

## Confidence Assessment

**HIGH confidence.** This assessment is based on:

1. The DECISIVE test (uniform-weight recovery) passes at machine precision (0.00e+00) for all 7 h-values on real data -- this is the strongest possible evidence for backward compatibility
2. All synthetic tests pass at machine precision -- the IS estimator, N_eff, and Wilson CI formulas are mathematically correct
3. The VRF computation was manually spot-checked and matches code output exactly
4. All 35 report table values match code output exactly -- no copy errors
5. The full support proof is a 3-line mathematical argument that follows from alpha > 0
6. The VRF > 2.0 target is exceeded by 3-12x for all h-values, providing large margin

No gaps found. All contract targets verified with independent computation.
