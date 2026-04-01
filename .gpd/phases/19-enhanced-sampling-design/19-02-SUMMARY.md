---
phase: 19-enhanced-sampling-design
plan: 02
depth: full
one-liner: "Neyman-optimal allocation achieves 11.8-24.9x VRF in boundary bins from Phase 18 pilot data, with full two-stage design specification and defensive mixture guarantee"
subsystem: [analysis, computation]
tags: [importance-sampling, neyman-allocation, variance-reduction, stratified-sampling, detection-probability]

requires:
  - phase: 18-detection-yield-grid-quality
    provides: "Per-bin n_total, n_detected, P_det for 15x10 grid across 7 h-values"
  - phase: 19-enhanced-sampling-design
    plan: 01
    provides: "IS estimator, Kish N_eff, weighted Wilson CI, weight diagnostic"
provides:
  - "Neyman-optimal allocation function with integer conservation"
  - "VRF computation from actual Phase 18 per-bin statistics"
  - "Defensive mixture weight function with bounded weights"
  - "Two-stage design specification (pilot + targeted)"
  - "Complete sampling-design-report.md with 8 sections"
affects: [20-validation, next-injection-campaign]

methods:
  added: [neyman-optimal-allocation, variance-reduction-factor, defensive-mixture-proposal]
  patterns: [sigma-weighted-VRF-average, integer-remainder-distribution, minimum-allocation-floor]

key-files:
  created:
    - analysis/sampling_design.py
    - .gpd/phases/19-enhanced-sampling-design/sampling-design-report.md
  modified: []

key-decisions:
  - "VRF computed from actual Phase 18 per-bin counts (not generic formulas)"
  - "Targeted budget = 70% of pilot per h-value"
  - "Alpha = 0.3 for defensive mixture (30% uniform + 70% targeted)"
  - "Minimum allocation floor of 5 per non-empty bin"
  - "Sigma-weighted VRF average across boundary bins"

patterns-established:
  - "Neyman allocation: N_k proportional to sigma_k = sqrt(P*(1-P))"
  - "Defensive mixture: q = alpha*p + (1-alpha)*g ensures full support"
  - "VRF = n_stratified / n_uniform (P*(1-P) cancels in ratio)"

conventions:
  - "SI units: distances in Gpc, masses in solar masses, h dimensionless"
  - "P_det in [0, 1], sigma_k in [0, 0.5]"
  - "Boundary bins: 0.05 < P_det < 0.95"
  - "Grid: 15x10 (d_L x M)"
  - "Wilson score 95% CI (z=1.96)"

plan_contract_ref: ".gpd/phases/19-enhanced-sampling-design/19-02-PLAN.md#/contract"
contract_results:
  claims:
    claim-neyman-allocation:
      status: passed
      summary: "Neyman-optimal allocation concentrates 82-687x more injections per boundary bin vs non-boundary, using Phase 18 pilot sigma estimates on 15x10 grid"
      linked_ids: [deliv-sampling-design, deliv-design-report, test-allocation-sums, test-boundary-concentration, test-vrf-exceeds-2]
    claim-vrf:
      status: passed
      summary: "VRF exceeds 2.0 for all 7 h-values: weighted-average VRF ranges from 11.8x (h=0.90) to 24.9x (h=0.60), computed from actual Phase 18 per-bin statistics"
      linked_ids: [deliv-sampling-design, deliv-design-report, test-vrf-exceeds-2, test-vrf-from-data]
    claim-two-stage:
      status: passed
      summary: "Two-stage design fully specified: (a) Phase 18 pilot, (b) Neyman-optimal targeted budget, (c) boundary ID from pilot P_det, (d) defensive mixture proposal alpha=0.3, (e) weight formula pilot w=1 / targeted w=p/q"
      linked_ids: [deliv-sampling-design, deliv-design-report, test-full-support, test-weight-formula]
    claim-full-support:
      status: passed
      summary: "Full support proved: q >= alpha*p > 0 wherever p > 0 for alpha=0.3; weight bound max(w) <= 1/alpha = 3.33"
      linked_ids: [deliv-sampling-design, deliv-design-report, test-full-support]
  deliverables:
    deliv-sampling-design:
      status: passed
      path: "analysis/sampling_design.py"
      summary: "Contains neyman_allocation, defensive_mixture_weight, variance_reduction_factor, two_stage_design functions; all operate on Phase 18 data"
      linked_ids: [claim-neyman-allocation, claim-vrf, claim-two-stage, claim-full-support]
    deliv-design-report:
      status: passed
      path: ".gpd/phases/19-enhanced-sampling-design/sampling-design-report.md"
      summary: "8-section report with per-h allocation tables, VRF values, full support proof, implementation checklist"
      linked_ids: [claim-neyman-allocation, claim-vrf, claim-two-stage, claim-full-support]
  acceptance_tests:
    test-allocation-sums:
      status: passed
      summary: "sum(N_k) = N_targeted exactly for all 7 h-values (integer conservation after rounding)"
      linked_ids: [claim-neyman-allocation, deliv-sampling-design]
    test-boundary-concentration:
      status: passed
      summary: "mean(N_k|boundary) / mean(N_k|non-boundary) ranges from 82.2x to 687.0x, far exceeding 5x threshold"
      linked_ids: [claim-neyman-allocation, deliv-sampling-design]
    test-vrf-exceeds-2:
      status: passed
      summary: "Weighted-average VRF: 11.8-24.9x across all h-values; minimum per-bin VRF: 6.9x (h=0.90 bin (0,5))"
      linked_ids: [claim-vrf, deliv-sampling-design, deliv-design-report]
    test-vrf-from-data:
      status: passed
      summary: "VRF computed from load_injection_data() -> build_grid_with_ci() -> neyman_allocation(), using actual Phase 18 CSV per-bin counts"
      linked_ids: [claim-vrf, deliv-sampling-design]
    test-full-support:
      status: passed
      summary: "Proved: q >= alpha*p, alpha=0.3 > 0, p > 0 on injection domain => q > 0 everywhere. Corollary: max(w) <= 1/alpha = 3.33"
      linked_ids: [claim-full-support, deliv-design-report]
    test-weight-formula:
      status: passed
      summary: "Weight formula documented: pilot w_i=1 (drawn from p), targeted w_i=p/q. Combined IS estimator consistent with Plan 19-01 implementation"
      linked_ids: [claim-two-stage, deliv-sampling-design, deliv-design-report]
  references:
    ref-owen2013:
      status: completed
      completed_actions: [use]
      missing_actions: []
      summary: "Neyman-optimal allocation formula N_k proportional to sigma_k used as primary allocation method"
    ref-hesterberg1995:
      status: completed
      completed_actions: [use]
      missing_actions: []
      summary: "Defensive mixture alpha=0.3 ensures full support and bounded weights per Hesterberg (1995)"
    ref-tiwari2018:
      status: completed
      completed_actions: [cite]
      missing_actions: []
      summary: "Self-normalized IS estimator from Tiwari (2018) cited for combined estimator"
    ref-farr2019:
      status: completed
      completed_actions: [use, compare]
      missing_actions: []
      summary: "Farr criterion N_eff > 4*N_det referenced in design report Section 5"
    ref-phase18-data:
      status: completed
      completed_actions: [read, use, compare]
      missing_actions: []
      summary: "Phase 18 injection CSVs loaded and used as pilot data; per-bin counts drive all VRF computations"
  forbidden_proxies:
    fp-handwave-vrf:
      status: rejected
      notes: "VRF computed from actual Phase 18 per-bin n_total/n_detected/P_det, not from generic formulas"
    fp-no-weight-correction:
      status: rejected
      notes: "Weight formula w=p/q specified for targeted samples; pilot w=1; combined IS estimator documented"
    fp-zero-support:
      status: rejected
      notes: "Full support proved: q >= 0.3*p > 0 on entire injection domain; max(w) <= 3.33 bounded"
  uncertainty_markers:
    weakest_anchors:
      - "Pilot sigma accuracy ~7-10% for boundary bins (n_total >= 52), sufficient for Neyman allocation (robust to ~30% misspecification per Owen 2013)"
      - "With 3-5 boundary bins per h-value, VRF is sensitive to boundary definition threshold (0.05, 0.95)"
    unvalidated_assumptions: []
    competing_explanations: []
    disconfirming_observations: []

comparison_verdicts:
  - subject_id: claim-vrf
    subject_kind: claim
    subject_role: decisive
    reference_id: ref-phase18-data
    comparison_kind: baseline
    metric: VRF_boundary_mean
    threshold: "> 2.0"
    verdict: pass
    recommended_action: "None -- all h-values exceed target by large margin (5.9-12.5x)"
    notes: "VRF_mean: 0.60=24.9, 0.65=18.9, 0.70=19.3, 0.73=21.2, 0.80=20.1, 0.85=16.8, 0.90=11.8"

duration: 5min
completed: 2026-04-01
---

# Phase 19-02: Neyman-Optimal Allocation and Two-Stage Design Summary

**Neyman-optimal allocation achieves 11.8-24.9x VRF in boundary bins from Phase 18 pilot data, with full two-stage design specification and defensive mixture guarantee**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-01T16:11:27Z
- **Completed:** 2026-04-01T16:16:30Z
- **Tasks:** 2
- **Files modified:** 2

## Key Results

- VRF > 2.0 for all 7 h-values: weighted-average ranges from 11.8x (h=0.90) to 24.9x (h=0.60) [CONFIDENCE: HIGH]
- Boundary concentration ratio 82-687x (threshold was 5x): Neyman allocation concentrates injections on 3-5 boundary bins out of 140-150 non-empty bins [CONFIDENCE: HIGH]
- CI half-width improvement 3.4-4.6x in boundary bins vs uniform sampling with same total budget [CONFIDENCE: HIGH]
- Full support guarantee: q >= 0.3*p > 0 everywhere, max weight <= 3.33 [CONFIDENCE: HIGH]

## Task Commits

1. **Task 1: Neyman-optimal allocation and VRF computation** - `eb0d633` (implement)
2. **Task 2: Two-stage design and sampling-design-report.md** - `d32c925` (docs)

## Files Created/Modified

- `analysis/sampling_design.py` -- Neyman allocation, VRF computation, defensive mixture weight, two-stage design specification, verification suite
- `.gpd/phases/19-enhanced-sampling-design/sampling-design-report.md` -- 8-section design report with numerical tables from Phase 18 data

## Next Phase Readiness

- Complete sampling design ready for implementation in next injection campaign
- Implementation checklist provided in design report (8 steps)
- IS estimator infrastructure (Plan 19-01) + allocation design (Plan 19-02) form the complete enhanced sampling system
- h=0.90 identified as weakest performer (VRF_min=6.9) due to fewer pilot events

## Contract Coverage

- Claims: claim-neyman-allocation PASSED, claim-vrf PASSED, claim-two-stage PASSED, claim-full-support PASSED
- Deliverables: deliv-sampling-design PASSED, deliv-design-report PASSED
- Acceptance tests: test-allocation-sums PASSED, test-boundary-concentration PASSED, test-vrf-exceeds-2 PASSED, test-vrf-from-data PASSED, test-full-support PASSED, test-weight-formula PASSED
- References: ref-owen2013 completed (use), ref-hesterberg1995 completed (use), ref-tiwari2018 completed (cite), ref-farr2019 completed (use, compare), ref-phase18-data completed (read, use, compare)
- Forbidden proxies: fp-handwave-vrf REJECTED, fp-no-weight-correction REJECTED, fp-zero-support REJECTED
- Comparison verdicts: claim-vrf PASS (VRF_mean 11.8-24.9, threshold >2.0)

## Equations Derived

**Eq. (19-02.1): Neyman-optimal allocation**

$$
N_k = \left\lfloor N_{\text{targeted}} \cdot \frac{\sigma_k}{\sum_k \sigma_k} \right\rfloor, \quad \sigma_k = \sqrt{\hat{P}_k (1 - \hat{P}_k)}
$$

**Eq. (19-02.2): Variance reduction factor**

$$
\text{VRF}_k = \frac{n_{\text{pilot},k} + N_k}{n_{\text{pilot},k} + N_{\text{new}} \cdot f_k}, \quad f_k = \frac{n_{\text{pilot},k}}{N_{\text{pilot}}}
$$

**Eq. (19-02.3): Defensive mixture proposal**

$$
q(\theta) = \alpha \cdot p_{\text{uniform}}(\theta) + (1 - \alpha) \cdot g_{\text{targeted}}(\theta), \quad \alpha = 0.3
$$

**Eq. (19-02.4): Weight bound**

$$
w_i = \frac{p(\theta_i)}{q(\theta_i)} \leq \frac{1}{\alpha} = 3.33
$$

## Validations Completed

- **Allocation conservation (DECISIVE):** sum(N_k) = N_targeted exactly for all 7 h-values
- **Boundary concentration:** 82-687x ratio (threshold 5x) -- all h-values pass
- **VRF > 2.0 (CONTRACT TARGET):** 11.8-24.9x for all h-values
- **VRF from data:** Computation loads Phase 18 CSVs and uses per-bin counts directly
- **Sigma range:** sigma_k = 0 for P=0 or P=1; sigma in (0, 0.5] for boundary bins
- **Minimum allocation:** All non-empty bins receive >= 5 targeted injections
- **Dimensional consistency:** VRF is dimensionless ratio; sigma is dimensionless sqrt(P*(1-P))
- **Full support proof:** q >= alpha*p > 0 wherever p > 0; proved mathematically (3 lines)

## Decisions Made

- VRF computed from actual Phase 18 per-bin counts, not generic formulas (contract requirement)
- Targeted budget = 70% of pilot per h-value (balances cost vs variance reduction)
- Alpha = 0.3 chosen as standard defensive mixture fraction (Hesterberg 1995)
- Sigma-weighted average for VRF summary (higher-variance bins matter more)
- Minimum allocation floor = 5 per non-empty bin (prevents empty strata)

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## Open Questions

- How will the targeted injection sampler map (d_L, M) bin assignments back to (z, M) for sampling? (Implementation detail for next phase)
- What is the actual computational cost of the targeted campaign (~12k-18k events per h at 7 h-values)?
- Should h=0.90 receive extra pilot events to improve sigma estimates?

---

_Phase: 19-enhanced-sampling-design_
_Plan: 02_
_Completed: 2026-04-01_
