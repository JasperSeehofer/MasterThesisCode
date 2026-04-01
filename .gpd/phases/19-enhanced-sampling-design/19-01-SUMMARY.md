---
phase: 19-enhanced-sampling-design
plan: 01
depth: full
one-liner: "Implemented self-normalized IS estimator for P_det grid construction with exact backward compatibility and per-bin Kish N_eff diagnostic"
subsystem: [computation, analysis]
tags: [importance-sampling, detection-probability, histogram-estimator, effective-sample-size]

requires:
  - phase: 18-detection-yield-grid-quality
    provides: "P_det grid quality assessment, 30x20 and 15x10 grid comparison, injection CSV data"
provides:
  - "IS-weighted _build_grid_2d in SimulationDetectionProbability (optional weights parameter)"
  - "Standalone IS utility module: weighted_histogram_estimator, kish_n_eff, is_weighted_wilson_ci"
  - "Per-bin Kish N_eff diagnostic stored in quality_flags"
  - "Farr criterion checker compatible with IS weights"
  - "Weight diagnostic function for quality assessment"
affects: [19-02-enhanced-sampling-design, 20-validation]

methods:
  added: [self-normalized-IS-estimator, kish-effective-sample-size, IS-weighted-wilson-CI]
  patterns: [optional-weights-parameter-for-backward-compat, digitize-based-bin-accumulation]

key-files:
  created:
    - analysis/importance_sampling.py
  modified:
    - master_thesis_code/bayesian_inference/simulation_detection_probability.py

key-decisions:
  - "Preserved original unweighted code path (np.histogram2d) when weights=None for bit-for-bit backward compatibility"
  - "Used np.digitize + np.add.at for weighted path instead of np.histogram2d(weights=) to accumulate both total and detected weights in one pass"
  - "Reliable mask stays based on integer n_total >= 10, not N_eff"

patterns-established:
  - "Optional weights parameter pattern: weights=None falls back to unweighted path"
  - "IS estimator backward compatibility: decisive test is max |diff| < 1e-14 at uniform weights"

conventions:
  - "SI units: distances in Gpc, masses in solar masses, h dimensionless"
  - "P_det dimensionless, in [0, 1]"
  - "Weights w_i = p(theta_i) / q(theta_i), dimensionless, positive"
  - "N_eff = (sum w)^2 / sum(w^2), Kish (1965)"

plan_contract_ref: ".gpd/phases/19-enhanced-sampling-design/19-01-PLAN.md#/contract"
contract_results:
  claims:
    claim-is-estimator:
      status: passed
      summary: "Self-normalized IS estimator P_hat(B) = sum(w_det)/sum(w_total) implemented in _build_grid_2d with optional weights parameter"
      linked_ids: [deliv-sdp-weighted, deliv-is-module, test-uniform-recovery, test-pdet-bounds, test-neff-bounds]
    claim-backward-compat:
      status: passed
      summary: "When weights=None, output is bit-for-bit identical to original N_det/N_total estimator (max diff = 0.0 for all 7 h-values)"
      linked_ids: [deliv-sdp-weighted, test-uniform-recovery]
    claim-neff-diagnostic:
      status: passed
      summary: "Per-bin Kish N_eff = (sum w)^2 / sum(w^2) computed and stored in quality_flags['n_eff']"
      linked_ids: [deliv-sdp-weighted, deliv-is-module, test-neff-bounds, test-neff-uniform]
  deliverables:
    deliv-sdp-weighted:
      status: passed
      path: "master_thesis_code/bayesian_inference/simulation_detection_probability.py"
      summary: "_build_grid_2d extended with optional weights parameter; quality_flags extended with n_eff array"
      linked_ids: [claim-is-estimator, claim-backward-compat, claim-neff-diagnostic]
    deliv-is-module:
      status: passed
      path: "analysis/importance_sampling.py"
      summary: "Standalone IS utilities: weighted_histogram_estimator, kish_n_eff, is_weighted_wilson_ci, weight_diagnostic, farr_criterion_check"
      linked_ids: [claim-is-estimator, claim-neff-diagnostic]
  acceptance_tests:
    test-uniform-recovery:
      status: passed
      summary: "max |P_det_IS(w=1) - P_det_standard| = 0.0 for all 7 h-values (threshold was < 1e-14)"
      linked_ids: [claim-backward-compat, deliv-sdp-weighted, ref-tiwari2018]
    test-pdet-bounds:
      status: passed
      summary: "0 <= P_det <= 1 for all bins across all 7 h-values"
      linked_ids: [claim-is-estimator, deliv-sdp-weighted]
    test-neff-bounds:
      status: passed
      summary: "0 < N_eff <= n_total for all non-empty bins across all 7 h-values"
      linked_ids: [claim-neff-diagnostic, deliv-sdp-weighted]
    test-neff-uniform:
      status: passed
      summary: "N_eff = n_total exactly (diff = 0.0) for all bins when weights = 1"
      linked_ids: [claim-neff-diagnostic, deliv-sdp-weighted]
  references:
    ref-tiwari2018:
      status: completed
      completed_actions: [use]
      missing_actions: []
      summary: "Self-normalized IS estimator formula from Tiwari (2018) Eq. 5-8 implemented in weighted_histogram_estimator"
    ref-kish1965:
      status: completed
      completed_actions: [use]
      missing_actions: []
      summary: "Kish N_eff formula implemented in kish_n_eff function"
    ref-phase18-grid:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "Phase 18 P_det grids used as baseline for uniform-weight recovery test; match to machine precision"
  forbidden_proxies:
    fp-no-weight-test:
      status: rejected
      notes: "Ran full uniform-weight recovery test on all 7 h-values; max diff = 0.0"
    fp-handwave-neff:
      status: rejected
      notes: "N_eff computed per-bin from actual weight arrays via np.add.at accumulation"
  uncertainty_markers:
    weakest_anchors:
      - "IS estimator bias O(1/N_eff) is asymptotic; may understate bias for N_eff < 10"
    unvalidated_assumptions: []
    competing_explanations: []
    disconfirming_observations: []

comparison_verdicts:
  - subject_id: claim-backward-compat
    subject_kind: claim
    subject_role: decisive
    reference_id: ref-phase18-grid
    comparison_kind: baseline
    metric: max_absolute_difference
    threshold: "< 1e-14"
    verdict: pass
    recommended_action: "None -- backward compatibility confirmed"
    notes: "max |diff| = 0.0 for all 7 h-values, exceeding threshold by infinite margin"

duration: 4min
completed: 2026-04-01
---

# Phase 19-01: IS-Weighted Histogram Estimator Summary

**Implemented self-normalized IS estimator for P_det grid construction with exact backward compatibility and per-bin Kish N_eff diagnostic**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-01T16:04:04Z
- **Completed:** 2026-04-01T16:08:25Z
- **Tasks:** 2
- **Files modified:** 2

## Key Results

- IS estimator backward compatible: max |P_det_IS(w=1) - P_det_standard| = 0.0 for all 7 h-values [CONFIDENCE: HIGH]
- Per-bin N_eff = n_total exactly when weights are uniform [CONFIDENCE: HIGH]
- Farr criterion (global) passes for all 7 h-values with uniform weights [CONFIDENCE: HIGH]
- Synthetic non-uniform weight test: P_det = 3/5 = 0.6 (not 2/4 = 0.5) for weights [2,1,1,1] -- confirms IS estimator is mathematically correct [CONFIDENCE: HIGH]

## Task Commits

1. **Task 1: Implement IS-weighted histogram estimator** - `f15df43` (implement)
2. **Task 2: Verify IS estimator properties** - `f15df43` (validate -- no additional code changes needed)

## Files Created/Modified

- `analysis/importance_sampling.py` -- Standalone IS utility module with weighted_histogram_estimator, kish_n_eff, is_weighted_wilson_ci, weight_diagnostic, farr_criterion_check
- `master_thesis_code/bayesian_inference/simulation_detection_probability.py` -- _build_grid_2d extended with optional weights parameter; quality_flags extended with n_eff

## Next Phase Readiness

- IS estimator infrastructure ready for Phase 19-02 (proposal distribution design)
- Any change to the injection sampling strategy only requires computing w_i = p(theta_i) / q(theta_i) and passing to _build_grid_2d
- Weight diagnostic and Farr criterion checker available for evaluating sampling quality

## Contract Coverage

- Claims: claim-is-estimator PASSED, claim-backward-compat PASSED, claim-neff-diagnostic PASSED
- Deliverables: deliv-sdp-weighted PASSED, deliv-is-module PASSED
- Acceptance tests: test-uniform-recovery PASSED, test-pdet-bounds PASSED, test-neff-bounds PASSED, test-neff-uniform PASSED
- References: ref-tiwari2018 completed (use), ref-kish1965 completed (use), ref-phase18-grid completed (read, compare)
- Forbidden proxies: fp-no-weight-test REJECTED, fp-handwave-neff REJECTED
- Comparison verdicts: claim-backward-compat PASS (max |diff| = 0.0)

## Equations Derived

**Eq. (19-01.1): Self-normalized IS estimator**

$$
\hat{P}_{\text{det}}(B) = \frac{\sum_{i \in B} w_i \cdot \mathbb{1}[\text{SNR}_i \geq \text{threshold}]}{\sum_{i \in B} w_i}
$$

Reference: Tiwari (2018), arXiv:1712.00482, Eq. 5-8.

**Eq. (19-01.2): Kish effective sample size**

$$
N_{\text{eff}}(B) = \frac{\left(\sum_{i \in B} w_i\right)^2}{\sum_{i \in B} w_i^2}
$$

Reference: Kish (1965), Survey Sampling.

## Validations Completed

- **Uniform-weight recovery (DECISIVE):** max |diff| = 0.0 for all 7 h-values (threshold: < 1e-14)
- **P_det bounds:** 0 <= P_det <= 1 for all bins (guaranteed by construction)
- **N_eff bounds:** 0 < N_eff <= n_total for all non-empty bins
- **N_eff uniform:** N_eff = n_total exactly (diff = 0.0) when weights = 1
- **Farr criterion:** Global pass for all 7 h-values
- **Synthetic weights:** P_det = 0.6 for [2,1,1,1] with detected [T,F,T,F] (hand-computed)
- **Existing tests:** All 203 tests pass, mypy clean

## Decisions Made

- Preserved original np.histogram2d code path when weights=None for bit-for-bit output
- Used np.digitize + np.add.at for weighted path (single pass through data)
- "reliable" mask stays based on integer n_total >= 10, not N_eff

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## Open Questions

- What proposal distribution q(theta) should be used for the enhanced injection campaign? (Phase 19-02 scope)
- How sensitive is the Farr per-bin pass fraction to non-uniform weights? (Needs real non-uniform weight data)

---

_Phase: 19-enhanced-sampling-design_
_Plan: 01_
_Completed: 2026-04-01_
