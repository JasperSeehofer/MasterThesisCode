---
phase: 25-likelihood-correction
plan: 01
depth: full
one-liner: "Implemented completeness-corrected dark siren likelihood (Gray et al. 2020 Eq. 9) with completion term, combination formula, and 11 unit tests"
subsystem: computation
tags: [dark-siren, bayesian-inference, completeness, likelihood, gray2020]

requires:
  - phase: 24-completeness-estimation
    provides: GladeCatalogCompleteness class with f(z,h) interface and comoving_volume_element()
provides:
  - Completeness-corrected per-event likelihood p_i = f_i * L_cat + (1-f_i) * L_comp in bayesian_statistics.py
  - Completion term L_comp via fixed_quad over comoving volume prior (Gray et al. 2020 Eqs. 31-32)
  - GladeCatalogCompleteness threading from evaluate() through p_D() to p_Di()
  - Test suite verifying limiting cases, positivity, dimensionlessness, and h-dependence
affects: [26-posterior-evaluation, bias-correction-validation]

methods:
  added: [completeness-corrected-likelihood, completion-term-quadrature]
  patterns: [combination-formula-f-weighting, detection-level-completion-term]

key-files:
  created:
    - master_thesis_code_test/test_completion_term.py
  modified:
    - master_thesis_code/physical_relations.py
    - master_thesis_code/bayesian_inference/bayesian_statistics.py

key-decisions:
  - "Completion term uses 'without BH mass' 3D Gaussian for both variants (uncataloged host has no galaxy mass info)"
  - "Completion term computed in p_Di() main process, not in worker processes (per forbidden proxy fp-no-worker-completion)"
  - "Integration limits for completion term match catalog term numerator (4-sigma d_L range)"
  - "z_lower floored at 1e-6 to avoid volume element singularity at z=0"

patterns-established:
  - "Completion term is per-detection (in p_Di scope), not per-galaxy (not in single_host_likelihood)"
  - "Completeness f_i evaluated at z_det = dist_to_redshift(d_L_det, h) for the trial h value"

conventions:
  - "SI units (km/s, Mpc, Gpc)"
  - "d_L from dist_vectorized in Gpc; P_det expects Gpc"
  - "d_L_fraction = d_L / d_L_detected (dimensionless, for GW Gaussian)"
  - "comoving_volume_element returns Mpc^3/sr (cancels in L_comp ratio)"
  - "completeness f_i in [0, 1] from GladeCatalogCompleteness"

plan_contract_ref: ".gpd/phases/25-likelihood-correction/25-01-PLAN.md#/contract"
contract_results:
  claims:
    claim-gray-eq9:
      status: passed
      summary: "p_Di returns f_i * L_cat + (1-f_i) * L_comp for both 'without BH mass' and 'with BH mass' variants"
      linked_ids: [deliv-bayesian-statistics, test-combination, test-f1-limit, test-f0-limit]
    claim-completion-term:
      status: passed
      summary: "L_comp computed as ratio of GW-likelihood-weighted volume integral to detection-probability-weighted volume integral via fixed_quad"
      linked_ids: [deliv-bayesian-statistics, test-lcomp-positive, test-lcomp-ratio]
    claim-completeness-threading:
      status: passed
      summary: "GladeCatalogCompleteness created in evaluate(), passed through p_D() to p_Di() without hardcoded values"
      linked_ids: [deliv-bayesian-statistics, test-f-varies-with-h]
    claim-reference-comments:
      status: passed
      summary: "All physics changes carry reference comments citing arXiv:1908.06050 equation numbers (8 reference lines total)"
      linked_ids: [deliv-bayesian-statistics, deliv-physical-relations, test-reference-comments]
  deliverables:
    deliv-bayesian-statistics:
      status: passed
      path: "master_thesis_code/bayesian_inference/bayesian_statistics.py"
      summary: "Completeness-corrected likelihood with GladeCatalogCompleteness, L_comp, completion_term integrands, and get_completeness_at_redshift threading"
      linked_ids: [claim-gray-eq9, claim-completion-term, claim-completeness-threading]
    deliv-physical-relations:
      status: passed
      path: "master_thesis_code/physical_relations.py"
      summary: "Gray et al. (2020) arXiv:1908.06050 Appendix A.2.3 reference added to comoving_volume_element docstring"
      linked_ids: [claim-reference-comments]
    deliv-tests:
      status: passed
      path: "master_thesis_code_test/test_completion_term.py"
      summary: "11 tests covering limiting cases, combination formula, positivity, dimensionlessness, h-dependence, and sky position convention"
      linked_ids: [test-combination, test-f1-limit, test-f0-limit, test-lcomp-positive, test-lcomp-ratio, test-f-varies-with-h]
  acceptance_tests:
    test-combination:
      status: passed
      summary: "test_combination_formula_weighted_sum and test_combination_always_between_components verify weighted sum for f=0.5, f=0.3, and 100 random values"
      linked_ids: [claim-gray-eq9, deliv-tests]
    test-f1-limit:
      status: passed
      summary: "test_f1_recovers_catalog_only verifies f=1 returns exactly L_cat for both variants"
      linked_ids: [claim-gray-eq9, deliv-tests]
    test-f0-limit:
      status: passed
      summary: "test_f0_gives_completion_only verifies f=0 returns exactly L_comp"
      linked_ids: [claim-gray-eq9, deliv-tests]
    test-lcomp-positive:
      status: passed
      summary: "test_completion_term_positive verifies L_comp > 0 with P_det=1 and real comoving volume element"
      linked_ids: [claim-completion-term, deliv-tests]
    test-lcomp-ratio:
      status: passed
      summary: "test_lcomp_invariant_under_dvc_scaling verifies L_comp is dimensionless (dVc cancels in ratio) at scale factors 1, 100, 0.01"
      linked_ids: [claim-completion-term, deliv-tests]
    test-f-varies-with-h:
      status: passed
      summary: "test_f_varies_with_h verifies f(z=0.1, h=0.6) != f(z=0.1, h=0.86) and f increases with h"
      linked_ids: [claim-completeness-threading, deliv-tests]
    test-reference-comments:
      status: passed
      summary: "grep confirms 8 reference lines citing arXiv:1908.06050 or Gray et al. across both modified files"
      linked_ids: [claim-reference-comments, deliv-bayesian-statistics, deliv-physical-relations]
  references:
    ref-gray2020:
      status: completed
      completed_actions: [use, cite]
      missing_actions: [compare]
      summary: "Gray et al. (2020) arXiv:1908.06050 cited in all new physics formulas. Eq. 9 (combination), Eqs. 24-25 (catalog term label), Eqs. 31-32 (completion term), Appendix A.2.3 (volume element). compare deferred to posterior evaluation phase."
    ref-hogg1999:
      status: completed
      completed_actions: [cite]
      missing_actions: []
      summary: "Hogg (1999) already cited in comoving_volume_element; no changes needed"
    ref-dalya2022:
      status: completed
      completed_actions: [use]
      missing_actions: []
      summary: "Dalya et al. (2022) data used via GladeCatalogCompleteness (from Phase 24)"
  forbidden_proxies:
    fp-no-single-host-mod:
      status: rejected
      notes: "single_host_likelihood() is unchanged. Completion term is in p_Di() scope."
    fp-no-galaxy-f-weight:
      status: rejected
      notes: "Completeness enters only at combination level in p_Di(), not inside galaxy sum."
    fp-no-worker-completion:
      status: rejected
      notes: "Completion term computed in main process in p_Di(), not in worker processes."
    fp-no-hardcoded-f:
      status: rejected
      notes: "f_i comes from GladeCatalogCompleteness.get_completeness_at_redshift(z_det, h)."
    fp-no-catalog-term-change:
      status: rejected
      notes: "Existing catalog term computation logic preserved; only wrapped in L_cat variables."
  uncertainty_markers:
    weakest_anchors:
      - "Digitized completeness may be number completeness rather than B-band luminosity completeness (Phase 24 deviation 1)"
    unvalidated_assumptions: []
    competing_explanations: []
    disconfirming_observations: []

duration: 6min
completed: 2026-04-04
---

# Phase 25 Plan 01: Likelihood Correction Summary

**Implemented completeness-corrected dark siren likelihood (Gray et al. 2020 Eq. 9) with completion term, combination formula, and 11 unit tests**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-04T16:38:58Z
- **Completed:** 2026-04-04T16:44:32Z
- **Tasks:** 2
- **Files modified:** 3

## Key Results

- Per-event likelihood now combines catalog and completion terms: p_i = f_i * L_cat + (1 - f_i) * L_comp [CONFIDENCE: HIGH]
- L_comp is dimensionless (ratio of two integrals with same dVc measure), always positive [CONFIDENCE: HIGH]
- f_i varies with h through d_L(z, h) relation: f(z=0.1, h=0.86) > f(z=0.1, h=0.6) [CONFIDENCE: HIGH]
- f=1 limit exactly recovers current code behavior (catalog-only), f=0 limit gives volume-prior-weighted result [CONFIDENCE: HIGH]

## Task Commits

Each task was committed atomically:

1. **Task 1: Add Gray et al. reference + implement completion term** - `f60a75a` (implement)
2. **Task 2: Tests for completion term and combination formula** - `5f3d8f7` (validate)

## Files Created/Modified

- `master_thesis_code/physical_relations.py` - Added Gray et al. (2020) reference to comoving_volume_element docstring
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` - Completion term L_comp + combination formula in p_Di(); GladeCatalogCompleteness threading
- `master_thesis_code_test/test_completion_term.py` - 11 tests for limiting cases, positivity, dimensionlessness, h-dependence, sky position

## Next Phase Readiness

- Completeness-corrected likelihood ready for posterior evaluation
- Next step: run posterior with corrected likelihood and compare MAP to h=0.73 (expect bias reduction)
- The f=1 limit test confirms backward compatibility: setting completeness to 1.0 everywhere recovers current behavior exactly

## Contract Coverage

- Claim IDs advanced: claim-gray-eq9 -> passed, claim-completion-term -> passed, claim-completeness-threading -> passed, claim-reference-comments -> passed
- Deliverable IDs produced: deliv-bayesian-statistics -> passed, deliv-physical-relations -> passed, deliv-tests -> passed
- Acceptance test IDs run: test-combination -> passed, test-f1-limit -> passed, test-f0-limit -> passed, test-lcomp-positive -> passed, test-lcomp-ratio -> passed, test-f-varies-with-h -> passed, test-reference-comments -> passed
- Reference IDs surfaced: ref-gray2020 -> used+cited (compare deferred), ref-hogg1999 -> cited, ref-dalya2022 -> used
- Forbidden proxies rejected: fp-no-single-host-mod, fp-no-galaxy-f-weight, fp-no-worker-completion, fp-no-hardcoded-f, fp-no-catalog-term-change (all rejected)

## Equations Derived

**Eq. (25.1):** Combination formula (Gray et al. 2020, Eq. 9)

$$
p_i(x_{\mathrm{GW}} | D_{\mathrm{GW}}, H_0) = f_i \cdot L_{\mathrm{cat}}^i + (1 - f_i) \cdot L_{\mathrm{comp}}^i
$$

**Eq. (25.2):** Completion term (Gray et al. 2020, Eqs. 31-32)

$$
L_{\mathrm{comp}}^i = \frac{\int p_{\mathrm{GW}}(x | z, \Omega_{\mathrm{det}}, h) \cdot P_{\mathrm{det}}(d_L(z,h)) \cdot \frac{dV_c}{dz} \, dz}{\int P_{\mathrm{det}}(d_L(z,h)) \cdot \frac{dV_c}{dz} \, dz}
$$

## Validations Completed

- f=1 limiting case: combination returns exactly L_cat (both variants) -- 2 tests
- f=0 limiting case: combination returns exactly L_comp -- 2 tests
- Weighted sum: f=0.5 gives average, f=0.3 gives 0.3*L_cat + 0.7*L_comp -- 2 tests
- Monotonic interpolation: combined always between min(L_cat, L_comp) and max(L_cat, L_comp) -- 100 random tests
- L_comp positivity: verified with P_det=1 and real comoving volume element -- 1 test
- L_comp dimensionless: invariant under dVc scaling (factors 1, 100, 0.01) -- 1 test
- h-dependence: f(z=0.1, h=0.6) < f(z=0.1, h=0.86) confirmed -- 1 test
- Sky position: completion term uses detection sky position, not galaxy position -- 1 test
- Dimensional consistency: L_comp = [prob * prob * Mpc^3/sr] / [prob * Mpc^3/sr] = [dimensionless]
- Reference comments: 8 lines citing arXiv:1908.06050 across 2 files (grep verified)
- single_host_likelihood unchanged (grep verified: still at lines 645, 861)
- ruff, mypy, pytest -m "not gpu and not slow" all pass (376 tests, 0 failures)

## Decisions & Deviations

None - plan executed exactly as written.

## Open Questions

- Will the corrected posterior shift MAP toward h=0.73? (Expected yes; to be verified in posterior evaluation phase)
- Is the digitized completeness data number completeness or B-band luminosity completeness? (Phase 24 deviation 1, deferred)

## Approximations Used

| Approximation | Valid When | Error Estimate | Breaks Down At |
|---|---|---|---|
| Angle-averaged completeness | LISA EMRI sky localization ~1 deg^2 | Negligible for EMRI | LIGO ~100 deg^2 |
| No source evolution p(s|z) = const | z < 0.2 (EMRI range) | O(10%) at z~0.2 for lambda~1 | High-z surveys |
| Equal galaxy weights p(s|M) = const | All z | Standard for EMRI | Luminosity weighting |
| Completion term uses 3D Gaussian | Uncataloged host has no galaxy mass info | Exact when host is unknown | Host with mass info |

## Issues Encountered

None.

## Self-Check: PASSED

- [x] master_thesis_code/physical_relations.py exists and contains Gray et al. reference
- [x] master_thesis_code/bayesian_inference/bayesian_statistics.py exists and contains completion term
- [x] master_thesis_code_test/test_completion_term.py exists with 11 tests
- [x] Commit f60a75a exists (Task 1)
- [x] Commit 5f3d8f7 exists (Task 2)
- [x] ruff clean on all modified files
- [x] mypy clean on all modified files
- [x] 376 tests pass (pytest -m "not gpu and not slow")

---

_Phase: 25-likelihood-correction_
_Completed: 2026-04-04_
