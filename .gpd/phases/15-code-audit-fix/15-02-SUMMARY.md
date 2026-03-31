---
phase: 15-code-audit-fix
plan: 02
depth: standard
one-liner: "Quick validation at 4 h-values shows /(1+z) fix changes likelihood scale but NOT posterior shape -- 'with BH mass' still monotonically decreasing"
subsystem: [validation]

requires:
  - phase: 15-01
    provides: ["/(1+z) removal from bayesian_statistics.py"]
provides:
  - "Post-fix posterior values at h={0.652, 0.678, 0.704, 0.730} with P_det=1"
  - "Directional test result: FAIL (posterior still monotonically decreasing)"
  - "Confirmation that 'without BH mass' channel peaks at h=0.678 as expected"
  - "Evidence that additional bias sources exist beyond /(1+z)"
affects: [Phase 16 - Validation]

key-files:
  created:
    - .gpd/phases/15-code-audit-fix/15-validation-results.md
    - .gpd/phases/15-code-audit-fix/15-quick-validation-results.json
    - scripts/quick_validation_15.py
    - cluster/quick_validation.sh
    - cluster/extract_validation_results.py
  modified: []

key-decisions:
  - "Ran with P_det=1 mock locally (injection data not available on dev machine)"
  - "Directional test FAIL accepted -- /(1+z) fix kept as correct physics, additional investigation deferred to Phase 16"

conventions:
  - "SI units (c, G, H0 in km/s/Mpc)"
  - "h = H0 / (100 km/s/Mpc)"
  - "P_det = 1 (mock, matching pre-fix baseline conditions)"

plan_contract_ref: ".gpd/phases/15-code-audit-fix/15-02-PLAN.md#/contract"
contract_results:
  claims:
    claim-posterior-shift:
      status: failed
      summary: "Post-fix 'with BH mass' posterior is still monotonically decreasing (peak at h<=0.652). Scale changed dramatically (log ~-185 to ~+50) but shape unchanged."
      linked_ids: [deliv-validation-results, test-direction, ref-pre-fix-baseline]
    claim-without-bh-unchanged:
      status: passed
      summary: "'Without BH mass' channel peaks at h=0.678 (log-product) as expected; code path was not modified."
      linked_ids: [deliv-validation-results, test-without-unchanged, ref-pre-fix-baseline]
  deliverables:
    deliv-validation-results:
      status: passed
      path: ".gpd/phases/15-code-audit-fix/15-validation-results.md"
      summary: "Full validation results with pre-fix baseline and post-fix values at 4 h-values"
      linked_ids: [claim-posterior-shift, claim-without-bh-unchanged]
  acceptance_tests:
    test-direction:
      status: failed
      summary: "Posterior at h=0.678 (49.28) < h=0.652 (50.26) -- peak has NOT shifted toward h=0.678"
      linked_ids: [claim-posterior-shift, deliv-validation-results, ref-pre-fix-baseline]
    test-without-unchanged:
      status: passed
      summary: "'Without BH mass' peaks at h=0.678 (64.04) -- unchanged by fix"
      linked_ids: [claim-without-bh-unchanged, deliv-validation-results, ref-pre-fix-baseline]
  references:
    ref-pre-fix-baseline:
      status: completed
      completed_actions: [read, compare]
      missing_actions: []
      summary: "Pre-fix baseline from run_v12_validation: 'with BH mass' monotonically decreasing, 'without BH mass' peaks h=0.730 (sum) / h=0.678 (product)"
  forbidden_proxies:
    fp-no-run:
      status: rejected
      notes: "Evaluation WAS run -- P_det=1 mock used due to missing injection data"
    fp-wrong-direction:
      status: rejected
      notes: "Direction didn't change at all -- no shift observed, not wrong direction"
    fp-two-errors-cancel:
      status: rejected
      notes: "Neither channel peaks at h=0.73"
  uncertainty_markers:
    weakest_anchors:
      - "P_det=1 mock may mask interaction between P_det and the remaining bias"
      - "Only 4 h-values tested -- full posterior shape unknown"
    unvalidated_assumptions: []
    competing_explanations:
      - "Remaining bias from galaxy mass distribution preferring lower-z galaxies"
      - "Conditional decomposition in M_z_frac coordinates introduces systematic tilt"
      - "Redshift-mass correlation in integrand inherently favors lower h"
    disconfirming_observations:
      - "/(1+z) removal did NOT shift posterior by 0.02 from h=0.600 -- additional bias sources exist"

comparison_verdicts:
  - subject_id: claim-posterior-shift
    subject_kind: claim
    subject_role: decisive
    reference_id: ref-pre-fix-baseline
    comparison_kind: baseline
    metric: posterior_peak_location
    threshold: "shift >= 0.02 from h=0.600"
    actual_value: "no shift detected -- still monotonically decreasing"
    verdict: fail
    decisive: true
    notes: "The fix changed absolute scale but not posterior shape"
---

## Performance

| Metric | Value |
|--------|-------|
| Duration | ~15 min (local, 4 h-values × 22 detections) |
| Tasks | 2 (evaluation + verification) |
| Files | 5 created |

## Key Results

### Post-Fix Posterior Values (P_det = 1)

| h     | with BH (log) | w/o BH (log) |
|-------|---------------|--------------|
| 0.652 | **50.259**    | 63.955       |
| 0.678 | 49.282        | **64.039**   |
| 0.704 | 48.109        | 63.976       |
| 0.730 | 46.747        | 63.773       |

### Directional Test: FAIL

The "with BH mass" posterior is still monotonically decreasing. The /(1+z) fix changed the absolute scale (pre-fix log ~-185, post-fix log ~+50) but not the shape. The bias toward low h persists.

### "Without BH mass" Channel: UNCHANGED (PASS)

Peaks at h=0.678 (log-product) as expected. The fix did not affect this code path.

## Conclusion

The /(1+z) removal was **theoretically correct** (Eq. 14.21, verified numerically in Phase 14 to rtol=1e-10) but **insufficient** to resolve the low-h bias in the "with BH mass" channel. Additional bias sources must be investigated in Phase 16.

## Task Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | ab476f9 | Create validation script and document pre-fix baseline |
| 2 | (orchestrator) | Run evaluation, analyze results, write summary |

## Issues Encountered

1. `SimulationDetectionProbability` requires injection campaign data not available locally -- used P_det=1 mock
2. Pre-fix baseline in STATE.md (h=0.678 for "without BH mass") differs from run_v12_validation data (h=0.730 sum-based) -- likely different datasets or product-vs-sum distinction

## Open Questions (for Phase 16)

1. What is the remaining bias source in the "with BH mass" channel?
2. Does the bias persist with realistic P_det (not just P_det=1)?
3. Is the conditional Gaussian decomposition in M_z_frac coordinates introducing a systematic tilt?
4. Does the galaxy mass distribution inherently prefer lower-z associations?
