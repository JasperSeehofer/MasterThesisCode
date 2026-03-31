# Phase 15: Quick Validation Results

## Pre-Fix Baseline (run_v12_validation, 22 detections)

Data source: `evaluation/run_v12_validation/simulations/posteriors/` and `posteriors_with_bh_mass/`
Git commit: `b2019bb` (pre-fix, contains spurious `/(1+z)`)

### Without BH Mass Channel

| h     | sum(posteriors) | n_detections |
|-------|-----------------|--------------|
| 0.600 | 4996.2900       | 22           |
| 0.626 | 5019.8282       | 22           |
| 0.652 | 5038.7891       | 22           |
| 0.678 | 5052.7217       | 22           |
| 0.704 | 5061.7825       | 22           |
| 0.730 | 5065.1721       | 22           |
| 0.756 | 5063.5014       | 22           |
| 0.782 | 5056.1839       | 22           |
| 0.808 | 5043.0052       | 22           |
| 0.834 | 5023.7381       | 22           |
| 0.860 | 4998.9928       | 22           |

**Peak: h ~ 0.730** (sum-based), monotonically increasing 0.600-0.730, then decreasing.

### With BH Mass Channel (PRE-FIX, contains spurious /(1+z))

| h     | sum(posteriors) | n_detections |
|-------|-----------------|--------------|
| 0.600 | 0.023029        | 22           |
| 0.626 | 0.022836        | 22           |
| 0.652 | 0.022565        | 22           |
| 0.678 | 0.022246        | 22           |
| 0.704 | 0.021870        | 22           |
| 0.730 | 0.021453        | 22           |
| 0.756 | 0.020960        | 22           |
| 0.782 | 0.020455        | 22           |
| 0.808 | 0.019886        | 22           |
| 0.834 | 0.019252        | 22           |
| 0.860 | 0.018599        | 22           |

**Peak: h <= 0.600** (monotonically decreasing across entire range). Consistent with STATE.md claim that pre-fix "with BH mass" peak is at h=0.600 (or below).

## Post-Fix Results (P_det = 1, local execution)

**Status: COMPLETE**

Ran locally with mock P_det=1 (matching pre-fix baseline conditions where SimulationDetectionProbability was not yet available). Script: `scripts/quick_validation_15.py`.

Git commit: post-`1d4e9a1` (/(1+z) removed from line 655)

### Post-Fix With BH Mass Channel (log-product of per-detection likelihoods)

| h     | log(prod L_i) | sum(L_i)     | n_detections |
|-------|---------------|--------------|--------------|
| 0.652 | **50.2587**   | 7.965e+02    | 22           |
| 0.678 | 49.2824       | 7.718e+02    | 22           |
| 0.704 | 48.1094       | 7.424e+02    | 22           |
| 0.730 | 46.7468       | 7.092e+02    | 22           |

**Peak: h <= 0.652** (still monotonically decreasing). The absolute scale shifted dramatically (pre-fix log ~-185, post-fix log ~+50), confirming the /(1+z) removal changed the likelihood values, but the shape (monotonically decreasing) is unchanged.

### Post-Fix Without BH Mass Channel

| h     | log(prod L_i) | sum(L_i)     | n_detections |
|-------|---------------|--------------|--------------|
| 0.652 | 63.9548       | 1.309e+03    | 22           |
| 0.678 | **64.0385**   | 1.319e+03    | 22           |
| 0.704 | 63.9757       | 1.321e+03    | 22           |
| 0.730 | 63.7725       | 1.316e+03    | 22           |

**Peak: h ~ 0.678** (product-based) / h ~ 0.704 (sum-based). Consistent with expected behavior.

## Acceptance Criteria Results

1. **Direction test (test-direction): FAIL.** Post-fix "with BH mass" posterior at h=0.678 (49.28) < h=0.652 (50.26). The peak has NOT shifted toward h=0.678.
2. **Unchanged test (test-without-unchanged): PASS.** "Without BH mass" channel peaks at h=0.678 as expected (code path was not modified).
3. **No overshoot: PASS.** Neither channel peaks at h=0.73.
4. **Minimum shift: FAIL.** The "with BH mass" peak remains at h <= 0.652 (no measurable shift toward higher h).

## Conclusion

**The /(1+z) fix is theoretically correct** (Phase 14 derivation proved it spurious via Jacobian absorption identity, Eq. 14.21) **but insufficient to resolve the low-h bias.** The "with BH mass" channel's posterior is still monotonically decreasing across the tested h-range.

This triggers the disconfirming observation from the plan uncertainty markers:
> "If posterior peak does NOT shift by at least 0.02 from h=0.600, the /(1+z) fix alone is insufficient and additional bias sources must be investigated"

### Possible additional bias sources (for future investigation)

1. **p_det(detection.M) vs p_det(M_gal*(1+z))**: The numerator uses the ML mass estimate (detection.M) for p_det, but the physics requires M_gal*(1+z) at the trial redshift. With P_det=1 mock this doesn't matter, but it may interact with other terms.
2. **Galaxy mass distribution mismatch**: The Gaussian p_gal(M) may systematically prefer lower-z galaxies where M*(1+z) better matches the detected M_z.
3. **Conditional decomposition in M_z_frac coordinates**: The rescaling to fractional coordinates (M_z_frac = M*(1+z)/M_z_det) may introduce a bias through the Gaussian product integral.
4. **Redshift-mass correlation**: The product of p_gal(z) and the mass integral may inherently favor lower h (which maps to lower z for a given d_L), creating a systematic tilt.

**Recommendation:** Phase 16 should investigate these additional bias sources before declaring the audit complete. The /(1+z) fix should be kept (it IS correct physics) but the remaining bias needs a separate root-cause analysis.
