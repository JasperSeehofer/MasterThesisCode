---
session_id: h0-posterior-residual-bias
status: resolved
created: 2026-03-30T10:00:00Z
last_updated: 2026-03-30T11:00:00Z
symptom: "H0 posterior peaks at h=0.678 instead of h_true=0.73 after removing /d_L and disabling P_det"
current_focus: "Resolution: confirmed bug fixed + residual explained as statistical fluctuation"
eliminated: ["numerator-denominator integration asymmetry", "H0 unit conversion error", "core formula error"]
root_cause: "Two distinct issues: (1) BUG: Line 631 uses 3D Gaussian [0] instead of 4D Gaussian [1] for 'with BH mass' numerator. (2) NOT A BUG: 'without BH mass' residual offset of -0.052 is a 0.7-sigma statistical fluctuation (posterior sigma_h ≈ 0.07 from 22 detections)."
---

## Current Focus

hypothesis: RESOLVED
test: N/A
expecting: N/A
next_action: Archive session

## Symptoms

expected: H0 posterior should peak at h=0.73 (the true Hubble constant used in simulation)
actual: "Without BH mass" peaks at h=0.678 (offset -0.052). "With BH mass" peaks at h=0.600 (offset -0.130).
errors: No runtime errors -- biased results only
reproduction: Run evaluation with --evaluate on v12 validation dataset (22 detections)
context: Always present. Grid spacing is 0.026 so true peak could be in [0.665, 0.691].

## Eliminated

- hypothesis: Numerator-denominator integration limit asymmetry causes h-dependent normalization mismatch
  evidence: Diagnostic v2 showed all 22 detections peak exactly at h=0.73 when using true host galaxy with perfect sky match. The integration limits are not the issue.
  timestamp: 2026-03-30T10:15:00Z

- hypothesis: H0 unit conversion error in dist()
  evidence: Verified GPC_TO_MPC conversion: H_0 = h*100*KM_TO_M/GPC_TO_MPC^(-1) gives exactly correct values for all h tested.
  timestamp: 2026-03-30T10:15:00Z

- hypothesis: Core formula error
  evidence: All 22 detections individually peak at h=0.73 when evaluated with true host galaxy. Combined log-posterior peaks at h=0.73. Formula is mathematically correct.
  timestamp: 2026-03-30T10:10:00Z

- hypothesis: Measurement noise creates systematic low-h bias
  evidence: With noisy detection params + true host galaxies, combined posterior peaks at h=0.74 (offset +0.01). The noise realization has mean shift -0.24 sigma (14/22 negative) which would bias h HIGH, not low.
  timestamp: 2026-03-30T10:25:00Z

## Evidence

- timestamp: 2026-03-30T10:10:00Z
  checked: Core formula with all 22 detections, true host galaxies, perfect sky match
  found: Every detection individually peaks at h=0.73. Combined log-posterior peaks at h=0.73.
  implication: The likelihood formula is mathematically correct. No code bug in the core math.

- timestamp: 2026-03-30T10:15:00Z
  checked: H0 unit conversion in dist() function
  found: H_0 = h*100*1e3/1e-3 = h*1e8 m/s/Gpc. Verified correct for h=0.6, 0.73, 0.86.
  implication: Distance calculation is correct.

- timestamp: 2026-03-30T10:25:00Z
  checked: Combined posterior with NOISY detection params + true host galaxies
  found: Peak at h=0.74, offset +0.01. Per-detection peaks scatter from 0.66 to 0.82.
  implication: Measurement noise alone does not cause the -0.052 bias.

- timestamp: 2026-03-30T10:35:00Z
  checked: Per-detection likelihood comparison between h=0.678 and h=0.73 from production run
  found: 11/22 detections favor h=0.678, 11/22 favor h=0.73. Total delta = +0.266 (tiny).
  implication: No systematic effect. The preference is random scatter.

- timestamp: 2026-03-30T10:40:00Z
  checked: Posterior width from curvature analysis
  found: sigma_h = 0.068-0.071 (from quadratic fit and finite difference second derivative). The offset -0.052 is 0.69-0.77 sigma from the peak.
  implication: The "residual bias" is a completely normal <1 sigma statistical fluctuation from 22 detections with galaxy catalog confusion.

- timestamp: 2026-03-30T10:45:00Z
  checked: Line 631 bug in "with BH mass" numerator integrand (production function)
  found: Uses `detection_likelihood_gaussians_by_detection_index[detection_index][0]` (3D Gaussian) instead of `[1]` (4D Gaussian). Also evaluates at `[phi, theta, luminosity_distance_fraction]` (3 components) instead of `[phi, theta, luminosity_distance_fraction, redshifted_mass_fraction]` (4 components).
  implication: CONFIRMED BUG. The "with BH mass" path never actually uses the BH mass information in the GW likelihood, explaining why its posterior (h=0.600) is worse than "without BH mass" (h=0.678).

- timestamp: 2026-03-30T10:45:00Z
  checked: Same bug in _integration_testing version at line 831
  found: Same pattern: uses [0] with 3 components instead of [1] with 4 components.
  implication: Both code paths have the same bug.

## Resolution

root_cause: |
  Two distinct issues identified:

  1. **BUG (fixed): Wrong Gaussian index in "with BH mass" numerator** (lines 631, 831)
     - `detection_likelihood_gaussians_by_detection_index[detection_index][0]` should be `[1]`
     - The 3D Gaussian (without BH mass) was used instead of the 4D Gaussian (with BH mass)
     - This meant BH mass correlations from the Fisher matrix were completely ignored
     - Fixed in both the production function (`single_host_likelihood`) and the testing function (`single_host_likelihood_integration_testing`)

  2. **NOT A BUG: "Without BH mass" residual offset of -0.052**
     - Posterior width sigma_h ~ 0.07 (from curvature analysis)
     - The offset -0.052 corresponds to 0.7 sigma -- a normal statistical fluctuation
     - With only 22 detections and galaxy catalog confusion (thousands of candidate hosts per detection), the posterior is broad
     - The 11-vs-11 split of per-detection preferences between h=0.678 and h=0.73 confirms random scatter
     - Increasing the number of detected events would reduce the statistical uncertainty

correction: |
  Fixed lines 631 and 831 in bayesian_statistics.py:
  - Changed Gaussian index from [0] to [1] in `numerator_integrant_with_bh_mass`
  - Added `redshifted_mass_fraction` (= 1.0 under delta-function approximation) as 4th evaluation coordinate
  - Both production (`single_host_likelihood`) and testing (`single_host_likelihood_integration_testing`) functions corrected

verification: |
  - All 198 CPU tests pass (no regressions)
  - Ruff lint: clean
  - mypy: clean
  - The "without BH mass" posterior peaks correctly at h=0.73 when evaluated with true host galaxies (confirmed by diagnostic scripts)
  - The residual -0.052 offset is explained by posterior width analysis (0.7 sigma fluctuation)
  - Re-running the full evaluation pipeline with the BH mass fix would verify the "with BH mass" posterior improves (requires GPU cluster or significant CPU time)

files_changed:
  - master_thesis_code/bayesian_inference/bayesian_statistics.py
