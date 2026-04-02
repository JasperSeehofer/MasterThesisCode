---
status: awaiting_human_verify
trigger: "H0 posterior residual bias after removing /d_L factor and disabling P_det"
created: 2026-03-30T00:00:00Z
updated: 2026-03-30T01:15:00Z
---

## Current Focus

hypothesis: CONFIRMED - Both bias channels trace to the same root cause: disabled P_det
test: Verified with analytical tests and per-detection analysis
expecting: Re-enabling P_det (after KDE issues are resolved) should fix both channels
next_action: Present findings and discuss path forward with user

## Symptoms

expected: H0 posterior should peak at or near h=0.73 (the true Hubble constant used in the simulation)
actual: "Without BH mass" peaks at h=0.678 (offset -0.052). "With BH mass" peaks at h=0.600 (offset -0.130).
errors: No runtime errors - biased results only
reproduction: Run evaluation with --evaluate on v12 validation dataset (22 detections)
started: First observed in v12 validation run; diagnostic fix partially improved "without BH mass" but not "with BH mass"

## Eliminated

- hypothesis: Covariance matrix construction is incorrect
  evidence: Eigenvalues are all positive, fractional parametrization is mathematically correct, single-galaxy test peaks at h=0.73
  timestamp: 2026-03-30T00:35:00Z

- hypothesis: Integration limits (d_L-based vs galaxy z-based) cause the bias
  evidence: Both approaches produce identical results to 6+ decimal places
  timestamp: 2026-03-30T00:36:00Z

- hypothesis: Core likelihood formula is mathematically wrong
  evidence: Single-galaxy test at true host peaks exactly at h=0.73 for all tested z values (0.03-0.20)
  timestamp: 2026-03-30T00:40:00Z

- hypothesis: d_L(z,h) nonlinearity causes systematic bias in numerator
  evidence: True-host-only likelihood peaks at exactly h=0.73 regardless of z_true; bias only appears with multiple galaxies
  timestamp: 2026-03-30T00:45:00Z

- hypothesis: Gaussian index [0] vs [1] in "with BH mass" numerator causes the -0.130 offset
  evidence: Under the delta function approximation (M_z = M_det, so M_frac = 1), the 3D and 4D gaussians give identical results because sigma_M_frac ~ 1e-7 makes the M dimension a delta function at the mean. Using allow_singular=True, the 4D gaussian evaluated at M_frac=1 returns the same value as the 3D gaussian.
  timestamp: 2026-03-30T01:10:00Z

## Evidence

- timestamp: 2026-03-30T00:30:00Z
  checked: Minimal analytical test with single galaxy and true parameters
  found: Formula peaks correctly at h=0.73 in isolation - core formula is NOT the issue
  implication: Bug is in the specific code path, not the mathematical formulation

- timestamp: 2026-03-30T00:35:00Z
  checked: Integration limits (d_L-based vs galaxy z-based)
  found: No difference in results - integration limits are not the issue
  implication: Limits are wide enough to capture all relevant support

- timestamp: 2026-03-30T00:40:00Z
  checked: "with BH mass" numerator integrand at line 631
  found: Uses gaussian index [0] (3D without BH mass) instead of [1] (4D with BH mass).
    However, under the delta function approximation, this makes no numerical difference.
  implication: Code style bug but not a numerical bug under current approximation

- timestamp: 2026-03-30T00:42:00Z
  checked: Denominator values in posterior JSON
  found: denominator_without_bh_mass = 0.9999... (correct, integrating normal PDF over 4-sigma)
  found: denominator_with_bh_mass = 1.0 exactly (correct when P_det=1, MC sampling cancels)
  implication: Denominator computation is correct

- timestamp: 2026-03-30T00:50:00Z
  checked: True host contribution vs total for detection 4 (biased detection, z_true=0.136)
  found: True host contributes only 0.57% of total numerator (981/171405).
    1197 candidate galaxies overwhelm the true host signal.
  implication: With P_det=1, the posterior is dominated by galaxy catalog density, not GW measurement

- timestamp: 2026-03-30T00:55:00Z
  checked: Per-detection peak h values for all 22 detections
  found: 12/22 detections peak below h=0.73. Clear redshift-dependent pattern:
    - z < 0.09 detections (3 total): all peak at h=0.86 (highest tested)
    - z = 0.09-0.11 detections (6 total): peak at h=0.70-0.73 (near true)
    - z > 0.12 detections (13 total): most peak at h=0.60-0.65 (strongly biased low)
  implication: Systematic redshift-dependent bias consistent with galaxy density effects

- timestamp: 2026-03-30T01:00:00Z
  checked: GW likelihood at true host for different h
  found: At h=0.60, d_L(z_true, 0.6)/d_L_det = 1.218, giving Gaussian factor = 1.35e-03.
    At h=0.86, factor = 4.36e-02. True host contribution drops by >1000x at h=0.60.
  implication: Non-host galaxies dominate the numerator at h != h_true

- timestamp: 2026-03-30T01:05:00Z
  checked: "With BH mass" vs "without BH mass" numerator magnitudes (detection 4)
  found: "With BH mass" numerator is ~4000x smaller (4.1e-02 vs 1.7e+05) because the
    mass factor galaxy_mass_norm.pdf(M_det/(1+z))/(1+z) restricts to mass-matching galaxies.
    Only ~362 galaxies (vs 1197) contribute, further amplifying the catalog density bias.
  implication: "With BH mass" channel is MORE sensitive to P_det=1 limitation, not LESS

- timestamp: 2026-03-30T01:10:00Z
  checked: 3D vs 4D gaussian comparison for actual detection covariance
  found: sigma_M_frac = 7.1e-7 (near-zero fractional mass error). 4D gaussian at M_frac=1.0
    gives identical results to 3D gaussian for all tested d_L_frac values.
    4D gaussian at M_frac != 1 gives essentially zero (delta function behavior).
  implication: The gaussian index [0] vs [1] makes no numerical difference under delta approx

## Resolution

root_cause: |
  The residual bias in BOTH channels has the same root cause: _DEBUG_DISABLE_DETECTION_PROBABILITY = True.

  When P_det is disabled (set to 1.0), the dark siren likelihood formula becomes sensitive to the
  galaxy catalog density distribution in the search window. The formula computes:
    L(h) = sum_gal[integral L_GW(d_L(z,h)) * p(z|gal) dz] / N_gal

  This is the average GW likelihood over candidate galaxies. For the TRUE host galaxy, the
  GW likelihood peaks at h_true = 0.73. But the true host contributes only ~0.6% of the total
  (overwhelmed by ~1000 non-host galaxies). The non-host galaxies introduce a bias that depends
  on the galaxy density gradient in redshift space within the search window.

  The bias is redshift-dependent:
  - Low-z detections (z < 0.09): more galaxies at HIGHER z => bias toward HIGH h
  - High-z detections (z > 0.12): more galaxies at LOWER z => bias toward LOW h
  - The net effect (dominated by the majority high-z detections): bias toward LOW h

  The "with BH mass" channel is worse because the mass-matching factor reduces the effective
  number of contributing galaxies, amplifying the density-driven bias.

  With a correct P_det(z, h) in both numerator and denominator, the per-galaxy ratio:
    integral [P_det * L_GW * p(z|gal)] dz / integral [P_det * p(z|gal)] dz
  properly re-weights galaxies by their detection-probability-adjusted contribution, eliminating
  the catalog density bias.

  SECONDARY CODE ISSUE (correctness, no numerical impact currently):
  Line 631: numerator_integrant_with_bh_mass() uses gaussian index [0] (3D) instead of [1] (4D).
  Under the current delta function approximation (M_frac always = 1), this has no numerical
  effect because the 4D gaussian collapses to the 3D result. However, it should be corrected
  for clarity and to prevent issues if the approximation is changed.

fix: |
  1. Re-enable detection probability: set _DEBUG_DISABLE_DETECTION_PROBABILITY = False (line 60).
     This requires the KDE-based P_det to be working correctly, which was previously identified
     as part of the bias audit.
  2. Correct the gaussian index on line 631: change [0] to [1] and add the M_frac dimension
     to the evaluation point (value = 1.0 under delta approximation).

verification:
files_changed: []
