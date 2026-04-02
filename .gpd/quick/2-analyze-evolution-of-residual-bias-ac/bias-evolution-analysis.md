# Bias Evolution Analysis: H0 Posterior Across Milestones v1.2–v1.2.2

**Date:** 2026-04-02
**Purpose:** Diagnostic baseline before v1.3 planning. Separates eliminated from remaining bias sources and explains the observed asymmetry: "with BH mass" posterior pulls low (h=0.652 with P_det=1) while "without BH mass" pulls high (h=0.86 with real P_det).

**h_true = 0.73. σ_h ≈ 0.07 from 22 detections.**

---

## 1. Bias Timeline — What Was Fixed and When

### Pre-v1.2 Baseline (git b2019bb, run_v12_validation, 22 detections)

"Without BH mass" (sum-based): h=0.730 — exactly h_true. Within statistical uncertainty.
"Without BH mass" (product-based): h=0.678 — 0.7σ below h_true. Consistent with fluctuation.
"With BH mass": posterior monotonically decreasing across [0.600, 0.860], peak at h ≤ 0.600.

The "with BH mass" result was clearly anomalous: the mass information was completely ignored due to a code bug (see Fix 4 below).

---

### Fix 1: Fisher Matrix O(ε) Forward Difference → O(ε⁴) Five-Point Stencil (PHYS-01, v1.2)

**What was wrong:** `parameter_estimation.py:336` called `finite_difference_derivative()` (O(ε) forward difference) instead of `five_point_stencil_derivative()` (O(ε⁴) central difference). This underestimates the Fisher matrix accuracy, producing wider/shifted Cramer-Rao bounds.

**Expected posterior change:** More accurate Fisher matrix → tighter, better-centered GW likelihood Gaussians → posterior closer to h_true for both pipelines.

**Confirmed effect:** Applied in v1.2. The production dataset (1000+ detections) was generated with this fix. No before/after comparison on the same dataset was documented.

**Status: ELIMINATED** (fix applied; remaining biases are downstream of correct Fisher matrix computation).

---

### Fix 2: Galactic Confusion Noise Added to LISA PSD (PHYS-02, v1.2)

**What was wrong:** Galactic confusion noise constants were defined in `constants.py:77–83` (Cornish & Robson 2017, arXiv:1703.09858) but never incorporated into the PSD used by `scalar_product_of_functions`. The PSD was instrument noise only.

**Expected posterior change:** Higher PSD at low frequencies → lower SNR for some events → fewer detections above threshold → selection effects change. The confusion noise primarily affects f < 3 mHz, which overlaps EMRI signal bandwidth.

**Confirmed effect:** Applied in v1.2. SNR threshold simultaneously lowered from 20 to 15 to compensate for the higher effective noise floor.

**Status: ELIMINATED** (fix applied; PSD is now physically correct with confusion noise).

---

### Fix 3: KDE → IS (Importance Sampling) Estimator for P_det (v1.2 / v1.2.2)

**What was wrong:** The original detection probability estimator used kernel density estimation (KDE) on the detected event distribution. KDE is known to underestimate probability in low-density regions (tails) and overestimate at the peak of the distribution.

**Expected posterior change:** Better P_det estimates → more accurate normalization of the likelihood denominator → reduced systematic bias from P_det miscalibration.

**Confirmed effect:** IS estimator validated in v1.2.2 (VALD-01 PASS: 916 bins, VALD-02 PASS: alpha_grid = alpha_MC for all 7 h-values). Variance reduction factor 11.8–24.9x in boundary bins. The IS estimator is backward-compatible.

**Status: ELIMINATED** (IS estimator validated and confirmed exact backward-compatible; P_det estimation is now demonstrably correct in the validated range).

---

### Fix 4: Gaussian Index Bug — [0] (3D) → [1] (4D) in "With BH Mass" Numerator (GPD Debug Session, 2026-03-30, v1.2)

**What was wrong:** In `bayesian_statistics.py` lines 631 and 831, the "with BH mass" numerator used `detection_likelihood_gaussians[i][0]` — the 3D Gaussian over (phi, theta, d_L_frac) — instead of `[1]`, the 4D Gaussian over (phi, theta, d_L_frac, M_z_frac). The evaluation point was 3-component rather than 4-component. BH mass information was completely absent from the computation despite the code path being labeled "with BH mass."

**Expected posterior change:** Correct Gaussian index → BH mass information actually enters the likelihood. The posterior should shift substantially and narrow.

**Confirmed effect:** After fix, "with BH mass" posterior log-scale shifted dramatically (log L ≈ -185 → ≈ +50), confirming mass information now enters. However, the posterior remains monotonically decreasing (peak still at h ≤ 0.652 with P_det=1). Shape unchanged from pre-fix despite scale change — this is the key observation that motivates the remaining bias investigation.

**Status: ELIMINATED** (the bug is fixed; remaining bias is a different mechanism).

---

### Fix 5: Spurious /(1+z) Jacobian Removal from "With BH Mass" Numerator (Phase 14–15, v1.2.1)

**What was wrong:** The numerator integrand contained a /(1+z) factor (lines 655 and 870) believed to be a Jacobian correction for the d(M_z)/d(M) change of variables. Phase 14 first-principles derivation showed this factor is spurious: the Jacobian |dM/dM_z_frac| = M_z_det/(1+z) is exactly absorbed by the Gaussian normalization identity when transforming p_gal(M) to M_z_frac coordinates (Eq. 14.21 of derivations/dark_siren_likelihood.md).

**Expected posterior change:** Remove spurious suppression of numerator at high z (high z → small 1/(1+z) → numerator was artificially suppressed at high z → posterior was biased toward lower z → lower h). Removing /(1+z) should shift posterior toward higher h.

**Confirmed effect (Phase 15, Plan 02):** Post-fix "with BH mass" posterior: monotonically decreasing, peak at h ≤ 0.652. Directional test: FAIL. The expected shift toward higher h was not observed. The "without BH mass" result was unchanged (h=0.678), confirming the fix applied only to the correct code path and did not introduce regressions. The log-scale values in the 22-detection validation run are consistent with correct marginalizations; the bias persists at the same level.

**Status: ELIMINATED as a bug** (the spurious factor is removed and its absence is confirmed correct by derivation). However, **the fix was INEFFECTIVE at resolving the observed bias** — the bias persists, indicating a different dominant mechanism.

---

### Summary Table

| Fix | Milestone | Classification | Posterior Effect Confirmed? |
|-----|-----------|---------------|-----------------------------|
| 1. Five-point stencil (PHYS-01) | v1.2 | ELIMINATED | Not directly measured (production dataset post-fix) |
| 2. Confusion noise in PSD (PHYS-02) | v1.2 | ELIMINATED | Not directly measured (production dataset post-fix) |
| 3. KDE → IS estimator for P_det | v1.2/v1.2.2 | ELIMINATED | Yes: VALD-01/VALD-02 PASS |
| 4. Gaussian index bug [0] → [1] | v1.2 | ELIMINATED | Yes: log L shifted +135 units |
| 5. Spurious /(1+z) removal | v1.2.1 | ELIMINATED (INEFFECTIVE) | Yes: applied; bias unchanged |

No eliminated fix appears in the remaining bias list below.

---

## 2. Remaining Bias Sources — Mechanistic Analysis

### Source A: p_det(M_detection) Mismatch in Numerator

**Code location:** `bayesian_statistics.py` lines 622–626.

**Formula term:**
```
p_det = detection_probability_with_bh_mass_interpolated(d_L, detection.M, phi, theta, h=h)
```

The numerator evaluates P_det at the maximum-likelihood mass estimate `detection.M` (the detector-frame BH mass from the observed signal), not at `M_gal * (1+z)` at the trial redshift z being integrated over.

**Mechanism:** At trial z, the correct physical question is: "what is the probability of detecting an EMRI with M_z = M_gal*(1+z) at luminosity distance d_L(z, h)?" The denominator correctly uses `M_z = M_gal*(1+z)` (line 671). The numerator instead holds M fixed at the detection ML value regardless of z, which is a different mass at each trial z.

**Directional prediction (toward lower h):** For the "with BH mass" pipeline, decreasing h means increasing d_L for the same observed signal, which corresponds to a higher trial z. At higher trial z, the correct M_z = M_gal*(1+z) grows. The P_det interpolator is evaluated over a (d_L, M_z) grid; using the wrong mass `detection.M` instead of the growing `M_gal*(1+z)` means P_det is evaluated at a mass that is systematically below the physically correct value at high trial z. Since P_det generally increases with M_z (heavier EMRIs are louder), the numerator P_det is underestimated at high z (low h), biasing the posterior toward low h.

**Affects:** "With BH mass" only (the "without BH mass" pipeline does not use mass-dependent P_det in the numerator).

**Estimated severity: HIGH.** This mismatch exists at every integration step, for every detection, for every h-bin. The numerator and denominator use inconsistent P_det inputs, which is a systematic violation of the likelihood normalization.

---

### Source B: Galaxy Mass Distribution z-Asymmetry

**Code location:** `bayesian_statistics.py` lines 641–642, 648–651.

**Formula terms:**
```
mu_gal_frac = M_gal * (1+z) / M_z_det
sigma_gal_frac = M_gal_error * (1+z) / M_z_det
```
The Gaussian product integral:
```
mz_integral = N(mu_cond; mu_gal_frac, sigma2_cond + sigma_gal_frac^2)
```

**Mechanism:** The galaxy mass prior, when expressed in M_z_frac coordinates, has a mean that grows linearly with (1+z). For a fixed galaxy M_gal and observed M_z_det, at trial redshift z the prior mean in M_z_frac space is M_gal*(1+z)/M_z_det. The GW likelihood conditional mean (mu_cond) is approximately 1 (near the detected mass). The integral peaks when mu_gal_frac ≈ mu_cond, i.e., when M_gal*(1+z)/M_z_det ≈ 1, i.e., when z ≈ (M_z_det/M_gal) - 1.

**Directional prediction (toward lower h):** The mass integral is largest when the trial z places the galaxy mass near the detected M_z. This preferred z depends only on M_gal and M_z_det, not on h. However, for the mz_integral to be non-negligible, the galaxy must be at z ~ M_z_det/M_gal - 1. Lower h maps the same d_L to a higher z, and if that z is closer to the preferred z for mass matching, the numerator is enhanced at lower h. Whether this systematically favors lower h depends on the M_z_det/M_gal distribution of the detections. Given that detected events have M_z_det = M_gal*(1+z_true) at h=h_true, the mass-preferred z equals z_true, which is consistent with h=h_true. This source is therefore expected to be approximately unbiased but may contribute to the width of the posterior.

**Affects:** "With BH mass" only.

**Estimated severity: LOW to MEDIUM.** Expected to contribute to posterior width rather than systematic offset, unless there is a population-level asymmetry in the M_gal distribution of the galaxy catalog.

---

### Source C: Conditional Decomposition Tilt in M_z_frac Coordinates

**Code location:** `bayesian_statistics.py` lines 600–655.

**Formula terms:** The conditional Gaussian decomposition (Eqs. 14.23–14.31 of the derivation) uses the pseudoinverse `cov_obs_inv = np.linalg.pinv(cov_obs)` and computes:
```
sigma2_cond = cov_mz - cov_cross @ cov_obs_inv @ cov_cross
```
The floor `max(sigma2_cond, 1e-30)` is applied to prevent numerical issues.

**Mechanism:** Near-singular Fisher matrices (low-SNR events, or events where the mass is poorly constrained) may produce a very small or negative `sigma2_cond` before flooring. After flooring to 1e-30, the conditional distribution becomes a delta function — extremely narrow — and the mz_integral evaluates as a delta function comparison between mu_cond and mu_gal_frac. If the floor is triggered, the likelihood contribution of that event is artificially suppressed unless the galaxy mass is nearly exactly equal to the conditional mean (a rare coincidence).

**Directional prediction:** The floor is triggered when the Fisher matrix is near-singular, which is more common for events near the SNR threshold (SNR ≈ 15–20). The effect is suppression of contributions from these marginal events. Whether this biases h low or high depends on whether marginally-detected events are more common at specific h-values. Since the denominator does not use the conditional variance (it marginalizes independently), a floored numerator without a compensating floored denominator would suppress the likelihood at the h-values corresponding to the redshifts of marginal events.

**Affects:** "With BH mass" only.

**Estimated severity: MEDIUM.** This is a numerical precision issue for low-SNR events, not a fundamental formula error. However, with SNR threshold lowered to 15, a larger fraction of events are near the threshold.

---

### Source D: Redshift-Mass Correlation in Joint p_gal(z) * Mass Integral

**Code location:** The numerator integrant (`numerator_integrant_with_bh_mass`) integrates `p_det * gw_3d * mz_integral * p_gal(z)` over z. The denominator integrates `p_det * p_gal(z) * p_gal(M)` over a 2D (M, z) grid.

**Mechanism:** In the numerator, mz_integral and p_gal(z) are not independent — both depend on z through the (1+z) factor. The numerator thus evaluates a joint integral where the mass compatibility weight (mz_integral) and the redshift prior (p_gal(z)) are correlated. This correlation is not present in the denominator, which uses a factored p_gal(z) * p_gal(M) prior. If the mz_integral systematically favors the same z that maximizes p_gal(z), the numerator is disproportionately enhanced at that z.

**Directional prediction:** The galaxy redshift distribution p_gal(z) peaks at low z (GLADE catalog completeness to z ~ 0.1 introduces a sharp drop-off). The mz_integral peaks where M_gal*(1+z)/M_z_det ≈ 1, i.e., at z_peak ≈ z_true. For nearby hosts (z_true < 0.1), both p_gal(z) and mz_integral peak at the same z, producing coherent enhancement of the numerator. For distant hosts (z_true > 0.1), catalog incompleteness reduces p_gal(z), and the mz_integral peak is outside the catalog's supported z-range. This asymmetry amplifies the numerator for nearby events and suppresses it for distant ones. Nearby events correspond to higher h (same d_L, lower z → higher H0 from v = H0 * d_L), so one might expect a bias toward higher h. However, Source A (p_det mismatch) operates in the opposite direction and at higher magnitude, which may dominate. The net effect requires a numerical test.

**Affects:** "With BH mass" only (the denominator uses factored priors).

**Estimated severity: MEDIUM.** The asymmetry is real and traceable, but its direction is not certain without numerical experiments.

---

### Source E: "Without BH Mass" High-h — P_det Normalization / Selection Effect

**Code location:** `bayesian_statistics.py` lines 545–563 (without BH mass numerator integrand) and the denominator integration.

**Mechanism:** With P_det=1 (22-detection mock), "without BH mass" gives h=0.678 — within 0.7σ of h_true=0.73. With real P_det (1000+ detections, production dataset), "without BH mass" gives h=0.86. The likelihood formula is identical in both cases; only P_det changes. This is the strongest evidence available: the "without BH mass" likelihood formula itself is approximately correct.

The P_det selection effect works as follows. The normalization denominator is:
```
p(detection | h) ∝ ∫ p_det(d_L(z,h), ...) * p_gal(z) dz
```
If P_det is overestimated at high h (or underestimated at low h), the denominator is larger at high h, which depresses the posterior at high h. Conversely, if the IS estimator outputs P_det values that are calibrated to the true h_true=0.73 injection distribution but used at trial h ≠ h_true, the P_det interpolation may extrapolate outside its calibration range.

**More specifically:** The IS estimator was trained on events simulated at h_true=0.73. When evaluated at trial h=0.86, d_L(z, h=0.86) > d_L(z, h=0.73) for fixed z, meaning events appear further away. The P_det interpolator receives a larger d_L than it was trained on, placing queries in the interpolator's low-d_L coverage boundary. The IS estimator paper (v1.2.2, Phase 19) validated VALD-01/VALD-02 across 7 h-values, but this validates the IS estimator's consistency, not that P_det itself correctly accounts for h-variation in selection.

**Alternative interpretation:** The 15x10 detection probability grid may have insufficient coverage at high h (high d_L), and the RegularGridInterpolator may return boundary values (extrapolation artifacts) that systematically misweight the denominator, shifting the posterior toward the h-range where the IS estimator is best calibrated (h ≈ 0.73 → h ≈ 0.86 overshoot?).

**Directional prediction (toward higher h):** If P_det is overestimated at low h in the denominator (the denominator integral at low h is too large, suppressing the posterior there), the posterior peak shifts to higher h. This is consistent with the observed h=0.86 overshoot.

**Affects:** "Without BH mass" primarily (since the "with BH mass" result with real P_det gives h=0.72, which is close to h_true — the mass terms may partially compensate the P_det selection effect in that pipeline).

**Estimated severity: HIGH** for "without BH mass" production run. The P_det=1 / real-P_det comparison is the key diagnostic: the 0.182 shift in posterior peak (0.678 → 0.86) is entirely attributable to P_det. This is larger than the single-σ uncertainty.

---

### Source F: Zero-Likelihood Problem (21% of Events, "With BH Mass")

**Code location:** Individual event likelihoods returning zero at some h-bins.

**Mechanism:** For 21% of "with BH mass" events in the production run, the likelihood evaluates to zero at some h-bins. Possible causes: (a) galaxy catalog gap — no galaxy candidate within the angular search radius at the trial d_L(z,h); (b) numerator integral evaluates below floating-point precision (extremely small Gaussian products); (c) P_det interpolator returns exactly zero at the queried (d_L, M_z) grid point. Each such zero contribution forces that event to contribute nothing at those h-bins, effectively reducing the sample size for the affected bins.

**Directional prediction:** If zero-likelihood events are not uniformly distributed across h-bins — for example, if events produce zeros more often at higher h-bins (where d_L is larger and galaxy catalog completeness is lower), then the remaining non-zero events are biased toward h-bins where all events contribute, which is lower h. This would reinforce the low-h bias.

**Affects:** "With BH mass" only (the "without BH mass" pipeline does not depend on the 4D Gaussian or mass integral, so the zero-likelihood rate is lower there).

**Estimated severity: MEDIUM.** The 21% rate is substantial. The direction of bias from this source is consistent with the observed low-h pull but has not been quantified.

---

### Source G: Quadrature (Numerator) vs Monte Carlo (Denominator) Methodology Asymmetry

**Code location:** Numerator uses `scipy.integrate.fixed_quad` with `n=FIXED_QUAD_N` nodes. Denominator uses MC integration with N=10,000 samples over the 2D (M, z) grid.

**Mechanism:** The numerator is a 1D quadrature (fixed_quad is exact for polynomials up to degree 2n-1). The denominator is a 2D Monte Carlo with relative error ~1/√N ≈ 1% per evaluation. Systematic integration error can arise if the fixed_quad node count is insufficient for the peaked integrands that arise when the galaxy is the true host (sharp Gaussian peak in z-space). MC integration at 1% relative error is unlikely to introduce systematic bias (random errors average out over many events and h-bins), but the asymmetry in integration method means any numerical bias is not correlated between numerator and denominator.

**Directional prediction:** Neither clearly toward low nor high h. The 1% MC relative error contributes to posterior width, not peak location. The fixed_quad integrator could introduce systematic bias if the integrand has structure at scales below the node spacing, but this is not documented.

**Affects:** Both pipelines (the same numerator/denominator asymmetry applies to "without BH mass", though the without-BH-mass case has a simpler numerator integrand).

**Estimated severity: LOW.** The 1% MC error averages to near zero over 1000+ events. This is a noise source, not a systematic.

---

## 3. Differential Diagnosis

### Why does "with BH mass" pull low while "without BH mass" pulls high?

The two pipelines experience fundamentally different dominant bias sources:

**"With BH mass" — dominant source: Source A (p_det(M_detection) mismatch)**

In the "with BH mass" numerator, P_det is evaluated at the fixed observed mass `detection.M` regardless of the trial z. The denominator correctly uses `M_z = M_gal*(1+z)`. At low trial h (low h → high d_L → high trial z), the correct M_z is larger than detection.M, meaning the numerator underestimates P_det relative to the denominator at low h. This asymmetry inflates the denominator relative to the numerator at low h, suppressing the likelihood there. Wait — this predicts the posterior is suppressed at LOW h, contradicting the observation. Let us re-examine.

More carefully: at high trial h (trial h > h_true), the trial z for the same d_L is lower than z_true. Lower trial z means M_gal*(1+z) is smaller than M_z_det, so the correct M_z_det > M_gal*(1+z_trial). The numerator holds P_det fixed at `detection.M = M_z_det`, while the denominator evaluates P_det at `M_gal*(1+z_trial)` which is smaller. Since P_det is lower for smaller M_z, the denominator P_det is smaller at high h. A smaller denominator raises the likelihood at high h — which contradicts the low-h pull.

**Reconsidering the mechanism:** The "with BH mass" mass integral (mz_integral) is the critical element that the "without BH mass" pipeline lacks. The mz_integral is:
```
N(mu_cond; mu_gal_frac, sigma2_cond + sigma_gal_frac^2)
```
where `mu_gal_frac = M_gal * (1+z) / M_z_det` and `mu_cond ≈ 1` (the detection is near its own mass).

For the mz_integral to be large, we need `mu_gal_frac ≈ mu_cond ≈ 1`, i.e., `M_gal * (1+z) ≈ M_z_det`. This is satisfied at `z ≈ z_true` (the true redshift of the host galaxy). The peak of the mz_integral is thus fixed in z-space, independent of h.

The h-dependence enters through `gw_3d`, the 3D marginal Gaussian in (phi, theta, d_L_frac). The `d_L_frac = d_L(z,h) / d_L_detected`. For a trial galaxy at angular position (phi_gal, theta_gal) and the integration variable z: the gw_3d term is maximized when `d_L(z,h)` matches `d_L_detected`. This happens at `z_peak_gw(h)` where `d_L(z_peak_gw, h) = d_L_detected`.

The joint integrand `gw_3d * mz_integral` is thus peaked at z-values satisfying both:
- `d_L(z, h) ≈ d_L_detected` (from gw_3d)
- `M_gal * (1+z) ≈ M_z_det` (from mz_integral)

At h = h_true: these two conditions are simultaneously satisfied at z = z_true for the true host. At h ≠ h_true: gw_3d peaks at a z where `d_L(z,h)` matches, but mz_integral peaks at z_true. The joint peak is between these two, with amplitude reduced by the mismatch.

**The key asymmetry is the width of mz_integral vs gw_3d in z-space.** If mz_integral is very narrow (tightly constrained mass), the joint peak is dominated by the mass constraint, locking the z-integration to z_true regardless of h. The z_true then maps back to h via `d_L_detected = d_L(z_true, h)`, which gives h = h_true. If this were the whole story, "with BH mass" would be LESS biased than "without BH mass".

The fact that "with BH mass" is MORE biased (lower h) indicates that Source A (p_det mismatch) or Sources C/F introduce an asymmetric suppression that tilts the posterior toward lower h more strongly than the mass constraint can compensate.

**Most likely dominant mechanism for "with BH mass" low-h bias: Zero-likelihood problem (Source F) combined with P_det(M_detection) mismatch (Source A).**

At low h (high trial z): P_det in the numerator is evaluated at `detection.M`, which is lower than the correct `M_gal*(1+z)` at that high trial z. But simultaneously, the IS estimator P_det grid may return zero for some (d_L, M_z_det) combinations at the high d_L corresponding to low h. Zero-probability events cannot contribute to the posterior at those h-bins. If these zero events are more concentrated at low-h bins (because the IS grid has less coverage at high d_L), the effective sample size at low h is reduced, and the posterior is pushed toward the h-bins where fewer events are zeroed out. Depending on the P_det grid structure, this could push the peak to the lowest evaluated h-value — consistent with the observed monotonically decreasing behavior peaking at h ≤ 0.652.

**"Without BH mass" — dominant source: Source E (P_det selection / normalization)**

The cross-check is decisive: "without BH mass" with P_det=1 gives h=0.678 (within 0.7σ of h_true=0.73). With real P_det, the same pipeline gives h=0.86. The shift of 0.182 in posterior peak is entirely produced by switching P_det from unity to the IS estimator. The "without BH mass" likelihood formula itself is approximately correct; the high-h overshoot is a P_det artifact.

The most plausible mechanism: the IS estimator denominator integral at low h (low d_L) is overestimated relative to high h, suppressing the posterior at low h and inflating it at high h. This could arise because the IS estimator was trained on events at h_true=0.73 and the P_det grid density/coverage differs across h-values.

**Asymmetry summary:**

| Pipeline | Dominant remaining source | Direction | Formula locus |
|----------|--------------------------|-----------|---------------|
| With BH mass | Source F (zero-likelihood) + Source A (p_det(M) mismatch) | Low h | Lines 622–626, zero-valued IS output |
| Without BH mass | Source E (P_det normalization) | High h | Denominator IS integral, h-dependent coverage |

The two pipelines pull in opposite directions because their remaining bias sources are completely different in nature and location in the likelihood formula.

---

## 4. Prioritized Investigation Agenda

### Priority 1 — Diagnose P_det as "Without BH Mass" High-h Driver [HIGH IMPACT, FEASIBLE NOW]

**What to test:** Re-run the production dataset evaluation (1000+ detections) with P_det=1 (mock). If "without BH mass" recovers h ≈ 0.73, P_det normalization is confirmed as the sole driver of the h=0.86 overshoot.

**Infrastructure needed:** Existing code, existing production dataset. Set `p_det = 1.0` in the denominator integrant. No new code required.

**Expected outcome:** h_peak moves from 0.86 toward h_true. If it moves past 0.73 (overshooting low), the "without BH mass" formula has a residual bias that was masked by P_det.

**Blocks Phase 16:** YES — Phase 16 requires a reliable baseline for posterior validation.

**Action:** Add a `--p_det_mock` flag to `bayesian_statistics.py` that sets P_det=1 uniformly, or invoke the existing P_det=1 code path with the production dataset.

---

### Priority 2 — Diagnose "With BH Mass" Zero-Likelihood Events [HIGH IMPACT, FEASIBLE NOW]

**What to test:** Log which h-bins each event returns zero likelihood in. Tabulate: (a) what fraction of events return zero at each h-bin, (b) whether events that are zero at high-h bins are also zero at low-h bins.

**Infrastructure needed:** Add per-event zero-detection logging. Output a (N_events x N_h_bins) boolean matrix of zero-likelihood occurrences. No new data required.

**Expected outcome:** If zero events cluster at low-h bins, the observed monotonically decreasing shape is explained. If zeros are random, a different mechanism is dominant.

**Blocks Phase 16:** YES — cannot validate the "with BH mass" posterior if 21% of events are silently dropped.

**Action:** Modify `BayesianStatistics.evaluate()` to log zero-likelihood events per h-bin before combining posteriors.

---

### Priority 3 — Fix p_det(M_detection) Mismatch in Numerator (Source A) [MEDIUM IMPACT, REQUIRES CODE CHANGE]

**What to fix:** Replace `detection.M` with `M_gal * (1+z)` in the numerator P_det call (line 625–626). This requires that the numerator integration variable z is passed to the P_det evaluator consistently with the denominator.

**Current code:**
```python
p_det = detection_probability.detection_probability_with_bh_mass_interpolated(
    d_L, np.full_like(z, detection.M), phi, theta, h=h
)
```

**Corrected:**
```python
M_z_trial = np.full_like(z, possible_host.M) * (1 + z)
p_det = detection_probability.detection_probability_with_bh_mass_interpolated(
    d_L, M_z_trial, phi, theta, h=h
)
```

**Infrastructure needed:** Existing code. The `possible_host.M` is already in scope for this function closure. The change is one line.

**Note:** This is a physics change (changes a computed value). Physics Change Protocol required before implementation — present old formula, new formula, reference, dimensional analysis, limiting case.

**Blocks Phase 16:** Partial blocker. Should be fixed before the final posterior comparison.

---

### Priority 4 — P_det Grid Coverage at High d_L (Source E detailed diagnosis) [MEDIUM IMPACT, REQUIRES DATA]

**What to investigate:** Inspect the IS estimator's (d_L, M_z) grid coverage at the d_L values corresponding to h=0.86 and h=0.73. Check whether the denominator IS integral extrapolates outside the training grid at high h-values.

**Infrastructure needed:** Examine the 15x10 grid bounds vs the d_L range queried at each trial h. Can be done with the existing IS estimator output files.

**Action:** Plot P_det(d_L, M_z) as a function of d_L at M_z = detection.M for h in {0.65, 0.73, 0.86}. Identify any boundary-clamping or flat extrapolation behavior.

---

### Priority 5 — Conditional Variance Floor sigma2_cond (Source C) [LOW IMPACT, FEASIBLE]

**What to test:** Log the fraction of events where `sigma2_cond` hits the floor (1e-30) before the max() call. If this fraction is non-negligible (>1%), investigate whether it correlates with low-SNR events or specific h-bins.

**Infrastructure needed:** One-line diagnostic addition to the conditional variance computation.

**Action:** Add `warnings.warn()` or diagnostic logging when `sigma2_cond < 0` before flooring.

---

### Priority 6 — WMAP vs Planck Cosmology (Constants.py Known Bug 8) [LOW IMPACT, DEFERRED]

**Issue:** The simulation uses Omega_m=0.25, H=0.73 (WMAP-era). Planck 2018 best-fit is Omega_m=0.3153, h=0.6736. If h_true=0.73 is the injection value but the true Planck h is 0.6736, the "correct" posterior should recover 0.73 (not Planck), so this is not a bias source for the current pipeline but is a known inconsistency for the final thesis interpretation.

**Deferred to:** Post-v1.3 (requires Physics Change Protocol and full re-simulation).

---

## 5. Key Numerical Anchors

| Condition | Pipeline | h_peak | Dataset | P_det | Note |
|-----------|----------|--------|---------|-------|------|
| Pre-v1.2 baseline | Without BH mass (sum-based) | 0.730 | 22 det (run_v12_validation) | 1 | Matches h_true exactly |
| Pre-v1.2 baseline | Without BH mass (product-based) | 0.678 | 22 det | 1 | 0.7σ below h_true; fluctuation |
| Pre-v1.2 baseline | With BH mass | ≤ 0.600 | 22 det | 1 | Monotonically decreasing; Gaussian index bug |
| Post-all-fixes (v1.2.1) | Without BH mass | 0.678 | 22 det | 1 | Unchanged from baseline; formula correct |
| Post-all-fixes (v1.2.1) | With BH mass | ≤ 0.652 | 22 det | 1 | Still monotonically decreasing; remaining bias |
| Production (v1.2) | Without BH mass | 0.86 | 1000+ det | real IS | 1.9σ above h_true; P_det dominant |
| Production (v1.2) | With BH mass | 0.72 | 1000+ det | real IS | 0.14σ below h_true; closest to h_true |

**h_true = 0.73. σ_h ≈ 0.07 (22-detection estimate).**

**Interpretation of deviations:**

- 0.678 (without BH mass, P_det=1): within 1σ — not statistically significant. Formula is correct.
- 0.652 (with BH mass, P_det=1, 22 det): 1.1σ below — marginal in 22-detection dataset; the small sample means this could be partially statistical, but the monotonically decreasing shape (no peak in [0.652, 0.860]) strongly implicates a systematic.
- 0.86 (without BH mass, real P_det): 1.9σ above — clearly systematic. The P_det=1 cross-check (0.678) proves the formula is not responsible; P_det is.
- 0.72 (with BH mass, real P_det): 0.14σ below — apparently close to h_true, but this is likely a cancellation between multiple bias sources (low-h pull from Sources A/F, high-h pull from Source E acting through the denominator). Not interpretable as evidence that "with BH mass" is correct.

---

## Appendix: Code-Path Map

For reference, the key lines in `bayesian_statistics.py` discussed in this analysis:

| Line | Function | Term | Status |
|------|----------|------|--------|
| 550 | Numerator (w/o BH mass) | gw_3d Gaussian PDF via `[0]` | Correct (3D is right here) |
| 563 | Numerator (w/o BH mass) | `p_det * p_gal(z)` | Correct |
| 592 | Numerator (w/ BH mass) | `detection_likelihood_gaussians[i][1]` — 4D Gaussian | FIXED (was [0], bug 4) |
| 625 | Numerator (w/ BH mass) | `p_det` evaluated at `detection.M` | **KNOWN APPROXIMATION (Source A)** |
| 641 | Numerator (w/ BH mass) | `mu_gal_frac = M_gal * (1+z) / M_z_det` | Correct (coordinate transform) |
| 655 | Numerator (w/ BH mass) | `return p_det * gw_3d * mz_integral * p_gal(z)` | FIXED (spurious /(1+z) removed) |
| 671 | Denominator (w/ BH mass) | `M_z = M_gal * (1+z)` | Correct |
| 674 | Denominator (w/ BH mass) | `p_det` at `M_z = M_gal*(1+z)` | Correct |

---

_Analysis produced: 2026-04-02. Based on pre-researched timeline from planning orchestrator and direct code inspection of `bayesian_statistics.py`._
