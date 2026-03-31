# Physics and Computational Pitfalls

**Physics Domain:** Gravitational wave selection effects, EMRI injection campaigns, P_det estimation for H0 inference
**Researched:** 2026-03-31
**Context:** Extending EMRI injection campaign results for simulation-based detection probability estimation with importance sampling

## Critical Pitfalls

Mistakes that invalidate the H0 posterior or produce biased detection probability grids.

### Pitfall 1: Parameter Distribution Mismatch Between Injection and Evaluation Pipelines

**What goes wrong:** The injection campaign draws EMRI events from `Model1CrossCheck.sample_emri_events()` which uses an emcee MCMC sampler over the EMRI event rate distribution `emri_distribution(M, z)`. The evaluation pipeline (bayesian_statistics.py) uses `galaxy_redshift_normal_distribution` and `galaxy_mass_normal_distribution` centered on each host galaxy. If the injection distribution does not cover the support of the evaluation distribution, P_det will be zero in regions where the posterior needs it, biasing H0 low (or causing numerical zeros in the denominator).

**Why it happens:** The injection campaign samples from the *population prior* p(M, z), but the Bayesian inference evaluates P_det at the *per-event posterior* locations. These distributions differ: the population prior is broad (z up to 1.5, M over several orders of magnitude), while per-event posteriors are narrow Gaussians centered on the detected source. The mismatch becomes critical if the injection grid has empty bins near detected events.

**Consequences:** If P_det is underestimated (returns 0 in a bin where the true value is nonzero), the selection-effect correction overcorrects, biasing H0. If P_det is overestimated (noisy bin with few samples), H0 is biased in the opposite direction. Since P_det enters the likelihood denominator, even small errors are amplified.

**Prevention:**
- Verify that every detected event's (z, M) falls within the well-populated region of the injection grid. Log a warning if any event falls in a bin with fewer than 10 injections.
- Plot the injection distribution p_inj(z, M) overlaid with the detected events. Gaps indicate coverage failure.
- Ensure the injection campaign's `max_redshift = 1.5` matches the evaluation pipeline's `REDSHIFT_UPPER_LIMIT = cosmological_model.max_redshift` (currently both 1.5).
- Check that the M sampling range in injection matches the evaluation prior. The injection uses `parameter_space.M.lower_limit` to `M.upper_limit`; confirm these match the mass prior in `bayesian_statistics.py`.

**Detection:**
- Anomalously peaked or asymmetric H0 posterior (should be roughly Gaussian for reasonable data).
- P_det grid has columns or rows of zeros adjacent to nonzero values (coverage gap, not physical transition).
- Compare P_det from injection data to the old KDE-based DetectionProbability at overlapping parameter values.

**References:**
- Mandel, Farr & Gair (2019), arXiv:1809.02063 -- foundational framework for selection effects in hierarchical Bayesian inference
- Essick & Farr (2022), arXiv:2204.00461 -- Monte Carlo uncertainty in selection functions

### Pitfall 2: Waveform Failure Correlation with Parameter Regions

**What goes wrong:** The `injection_campaign()` function catches at least 6 types of waveform failures: Warning (mass ratio out of bounds), ParameterOutOfBoundsError, RuntimeError, ValueError (EllipticK, Brent solver), ZeroDivisionError, and TimeoutError (90s). These failures cause the event to be *silently skipped* (not counted in the denominator). If failures correlate with specific parameter regions (e.g., high eccentricity, extreme mass ratios, edge of the FEW flux grid), then P_det is estimated from a *biased subsample* of the population.

**Why it happens:** The FEW/fastlisaresponse waveform generator has known parameter space limitations. For low-mass MBHs, up to 75% of EMRIs at 2 years from plunge fall outside the currently available flux parameter space (Speri et al., arXiv:2509.08875). Waveform failures cluster in specific regions: high eccentricity (e0 close to upper limit 0.2), extreme mass ratios, high spin values (a > 0.9), and configurations where the trajectory integrator fails (Brent solver, EllipticK errors).

**Consequences:** The denominator of P_det = N_detected / N_total undercounts N_total in failure-prone regions. If those regions would have produced mostly undetected events (low SNR), then P_det is overestimated there. If some would have been detected, P_det is underestimated. The direction of bias depends on the correlation between failure probability and SNR, which is not known a priori.

**Prevention:**
- **Track failures separately:** Record failed events with their (z, M, e0, a, q) parameters in a separate CSV or add a `status` column to the injection CSV. Currently, failures are simply `continue`d past with no record.
- **Quantify failure fraction by parameter region:** After the campaign, bin failure rate vs (z, M) and check for spatial correlation. If failure rate exceeds 10% in any populated bin, that bin's P_det is unreliable.
- **Use conservative correction:** For bins with significant failure rates, P_det_corrected = N_detected / (N_success + N_failed), treating all failed waveforms as if they would have been undetected (conservative, biases P_det low rather than unknown direction).
- **Report failure fraction:** Include total failure count and failure-by-type statistics in the injection campaign metadata.

**Detection:**
- Log analysis: count `Continue with new parameters...` messages vs successful SNR computations. Current code logs these but does not aggregate.
- If failure rate > 30%, the injection results in affected parameter regions cannot be trusted.
- Compare P_det(z) shape: physical expectation is monotonically decreasing with z. Non-monotonic features at specific z values may indicate parameter-correlated failures.

**References:**
- Speri et al. (2025), arXiv:2509.08875 -- systematic errors in fast relativistic EMRI waveforms
- Katz et al. (2021), arXiv:2104.04582 -- FastEMRIWaveforms parameter space limitations

### Pitfall 3: z > 0.5 Cutoff Introduces Implicit Importance Sampling Without Weight Correction

**What goes wrong:** The injection campaign applies `if sample.redshift > z_cut: continue` with z_cut = 0.5 (line 474 in main.py). The code comment states "Truncating here does not bias P_det because the numerator and denominator are both zero in the truncated region." This is incorrect as a general claim: it is an importance sampling approximation that is valid only if P_det(z > 0.5) is exactly zero.

**Why it happens:** The initial campaign found 24/69500 detections all at z < 0.18. The z_cut = 0.5 gives a "generous margin." However:
1. The claim that P_det = 0 for z > 0.5 is empirically motivated but not proven. Different h values change d_L(z), and at h = 0.60 an EMRI at z = 0.5 is closer in d_L than at h = 0.90.
2. Future parameter changes (different SNR threshold, improved waveforms, different mass distribution) could make P_det(z > 0.5) nonzero.
3. The histogram estimator in `SimulationDetectionProbability._build_grid_2d()` sets `z_max = max(z_vals) * 1.1`. Since no events exist above z = 0.5, the grid covers only [0, 0.55]. Evaluation at z > 0.55 returns `fill_value=0.0`, which is correct only if the cutoff is valid.

**Consequences:** If any events at z > 0.5 would have been detected, P_det is underestimated (set to zero instead of a small positive value). This biases the selection-effect correction. The bias is h-dependent: lower h values push more events into the truncated region because d_L(z, h) decreases with h.

**Prevention:**
- Document the z_cut as an explicit importance sampling assumption. State the physical justification (maximum detected z at each h value) and the condition under which it breaks.
- Validate by running a small sample (e.g., 1000 events) without the z_cut for one h value. If zero detections above z = 0.5, the approximation is confirmed for that configuration.
- When extending the campaign, test whether the maximum detected z changes with h. If max detected z increases at low h, z_cut may need to be h-dependent.
- Add a runtime check: if any evaluated z value falls above the injection grid's z_max, log a warning.

**Detection:**
- Check that `z_max` in the P_det grid exceeds the maximum z of any galaxy in the catalog weighted by the posterior. If the galaxy catalog's `GALAXY_CATALOG_REDSHIFT_UPPER_LIMIT = 0.55` (in constants.py), the cutoff is consistent. But if any inference path evaluates P_det at z > 0.55, the grid returns 0 by default.
- Compare P_det at z = 0.45 across h values. If it is nonzero for low h but zero for high h, the cutoff matters.

**References:**
- Code: main.py lines 456-476

## Moderate Pitfalls

### Pitfall 4: Histogram P_det Estimator with Low Bin Counts

**What goes wrong:** The `_build_grid_2d` method uses `np.histogram2d` with 30 z-bins x 20 M-bins = 600 cells. With ~10,000 events per h value (20 tasks x 500 events), the average bin occupancy is ~17 events/bin. But the distribution is not uniform: high-z bins and extreme-M bins will have very few events. Bins with N < 5 produce P_det estimates with relative uncertainty > 45% (binomial standard error sqrt(p(1-p)/N)), and bins with N = 0 produce P_det = 0 by construction.

**Why it happens:** The histogram estimator is the simplest unbiased estimator for P_det, but it has high variance in sparse bins. The 2D binning compounds this because the event distribution concentrates at low z and intermediate M (reflecting the population prior), leaving the tails undersampled.

**Prevention:**
- After building the grid, count the number of bins with N_total < 10. Report this as a data quality metric.
- Consider adaptive binning: use finer bins where data is dense and coarser bins where data is sparse.
- Consider smoothing: apply a Gaussian kernel or use a kernel density estimator instead of raw histogram ratios. The KDE approach was used in the old `DetectionProbability` class and has better variance properties.
- For bins with 0 total events: use `fill_value=0.0` (conservative, current behavior). This is correct for high-z bins outside the detection horizon but incorrect for bins that are empty due to undersampling of the injection set.
- Use Wilson score interval or Clopper-Pearson interval instead of the point estimate N_det/N_total to quantify uncertainty per bin.

**Detection:**
- Noisy or non-monotonic P_det(z) curve at fixed M.
- P_det changing significantly when the number of bins is varied (sensitivity to binning is a sign of low statistics).
- Compare 1D marginalized P_det(z) (30 bins, ~333 events/bin) with 2D P_det(z,M): if 1D is smooth but 2D is noisy, the 2D grid has insufficient statistics.

### Pitfall 5: Linear Interpolation Between h Grid Points

**What goes wrong:** `SimulationDetectionProbability._interpolate_at_h` linearly interpolates P_det between the bracketing h values on the grid. If the h grid is coarse (e.g., [0.60, 0.65, 0.70, 0.73, 0.80, 0.85, 0.90]), the spacing is 0.05, and linear interpolation may miss curvature in P_det(h). P_det depends on h through d_L(z, h), which enters the SNR nonlinearly.

**Why it happens:** P_det(z | h) depends on h because changing h changes the luminosity distance d_L = d_L(z, h), and SNR scales as 1/d_L. The relationship SNR proportional to 1/d_L(z,h) is nonlinear in h, so P_det(h) at fixed z may have curvature that linear interpolation misses.

**Prevention:**
- Test interpolation accuracy: for a few z values, compare the linearly interpolated P_det(h=0.72) against the grid values at h=0.70 and h=0.73. If the true P_det at h=0.72 were known (e.g., from a small injection run), compare.
- If the H0 posterior is sensitive to P_det values between grid points, add intermediate h values to the grid.
- Consider using cubic spline interpolation in h instead of linear (but ensure P_det remains in [0, 1]).

**Detection:**
- H0 posterior with artificial kinks or steps at the grid h values.
- Discontinuities in the log-likelihood as a function of h.

### Pitfall 6: d_L-to-z Inversion Inconsistency

**What goes wrong:** The `detection_probability_with_bh_mass_interpolated` method converts d_L to z using `dist_to_redshift(d_L, h=h)`, then converts M_z (observer-frame mass) to source-frame mass via `M_z / (1+z)`. The injection campaign stores z directly (from the population sampler) and M (source-frame mass). If `dist_to_redshift` uses a different cosmology or numerical inversion than the one used to generate d_L in the injection campaign, the z values will not match, and the lookup will hit the wrong bin.

**Why it happens:** `dist_to_redshift` is a numerical root-finding inversion of `dist(z, h)`. The injection campaign computes `dist(sample.redshift, h=h_value)` and stores both z and d_L. The evaluation pipeline receives d_L from the Cramer-Rao bounds and inverts it. Any numerical precision difference or cosmological parameter difference (Omega_m, w0, wa) between the two paths causes a z mismatch.

**Prevention:**
- Verify that the same cosmological parameters (Omega_m, H0 convention, dark energy EOS) are used in both the injection campaign's `dist()` call and the evaluation pipeline's `dist_to_redshift()`.
- Known issue: `constants.py` uses WMAP-era cosmology (Omega_m = 0.25, H = 0.73). This is consistent within the project, but if any code path uses astropy's default Planck cosmology, there will be a mismatch.
- Add a round-trip test: z_original -> dist(z, h) -> dist_to_redshift(d_L, h) should recover z_original to within 1e-4.

**Detection:**
- P_det lookup returning 0 for events that should clearly be detectable.
- Systematic offset between z stored in injection CSV and z recovered from d_L.

## Minor Pitfalls

### Pitfall 7: Geometric Mean for Mass Bin Centers

**What goes wrong:** Line 173 of `simulation_detection_probability.py` uses geometric mean for M bin centers: `M_centers = np.sqrt(M_edges[:-1] * M_edges[1:])`. The bin edges are log-spaced (`np.geomspace`), so the geometric mean is the correct center in log-space. However, `RegularGridInterpolator` with `method="linear"` interpolates linearly in the provided coordinates. This means interpolation is linear in M (not log-M), which may be inaccurate for a function that varies slowly in log-M but rapidly in M.

**Prevention:**
- Consider passing log(M_centers) to the interpolator and converting query points to log-M before lookup. This makes the interpolation linear in log-M, which better matches the physics (SNR depends on chirp mass, which varies on a log scale).
- Test: evaluate P_det at the midpoint of two adjacent M bins and compare to the average of the bin values. If the discrepancy exceeds 5%, log-space interpolation is needed.

### Pitfall 8: Extrinsic Parameter Marginalization Incompleteness

**What goes wrong:** The injection campaign randomizes sky angles (phiS, qS) via `parameter_space.randomize_parameters(rng=rng)`, and the P_det grid marginalizes over them (bins in z and M only). The marginalization is correct only if the sky angle distribution in the injections matches the assumed prior (isotropic). If `randomize_parameters` draws from a non-isotropic distribution (e.g., uniform in theta rather than uniform in cos(theta)), the marginalization is biased.

**Prevention:**
- Verify that `randomize_parameters` draws sky angles from the correct prior: phi ~ Uniform(0, 2pi), cos(theta) ~ Uniform(-1, 1) (i.e., theta is NOT uniform, but cos(theta) is).
- The injection campaign stores phiS and qS in the CSV. Plot their distributions to confirm isotropy.

### Pitfall 9: Seed Isolation Insufficient for Large Campaigns

**What goes wrong:** The injection campaign offsets seeds by `h_index * 10000` (line 118 of submit_injection.sh). With 20 tasks per h value, seeds span [BASE_SEED + offset, BASE_SEED + offset + 19]. If tasks_per_h > 10000 (unlikely but possible in extended campaigns), seed ranges for different h values overlap, producing correlated samples.

**Prevention:**
- For the current campaign (20 tasks/h), this is safe. For extensions, verify that tasks_per_h * max(h_index) < 10000.
- The RNG within each task also depends on the initial state of emcee's MCMC sampler, which adds another layer of randomness. The seed controls `np.random.Generator`, not emcee's internal state.

## Numerical Pitfalls

| Issue | Symptom | Cause | Fix |
|-------|---------|-------|-----|
| P_det = 0 in populated bin | H0 posterior has sharp feature | Bin has zero detected events but nonzero total (legitimate P_det ~ 0) vs. bin has zero total events (coverage gap) | Distinguish the two cases: check `total_counts` array |
| P_det > 1 after interpolation | Unphysical value | Linear interpolation between grid points can overshoot | Already handled: `np.clip(result, 0.0, 1.0)` in line 268 |
| NaN in P_det | Crash or silent corruption | 0/0 in histogram ratio | Already handled: `np.divide(..., where=total_counts > 0)` with `out=np.zeros_like(...)` |
| Binomial noise in P_det | Noisy H0 posterior | Low bin counts N < 10 | Increase injection count or reduce bin count |
| fill_value=0.0 at grid boundary | Underestimated P_det at low z | z_edges start at 0, but first bin center is > 0; queries at z ~ 0 extrapolate | Verify that z = 0 is within the grid bounds or that no events have z ~ 0 |
| d_L-to-z inversion precision | Wrong bin lookup | Root-finding tolerance in `dist_to_redshift` | Verify round-trip consistency to 1e-4 in z |
| MC variance in denominator integral | Noisy single-event likelihoods | N_SAMPLES = 10,000 in bayesian_statistics.py line 689 may be too few if P_det varies rapidly | Increase N_SAMPLES or use stratified sampling; check variance across repeated evaluations |

## Convention and Notation Pitfalls

| Pitfall | Sources That Differ | Resolution |
|---------|-------------------|------------|
| M (source-frame) vs M_z (observer-frame) | Injection CSV stores source-frame M; evaluation uses M_z = M*(1+z) | `simulation_detection_probability.py` line 351 correctly converts: `M_true = M_z / (1+z)`. Verify this conversion uses the same z as the injection. |
| h convention: dimensionless vs km/s/Mpc | Code uses dimensionless h (H0 = 100*h km/s/Mpc) throughout | Consistent within project. But h=0.73 means H0=73 km/s/Mpc. |
| d_L units: Gpc vs Mpc | `dist()` returns Gpc; `detection_probability_with_bh_mass_interpolated` expects Gpc | Consistent. Verify any new code paths also use Gpc. |
| Omega_m: WMAP vs Planck | constants.py uses Omega_m = 0.25 (WMAP); Planck 2018 gives 0.3153 | Known bug (LOW priority). Consistent within project but physically outdated. |

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Analyzing existing injection results | Waveform failure fraction unknown | Parse logs or add failure tracking to future runs |
| Extending injection campaign | z_cut=0.5 may need h-dependent adjustment | Run validation batch without cutoff for extreme h values |
| Building P_det grid | Sparse bins at high z and extreme M | Monitor bin occupancy; fall back to 1D P_det(z) where 2D is too noisy |
| Using P_det in H0 posterior | MC noise in denominator integral | Increase N_SAMPLES from 10,000 if posterior is noisy; test with 50,000 |
| Comparing old KDE P_det with new simulation P_det | Different smoothing assumptions | Expect quantitative differences; qualitative shape should agree |
| Adding new h grid points | Interpolation error between existing points | Verify that adding h = 0.75 does not change results at h = 0.73 significantly |

## Sources

- Mandel, Farr & Gair (2019), arXiv:1809.02063 -- Selection effects in hierarchical Bayesian inference for GW populations
- Essick & Farr (2022), arXiv:2204.00461 -- Monte Carlo uncertainty in selection functions; correlated uncertainties
- Farr (2019), arXiv:1904.10879 -- Accuracy requirements for empirically measured selection functions
- Vitale et al. (2022), "Quick recipes for gravitational-wave selection effects", arXiv:2404.16930 -- Approximations to GW selection effects, noise realization effects
- Speri et al. (2025), arXiv:2509.08875 -- Systematic errors in fast relativistic EMRI waveforms
- Essick et al. (2023), MNRAS 526, 3495 -- Growing pains: likelihood uncertainty impact on hierarchical inference
- Katz et al. (2021), arXiv:2104.04582 -- FastEMRIWaveforms: parameter space coverage and limitations
