# Research Gaps and Open Issues

**Analysis Date:** 2026-03-30

---

## Unjustified Approximations

**wCDM parameters silently ignored in distance computation:**
- Where used: `master_thesis_code/physical_relations.py` (lines 40-77), function `dist()`
- The function signature accepts `w_0` and `w_a` parameters, but the actual computation calls `lambda_cdm_analytic_distance()` (line 72), which is a closed-form hypergeometric expression valid only for flat LCDM (w_0 = -1, w_a = 0). The `hubble_function()` at line 216 correctly implements the CPL parameterization, but it is never used in `dist()`.
- Justification status: None given. The function silently returns LCDM distances regardless of w_0, w_a input.
- What could go wrong: Any dark energy analysis (e.g., `DarkEnergyScenario` in `cosmological_model.py:333`) that varies w_0 or w_a will get LCDM distances instead of the correct wCDM distances. All derived quantities (luminosity distances, comoving volumes, H0 posteriors) will be wrong for non-LCDM cosmologies.
- How to justify: Either (a) remove w_0/w_a from `dist()` signature and document LCDM-only scope, or (b) fall back to numerical integration of `1/E(z)` using `hubble_function()` when w_0 != -1 or w_a != 0.
- Priority: **Medium** (LCDM is the current fiducial; becomes **High** if dark energy scenarios are used)

**Galaxy redshift uncertainty scaling `0.013 * (1+z)^3`:**
- Where used: `master_thesis_code/datamodels/galaxy.py` (line 67), `Galaxy.redshift_uncertainty` property
- Justification status: None given. No citation. The cap at `min(..., 0.015)` means this formula saturates at z ~ 0.048, so nearly all galaxies in the catalog (z up to 0.55) use the constant value 0.015.
- What could go wrong: Standard photometric redshift uncertainty scales as `sigma_z ~ 0.05*(1+z)` (much larger) and spectroscopic as `sigma_z ~ 0.001*(1+z)` (much smaller). Using the wrong uncertainty model changes the galaxy identification likelihood in the Bayesian pipeline.
- How to justify: Cite the source of this formula or replace with a standard form. The cap at 0.015 makes the (1+z)^3 scaling effectively irrelevant, suggesting this may be a miscoded spectroscopic error model.
- Priority: **Low** (the cap dominates; actual impact is small unless z < 0.05 galaxies are important)

**WMAP-era fiducial cosmological parameters:**
- Where used: `master_thesis_code/constants.py` (lines 29-32)
- Current values: Omega_m = 0.25, Omega_de = 0.75, H = 0.73 (dimensionless h)
- Planck 2018 best-fit: Omega_m = 0.3153, Omega_de = 0.6847, H = 0.6736
- Justification status: These may be intentional choices to match the reference EMRI population model (Babak et al. 2017). However, no explicit justification is given.
- What could go wrong: All distances, volumes, and event rates are computed with these values. The ~10% shift in Omega_m changes the comoving volume by O(10%) and the luminosity distances by O(5%). If the goal is to infer H0, using a biased fiducial may affect the posterior.
- How to justify: Document whether these are inherited from the population model or intended to be updated. Add a reference comment.
- Priority: **Low** (standard practice to use a fiducial; but should be documented)

**Hardcoded 10% fractional luminosity distance error in Pipeline A:**
- Where used: `master_thesis_code/bayesian_inference/bayesian_inference.py` (lines 154, 176, 208, 252), via `FRACTIONAL_LUMINOSITY_ERROR = 0.1` from `constants.py:52`
- Justification status: Pipeline A is documented as a dev cross-check, not production. However, `EMRIDetection.from_host_galaxy()` in `master_thesis_code/datamodels/emri_detection.py` (line 56) also uses this constant when `use_measurement_noise=True`.
- What could go wrong: The actual Fisher-matrix-derived sigma(d_L) varies widely across the parameter space. A flat 10% underestimates errors for distant/faint sources and overestimates for nearby/loud ones.
- How to justify: Acceptable for Pipeline A (dev cross-check). Flag if Pipeline A results are ever compared to Pipeline B without accounting for this difference.
- Priority: **Medium** (Pipeline A is not production, but `EMRIDetection` is used more broadly)

**Quick SNR extrapolation factor of 5:**
- Where used: `master_thesis_code/main.py` (lines 265, 267)
- The quick SNR (computed with a cheaper waveform generator) is multiplied by 5 before saving as the "not detected" SNR: `save_not_detected(quick_snr * 5, ...)`.
- Justification status: None given. No reference. The factor of 5 is a heuristic relating the quick-check SNR to the full SNR.
- What could go wrong: If the quick SNR underestimates by more or less than 5x, the recorded undetected-event SNR distribution is biased. This affects the detection probability estimation (KDE in `detection_probability.py`) if undetected events are used.
- How to justify: Calibrate the quick-to-full SNR ratio empirically. Log both values for a sample of events and measure the actual ratio distribution.
- Priority: **Low** (affects undetected events only; detection probability uses detected events)

---

## Unchecked Limits

**z -> 0 limit of luminosity distance:**
- Limit: dist(z -> 0) -> c*z/H0 (Hubble law)
- Expected behavior: Linear in z for small z
- Current status: A docstring example `dist(0.0) == 0.0` exists in `physical_relations.py` (line 67), but no test verifies the linear slope dD_L/dz|_{z=0} = c/H0.
- Files: `master_thesis_code/physical_relations.py` (line 42)
- Why it matters: Confirms the normalization of the distance-redshift relation.

**High-z behavior of comoving volume element:**
- Limit: dV_c/dz should peak and turn over at z ~ 2 (for standard cosmology)
- Current status: Not checked. The galaxy catalog is limited to z < 0.55 (`GALAXY_CATALOG_REDSHIFT_UPPER_LIMIT`), so this limit is outside the operational range but would validate the formula.
- Files: `master_thesis_code/datamodels/galaxy.py` (line 95, `comoving_volume_element_spline`)

**Fisher matrix condition number and Cramer-Rao bound validity:**
- Limit: For very low SNR (near threshold), the Fisher matrix may be ill-conditioned and the Cramer-Rao bound may not be achievable.
- Current status: Condition number is logged (`parameter_estimation.py` line 392) and singular matrices raise `LinAlgError` (caught in `main.py` line 340). However, there is no check for "nearly singular" matrices where the inverse exists but is numerically unreliable.
- Files: `master_thesis_code/parameter_estimation/parameter_estimation.py` (lines 392-399)
- Why it matters: A condition number of 10^12 would give ~4 digits of precision loss. The 14x14 Fisher matrix for EMRIs can easily be ill-conditioned. Negative diagonal entries in the inverse are checked (line 398), but a condition number threshold warning is missing.

---

## Missing Cross-Checks

**Pipeline A vs Pipeline B consistency:**
- What to verify: Both Bayesian pipelines should give consistent H0 posteriors when given the same input data and matching assumptions (same detection probability model, same distance errors).
- Method: Run both pipelines on a small synthetic dataset with known H0, compare posterior shapes.
- Expected outcome: Peaks should agree within statistical uncertainty.
- Files: `master_thesis_code/bayesian_inference/bayesian_inference.py` (Pipeline A), `master_thesis_code/bayesian_inference/bayesian_statistics.py` (Pipeline B)
- Priority: **Medium** (Pipeline A is dev cross-check; but cross-checking validates both)

**Fisher matrix derivative accuracy:**
- What to verify: The five-point stencil derivative (when enabled via `use_five_point_stencil=True`) should give more accurate Fisher matrices than the forward difference.
- Method: Compare Fisher matrix elements and Cramer-Rao bounds between both methods for a set of reference EMRI parameters.
- Expected outcome: Five-point stencil results should converge with smaller epsilon; forward difference should show O(epsilon) bias.
- Files: `master_thesis_code/parameter_estimation/parameter_estimation.py` (lines 142-258, 350-358)
- Priority: **High** (Fisher matrix accuracy directly determines all Cramer-Rao bounds)

**Confusion noise impact on SNR:**
- What to verify: Adding galactic confusion noise (now implemented in `LISA_configuration.py` lines 86-136) should reduce SNR for low-frequency-dominated sources.
- Method: Compare SNR with and without confusion noise (`include_confusion_noise=True/False`) for a reference EMRI.
- Expected outcome: SNR should decrease, especially for sources with significant power below 3 mHz. The detection rate should decrease.
- Files: `master_thesis_code/LISA_configuration.py` (lines 68, 86-136)
- Priority: **High** (confusion noise is newly implemented; needs validation)

---

## Numerical Concerns

**Multivariate Gaussian `allow_singular=True` in likelihood construction:**
- Problem: The covariance matrices for the detection likelihood Gaussians in `bayesian_statistics.py` (lines 222, 227) require `allow_singular=True`. This means at least some covariance matrices are numerically singular or near-singular, which can produce degenerate likelihood evaluations.
- Files: `master_thesis_code/bayesian_inference/bayesian_statistics.py` (lines 219-228)
- Symptoms: Without `allow_singular=True`, scipy raises `LinAlgError`. The TODO comment says "this should not be needed in the end."
- Resolution: Investigate which detections produce singular covariance matrices. Possible causes: (a) some Cramer-Rao bound diagonal entries are zero or near-zero, (b) perfect correlations between parameters, (c) the 3x3/4x4 covariance construction at lines 173-217 has a structural rank deficiency. Consider adding a regularization floor (e.g., `max(sigma^2, epsilon)`) to diagonal entries.

**Global state in multiprocessing workers (`bayesian_statistics.py`):**
- Problem: The module uses 6 global variables (`redshift_upper_integration_limit`, `redshift_lower_integration_limit`, `bh_mass_upper_integration_limit`, `bh_mass_lower_integration_limit`, `detection_probability`, `detection_likelihood_gaussians_by_detection_index`) to pass data to multiprocessing worker functions. These are set via `global` statements in 4 separate functions (lines 499-504, 519-524, 672-677, 922-927).
- Files: `master_thesis_code/bayesian_inference/bayesian_statistics.py`
- Symptoms: Race conditions are unlikely (workers are read-only), but the pattern makes the code fragile: any refactoring must keep global state in sync across all 4 initializer functions. Missing a global update silently uses stale values.
- Resolution: Refactor to use `multiprocessing.Pool(initializer=...)` with a single shared state object, or use `functools.partial` to bind parameters directly.

**`signal.alarm()` timeout mechanism is Unix-only and fragile:**
- Problem: The simulation loop uses `signal.alarm(90)` (in `main.py` lines 255, 332) to timeout waveform computations. This mechanism: (a) only works on Unix, (b) is not thread-safe, (c) can interrupt I/O operations leading to partial writes, (d) the alarm is not reliably cancelled if an exception is raised between `signal.alarm(90)` and `signal.alarm(0)`.
- Files: `master_thesis_code/main.py` (lines 198-200, 255, 261, 270, 332, 334)
- Symptoms: On rare occasions, a stale alarm could fire during the next iteration's setup phase. The broad `except` blocks mitigate this but make debugging harder.
- Resolution: Consider using `multiprocessing` with a timeout, or `concurrent.futures.ProcessPoolExecutor` with `timeout` parameter, for a more robust mechanism.

**Sky localization weight placement concern:**
- Problem: Two TODO comments in `bayesian_statistics.py` (lines 547, 684) warn: "KEEP IN MIND SKYLOCALIZATION WEIGHT IS IN THE GW LIKELIHOOD ATM. possible source of error." This suggests the sky localization weight may be double-counted or placed in the wrong term of the likelihood decomposition.
- Files: `master_thesis_code/bayesian_inference/bayesian_statistics.py` (lines 547, 684)
- Symptoms: If the sky localization weight is in both the GW likelihood and the galaxy catalog term, it would be double-counted, biasing the posterior.
- Resolution: Audit the full likelihood expression: p(data|H0) = sum_galaxies p(data|galaxy, H0) * p(galaxy). Verify that sky localization appears in exactly one factor. Compare against the reference (likely Schutz 1986 or Chen et al. 2018).

**Unverified Jacobian factor `1 / (d_L * (1+z))` in likelihood integrand:**
- Problem: `bayesian_statistics.py` line 613 contains `/ (d_L * (1 + z))` with the comment "TODO: check if this is correct." This Jacobian factor appears in `numerator_integrant_with_bh_mass` and affects the normalization of the mass-dependent likelihood.
- Files: `master_thesis_code/bayesian_inference/bayesian_statistics.py` (line 613)
- Symptoms: An incorrect Jacobian would systematically bias the H0 posterior, especially at high redshift where the (1+z) factor is significant.
- Resolution: Derive the Jacobian from first principles. The factor likely comes from converting between source-frame mass M and redshifted mass M_z = M*(1+z), combined with the volume element dV/dz ~ d_L^2 / (1+z). Write out the full change of variables and verify.

---

## Incomplete Derivations

**`Model1CrossCheck.simplified_event_mass_dependency` is a stub:**
- What exists: Method signature at `cosmological_model.py` line 295
- What's missing: Raises `NotImplementedError`. The companion method `setup_simplified_event_sampler` (line 298) is an empty `pass`.
- Files: `master_thesis_code/cosmological_model.py` (lines 295-299)
- Impact: These methods are never called in production. They appear to be placeholders for a simplified event sampling path.
- Difficulty estimate: Unknown (depends on what "simplified" means in context)

**PHYS-9: Seven pre-existing physics TODOs from the original codebase:**
- What exists: Listed in `TODO.md` under PHYS-9:
  1. Coordinate transformation to orbital motion around sun
  2. Check `_s` parameters: barycenter vs. binary orientation
  3. Check spin limits for parameter `a`
  4. Inclination for Schwarzschild waveforms
  5. Compute derivative w.r.t. sky localization in SSB again
  6. Use second detector from LISA (currently only A/E channels)
  7. Frequency integration: claims negative frequency == complex conjugate, but `fs` contains negative frequencies
- Files: Various (not precisely located in the code; these are conceptual TODOs)
- Impact: Items 5-7 could affect Fisher matrix accuracy. Items 1-4 affect parameter interpretation.
- Difficulty estimate: Mixed (items 1-4 are documentation/verification; items 5-7 require computation)

---

## Physical Consistency Issues

**Two different fiducial Hubble constants:**
- Concern: `constants.py` defines `H = 0.73` (line 25, used as the fiducial for `dist()` and simulation) and `TRUE_HUBBLE_CONSTANT = 0.7` (line 26, used as the "true" value for Bayesian inference and `EMRIDetection` generation). These differ by ~4%. The simulation generates events assuming H = 0.73, but the Bayesian inference tries to recover H = 0.7.
- Files: `master_thesis_code/constants.py` (lines 25-26), `master_thesis_code/datamodels/emri_detection.py` (line 45), `master_thesis_code/bayesian_inference/bayesian_statistics.py` (line 111)
- Impact: This is likely intentional (the inference should find the "true" value despite the simulation using a slightly different fiducial). However, it introduces a subtle inconsistency: Pipeline A's `EMRIDetection.from_host_galaxy()` uses `TRUE_HUBBLE_CONSTANT = 0.7` for measured distances (line 45), while the simulation's `dist()` defaults to `H = 0.73`. This means "true" distances and "measured" distances use different cosmologies.
- Resolution path: Document whether this is intentional (blind analysis setup) or a bug. If intentional, add a clear comment. If not, unify to a single fiducial.

**`remove_garbage=True` in waveform generator without understanding:**
- Concern: `waveform_generator.py` line 64 sets `remove_garbage=True` with the comment "TODO: understand why to use this." The `remove_garbage` option in `fastlisaresponse` trims the start and end of the time series to remove artifacts from the response function initialization.
- Files: `master_thesis_code/waveform_generator.py` (line 64)
- Impact: If garbage removal is too aggressive, it could clip valid signal. If insufficient, artifacts contaminate the inner product and SNR computation.
- Resolution path: Read the `fastlisaresponse` documentation; verify that the trimmed region does not contain significant signal power for typical EMRI parameters (T_obs ~ 1 year).

---

## Missing Generalizations

**Dark energy equation of state:**
- Current scope: All distance computations use flat LCDM despite `DarkEnergyScenario` class existing (`cosmological_model.py` line 333) with w_0, w_a parameters.
- Natural extension: Implement numerical integration fallback in `dist()` for w_0 != -1 or w_a != 0.
- Difficulty: Straightforward (the `hubble_function()` already implements CPL; just needs `scipy.integrate.quad` in `dist()`)
- Blocks: Any dark energy analysis or wCDM posterior inference.

**Second LISA TDI channel (T channel):**
- Current scope: Only A and E channels are used (`ESA_TDI_CHANNELS = "AE"` in `constants.py` line 34).
- Natural extension: Include T channel for improved parameter estimation (T is primarily noise-only for GW signals, useful as a null channel for noise characterization).
- Difficulty: Moderate (PSD for T channel differs; need to verify `fastlisaresponse` T-channel output)
- Blocks: Noise characterization, systematic error studies.

---

## Documentation Gaps

**`Model1CrossCheck` polynomial coefficients have no citation:**
- What's undocumented: `cosmological_model.py` lines 91-147 contain 5 sets of 9th-degree polynomial coefficients for the EMRI event rate dN/dz. No reference is given. Presumably fits to Babak et al. (2017) PRD 95, 103012.
- Files: `master_thesis_code/cosmological_model.py` (lines 91-147)
- Impact: Impossible to verify correctness or reproduce without knowing the source.

**Covariance matrix construction in `BayesianStatistics.evaluate()`:**
- What's undocumented: The 3x3 and 4x4 covariance matrices at `bayesian_statistics.py` lines 173-217 are constructed from Cramer-Rao bound entries, but the mapping from Fisher matrix inverse elements to these specific matrix positions is not documented. The off-diagonal terms (e.g., `d_L_phi_covariance / detection.d_L`) involve normalization by measured values.
- Files: `master_thesis_code/bayesian_inference/bayesian_statistics.py` (lines 173-217)
- Impact: Hard to verify whether the covariance is constructed correctly. The `allow_singular=True` TODO suggests there may be an issue.

---

## Stale or Dead Content

**`LamCDMScenario` and `DarkEnergyScenario` classes are partially implemented:**
- What: `cosmological_model.py` lines 302-355. `LamCDMScenario` and `DarkEnergyScenario` are plain classes (not dataclasses) with hardcoded parameter bounds but no methods. They are used only in `BayesianStatistics.__init__()` to store fiducial cosmological parameter bounds.
- Files: `master_thesis_code/cosmological_model.py` (lines 302-355)
- Action: Either flesh out with methods (distance computation, sampling) or simplify to plain dataclasses/NamedTuples.
- Risk: Low. They work as data containers. But `DarkEnergyScenario` implies wCDM capability that does not exist (see wCDM issue above).

**`simplified_event_mass_dependency` and `setup_simplified_event_sampler` stubs:**
- What: Two methods in `Model1CrossCheck` that raise `NotImplementedError` / `pass`.
- Files: `master_thesis_code/cosmological_model.py` (lines 295-299)
- Action: Delete if not planned for implementation; document as future work if intended.
- Risk: None (never called).

---

## Placeholder and Stub Content

**`NullCallback` in `callbacks.py`:**
- What: All 5 protocol methods are `pass` stubs (lines 47-61). This is intentional (null object pattern), not a missing implementation.
- Files: `master_thesis_code/callbacks.py` (lines 47-61)
- Needed for: Default callback when no plotting or monitoring is requested. Working as designed.

**Empty test files:**
- What: No tests exist for `galaxy_catalogue/handler.py` (669 lines), `plotting/` factory functions (6 modules with 0% coverage), or `cosmological_model.py` core classes.
- Files: `master_thesis_code_test/` (see test inventory above; notable absences: no `test_handler.py`, no `test_cosmological_model_classes.py`, limited plotting tests)
- Needed for: Coverage target of 50% (currently 37%). Handler and cosmological model are high-risk untested modules.

---

## Missing Literature Connections

**Vallisneri (2008) Fisher matrix accuracy:**
- What: arXiv:gr-qc/0703086 establishes that O(epsilon) forward differences in Fisher matrix computation introduce systematic bias. The five-point stencil is now available (`use_five_point_stencil=True`, default True), but no validation has been done comparing the two methods on this codebase.
- Why relevant: Fisher matrix accuracy is the foundation of all Cramer-Rao bounds and downstream H0 posteriors.
- Priority: **High**

**Schutz (1986) / Chen et al. (2018) likelihood decomposition:**
- What: The canonical formulation of the "dark siren" H0 inference likelihood. The sky localization weight placement (TODO comments at lines 547, 684 of `bayesian_statistics.py`) should be verified against these references.
- Why relevant: Correct likelihood decomposition is essential for unbiased H0 posteriors.
- Priority: **High**

**Cornish & Robson (2017) / Robson et al. (2019) confusion noise:**
- What: arXiv:1703.09858 Eq. (3) / arXiv:1803.01944 Eq. (14). The confusion noise PSD is now implemented (`LISA_configuration.py` lines 86-136). The constants in `constants.py` (lines 74-84) have a citation note acknowledging the actual source differs from the originally cited arXiv:2303.15929.
- Why relevant: Confusion noise implementation should be validated against these references.
- Priority: **Medium** (implementation exists; needs cross-check)

---

## Priority Ranking

**Critical (blocks correctness):**
1. Sky localization weight double-counting risk (`bayesian_statistics.py` lines 547, 684) -- could systematically bias the H0 posterior
2. Unverified Jacobian `1/(d_L*(1+z))` in mass-dependent likelihood integrand (`bayesian_statistics.py` line 613) -- could bias mass-dependent results
3. `allow_singular=True` in detection likelihood Gaussians (`bayesian_statistics.py` lines 222, 227) -- indicates possible numerical degeneracy in covariance matrices

**High (blocks completeness):**
1. Fisher matrix derivative validation: five-point stencil vs forward difference comparison needed
2. Confusion noise impact validation on SNR and detection rates
3. No tests for `galaxy_catalogue/handler.py` (669 lines of untested code handling the real GLADE catalog)
4. No tests for `cosmological_model.py` core classes (event sampling, rate model)

**Medium (improves quality):**
1. wCDM distance computation is silently LCDM-only (blocks dark energy analysis)
2. Two different fiducial Hubble constants (H=0.73 vs TRUE_HUBBLE_CONSTANT=0.7) need documentation
3. Global state pattern in multiprocessing workers (fragile, hard to maintain)
4. `Model1CrossCheck` polynomial coefficients lack citation

**Low (nice to have):**
1. WMAP-era cosmological parameters (standard practice but should be documented)
2. Galaxy redshift uncertainty formula lacks citation (effectively constant at 0.015)
3. `remove_garbage=True` not understood in waveform generator
4. Stubs (`simplified_event_mass_dependency`) should be removed or documented
5. Quick SNR * 5 extrapolation factor is uncalibrated

---

*Gap analysis: 2026-03-30*
