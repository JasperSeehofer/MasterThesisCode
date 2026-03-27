# Codebase Concerns

**Analysis Date:** 2026-03-25

## Tech Debt

**Global variables in multiprocessing workers [HIGH]:**
- Issue: `bayesian_inference/bayesian_statistics.py` uses 24 `global` statements across 4 functions (`child_process_init` at lines 921-946, `single_host_likelihood` at lines 509-534, 682-687, 932-937) to pass state to multiprocessing child processes.
- Files: `master_thesis_code/bayesian_inference/bayesian_statistics.py`
- Impact: Untestable without multiprocessing; global state makes reasoning about correctness difficult. The same 6 globals are declared in 4 separate locations.
- Fix approach: Replace with a frozen dataclass or `functools.partial` bound to the worker function. See TODO.md ARCH-2.

**Print statements instead of logging [MEDIUM]:**
- Issue: 20+ bare `print()` calls scattered through production code, including debug prints like `print(possible_host.z, possible_host.z_error)` (line 517).
- Files: `master_thesis_code/bayesian_inference/bayesian_statistics.py` (13 calls), `master_thesis_code/datamodels/galaxy.py` (3 calls), `master_thesis_code/parameter_estimation/parameter_estimation.py` (2 calls), `master_thesis_code/bayesian_inference/bayesian_inference_mwe.py` (2 calls)
- Impact: No log-level control; debug output pollutes stdout in production runs.
- Fix approach: Replace all `print()` with `_LOGGER.info()` or `_LOGGER.debug()`. See TODO.md ARCH-3.

**Dead testing code in bayesian_statistics.py [MEDIUM]:**
- Issue: `single_host_likelihood_integration_testing` (lines ~700-918) is a 200+ line duplicate of `single_host_likelihood` with inline print statements and Monte Carlo cross-checks. Not called from production code.
- Files: `master_thesis_code/bayesian_inference/bayesian_statistics.py`
- Impact: Maintenance burden; diverges from production `single_host_likelihood` over time.
- Fix approach: Either promote useful assertions to proper tests or delete entirely. See TODO.md ARCH-4.

**Massive data-as-code files [LOW]:**
- Issue: `M1_model_extracted_data/emri_distribution.py` (9,510 lines) and `detection_fraction.py` (7,066 lines) contain hardcoded numpy arrays as Python source.
- Files: `master_thesis_code/M1_model_extracted_data/emri_distribution.py`, `master_thesis_code/M1_model_extracted_data/detection_fraction.py`
- Impact: Inflates repo size; slow to parse; impossible to diff meaningfully.
- Fix approach: Store as `.npy` or `.npz` data files and load at runtime.

**`allow_singular=True` on covariance matrices [MEDIUM]:**
- Issue: Two `multivariate_normal()` calls use `allow_singular=True` with a TODO comment "this should not be needed in the end."
- Files: `master_thesis_code/bayesian_inference/bayesian_statistics.py` lines 221, 226
- Impact: Masks degenerate covariance matrices from Cramer-Rao bounds, which may indicate upstream Fisher matrix issues. Silently produces zero-probability regions instead of raising.
- Fix approach: Investigate why covariance matrices are singular. Likely related to the forward-difference Fisher stencil (PHYS-3 below). After switching to five-point stencil, test if `allow_singular` is still needed.

**`ParameterSample` default_factory creates unseeded RNG [LOW]:**
- Issue: `phi_S` and `theta_S` defaults each create a fresh `np.random.default_rng()` (no seed), making instances non-reproducible even when the rest of the code is seeded.
- Files: `master_thesis_code/galaxy_catalogue/handler.py` lines 37-42
- Impact: Minor reproducibility gap when default sky positions are used.
- Fix approach: Accept an `rng` parameter or remove random defaults entirely.

## Known Bugs

**`bayesian_statistics.py` line 618 uses wrong Gaussian index [HIGH]:**
- Symptoms: The `numerator_integrant_with_bh_mass` function (line 618) indexes `detection_likelihood_gaussians_by_detection_index[detection_index][0]` but should use `[1]` (the "with BH mass" Gaussian). Index `[0]` is the "without BH mass" variant.
- Files: `master_thesis_code/bayesian_inference/bayesian_statistics.py` line 618
- Trigger: Any evaluation with `evaluate_with_bh_mass=True` in the `single_host_likelihood` function.
- Workaround: None. Results from the "with BH mass" branch use the wrong likelihood.

**`bayesian_statistics.py` line 623 unchecked Jacobian factor [MEDIUM]:**
- Symptoms: `/ (d_L * (1 + z))` Jacobian has a TODO "check if this is correct." This factor appears only in the "with BH mass" numerator, not in the "without BH mass" numerator (line 573 has `/ d_L` only). The asymmetry is suspicious.
- Files: `master_thesis_code/bayesian_inference/bayesian_statistics.py` line 623
- Trigger: All "with BH mass" likelihood evaluations.
- Workaround: None.

## Security Considerations

**No secrets in source code:**
- Risk: Minimal. No API keys or credentials detected. `.env` files are not present in the repository.
- Files: N/A
- Current mitigation: `.gitignore` excludes common secret file patterns.
- Recommendations: No action needed.

## Performance Bottlenecks

**Scalar product PSD loop (GPU hot path) [HIGH]:**
- Problem: `scalar_product_of_functions` is called O(N^2) times per Fisher matrix (105 calls for 14 parameters). Each call computes FFTs and integrates over the PSD. This is the dominant cost of the simulation pipeline.
- Files: `master_thesis_code/parameter_estimation/parameter_estimation.py` lines 260-310
- Cause: Sequential Fisher matrix computation; each derivative evaluation requires a full waveform generation + scalar product.
- Improvement path: Parallelize the 14 derivative evaluations (currently sequential in `compute_fisher_information_matrix`, line 333). With five-point stencil this becomes 56 independent waveform evaluations. See TODO.md PERF-4.

**Unconditional `cp.*` calls in parameter_estimation.py [HIGH]:**
- Problem: `_crop_to_same_length` (line 253), `compute_fisher_information_matrix` (line 335), and `compute_signal_to_noise_ratio` (line 383) use `cp.array()`, `cp.zeros()`, and `cp.sqrt()` directly, bypassing the `xp = _get_xp()` pattern.
- Files: `master_thesis_code/parameter_estimation/parameter_estimation.py` lines 253, 335, 383
- Cause: Pre-dates the `xp` namespace refactor.
- Improvement path: Replace `cp.*` with `xp.*` using the existing `_get_xp` helper. See TODO.md PERF-2.

**Python loops in BayesianInference.__post_init__ [MEDIUM]:**
- Problem: `__post_init__` loops over 1000 redshift values x N galaxies in pure Python to precompute galaxy distributions.
- Files: `master_thesis_code/bayesian_inference/bayesian_inference.py` lines 87-126
- Cause: Uses `NormalDist` in a Python loop instead of vectorized `scipy.stats.norm`.
- Improvement path: Vectorize with scipy. See TODO.md PERF-3.

**`.iterrows()` in cosmological_model.py [LOW]:**
- Problem: Two uses of `.iterrows()` (the slowest pandas iteration method) for filtering detections.
- Files: `master_thesis_code/cosmological_model.py` (filtering loops)
- Cause: Legacy code.
- Improvement path: Replace with vectorized boolean indexing. See TODO.md PERF-5.

## Physics / Mathematics Bugs (Open)

**Fisher matrix uses O(h) forward difference instead of O(h^4) five-point stencil [HIGH]:**
- Problem: `compute_fisher_information_matrix()` calls `finite_difference_derivative()` instead of the existing `five_point_stencil_derivative()`. The forward-difference method has O(epsilon) truncation error, producing noisier Fisher matrices and potentially singular covariance matrices.
- Files: `master_thesis_code/parameter_estimation/parameter_estimation.py` line 333
- Cause: The five-point stencil method was implemented but never wired in.
- Fix approach: Replace call at line 333. Requires 4x more waveform evaluations (56 vs 14). Ref: Vallisneri (2008) arXiv:gr-qc/0703086. See TODO.md PHYS-3.

**Galactic confusion noise absent from LISA PSD [HIGH]:**
- Problem: The PSD in `LISA_configuration.py` includes only instrumental noise (optical metrology + test mass acceleration). Galactic confusion noise constants are defined in `constants.py` lines 74-81 but never used. This noise dominates LISA sensitivity at 0.1-3 mHz.
- Files: `master_thesis_code/LISA_configuration.py`, `master_thesis_code/constants.py` lines 74-81
- Cause: Implementation was planned but not completed.
- Fix approach: Implement `galactic_confusion_noise(frequencies, T_obs)` and add to `power_spectral_density_a_channel()`. All existing Cramer-Rao CSV data must be regenerated. Ref: Babak et al. (2023) arXiv:2303.15929 Eq. (17). See TODO.md PHYS-4.

**wCDM parameters silently ignored [MEDIUM]:**
- Problem: `dist()` and `cached_dist()` accept `w_0` and `w_a` parameters but pass them to `lambda_cdm_analytic_distance()` which is hardcoded for LCDM (w=-1).
- Files: `master_thesis_code/physical_relations.py` lines 72, 80-88
- Cause: API accepts dark energy equation-of-state parameters for future use but the analytical distance formula does not support them.
- Fix approach: Either remove the parameters and document the LCDM assumption, or fall back to numerical integration when w_0 != -1 or w_a != 0. See TODO.md PHYS-6.

**Hardcoded 10% luminosity distance error in Pipeline A [MEDIUM]:**
- Problem: `BayesianInference` uses `FRACTIONAL_LUMINOSITY_ERROR * d_L` (constant 10%) instead of per-source Cramer-Rao bounds from the Fisher matrix.
- Files: `master_thesis_code/bayesian_inference/bayesian_inference.py` lines 154, 176, 186
- Cause: Pipeline A was built as a simplified cross-check before Fisher matrix integration existed.
- Fix approach: Thread per-detection `delta_luminosity_distance` from `Detection` dataclass. See TODO.md PHYS-5.

**Outdated WMAP-era cosmological parameters [LOW]:**
- Problem: Fiducial values `OMEGA_M = 0.25`, `H = 0.73` are WMAP-era. Planck 2018 best-fit is `OMEGA_M = 0.3153`, `H = 0.6736`.
- Files: `master_thesis_code/constants.py` lines 25, 29-30
- Cause: Values were set at project inception and never updated.
- Fix approach: Update to Planck 2018 values or make configurable via CLI. All simulation data must be regenerated. See TODO.md PHYS-8.

**Non-standard galaxy redshift uncertainty scaling [LOW]:**
- Problem: `Galaxy.redshift_uncertainty` uses `0.013 * (1+z)^3` capped at 0.015 (line 67). This caps at z ~ 0.048, so nearly all galaxies (z up to 0.55) use the constant cap value. Standard photometric: `0.05*(1+z)`, spectroscopic: `0.001*(1+z)`.
- Files: `master_thesis_code/datamodels/galaxy.py` line 67
- Cause: No citation provided; possibly thesis-specific choice.
- Fix approach: Document reference or switch to standard scaling. See TODO.md PHYS-7.

## Fragile Areas

**Multiprocessing initialization in BayesianStatistics [HIGH]:**
- Files: `master_thesis_code/bayesian_inference/bayesian_statistics.py` lines 370-400, 921-946
- Why fragile: Worker initialization relies on 6 global variables set by `child_process_init`. If any global is missed or renamed, workers silently read stale/undefined state. No type checking or validation on globals.
- Safe modification: Do not rename or reorder globals without updating all 4 `global` declaration sites. Run the full `--evaluate` pipeline after any change.
- Test coverage: Zero unit tests for multiprocessing workers. Only tested via integration test with monkeypatching.

**Fisher matrix inversion [HIGH]:**
- Files: `master_thesis_code/parameter_estimation/parameter_estimation.py` lines 356-374
- Why fragile: Uses deprecated `np.matrix(...).I` for inversion. If the Fisher matrix is ill-conditioned (common with forward-difference derivatives), inversion silently produces large/wrong Cramer-Rao bounds. No condition number check.
- Safe modification: Switch to `np.linalg.inv()` on a regular ndarray. Add condition number logging.
- Test coverage: Only tested on GPU (requires cupy + waveform generation).

**LISA response generator coupling [MEDIUM]:**
- Files: `master_thesis_code/waveform_generator.py`, `master_thesis_code/parameter_estimation/parameter_estimation.py`
- Why fragile: `create_lisa_response_generator` default `use_gpu=True` means any CPU-path test that touches waveform generation fails. The `fastlisaresponse` import is correctly lazy, but the downstream `cp.*` calls in parameter_estimation are not guarded.
- Safe modification: Always pass explicit `use_gpu` from CLI. Never rely on defaults.
- Test coverage: All waveform tests require GPU marker; no CPU-path waveform tests exist.

## Scaling Limits

**Galaxy catalog BallTree lookup [MEDIUM]:**
- Current capacity: GLADE catalog ~2.3M galaxies, BallTree built at startup.
- Limit: Memory-bound; BallTree fits in RAM for current catalog size.
- Scaling path: For larger catalogs (e.g., full GLADE+), consider HEALPix-based spatial indexing or database backend.

**Multiprocessing pool for likelihood evaluation [MEDIUM]:**
- Current capacity: Scales to available CPU cores (auto-detected via `os.sched_getaffinity`).
- Limit: Each worker pickles the full `DetectionProbability` interpolator (~4 arrays of RegularGridInterpolator). Serialization overhead grows with interpolator resolution.
- Scaling path: Use shared memory (`multiprocessing.shared_memory`) for large arrays.

## Dependencies at Risk

**`np.matrix` usage [LOW]:**
- Risk: `np.matrix` is deprecated since NumPy 1.20 and will be removed in a future release.
- Impact: `compute_Cramer_Rao_bounds()` in `parameter_estimation.py` line 359 uses `np.matrix(...).I`.
- Migration plan: Replace with `np.linalg.inv(np.array(...))`.

**Dual Bayesian pipelines [LOW]:**
- Risk: Pipeline A (`BayesianInference`) and Pipeline B (`BayesianStatistics`) implement overlapping functionality with different assumptions. Changes to shared functions (e.g., `dist()`) affect both differently.
- Impact: Confusion about which pipeline produces thesis results.
- Migration plan: Pipeline A is documented as dev cross-check; Pipeline B is production. Consider deprecating Pipeline A once thesis is complete.

## Missing Critical Features

**`--generate_figures` not implemented [MEDIUM]:**
- Problem: CLI flag exists (`arguments.py`) but `main.py` logs "not implemented" (lines 313-321). All plotting factory functions exist in `master_thesis_code/plotting/`.
- Blocks: Cannot regenerate thesis figures from saved data without manual scripting.

**`--use_gpu` CLI argument missing [HIGH]:**
- Problem: No CLI flag to control GPU usage. `waveform_generator.py` defaults `use_gpu=True`, making the code crash on CPU-only machines when the simulation pipeline is invoked.
- Blocks: Running simulation pipeline on CPU-only dev machines.

**EMRI distribution polynomial coefficients undocumented [MEDIUM]:**
- Problem: `Model1CrossCheck` in `cosmological_model.py` lines 91-147 contains 5 sets of 9th-degree polynomial fits with no reference citation.
- Blocks: Cannot verify scientific correctness of the EMRI event rate model.

## Test Coverage Gaps

**bayesian_statistics.py (Pipeline B production code) [HIGH]:**
- What's not tested: `BayesianStatistics.evaluate()`, `single_host_likelihood()`, `child_process_init()`, multiprocessing worker pool, all helper functions.
- Files: `master_thesis_code/bayesian_inference/bayesian_statistics.py` (988 lines, ~0% unit test coverage)
- Risk: Production Hubble constant posterior could be silently wrong. The Gaussian index bug (line 618) and unchecked Jacobian (line 623) would have been caught.
- Priority: HIGH

**cosmological_model.py core classes [HIGH]:**
- What's not tested: `Model1CrossCheck.sample_emri_events()`, `emri_distribution()`, `R_emri()`, `LamCDMScenario`, `DarkEnergyScenario`.
- Files: `master_thesis_code/cosmological_model.py` (412 lines)
- Risk: EMRI event rate model and sampling could have subtle bugs.
- Priority: HIGH

**galaxy_catalogue/handler.py [MEDIUM]:**
- What's not tested: `GalaxyCatalogueHandler`, BallTree setup, galaxy catalog filtering, `HostGalaxy.from_attributes`.
- Files: `master_thesis_code/galaxy_catalogue/handler.py` (669 lines, 0 tests)
- Risk: Galaxy catalog lookup errors would propagate silently into the likelihood.
- Priority: MEDIUM

**arguments.py + main.py [MEDIUM]:**
- What's not tested: CLI argument parsing, `main()` entry point, `_write_run_metadata()`, `data_simulation()` orchestration.
- Files: `master_thesis_code/arguments.py` (148 lines), `master_thesis_code/main.py` (331 lines)
- Risk: CLI regressions; metadata recording failures.
- Priority: MEDIUM

**plotting/ subpackage [LOW]:**
- What's not tested: 6 of 9 plotting modules have 0% coverage. Only `_style.py` has tests.
- Files: `master_thesis_code/plotting/bayesian_plots.py`, `master_thesis_code/plotting/evaluation_plots.py`, `master_thesis_code/plotting/model_plots.py`, `master_thesis_code/plotting/catalog_plots.py`, `master_thesis_code/plotting/physical_relations_plots.py`, `master_thesis_code/plotting/simulation_plots.py`
- Risk: Low (plotting bugs are visually obvious), but smoke tests would catch import errors.
- Priority: LOW

**Overall coverage:** 26% (gate at 25%). Target is 50%. 151 CPU tests pass, 18 deselected (GPU/slow).

---

*Concerns audit: 2026-03-25*
