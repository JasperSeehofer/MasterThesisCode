# TODO's

## Execution Order

### Phase 1: Foundation (do first — prerequisites for everything) ✅
1. ~~STAT-1 — Document which pipeline is production~~ (RESOLVED)
2. ~~TEST-1 + TEST-2 — Add regression guards before fixing bugs~~ (RESOLVED)
3. ~~ARCH-1 — Extract BayesianStatistics~~ (RESOLVED)

### Phase 2: Critical Physics Fixes (after Phase 1) ✅
4. ~~PHYS-1 — Comoving volume element formula~~ (RESOLVED)
5. ~~PHYS-2 — Mass distribution sigma bug~~ (RESOLVED)
6. ~~STAT-3 — Mass distribution normalization review~~ (RESOLVED)
7. ~~STAT-4 — d_L fraction direction audit~~ (RESOLVED)

### Phase 3: Testing Expansion (parallel with Phase 2)
8. TEST-3 — BayesianInference correctness tests
9. TEST-4 through TEST-7 — Expand coverage to untested modules

### Phase 4: Important Physics + Performance (after Phase 2)
10. ~~PHYS-3 — Five-point stencil~~ (RESOLVED)
11. ~~PHYS-4 — Confusion noise~~ (RESOLVED)
12. PERF-1 + PERF-2 — GPU portability fixes
13. REPRO-1 — RNG refactor

### Phase 5: Polish (after Phases 3-4)
14. All P2 items
15. Coverage gate increases (TEST-9)
16. PHYS-9 — Document remaining physics TODOs as paper scope or future work

### Verification (after each phase)
- `uv run pytest -m "not gpu and not slow"` — all CPU tests pass
- `uv run mypy master_thesis_code/` — clean
- `uv run ruff check master_thesis_code/` — clean
- Coverage check: `uv run pytest --cov` reports above current gate
- After physics fixes (PHYS-1 through PHYS-5): re-run simulation with `--seed 42`, compare outputs
- After ARCH-1: verify `cosmological_model.py` shrunk, evaluation pipeline still works

---

## Workstream 1: Physics Correctness

All items require the `/physics-change` skill (5-step protocol: old formula, new formula,
reference, dimensional analysis, limiting case).

- [x] **PHYS-1 [P0, S]** Fix comoving volume element formula in `datamodels/galaxy.py`
      The formula computes dV_c/dz (volume element), not V_c (total volume). The exponent 2
      and 4π prefactor were correct, but the formula was missing the 1/E(z) factor.
      Fix: `cv_grid = 4π · (c/H₀)³ · I(z)² / E(z)`. Ref: Hogg (1999) arXiv:astro-ph/9905116 Eq. (27).
      Also renamed all methods from `comoving_volume` → `comoving_volume_element` for clarity.

- [ ] **PHYS-2 [P0, S]** Fix `setup_galaxy_mass_distribution` NormalDist branch in `datamodels/galaxy.py:292`
      Sigma uses hardcoded `10**5.5` instead of `galaxy.central_black_hole_mass`.
      `append_galaxy_to_galaxy_mass_distribution` (line 230) already uses the correct value.
      The truncnorm branch also doesn't pass `loc`/`scale` to scipy (inherits defaults
      `loc=0, scale=1`), making it inconsistent with the `append_*` method.
      **Depends on:** TEST-2 (regression guard must come first)

- [x] **PHYS-3 [P1, M]** Switch Fisher matrix to five-point stencil in `parameter_estimation.py:328`
      (RESOLVED — Phase 10, `use_five_point_stencil=True` is default. Commit `a87eeab`.)
      Ref: Vallisneri (2008) arXiv:gr-qc/0703086; Cutler & Flanagan (1994) PRD 49, 2658.

- [x] **PHYS-4 [P1, M]** Add galactic confusion noise to LISA PSD in `LISA_configuration.py`
      (RESOLVED — `_confusion_noise()` in `LisaTdiConfiguration`, Babak et al. 2023 Eq. 17.
      Controlled by `include_confusion_noise` parameter, default True. Issue #3 closed.)

- [ ] **PHYS-5 [P1, M]** Use per-source Fisher-matrix σ(d_L) in `bayesian_inference/bayesian_inference.py`
      Replace hardcoded `FRACTIONAL_LUMINOSITY_ERROR * d_L` (constant 10%) at lines 154, 186, 235
      with `delta_luminosity_distance_delta_luminosity_distance` from the `Detection` dataclass.
      Requires threading per-detection error information from `EMRIDetection` into `BayesianInference`.

- [ ] **PHYS-6 [P2, S]** Fix or document silent wCDM fallback in `physical_relations.py:72`
      `w_0`, `w_a` params are accepted but `lambda_cdm_analytic_distance` ignores them entirely.
      Either (a) remove args and document ΛCDM assumption, or
      (b) fall back to numerical integration via `hubble_function()` when `w_0 ≠ -1` or `w_a ≠ 0`.
      Ref: Hogg (1999) arXiv:astro-ph/9905116 Eq. (14–16).

- [ ] **PHYS-7 [P2, S]** Document or fix galaxy redshift uncertainty in `datamodels/galaxy.py:64`
      Current `0.013 * (1+z)³` caps at z ≈ 0.048, meaning almost all galaxies (z up to 0.55)
      use the capped value of 0.015. Standard forms: photometric `σ_z = 0.05(1+z)`,
      spectroscopic `σ_z = 0.001(1+z)`. Add citation or switch to standard form.

- [ ] **PHYS-8 [P2, S]** Update fiducial cosmological parameters in `constants.py:29–30`
      Currently WMAP-era: `Ω_m = 0.25`, `H = 0.73`.
      Planck 2018: `Ω_m = 0.3153`, `Ω_de = 0.6847`, `H = 0.6736`.
      Changes all distances and volumes; must re-run full simulation pipeline afterward.
      Consider making configurable rather than hardcoded.
      Ref: Planck Collaboration (2020) arXiv:1807.06209 Table 2.

- [ ] **PHYS-9 [P2, L]** Address 7 pre-existing physics TODOs — document which are paper scope vs future work:
      - [ ] Coordinate transformation to orbital motion around sun
      - [ ] Check `_s` parameters: barycenter same as orientation of binary wrt fixed frame
      - [ ] Check spin limits for parameter `a`
      - [ ] Inclination for Schwarzschild waveforms (defined w.r.t. MBH angular momentum)
      - [ ] Compute derivative w.r.t. sky localization in SSB again
      - [ ] Use second detector from LISA
      - [ ] Function integration: reduced to positive integral because negative frequency == complex
            conjugate, but `fs` currently contains negative frequencies (wrong)

---

## Workstream 2: Statistical Methodology

- [x] **STAT-1 [P0, S]** Document which Bayesian pipeline is production
      Pipeline A (`BayesianInference` in `bayesian_inference/bayesian_inference.py`): simpler,
      synthetic galaxy catalog, erf-based detection probability.
      Pipeline B (`BayesianStatistics` in `cosmological_model.py`): real GLADE catalog,
      KDE-based detection probability, multiprocessing.
      Add module-level docstrings clarifying roles. Note in README.

- [ ] **STAT-2 [P1, S]** Validate emcee MCMC convergence for comoving volume sampling
      `galaxy.py:137` uses 5 walkers × 1000 burn-in for 1D sampling.
      emcee authors recommend at least 10 walkers. Add autocorrelation time check
      (`emcee.autocorr.integrated_time`) to verify convergence.

- [x] **STAT-3 [P1, M]** Review `evaluate_galaxy_mass_distribution` normalization in `galaxy.py:310-321`
      Two bugs found and fixed: (1) `truncnorm()` missing `loc`/`scale` params in
      `setup_galaxy_mass_distribution`, `append_galaxy_to_galaxy_mass_distribution`,
      `setup_galaxy_distribution`, and `append_galaxy_to_galaxy_distribution` — defaulted
      to N(0,1) instead of mass/redshift-space distributions. (2) `evaluate_galaxy_mass_distribution`
      divided by `std()` and CDF range, but `truncnorm.pdf()` is already normalized — removed
      double normalization. 4 regression tests added.

- [x] **STAT-4 [P1, M]** Audit d_L fraction direction in `bayesian_statistics.py`
      `single_host_likelihood` (lines 560, 607) used `detection.d_L / d_L` (measured/model).
      Correct direction is `d_L / detection.d_L` (model/measured), matching the covariance
      structure `σ²/d_L_measured²` and the `single_host_likelihood_integration_testing` function.
      The old formula introduced a spurious `(d_L_measured/d_L_model)²` factor in the exponent.
      Fixed both occurrences.

- [ ] **STAT-5 [P1, S]** Document `Model1CrossCheck` polynomial coefficients in `cosmological_model.py:91-147`
      5 sets of 9th-degree polynomial fits for dN/dz with no reference citation.
      Presumably fits to Babak et al. (2017) PRD 95, 103012 — must be documented.
      Add reference comment and ideally the data source files.

- [ ] **STAT-6 [P2, S]** Review emcee walkers for 2D EMRI event sampling
      `cosmological_model.py:277`: 20 walkers for 2D (mass, redshift) — fine (>= 2×ndim).
      1000 burn-in steps — verify with autocorrelation time. Sampler is never reset between
      calls to `sample_emri_events()` (intentional warm start but undocumented).

- [ ] **STAT-7 [P2, S]** Verify Monte Carlo integration sample size convergence
      `cosmological_model.py:1276`: `N_SAMPLES = 10_000` for the production path.
      Add convergence check: compare result with 2× samples and verify relative change < threshold.
      The importance sampling approach (sampling from prior, reweighting) is correct but
      should be documented.

- [x] **STAT-8 [P1, M]** Fix posterior combination numerical stability — RESOLVED
      (a) Zero-handling: four strategies implemented in `posterior_combination.py`
      (naive, exclude, per-event-floor, physics-floor). Physics-floor is production default.
      (b) Log-space accumulation: `combine_log_space()` uses `Σ log p` throughout.
      (c) D(h) selection-function correction (Gray et al. 2020 Eq. A.19): added
      `−N·log D(h)` term to `combine_log_space()` (commit 2853c32). The MAP=0.86 bias
      was caused by the missing D(h) term, not zero-counts. Expected MAP ~0.73 after fix.

---

## Workstream 3: Testing and Verification

Current: 149 tests, 37% coverage (gate 25%), target 50%.

- [ ] **TEST-COORD [P0, S]** Remove xfail markers from `test_coordinate_roundtrip.py` after Phase 36 lands
      Six `@pytest.mark.xfail(strict=True, reason=_XFAIL_REASON)` markers in
      `master_thesis_code_test/test_coordinate_roundtrip.py` must be removed once Phase 36
      fixes both coordinate-frame bugs (equatorial→ecliptic rotation + BallTree polar embedding).
      When the fix lands, tests flip XPASS → CI fails until markers are removed (by design, D-01).
      **Depends on:** Phase 36 completion.

- [x] **TEST-1 [P0, S]** Regression test for comoving volume element
      `test_comoving_volume_element_spline_matches_integration` now asserts the correct
      formula (including 1/E(z) factor). Resolved alongside PHYS-1.

- [x] **TEST-2 [P0, S]** Regression test for mass distribution sigma
      Create galaxies at different masses and verify the distribution sigma scales with
      galaxy mass (not with hardcoded `10**5.5`). This test should FAIL before PHYS-2.
      **Must precede PHYS-2.**

- [x] **TEST-3 [P0, M]** Correctness tests for `BayesianInference.likelihood`
      Added 10 tests: detection probability monotonicity, gw_likelihood peaks at true z,
      gw_likelihood symmetry, likelihood peaks near TRUE_HUBBLE_CONSTANT, all-positive
      across H₀ grid, selection effects changes likelihood, BH mass changes likelihood,
      posterior all positive, closer source → higher likelihood, wrong H₀ → lower likelihood.

- [ ] **TEST-4 [P1, L]** Tests for `cosmological_model.py` core classes
      `Model1CrossCheck`: `sample_emri_events()` returns samples within declared bounds,
      `emri_distribution` is positive, `R_emri` continuity.
      `DetectionProbability`: `evaluate_with_bh_mass` and `evaluate_without_bh_mass` return [0,1].
      `LamCDMScenario`, `DarkEnergyScenario`: parameter bounds are consistent.
      Currently 0 tests on core classes (only helpers like `gaussian`, `polynomial` tested).

- [ ] **TEST-5 [P1, M]** Tests for `galaxy_catalogue/handler.py`
      `HostGalaxy.from_attributes` round-trips correctly, `_get_pruned_galaxy_catalog` filters
      correctly, `setup_galaxy_catalog_balltree` produces a valid BallTree.
      Needs a mock `reduced_galaxy_catalogue.csv` fixture (real one is ~2.3M galaxies).
      Currently 0 tests.

- [ ] **TEST-6 [P1, M]** Tests for `arguments.py` + `main.py`
      `Arguments.create()` with mock `sys.argv`, `Arguments.validate()` edge cases,
      `main()` with `simulation_steps=0` (starts and stops), `_write_run_metadata` writes
      expected keys. Currently 0 tests.

- [ ] **TEST-7 [P1, S]** Expand `LISA_configuration_test.py` CPU tests
      PSD positivity across full valid frequency range, A==E channel symmetry.
      Some tests already exist — verify completeness and fill gaps.

- [ ] **TEST-8 [P2, M]** Plotting smoke tests
      Call each factory function in `plotting/` with minimal synthetic data and verify
      `(fig, ax)` is returned without errors. Do not test visual correctness.
      Currently 0% coverage on 6 out of 9 plotting modules.

- [ ] **TEST-9 [P2, S]** Raise coverage gate incrementally
      `pyproject.toml` `fail_under`: 25% → 40% (after TEST-2 through TEST-6) → 50%.

---

## Workstream 4: Performance and HPC

- [ ] **PERF-1 [P0, S]** Fix `USE_GPU = True` hardcoded in `waveform_generator.py:9`
      Also hardcoded in `pn5_aak_configuration` (line 22) and multiple `GenerateEMRIWaveform` calls.
      Should accept `use_gpu` parameter from CLI `--use_gpu` argument.
      Currently crashes on CPU machines when `main.py` tries to call `create_lisa_response_generator`.

- [ ] **PERF-2 [P1, S]** Guard remaining unconditional `cp.*` calls in `parameter_estimation.py`
      `_crop_to_same_length` (line 248): `cp.array()` unconditionally.
      `compute_fisher_information_matrix` (line 330): `cp.zeros()` unconditionally.
      `compute_signal_to_noise_ratio` (line 378): `cp.sqrt()` unconditionally.
      All need the `xp = _get_xp(use_gpu)` pattern.

- [ ] **PERF-3 [P1, M]** Vectorize Bayesian inference pre-computation loops
      `bayesian_inference.py:87-126`: `__post_init__` loops over 1000 redshift values × N galaxies
      in Python loops. `galaxy_distribution_at_redshifts` and
      `galaxy_detection_mass_distribution_at_redshifts` can be vectorized.
      `detection_skylocalization_weight_by_galaxy` uses `NormalDist` in Python loop; can use
      `scipy.stats.norm` vectorized.

- [ ] **PERF-4 [P1, L]** Parallelize 14 Fisher matrix derivative evaluations
      `parameter_estimation.py:324-348`: currently sequential, but each derivative is independent.
      With five-point stencil (PHYS-3): 56 total waveform evaluations, all parallelizable.
      Important for production-scale runs on multi-GPU systems.

- [ ] **PERF-5 [P2, S]** Replace `.iterrows()` with `.itertuples()` or vectorized ops
      `cosmological_model.py:755,796`: `.iterrows()` is the slowest way to iterate a DataFrame.
      The filtering loop `use_detection()` at line 755-758 can be fully vectorized.

- [ ] **PERF-6 [P2, S]** Add `--use_gpu` CLI argument + `CUDA_VISIBLE_DEVICES` selection
      `arguments.py`: currently missing entirely. Thread through `main.py` →
      `ParameterEstimation` → `waveform_generator.py`.

---

## Workstream 5: Code Architecture

- [x] **ARCH-1 [P0, L]** Extract `BayesianStatistics` + `DetectionProbability` from `cosmological_model.py`
      Currently 1616 lines with 24 `global` statements, multiprocessing workers, and helper
      functions all in one file. Extract into:
      - `bayesian_inference/bayesian_statistics.py` — `BayesianStatistics`, `single_host_likelihood`
      - `bayesian_inference/detection_probability.py` — `DetectionProbability`
      - Keep `Model1CrossCheck`, `LamCDMScenario`, `DarkEnergyScenario` in `cosmological_model.py`
      **Biggest refactor; makes all subsequent changes cleaner.**

- [ ] **ARCH-2 [P1, M]** Eliminate 24 `global` variables in multiprocessing workers
      `cosmological_model.py`: `child_process_init` sets 6 globals that `single_host_likelihood`
      reads. Refactor to frozen dataclass or `functools.partial`.
      Makes the code testable without multiprocessing.

- [ ] **ARCH-3 [P1, S]** Replace 22 `print()` calls with `_LOGGER.info/debug`
      Mostly in `cosmological_model.py` (14 calls). Some are debug prints left from development
      (e.g., `print(possible_host.z, possible_host.z_error)` at line 1147).

- [ ] **ARCH-4 [P1, S]** Remove 243-line `single_host_likelihood_integration_testing` dead code
      `cosmological_model.py:1303-1546`: testing variant of `single_host_likelihood` with
      inline print statements and commented-out code. Either promote to a proper test or delete.
      (Also relevant to STAT-4 — audit d_L fraction direction first.)

- [ ] **ARCH-5 [P2, S]** Fix `ParameterSample` mutable defaults in `galaxy_catalogue/handler.py:37-38`
      `phi_S: float = np.random.random() * 2 * np.pi` and
      `theta_S: float = np.arccos(np.random.random() * 2 - 1)` are evaluated once at class
      definition time, not per instance. All instances share the same default values.
      Per CLAUDE.md conventions: must use `field(default_factory=...)`.

- [ ] **ARCH-6 [P2, S]** Unify distance/detection threshold constants
      `LUMINOSITY_DISTANCE_THRESHOLD_GPC = 1.55` in `constants.py`.
      `luminostity_detection_threshold = 1.55` in `Model1CrossCheck._apply_model_assumptions()`.
      `dist(redshift=1.5)` is the actual threshold calculation.
      Should be a single source of truth.

---

## Workstream 6: Reproducibility

- [ ] **REPRO-1 [P0, M]** Replace 21 bare `np.random.*` calls with `np.random.default_rng(seed)`
      Generator instances, threaded through all functions that need randomness.
      `np.random.seed()` in `main.py` sets global state but is fragile.
      Ensures reproducibility even under parallelism.

- [ ] **REPRO-2 [P1, S]** Expand `run_metadata.json` in `main.py:86-102`
      Currently records: `git_commit`, `timestamp`, `random_seed`, `cli_args`.
      Add: Python version, numpy/scipy/few versions, GPU info (if available), uv.lock hash.
      Critical for paper reproducibility claims.

- [ ] **REPRO-3 [P1, M]** Implement `--generate_figures` stub in `main.py:313-321`
      Currently logs "not implemented". All plotting factory functions exist in `plotting/`.
      Implement: load saved CSV/JSON data, call each factory function, save to output directory.
      Allows regenerating all paper figures from saved data.

- [ ] **REPRO-4 [P2, S]** Add data provenance to CSV outputs
      Cramér-Rao bounds CSVs do not record which git commit, seed, or configuration produced them.
      Add a metadata header row or a companion JSON file per CSV.

---

## Code Health (remaining, not in workstreams above)

- [ ] Fix unconditional `import cupy` at module level in `LISA_configuration.py`
      (blocks import on CPU machines without `try/except` guard)
- [ ] Tag git release `v0.1.0` once current branch is merged: `git tag v0.1.0`
- [ ] Add Codecov integration to CI for a coverage badge in README
- [ ] Rename GitHub repository from `MasterThesisCode` to `emri-dark-siren-h0` (or similar).
      Touches: CI badge URL in README, git remote URLs, all documentation references.
      GitHub auto-redirects the old URL indefinitely. Do as a dedicated task.

---

## Workstream 7: Publication

All items tracked under the "Paper Submission" GitHub milestone.

- [x] **PUB-1 [P0, S]** Create `CITATION.cff` with project metadata and placeholder paper reference
      (RESOLVED — `CITATION.cff` created 2026-04-05.)
- [ ] **PUB-2 [P0, S]** Tag first GitHub Release (`v2.0.0-rc1`, paper-stage baseline)
- [ ] **PUB-3 [P1, M]** Prepare reproducibility package (simulation seeds, config, expected outputs)
- [ ] **PUB-4 [P2, S]** Archive Pipeline A or clearly label as development-only cross-check
- [ ] **PUB-5 [P1, M]** Write paper methods section describing the completeness-corrected likelihood
- [ ] **PUB-6 [P2, S]** Final data release preparation (simulation outputs, metadata, figure generation)

---

## Done (Phase 11 — 2026-03-12, CI & bugfix)

- [x] Upgrade GitHub Actions to Node.js 24 compatible versions
      (`checkout@v6`, `upload-artifact@v7`, `upload-pages-artifact@v4`, `setup-uv@v7`)
- [x] Fix `setup_galaxy_mass_distribution` to respect `_use_truncnorm` flag
- [x] Fix `.stdev` → `.std()` for scipy truncnorm frozen distributions

## Done (Phase 10 — 2026-03-11, Plotting Refactor)

- [x] Create `master_thesis_code/plotting/` subpackage with factory functions
      (`_style.py`, `_helpers.py`, `emri_thesis.mplstyle`, `simulation_plots.py`,
      `bayesian_plots.py`, `evaluation_plots.py`, `model_plots.py`, `catalog_plots.py`,
      `physical_relations_plots.py`)
- [x] Create `callbacks.py` with `SimulationCallback` Protocol; wire into `data_simulation()`
- [x] Delete `ScientificPlotter`, `IS_PLOTTING_ACTIVATED`, `if_plotting_activated` decorator
- [x] Remove all matplotlib imports from computation modules (only in `plotting/` now)
- [x] Remove `__init__` plot side effects from `glade_completeness.py`, `detection_horizon.py`,
      `detection_distribution_simplified.py`, `emri_distribution.py`, `detection_fraction.py`
- [x] Extract ~1900 lines of plotting from `cosmological_model.py` (shrunk to ~1611 lines)
- [x] Extract plotting from `evaluation.py`, `handler.py`, `physical_relations.py`,
      `galaxy.py`, `emri_detection.py`, `bayesian_inference.py`, `memory_management.py`,
      `LISA_configuration.py`, `parameter_estimation.py`
- [x] Add `--generate_figures` CLI argument (stub handler)
- [x] Add `master_thesis_code_test/plotting/test_style.py` (9 tests)
- [x] Coverage increased from 28.83% to 36.19%

## Done (Phase 9 — 2026-03-10)

- [x] Physics & mathematics review: README "Scientific Background and Known Limitations" section
- [x] All eight confirmed issues documented in README, TODO, CHANGELOG, and CLAUDE.md
- [x] GitHub issues filed for all confirmed bugs

## Done (Phase 8 — 2026-03-10)

- [x] Add LICENSE file (MIT)
- [x] Add CONTRIBUTING.md and .editorconfig
- [x] Add pytest-cov + coverage gate (25%); CI uploads coverage.xml artifact
- [x] Add pip-audit to dev extras + CI security step
- [x] Add Dependabot (weekly pip + GitHub Actions updates)
- [x] Add `--seed` CLI arg; seed numpy in main(); write run_metadata.json per run
- [x] Fix `get_samples_from_comoving_volume` PNG side-effect (`save_plot=False`)
- [x] Rename `ParameterSpace.dist` → `luminosity_distance` (field, symbol, CSV cols)
- [x] Fix missed `"dist"` column references in `scripts/prepare_detections.py` and
      `scripts/estimate_hubble_constant.py`; patch existing simulation CSVs
