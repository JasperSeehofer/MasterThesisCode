# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- `galaxy_catalogue/glade_completeness.py`: GLADE+ catalog completeness estimation
  $f(z, H_0)$ using galaxy counts in comoving volume shells (Phase 24).
- Completeness-corrected dark siren likelihood implementing Gray et al. (2020) Eq. 9,
  combining catalog and completion terms weighted by GLADE+ completeness (Phase 25).
- 23 tests for completeness estimation, 11 tests for completion term and combination
  formula (`test_glade_completeness.py`, `test_completion_term.py`).
- `scripts/bias_investigation/`: 7 diagnostic scripts and `FINDINGS.md` documenting
  root cause of H₀ posterior bias (GLADE catalog density gradient, not a formula bug).
- Docs badge in README linking to GitHub Pages.
- `CITATION.cff` for machine-readable citation metadata.
- GitHub issue templates (bug report, physics bug) and PR template.
- 14 new correctness tests for `BayesianInference` (TEST-3): likelihood peak location,
  detection probability monotonicity, selection effects, BH mass term, posterior
  positivity, and cross-H₀ consistency checks.
- 4 regression tests for truncnorm distribution correctness (STAT-3): PDF peak location,
  integration-to-one, correct `loc` for redshift and mass distributions.

### Fixed
- **[PHYSICS]** Comoving volume formula corrected to proper volume element $dV_c/dz$ with
  $1/E(z)$ factor; methods renamed `comoving_volume` → `comoving_volume_element`
  (PHYS-1, Issue #1).
- **[PHYSICS]** Galactic confusion noise integrated into LISA PSD via
  `LisaTdiConfiguration._confusion_noise()`, implementing Babak et al. (2023)
  Eq. (17) (PHYS-4, Issue #3).
- **[PHYSICS]** `GalaxyCatalog` truncnorm distributions (STAT-3): `truncnorm()` was created
  without `loc`/`scale` parameters in `setup_galaxy_mass_distribution`,
  `append_galaxy_to_galaxy_mass_distribution`, `setup_galaxy_distribution`, and
  `append_galaxy_to_galaxy_distribution`, defaulting to N(0,1) instead of the intended
  mass/redshift-space distributions. Also removed double normalization in
  `evaluate_galaxy_mass_distribution` — `truncnorm.pdf()` is already normalized.
- **[PHYSICS]** `single_host_likelihood` d_L fraction direction (STAT-4): the production
  likelihood in `bayesian_statistics.py` used `detection.d_L / d_L` (measured/model)
  instead of the correct `d_L / detection.d_L` (model/measured). The incorrect direction
  introduced a spurious `(d_L_measured/d_L_model)²` factor in the Gaussian exponent,
  biasing the H₀ posterior. Now consistent with `single_host_likelihood_integration_testing`.

### Changed
- Project framing updated from master thesis to paper publication stage.
- GitHub issues triaged: #1 and #3 closed as resolved, remaining issues labeled
  (`paper-blocker`, `design-choice`) and assigned to "Paper Submission" milestone.
- `.claude/skills/` directory with 6 custom skills for codified, repeatable workflows:
  - `physics-change`: enforces the 5-step Physics Change Protocol before any formula modification
  - `gpu-audit`: scans files for GPU/HPC compliance violations (guarded imports, xp pattern, vectorization)
  - `run-pipeline`: runs simulation/evaluation/SNR pipelines with correct flags and validates output
  - `check`: full quality gate (ruff lint + format + mypy + pytest) in one invocation
  - `known-bugs`: shows current status of all known physics/code bugs with priorities
  - `pre-commit-docs`: verifies CHANGELOG, TODO, CLAUDE.md, README consistency with staged changes
- Excalidraw MCP server configured (HTTP transport) for architecture diagram generation.
- `CLAUDE.md`: new "Skill-Driven Workflows" section with trigger rules table and
  physics-change trigger file list. Skills are mandatory workflow gates, not optional.
- `CLAUDE.md`: new "GitHub Integration" section — GSD/GPD workflows must keep GitHub
  issues, milestones, and labels in sync as work progresses.

---

## [2026-03-11] — plotting architecture refactor (Phases 1–4)

### Added
- `master_thesis_code/plotting/` subpackage: all visualization code now lives here.
  - `_style.py`: `apply_style()` sets Agg backend and loads the project style sheet.
  - `_helpers.py`: `get_figure()`, `save_figure()`, `make_colorbar()` utilities.
  - `emri_thesis.mplstyle`: single source of truth for all matplotlib rcParams
    (figure size, DPI, constrained layout, chunksize).
  - `simulation_plots.py`: factory functions for GPU usage, LISA PSD, noise components,
    and Cramér-Rao coverage plots. Also contains `PlottingCallback`.
  - `bayesian_plots.py`: factory functions for combined/event/subset posteriors,
    detection redshift distribution, and host galaxy count plots.
  - `evaluation_plots.py`: factory functions for Cramér-Rao heatmap, uncertainty violins,
    sky localization 3D scatter, detection contour, and generation time histogram.
  - `model_plots.py`: factory functions for EMRI distribution, rate, sampling, and
    detection probability grid plots.
  - `catalog_plots.py`: factory functions for BH mass distribution, redshift distribution,
    GLADE completeness, and comoving volume sampling plots.
  - `physical_relations_plots.py`: factory function for distance-redshift relation plot.
- `master_thesis_code/callbacks.py`: `SimulationCallback` Protocol class with five hooks
  (`on_simulation_start`, `on_snr_computed`, `on_detection`, `on_step_end`,
  `on_simulation_end`) and a `NullCallback` no-op implementation.
- `data_simulation()` now accepts an optional `callbacks: list[SimulationCallback]`
  parameter; hook calls inserted throughout the simulation loop.
- `--generate_figures <dir>` CLI argument in `arguments.py` (stub handler in `main.py`).
- `master_thesis_code_test/plotting/test_style.py`: 9 tests covering `apply_style`,
  `get_figure`, `save_figure`, and style sheet rcParams.

### Changed
- `main.py`: backend setup moved from inline `matplotlib.use("Agg")` to
  `from master_thesis_code.plotting import apply_style; apply_style()`.
- `memory_management.py`: removed `plot_GPU_usage()` method; added `time_series`,
  `memory_pool_gpu_usage`, `gpu_usage` properties for callback-based data access.
- `cosmological_model.py`: shrunk from ~3530 to ~1611 lines by extracting 7 plotting
  methods (~1900 lines): `plot_expected_detection_distribution`,
  `visualize_emri_distribution_sampling`, `visualize_emri_distribution`,
  `plot_detection_probability`, `plot_detection_fraction`, `visualize`,
  `visualize_galaxy_weights`.
- `parameter_estimation/evaluation.py`: removed `visualize()`,
  `visualize_detection_distribution()`, `evaluate_snr_analysis()` methods.
- `galaxy_catalogue/handler.py`: removed `visualize_galaxy_catalog()` method.
- `physical_relations.py`: removed `visualize()` function.
- Test coverage increased from 28.83% to 36.19%.

### Removed
- `master_thesis_code/bayesian_inference/scientific_plotter.py`: deleted entirely
  (dead `ScientificPlotter` wrapper class).
- `IS_PLOTTING_ACTIVATED` constant from `constants.py`.
- `if_plotting_activated` decorator from `decorators.py`.
- `__init__` plot side effects from `glade_completeness.py` (including module-level
  `asdf = GladeCatalogCompleteness()` instantiation), `detection_horizon.py`,
  `detection_distribution_simplified.py`, `emri_distribution.py`, `detection_fraction.py`.
- All `import matplotlib` statements from computation modules: `LISA_configuration.py`,
  `parameter_estimation.py`, `memory_management.py`, `galaxy.py`, `emri_detection.py`,
  `bayesian_inference.py`, `glade_completeness.py`, `handler.py`.
- Plotting methods removed from: `LISA_configuration.py` (`_visualize_lisa_configuration`),
  `parameter_estimation.py` (`_visualize_cramer_rao_bounds`),
  `galaxy.py` (`save_comoving_volume_sampling_plot`, `plot_comoving_volume`,
  `plot_galaxy_catalog`, `plot_galaxy_catalog_mass_distribution`),
  `emri_detection.py` (`plot_detection_distribution`, `plot_detection_sky_distribution`),
  `bayesian_inference.py` (`plot_gw_detection_probability`).

---

## [2026-03-11] — fix incomplete dist → luminosity_distance rename in scripts

### Fixed
- `scripts/prepare_detections.py`: column write `"dist"` → `"luminosity_distance"` so
  prepared CSVs produced by this script match the column name expected by the evaluation
  pipeline.
- `scripts/estimate_hubble_constant.py`: updated all column reads (`"dist"`,
  `"delta_dist_delta_dist"`) and dict keys (`"dist"`, `"dist_error"`) to use
  `"luminosity_distance"` / `"luminosity_distance_error"` /
  `"delta_luminosity_distance_delta_luminosity_distance"`.
- Patched existing simulation CSVs (`cramer_rao_bounds.csv`,
  `prepared_cramer_rao_bounds.csv`, `undetected_events.csv`) to rename the `dist` column
  to `luminosity_distance` so the evaluation pipeline can load them without error.

---

## [2026-03-11] — remove redundant binary data files from repo

### Changed
- `master_thesis_code/waveform_generator.py`: orbit file path changed from
  `"./lisa_files/esa-trailing-orbits.h5"` to bare `"esa-trailing-orbits.h5"`.
  `lisatools` resolves bare filenames against its own bundled `orbit_files/` directory,
  so the repo-local copy is no longer needed.
- `.gitignore`: added `few_data/` and `lisa_files/` to prevent accidental re-addition.

### Removed
- `few_data/` (~105 MB): 4 FEW waveform model binary files removed from git tracking.
  FEW auto-downloads its data to `~/.local/share/few/` on first use via its built-in
  `FileManager`; the repo-local copies were never registered as a search path.
- `lisa_files/` (~2.4 MB): 2 LISA orbit HDF5 files removed from git tracking.
  `lisatools` bundles all three orbit files inside the installed package; the
  relative-path workaround in `waveform_generator.py` is no longer needed.

---

## [2026-03-10] — physics & mathematics review (Phase 9)

### Added
- `README.md`: new top-level section "Scientific Background and Known Limitations" containing:
  - Two-paragraph project narrative (EMRIs as GW standard sirens, dark-siren H₀ method)
  - Key equations with references: Hubble function, luminosity distance, LISA inner product,
    Fisher matrix, SNR, and marginalised H₀ likelihood with selection-effects denominator
  - Model assumptions table (flat ΛCDM, Gaussian noise, SNR threshold, uniform H₀ prior,
    synthetic galaxy catalog, 5-year LISA mission)
  - Eight documented known limitations, each with file:line reference, impact description,
    and status tag (bug / design choice)
  - "What is mathematically correct" verification checklist for six core components
  - Bibliography with six key references (Hogg 1999, Babak 2023, Cutler & Flanagan 1994,
    Vallisneri 2008, Chen 2018, Planck 2018)
- `TODO.md`: physics fix items for all confirmed bugs (Issues 1–8), ordered by severity

### Changed
- `CLAUDE.md`: "Known Bugs to Be Aware Of" section updated with all eight confirmed issues
  from the physics review, with file:line references and fix descriptions

---

## [2026-03-10] — dev infrastructure & code health (Phase 8)

### Added
- `LICENSE`: MIT licence added so the project can legally be shared, forked, and cited.
- `CONTRIBUTING.md`: human-readable contribution guide covering env setup, branching,
  pre-commit, test commands, and the physics-change protocol.
- `.editorconfig`: enforces LF line endings, 4-space Python indent, UTF-8, and
  trailing-whitespace trimming across all editors.
- `.github/dependabot.yml`: weekly automated dependency-update PRs for both the `pip`
  ecosystem (uv lock file) and GitHub Actions.
- `pytest-cov` and `pytest-benchmark` added to `dev` extras in `pyproject.toml`.
- `[tool.coverage.run]` and `[tool.coverage.report]` sections in `pyproject.toml`:
  source is `master_thesis_code/`, test files omitted, gate at 25% (current: 36.19%).
- `addopts` in `[tool.pytest.ini_options]` now includes `--cov` and `--cov-report`
  so every `pytest` invocation reports coverage automatically.
- `pip-audit` added to `dev` extras; new CI step `pip-audit (security)` runs on every
  push to surface known CVEs in installed packages.
- CI step `Upload coverage report` uploads `coverage.xml` as a GitHub Actions artifact
  after the test run.
- `--seed` CLI argument in `arguments.py` (optional `int`; random value chosen and
  logged when omitted).
- `_write_run_metadata()` in `main.py`: writes `run_metadata.json` into the working
  directory at startup, recording `git_commit`, `timestamp`, `random_seed`, and all
  CLI arguments for simulation reproducibility.
- `master_thesis_code_test/test_benchmarks.py`: two `@pytest.mark.slow` benchmark
  tests — `BayesianInference.likelihood` for N=50 detections and
  `GalaxyCatalog.evaluate_galaxy_distribution` for a 500-galaxy catalog.

### Changed
- `main.py`: `main()` now seeds `numpy.random` from `arguments.seed` before any
  sampling begins, and calls `_write_run_metadata()`.
- `master_thesis_code/datamodels/galaxy.py`:
  `GalaxyCatalog.get_samples_from_comoving_volume` gains `save_plot: bool = False`
  parameter; the PNG side-effect is suppressed by default and only fires when the
  caller explicitly passes `save_plot=True`.
- `master_thesis_code/datamodels/parameter_space.py`:
  `ParameterSpace.dist` field and its `Parameter.symbol` both renamed to
  `luminosity_distance`. `_parameters_to_dict` key updated accordingly.
  This removes the Python name-shadowing of the imported `dist()` function.
- All CSV column names derived from the renamed field updated throughout the codebase:
  `"dist"` → `"luminosity_distance"`,
  `"delta_dist_delta_dist"` → `"delta_luminosity_distance_delta_luminosity_distance"`,
  and the four mixed cross-covariance column names in `datamodels/detection.py`,
  `cosmological_model.py`, `parameter_estimation/evaluation.py`.
- `master_thesis_code_test/datamodels/parameter_space_test.py`,
  `master_thesis_code_test/datamodels/test_detection.py`,
  `master_thesis_code_test/cosmological_model_test.py`: all test fixtures updated
  to use the new `luminosity_distance` column names.
- CI `pytest` step now runs `not gpu and not slow` (slow benchmarks excluded from
  the fast CI path).

---

## [2026-03-10] — code cleanup & quality improvement (Phases 1–7)

### Added
- `master_thesis_code/datamodels/galaxy.py`: extracted `Galaxy` and `GalaxyCatalog` classes
  from `bayesian_inference_mwe.py` into a focused datamodel module.
- `master_thesis_code/datamodels/emri_detection.py`: extracted `EMRIDetection` dataclass.
- `master_thesis_code/bayesian_inference/bayesian_inference.py`: extracted `BayesianInference`
  class and `dist_array` helper; `bayesian_inference_mwe.py` reduced to a thin re-export shim
  plus the `__main__` demonstration script.
- `master_thesis_code/datamodels/detection.py`: extracted `Detection` dataclass and
  `_sky_localization_uncertainty()` from the 3617-line `cosmological_model.py` monolith.
- `scripts/` directory with four utility scripts moved out of the package root:
  `prepare_detections.py`, `remove_detections_out_of_bounds.py`,
  `merge_cramer_rao_bounds.py`, `estimate_hubble_constant.py`.
- `master_thesis_code_test/test_constants.py`: 5 tests — flat universe (Ω_m + Ω_de ≈ 1),
  speed of light value, GPC_TO_MPC, KM_TO_M, RADIAN_TO_DEGREE.
- `master_thesis_code_test/datamodels/test_detection.py`: 6 tests for `Detection` construction,
  field parsing from `pd.Series`, relative distance error, sky localisation error.
- `master_thesis_code_test/datamodels/test_emri_detection.py`: 4 tests — regression for
  `float` fields when `use_measurement_noise=False`, sky angles preserved, noise path positive.

### Changed
- `master_thesis_code/constants.py` (Phase 1): `C` and `G` now derived from
  `astropy.constants` for traceability; added `TRUE_HUBBLE_CONSTANT`, `SPEED_OF_LIGHT_KM_S`,
  `GALAXY_REDSHIFT_ERROR_COEFFICIENT`, `LUMINOSITY_DISTANCE_THRESHOLD_GPC`,
  `FRACTIONAL_LUMINOSITY_ERROR`, `FRACTIONAL_BLACK_HOLE_MASS_CATALOG_ERROR`,
  `FRACTIONAL_MEASURED_MASS_ERROR`, `SKY_LOCALIZATION_ERROR`, and LISA hardware constants
  (`LISA_ARM_LENGTH`, `YEAR_IN_SEC`, `LISA_STEPS`, `LISA_DT`, PSD coefficients).
  Removed duplicate constants previously scattered across `bayesian_inference_mwe.py`,
  `galaxy_catalogue/handler.py`, and `LISA_configuration.py`.
- `master_thesis_code/physical_relations.py` (Phase 2): `hubble_function()` now accepts and
  returns `float | npt.NDArray[np.floating[Any]]`; uses `np.ndim(result) == 0` to decide
  whether to wrap in `float()`. Added `redshifted_mass()` and `redshifted_mass_inverse()`.
- `master_thesis_code/bayesian_inference/bayesian_inference_mwe.py` (Phase 3): removed 12
  locally-duplicated constants and functions (`dist`, `lambda_cdm_analytic_distance`,
  `dist_to_redshift`, `redshifted_mass`, `redshifted_mass_inverse`); imports canonical
  versions from `constants.py` and `physical_relations.py`.
- `master_thesis_code/LISA_configuration.py` (Phase 1): removed 11 inline hardware constants;
  imports them from `constants.py`.
- `master_thesis_code/galaxy_catalogue/handler.py` (Phase 1): removed local `GPC_TO_MPC` and
  `RADIAN_TO_DEGREE`; imports from `constants.py`.
- `master_thesis_code/datamodels/galaxy.py` (Phase 6): unit comments added to all fields
  (`dimensionless`, `M_sun`, `rad`).
- `master_thesis_code/datamodels/detection.py` (Phase 6): unit comments added to all fields
  (`Gpc`, `M_sun`, `rad`, `dimensionless`).
- `master_thesis_code/bayesian_inference/bayesian_inference.py` (Phase 6):
  `luminosity_distance_threshold` field annotated `# Gpc`.

### Fixed (Phase 2 — four documented known bugs)
- **Bug 1** `hubble_function` ndarray crash: union return type prevents `float()` wrapping
  array results; `test_dist_derivative_positive` xfail removed and now passes.
- **Bug 2** `LISAConfiguration` staleness: `test_lisa_config_does_not_go_stale_after_randomize`
  xfail removed; test passes (fix was applied in a prior session).
- **Bug 3** `comoving_volume` hardcoded H₀: `GalaxyCatalog.__init__` now accepts `h0` param
  and passes it to `_build_comoving_volume_spline`; `test_comoving_volume_varies_with_hubble_constant`
  xfail removed and now passes.
- **Bug 4** `dist()` unit inconsistency (Mpc vs Gpc): removed the local `dist()` from
  `bayesian_inference_mwe.py`; all callers now use the canonical Gpc implementation in
  `physical_relations.py`; `luminosity_distance_threshold` updated 1550.0 Mpc → 1.55 Gpc.

### Removed
- Dead commented-out code blocks: multiprocessing derivatives stub in
  `parameter_estimation.py`, waveform-plotting block in `compute_signal_to_noise_ratio`,
  `# import statsmodels.api as sm` in `cosmological_model.py`, alternative galaxy
  distribution implementations in `galaxy.py`.
- `sys.exit()` calls in utility scripts replaced with `return` / natural function end.
- Duplicate `Galaxy`, `GalaxyCatalog`, `EMRIDetection`, `BayesianInference` class bodies
  from `bayesian_inference_mwe.py` (now delegated to the new datamodel/inference modules).

---

## [2026-03-10] — tests for HPC performance refactoring

### Added
- `master_thesis_code_test/bayesian_inference/test_bayesian_inference_mwe.py`: 7 new regression
  and correctness tests for the vectorized hot paths introduced in the HPC refactor —
  `dist_array` shape/dtype, element-wise agreement with scalar `dist()` to `1e-10`,
  strict monotonicity, zero-distance at z=0, comoving-volume spline accuracy vs direct
  trapezoid quadrature (<0.1% relative error at 20 redshifts), spline returns 0 at z=0, and
  `BayesianInference.likelihood()` returning a finite positive float (exercises the full
  vectorized numerator/denominator path).
- `master_thesis_code_test/parameter_estimation/parameter_estimation_test.py`: 3 CPU tests for
  the new buffered-CSV flush mechanism — empty buffer is a no-op (no file, no exception),
  explicit `flush_pending_results()` writes all 3 buffered rows to CSV, and
  `_crb_flush_interval=2` auto-flushes at the threshold with the remainder written on explicit
  flush. Plus 3 `@pytest.mark.gpu` tests: PSD cache identity (second call returns the same
  object), PSD cache shape `(n_channels, n_freqs_cropped)`, and Fisher matrix symmetry
  (mocked derivatives, asserts `np.allclose(F, F.T)`).
- `master_thesis_code_test/LISA_configuration_test.py`: 5 CPU tests (no GPU required) for
  the new `_get_xp()` numpy path — `power_spectral_density('A')`, `power_spectral_density('T')`,
  `S_OMS`, `S_TM` all positive with plain `np.logspace` input; channels A and E return
  identical PSD via `np.allclose`. Module-level `pytest.importorskip("cupy")` replaced with a
  `try/except` guard so the file is collected on CPU-only machines.

---

## [2026-03-10] — comprehensive test coverage & Python 3.13 fix

### Added
- `master_thesis_code_test/decorators_test.py`: 5 new tests for `if_plotting_activated`
  (disabled → returns `None`, enabled → passes return value through) and `timer_decorator`
  (return value, `__name__` preservation, function is actually called).
- `master_thesis_code_test/physical_relations_test.py`: 16 new tests covering `dist(0)==0`,
  monotonicity, float return type, `hubble_function` normalisation and positivity,
  `dist_to_redshift` at zero and round-trip (parametrised over z=0.5/1.0/2.0), vectorised
  shape/value consistency, `dist()` varying with `h` and approximate 1/H₀ scaling,
  mass-conversion algebra (both directions and round-trip), and error-propagation positivity.
  Two `dist_derivative` tests are `xfail` (known bug: `hubble_function` cannot accept ndarray).
- `master_thesis_code_test/datamodels/parameter_space_test.py`: 12 tests for `uniform`,
  `log_uniform`, `polar_angle_distribution`, `ParameterSpace` construction, per-parameter and
  bulk randomisation bounds, `_parameters_to_dict` keys/count/types/NaN safety, and
  `set_host_galaxy_parameters()` updating `dist`, `qS`, `phiS`, `M`.
- `master_thesis_code_test/LISA_configuration_test.py`: 7 tests — 1 CPU (instantiation), 6
  `@pytest.mark.gpu` (PSD positivity for A/E/T channels, A==E identity, S_OMS/S_TM/S_zz
  positivity). Plus `xfail` regression for Known Bug #1 (LISAConfiguration staleness).
- `master_thesis_code_test/parameter_estimation/parameter_estimation_test.py`: 2 CPU tests
  (CSV create/append via monkeypatched path) and 5 `@pytest.mark.gpu` tests (`scalar_product`
  positive-definiteness/symmetry, `_crop_frequency_domain` bounds and length, `_crop_to_same_length`).
- `master_thesis_code_test/bayesian_inference/test_bayesian_inference_mwe.py`: 15 new tests —
  `Galaxy` hashability (set deduplication, hash consistency, inequality), `redshifted_mass`/
  `redshifted_mass_inverse` algebra and round-trip, `dist`/`dist_to_redshift` in mwe module,
  `comoving_volume` positivity and monotonicity, `EMRIDetection.from_host_galaxy` tuple-comma
  regression (`use_measurement_noise=False` → float not tuple), truncnorm distribution type,
  `gw_detection_probability` near-zero and large-redshift bounds, posterior length matching.
  Plus `xfail` regression for Known Bug #3 (comoving_volume hardcoded H₀).
- `master_thesis_code_test/cosmological_model_test.py`: 12 tests — `gaussian` (peak, symmetry,
  positivity), `polynomial` (constant/linear/quadratic), `MBH_spin_distribution` range [0,1],
  and `Detection` dataclass (construction, field values, `get_relative_distance_error`,
  `get_skylocalization_error`, `convert_to_best_guess_parameters`).

### Fixed
- `master_thesis_code/bayesian_inference/bayesian_inference_mwe.py`: `Galaxy` dataclass
  changed to `@dataclass(unsafe_hash=True)` so `Galaxy` instances can be used in sets and as
  dict keys. Fixes `test_add_unique_host_galaxies_from_catalog` (was the one failing test).
- `master_thesis_code/datamodels/parameter_space.py`: all 14 `Parameter` field defaults
  changed from bare mutable instances to `field(default_factory=lambda: Parameter(...))`.
  This is required by Python 3.13 (`dataclasses` now rejects mutable defaults that are not
  wrapped in `field()`). Removes 20 previously-skipped tests and makes `ParameterSpace`
  importable on Python 3.13 without error.

---

## [2026-03-10] — modern dev tooling: ruff, pre-commit, CI, mypy clean

### Added
- `ruff` and `pre-commit` added to `dev` dependency group in `pyproject.toml`.
- `[tool.ruff]` and `[tool.ruff.lint]` configuration in `pyproject.toml`: selects
  `E`, `F`, `I`, `UP`, `B`, `N` rules; line length 100; `target-version = "py313"`.
- `.pre-commit-config.yaml`: ruff (lint + format) and mypy run automatically on
  every `git commit`. mypy uses the project's local environment to avoid false positives.
- `.github/workflows/ci.yml`: CI pipeline runs ruff check, ruff format check, mypy,
  and pytest (CPU only, `not gpu`) on every push and pull request.
- `CLAUDE.md`: added "Dev Workflow" section documenting the linting commands and
  pre-commit usage.

### Changed
- `pyproject.toml`: `[tool.mypy]` `python_version` updated `"3.10"` → `"3.13"`;
  extended `ignore_missing_imports` overrides to cover `pandas`, `scipy`, `sklearn`,
  `mpl_toolkits`, `emcee`, and `tabulate`.
- All 39 source files and 10 test files brought to 0 mypy errors under strict settings
  (`disallow_untyped_defs`, `disallow_incomplete_defs`, `warn_return_any`).
  Key changes across the codebase:
  - Removed all `from typing import List, Dict, Optional, Union`; replaced with
    native Python 3.10 syntax (`list[X]`, `dict[K,V]`, `X | None`, `X | Y`).
  - Removed `from __future__ import annotations` from `arguments.py` and
    `bayesian_inference_mwe.py` (per CLAUDE.md convention).
  - All public and private functions annotated with complete return types and
    parameter types.
  - `np.ndarray` → `npt.NDArray[np.float64]` / `npt.NDArray[np.floating[Any]]`
    throughout.
  - `np.trapz` → `np.trapezoid` (numpy 2.x rename).
  - Guarded `import cupy as cp` in `decorators.py`, `memory_management.py`,
    `LISA_configuration.py`, and `parameter_estimation.py` with
    `try/except ImportError` + `_CUPY_AVAILABLE` sentinel.
  - `decorators.py`: converted `TypeVar`-based generics to Python 3.12 type
    parameter syntax (`def f[F: Callable[..., Any]](func: F) -> F:`).
  - `lambda_cdm_analytic_distance` in `physical_relations.py` and
    `bayesian_inference_mwe.py`: removed `float()` cast (broke numpy 2.x when
    called with 1-D array from `fsolve`); `dist()` uses `np.asarray(...).flat[0]`
    for safe scalar extraction.
  - Fixed trailing-comma bug in `EMRIDetection.from_host_galaxy` that made
    `measured_luminosity_distance` and `measured_redshifted_mass` tuples instead
    of floats when `use_measurement_noise=False`.
  - Ruff-formatted all files to 100-character line length with isort ordering.

---

## [2026-03-09] — uv migration & Python 3.13 compatibility

### Added
- `pyproject.toml`: `[project]` section with dependency groups (`cpu`, `gpu`, `dev`),
  replacing the conda `environment.yml` as the authoritative dependency declaration.
- `uv.lock`: generated lock file replacing `conda-linux-64.lock`; commits exact resolved
  versions of all 150 transitive dependencies for bit-for-bit reproducibility.
- `.python-version`: pins to Python 3.13 (`fastlisaresponse` has no cp314 wheel yet).

### Changed
- Dependency manager switched from conda to [uv](https://docs.astral.sh/uv/).
  Motivation: faster installs, pure-pip workflow, `uv.lock` is simpler than
  `conda-lock`, and `fastemriwaveforms`/`fastlisaresponse` now ship cp313 wheels on PyPI.
- Updated `fastlisaresponse` 1.1.17 → 1.1.9 (latest stable with cp313 wheel; 1.1.17
  transitively pinned `numpy==1.26.0` via `lisaanalysistools`, which has no cp313 wheel).
- Updated `fastemriwaveforms` to 2.0.0rc1 (latest release; removes the numpy pin and
  ships a cp313 wheel on PyPI for the CPU variant).
- Scientific stack updated to current versions with cp313 wheels: numpy 2.4.3,
  scipy 1.17.1, matplotlib 3.10.8, pandas 3.0.1.
- `CLAUDE.md` Environment Setup section replaced with uv instructions.

### Fixed
- `BayesianInference` dataclass: `redshift_values` and `galaxy_distribution_at_redshifts`
  fields used bare `np.array([])` as defaults. Python 3.13 now explicitly rejects mutable
  defaults in dataclasses; replaced with `field(default_factory=lambda: np.array([]))`.

---

## [2025-05-08] — cosmological model & galaxy catalog refinements

### Changed
- `cosmological_model.py`: minor tuning of detection probability evaluation logic and
  integration limits in `BayesianStatistics`.
- `galaxy_catalogue/handler.py`: small adjustment to host-galaxy lookup parameters.

---

## [2025-05-04] — bugfix: detection probability (second round)

### Fixed
- `cosmological_model.py`: added plot of interpolated detection probability surface
  alongside the directly-evaluated one, to verify the interpolation is faithful.
  The root cause of the earlier divergence between the two was confirmed fixed.

---

## [2025-04-28] — bugfix: detection probability

### Fixed
- `cosmological_model.py`: phi boundary check was inverted (`phi >= 0` should be
  `phi < 0`); valid azimuth range is `[0, 2π)` so the out-of-range guard was
  accepting invalid values and rejecting valid ones.
- `cosmological_model.py`: `kde.evaluate(...)` returns a length-1 array, not a scalar;
  added `[0]` indexing so the detection probability is a float rather than an array,
  preventing silent broadcasting errors downstream.

### Added
- `cosmological_model.py`: `plot_detection_probability()` method for visual sanity-checking
  of the KDE-based detection probability over the (`d_L`, `M`, `φ`, `θ`) parameter space.

---

## [2025-04-30] — performance improvements & physical relations refactor

### Changed
- `physical_relations.py`: `dist()` now uses an analytic closed-form expression
  (`lambda_cdm_analytic_distance`) instead of the numerical `np.trapz` integral over
  redshift. Faster and avoids discretisation error.
- `physical_relations.py`: added `cached_dist()` with `@lru_cache(maxsize=1000)` so
  repeated calls at the same redshift/cosmology parameters hit the cache instead of
  recomputing the integral. Significant speedup for the inference loop.
- `cosmological_model.py`: extensive rework of the Bayesian inference evaluation loop;
  likelihood computation restructured around interpolated detection-probability functions
  rather than repeated KDE evaluation calls.
- `galaxy_catalogue/handler.py`: added `HostGalaxy.__eq__` and `__hash__` based on
  `catalog_index`, enabling deduplication of host candidates with a set; added
  `HostGalaxy.from_attributes()` classmethod for constructing instances without a
  full catalog row.

---

## [2025-04-25] — BallTree catalog lookups; inference via interpolated functions

### Changed
- `galaxy_catalogue/handler.py`: replaced linear-scan host-galaxy search with a
  scikit-learn `BallTree` on (φ, θ) sky coordinates. Lookup complexity drops from O(N)
  to O(log N) per query; dominant cost for large catalogs.
- `bayesian_inference/bayesian_inference_mwe.py`: inference now evaluates detection
  probability via interpolated functions over a precomputed grid instead of drawing
  Monte Carlo samples. Removes sample-size variance from the posterior and speeds up
  each likelihood call.
- `cosmological_model.py`: significant reduction in size (1 896 → ~300 lines) by
  removing dead evaluation scripts and consolidating the H₀ inference driver into
  `BayesianStatistics.evaluate()`.
- Tests in `test_bayesian_inference_mwe.py` updated to match the new function-based API.
