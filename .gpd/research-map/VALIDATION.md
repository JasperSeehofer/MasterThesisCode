# Validation and Cross-Checks

**Analysis Date:** 2026-03-30

## Analytic Cross-Checks

**Limiting Cases Verified:**

- dist(z=0) == 0.0: Expected 0.0 -> Obtained 0.0: Match Yes
  - File: `master_thesis_code_test/physical_relations_test.py::test_dist_at_zero_redshift` (line 53)
  - Also: `master_thesis_code_test/bayesian_inference/test_bayesian_inference_mwe.py::test_dist` with (0.0, 0.0) pair (line 30)

- hubble_function(z=0) == 1.0: Expected 1.0 -> Obtained 1.0 (within 1e-10): Match Yes
  - File: `master_thesis_code_test/physical_relations_test.py::test_hubble_function_at_zero` (line 67)
  - Physics: E(0) = sqrt(Omega_m + Omega_DE) = sqrt(1) = 1 for flat universe

- dist_to_redshift(0.0) == 0.0: Expected 0.0 -> Obtained ~0.0 (within 1e-6): Match Yes
  - File: `master_thesis_code_test/physical_relations_test.py::test_dist_to_redshift_at_zero` (line 81)

- dist(z=1) ~ 6.5 Gpc: Expected 6.5 -> Obtained 6.5 (rounded to 1 decimal): Match Yes
  - File: `master_thesis_code_test/physical_relations_test.py::test_dist` (line 18)
  - Note: This is a numerical check for h=0.73, Omega_m=0.25, not an analytic limit

- M_z / (1+z) == M round-trip: Expected identity -> Verified: Match Yes
  - File: `master_thesis_code_test/physical_relations_test.py::test_convert_redshifted_mass_to_true_mass` (line 114)
  - Also: `test_mass_conversion_round_trip` (line 135)

- Omega_m + Omega_DE == 1 (flat universe): Expected 1.0 -> Obtained 1.0 (within 1e-10): Match Yes
  - File: `master_thesis_code_test/test_constants.py::test_flat_universe` (line 15)

**Limiting Cases NOT Verified:**

- dist(z) proportional to 1/H_0: Partially tested at z=0.1 with 2% tolerance
  - File: `master_thesis_code_test/physical_relations_test.py::test_dist_hubble_scaling_approximate` (line 163)
  - Missing: Exact proportionality limit at z -> 0 (Hubble law d_L ~ cz/H_0)

- PSD positivity: Not tested. power_spectral_density(f) > 0 for all valid f is not checked in any test.
  - Recommended: Add `test_psd_positive` parametrized over frequency range [1e-5, 1] Hz.

- Inner product positivity <h|h> > 0: Not directly tested (requires GPU for waveform generation).
  - File: `master_thesis_code/parameter_estimation/parameter_estimation.py` (line 281)

- Fisher matrix positive semi-definiteness: Partially checked via diagonal positivity only.
  - File: `master_thesis_code/parameter_estimation/parameter_estimation.py` (lines 398-408)
  - Missing: Full eigenvalue positivity check.

- Non-relativistic Hubble law limit: dist(z << 1) ~ c*z/H_0 not tested.
  - Could be added to `physical_relations_test.py` with z = 0.001.

- Galaxy distribution normalization: Not tested. `evaluate_galaxy_distribution` should integrate to ~1 over the catalog.

- Detection probability in [0, 1]: Not tested for the KDE-based `DetectionProbability` class.
  - File: `master_thesis_code/bayesian_inference/detection_probability.py`

**Symmetry Checks:**

- dist is monotonically increasing in z: Verified
  - File: `master_thesis_code_test/physical_relations_test.py::test_dist_monotonically_increasing` (line 59)
  - Checks: dist(0.5) < dist(1.0) < dist(2.0)

- Higher H_0 -> smaller d_L: Verified
  - File: `master_thesis_code_test/physical_relations_test.py::test_dist_varies_with_hubble_constant` (line 151)

- Gaussian symmetry: gaussian(mu-delta) == gaussian(mu+delta): Verified
  - File: `master_thesis_code_test/cosmological_model_test.py::test_gaussian_symmetry` (line 57)

- dist/dist_to_redshift round-trip: Verified at z = 0.5, 1.0, 2.0 (within 1e-5)
  - File: `master_thesis_code_test/physical_relations_test.py::test_dist_round_trip` (line 87)

**Sum Rules / Consistency Relations:**

- dist_vectorized matches scalar dist: Verified element-wise within 1e-10
  - File: `master_thesis_code_test/physical_relations_test.py::test_dist_vectorized_matches_scalar` (line 101)

- EMRIDetection.from_host_galaxy preserves sky angles when noise=False: Verified
  - File: `master_thesis_code_test/datamodels/test_emri_detection.py::test_from_host_galaxy_sky_angles_preserved` (line 31)

## Numerical Validation

**Convergence Tests:**

- No explicit convergence tests exist in the test suite for:
  - Fisher matrix convergence with derivative step size epsilon
  - Inner product convergence with frequency grid resolution
  - Comoving volume MCMC sampling convergence (burn-in = 1000, production = number_of_samples/nwalkers)
  - Bayesian posterior convergence with number of H_0 grid points

**Stability Analysis:**

- Parameter randomization deterministic with fixed seed: Verified
  - File: `master_thesis_code_test/datamodels/parameter_space_test.py::test_randomize_parameters_deterministic_with_same_seed` (line 173)
  - Different seeds produce different output: Verified (line 189)

- Fisher matrix condition number logged but not acted upon:
  - File: `master_thesis_code/parameter_estimation/parameter_estimation.py` (line 393)
  - No threshold or warning for ill-conditioned matrices (condition number logged but not checked)

- Negative CRB diagonal entries trigger exception:
  - File: `master_thesis_code/parameter_estimation/parameter_estimation.py` (lines 398-408)
  - This catches the most severe numerical instability symptom

**Precision and Error Control:**

- Derivative epsilon: 1e-6 for all 14 parameters (default)
  - File: `master_thesis_code/datamodels/parameter_space.py` (line 37)
  - No adaptive step-size selection. Same epsilon used for parameters spanning 12 orders of magnitude (M ~ 1e4-1e7 vs phases ~ 0-2*pi).

- MCMC burn-in for comoving volume sampling: 1000 steps, 5 walkers
  - File: `master_thesis_code/datamodels/galaxy.py` (lines 150-153)
  - No convergence diagnostic (Gelman-Rubin, autocorrelation time)

- Comoving distance integral: 1000-point trapezoidal rule for dist_derivative
  - File: `master_thesis_code/physical_relations.py` (line 189)
  - Analytic hypergeometric form used for dist() itself -- exact (no quadrature error)

- Frequency band: [1e-5, 1] Hz (MINIMAL_FREQUENCY, MAXIMAL_FREQUENCY)
  - File: `master_thesis_code/constants.py` (lines 46-47)

## Comparison with Literature

**Reproduced Results:**

- Luminosity distance formula matches Hogg (1999) arXiv:astro-ph/9905116 Eq. (16)
  - File: `master_thesis_code/physical_relations.py` (line 63, docstring reference)
  - d_L(z=1, h=0.73, Om=0.25) = 6.5 Gpc -- consistent with standard cosmology calculators

- Comoving volume element matches Hogg (1999) Eq. (27)
  - File: `master_thesis_code/datamodels/galaxy.py` (line 125, comment)

- LISA A/E-channel PSD follows Babak et al. (2023) arXiv:2303.15929
  - File: `master_thesis_code/LISA_configuration.py` (line 76, docstring)

- Galactic confusion noise follows Cornish & Robson (2017) arXiv:1703.09858 Eq. (3) / Robson et al. (2019) arXiv:1803.01944 Eq. (14)
  - File: `master_thesis_code/LISA_configuration.py` (lines 83-85)

- 5-point stencil derivative formula references Vallisneri (2008) arXiv:gr-qc/0703086
  - File: `master_thesis_code/parameter_estimation/parameter_estimation.py` (line 242)

**Discrepancies:**

- Cosmological parameters (Omega_m=0.25, H=0.73) are WMAP-era values, not Planck 2018 best-fit (Omega_m=0.3153, H=0.6736)
  - File: `master_thesis_code/constants.py` (lines 25, 29)
  - Impact: Distance calculations differ by ~5-10% from Planck cosmology. This is deliberate (matching the original thesis setup) but should be documented.

- T-channel PSD is marked "NOT UPDATED" in the code
  - File: `master_thesis_code/LISA_configuration.py` (line 142)
  - Impact: T-channel is not used in the A/E analysis (ESA_TDI_CHANNELS = "AE") so this is dormant code.

## Internal Consistency

**Cross-Method Verification:**

- Forward-difference vs 5-point stencil derivatives: Both methods exist but are NOT cross-checked against each other in any test
  - Method A: `finite_difference_derivative()` -- `parameter_estimation.py` (line 142)
  - Method B: `five_point_stencil_derivative()` -- `parameter_estimation.py` (line 189)
  - The `use_five_point_stencil` flag (default True) selects which is used. No test verifies they produce consistent results.

- Pipeline A vs Pipeline B: Two independent Bayesian inference implementations exist
  - Pipeline A: `bayesian_inference/bayesian_inference.py` -- erf detection probability, 10% sigma(d_L), synthetic catalog
  - Pipeline B: `bayesian_inference/bayesian_statistics.py` -- KDE detection probability, Fisher-matrix sigma(d_L), GLADE catalog
  - No cross-check between the two pipelines is performed in any test.

- `dist()` vs `dist_vectorized()`: Verified element-wise match
  - File: `master_thesis_code_test/physical_relations_test.py::test_dist_vectorized_matches_scalar` (line 101)

**Self-Consistency:**

- Fisher matrix symmetry: Exploited (only upper triangle computed and mirrored)
  - File: `master_thesis_code/parameter_estimation/parameter_estimation.py` (lines 365-375)
  - Not independently verified that the transposed elements match.

- CRB stores lower triangle only, consistent with the Fisher matrix symmetry
  - File: `parameter_estimation.py` (lines 413-419)

## Test Suite

**Existing Tests:**

- `master_thesis_code_test/test_constants.py` (5 tests): Physical constant consistency
  - Coverage: flat universe, speed of light, unit conversions

- `master_thesis_code_test/physical_relations_test.py` (15 tests): Distance functions, Hubble function, mass conversions
  - Coverage: Zero-redshift limits, monotonicity, round-trips, H_0 scaling, error propagation, vectorization

- `master_thesis_code_test/datamodels/parameter_space_test.py` (12 tests): Parameter space construction, randomization, bounds
  - Coverage: Distribution bounds, dict keys/types, deterministic seeding, host galaxy parameter setting

- `master_thesis_code_test/datamodels/test_detection.py` (6 tests): Detection parsing from CSV
  - Coverage: Field extraction, uncertainty from variance, relative error, sky localization

- `master_thesis_code_test/datamodels/test_emri_detection.py` (4 tests): EMRIDetection from Galaxy
  - Coverage: No-noise float fields, positivity, sky angle preservation, with-noise positivity

- `master_thesis_code_test/cosmological_model_test.py` (10 tests): Gaussian, polynomial helpers, MBH spin, Detection
  - Coverage: Mathematical utility functions, Detection construction and field access

- `master_thesis_code_test/bayesian_inference/test_bayesian_inference_mwe.py` (~30 tests): Pipeline A end-to-end
  - Coverage: Galaxy catalog creation, distribution evaluation, likelihood computation, detection probability, BayesianInference posterior

- `master_thesis_code_test/plotting/test_style.py` (9 tests): Matplotlib style application
  - Coverage: Style parameters, Agg backend, figure factory

- `master_thesis_code_test/test_benchmarks.py` (2 tests, slow): Performance benchmarks
  - Coverage: Likelihood evaluation, galaxy distribution evaluation

- `master_thesis_code_test/test_main_metadata.py` (6 tests): Run metadata and SLURM env capture
  - Coverage: Metadata JSON writing, SLURM variable capture, indexed filenames

- `master_thesis_code_test/scripts/test_merge_cramer_rao_bounds.py` (7 tests): CSV merge script
  - Coverage: Merge with/without delete, empty input, undetected events

- `master_thesis_code_test/scripts/test_prepare_detections.py`: Detection preparation script tests

- `master_thesis_code_test/test_arguments.py`: CLI argument parsing tests

- `master_thesis_code_test/test_memory_management.py`: Memory management utility tests

- `master_thesis_code_test/integration/test_evaluation_pipeline.py` (integration, marked): Pipeline B integration test
  - Coverage: Posterior structure, JSON output, posterior narrowing with more detections

- `master_thesis_code_test/decorators_test.py`: Timer decorator tests

**Older-style test files (module_test.py naming):**

- `master_thesis_code_test/physical_relations_test.py`
- `master_thesis_code_test/cosmological_model_test.py`
- `master_thesis_code_test/decorators_test.py`
- `master_thesis_code_test/datamodels/parameter_space_test.py`

**Run Commands:**

```bash
uv run pytest -m "not gpu and not slow"   # Fast CPU tests (default dev workflow)
uv run pytest                              # All tests (cluster with GPU)
uv run pytest -m slow --benchmark-only     # Benchmarks only
uv run pytest master_thesis_code_test/physical_relations_test.py  # Single file
```

**Test Patterns:**

```python
# Analytic limit pattern (preferred for physics validation)
def test_dist_at_zero_redshift() -> None:
    """Fundamental analytical limit: dist(0) == 0.0."""
    result = dist(0)
    assert result == 0.0

# Parametrized bounds checking pattern
@pytest.mark.parametrize("lower, upper", [(0.0, 1.0), (-5.0, 5.0)])
def test_uniform_in_bounds(lower: float, upper: float) -> None:
    rng = np.random.default_rng(42)
    for _ in range(100):
        value = uniform(lower, upper, rng)
        assert lower <= value <= upper

# Round-trip consistency pattern
@pytest.mark.parametrize("z", [0.5, 1.0, 2.0])
def test_dist_round_trip(z: float) -> None:
    d = dist(z)
    z_recovered = dist_to_redshift(d)
    assert abs(z_recovered - z) < 1e-5

# GPU/CPU dual-backend fixture pattern
@pytest.fixture(params=["numpy"] + (["cupy"] if _CUPY_AVAILABLE else []))
def xp(request):
    if request.param == "cupy":
        return cp
    return np
```

**Missing Tests (Priority Order):**

1. **PSD positivity** -- power_spectral_density(f) > 0 for all f in [1e-5, 1] Hz. No test exists.
2. **Inner product positivity** -- <h|h> > 0. Requires GPU (marked `@pytest.mark.gpu`).
3. **Fisher matrix eigenvalue positivity** -- All eigenvalues > 0 for well-conditioned cases. Requires GPU.
4. **Forward-difference vs 5-point stencil agreement** -- Cross-check derivative methods. Requires GPU.
5. **Detection probability in [0,1]** -- KDE-based DetectionProbability bounds. No test exists.
6. **Hubble law limit** -- dist(z << 1) ~ c*z/H_0 within ~z^2 corrections. Easy to add.
7. **Comoving volume element positivity and normalization** -- Not tested.
8. **Bayesian posterior normalization** -- integral p(H_0 | data) dH_0 should be finite and normalizable.
9. **wCDM silently ignored** -- Test that dist(z, w_0=-0.9) == dist(z, w_0=-1.0) to document the limitation.

## Reproducibility

**Random Seeds:**

- Fixed via `--seed` CLI argument: Supported since Phase 2 gap fixes
  - File: `master_thesis_code/arguments.py` (CLI argument)
  - File: `master_thesis_code/main.py` (seeds numpy, writes `run_metadata.json`)
- On cluster: per-task seed = BASE_SEED + SLURM_ARRAY_TASK_ID
  - File: `cluster/simulate.sbatch`
- Run metadata (git commit, timestamp, seed, all CLI args) recorded in `run_metadata.json`
  - Tested: `master_thesis_code_test/test_main_metadata.py`

**Platform Dependence:**

- GPU (CuPy) vs CPU (NumPy) may produce slightly different floating-point results due to different reduction order in FFT and trapezoid integration
- `few` waveform generation is GPU-only in practice (SIGILL on some CPUs without AVX)
  - Guarded import in `waveform_generator.py` (line 49)
- fastlisaresponse lazy-imported to avoid SIGILL on CPU-only machines
  - File: `master_thesis_code/waveform_generator.py` (line 49)

**Version Pinning:**

- All dependencies pinned via `uv.lock` (committed, ~4220 lines)
  - File: `uv.lock`
- Python version pinned to 3.13 via `.python-version`
- Key physics packages:
  - `fastemriwaveforms==2.0.0rc1`
  - `fastlisaresponse==1.1.9`
  - `astropy>=6.1.7`
  - `cupy-cuda12x` (GPU only)

---

_Validation analysis: 2026-03-30_
