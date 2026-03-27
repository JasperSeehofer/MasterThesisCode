# Architecture

**Analysis Date:** 2026-03-25

## Pattern Overview

**Overall:** Procedural pipeline with dataclass-based domain models

**Key Characteristics:**
- Two distinct pipelines sharing a common physical model layer
- GPU/CPU portability via guarded CuPy imports and `_get_xp()` helper
- CLI-driven entry point dispatching to pipeline functions
- Callback protocol for decoupling side effects (plotting, monitoring) from computation
- Multiprocessing in the Bayesian inference pipeline for parallelizing likelihood evaluation
- Heavy use of dataclasses for domain objects (parameters, detections, galaxies)

## Pipelines

### Pipeline 1: EMRI Simulation (data generation)

**Entry:** `main.py:data_simulation()` via `--simulation_steps N`

**Data Flow:**

1. `Model1CrossCheck` (in `cosmological_model.py`) samples EMRI events using `emcee` MCMC, producing `ParameterSample` objects with (M, redshift, a)
2. `GalaxyCatalogueHandler` (in `galaxy_catalogue/handler.py`) resolves each sample to a `HostGalaxy` using a BallTree lookup against the GLADE galaxy catalog
3. `ParameterSpace.randomize_parameters()` randomizes the 14 EMRI parameters, then `set_host_galaxy_parameters()` overwrites M, phiS, qS, and luminosity_distance from the host galaxy
4. `ParameterEstimation.compute_signal_to_noise_ratio()` generates a LISA TDI waveform via `few` + `fastlisaresponse` and computes SNR against the noise PSD
5. Quick SNR pre-check (1-year observation) gates the full 5-year computation
6. If SNR >= threshold (20): `compute_Cramer_Rao_bounds()` evaluates the 14x14 Fisher matrix via finite-difference derivatives, then saves results to CSV
7. Callbacks fire at each stage: `on_snr_computed`, `on_detection`, `on_step_end`, `on_simulation_end`

**Key constraint:** Steps 4-6 require GPU (CuPy + CUDA). The `ParameterEstimation` constructor calls `create_lisa_response_generator()` which imports `fastlisaresponse` and `few`.

### Pipeline 2: Bayesian Inference (H0 posterior evaluation)

**Entry:** `main.py:evaluate()` via `--evaluate [--h_value 0.73]`

**Data Flow:**

1. `BayesianStatistics` (in `bayesian_inference/bayesian_statistics.py`) loads Cramer-Rao bounds from CSV files in `simulations/`
2. Constructs `Detection` objects (from `datamodels/detection.py`) with full Fisher-matrix covariance
3. Builds `DetectionProbability` (in `bayesian_inference/detection_probability.py`) using KDE on detected/undetected events
4. For each H0 value in grid: evaluates `single_host_likelihood()` for each detection, integrating over galaxy catalog hosts using `scipy.integrate.quad/dblquad`
5. Multiprocessing pool distributes detection likelihoods across CPU cores via `child_process_init` + module-level globals
6. Output: posterior JSON written to `simulations/posteriors/` and `simulations/posteriors_with_bh_mass/`

**No GPU required** for Pipeline 2 -- runs on CPU only.

### Pipeline A (dev cross-check, not production)

**Entry:** `bayesian_inference/bayesian_inference_mwe.py` standalone or programmatic

A simplified Bayesian inference pipeline using:
- Synthetic `GalaxyCatalog` (from `datamodels/galaxy.py`) instead of GLADE
- erf-based analytic detection probability instead of KDE
- Hardcoded 10% fractional sigma(d_L) instead of per-source Cramer-Rao bounds
- `EMRIDetection` dataclass (from `datamodels/emri_detection.py`) instead of `Detection`

**Not invoked by `--evaluate`.** Exists for rapid prototyping.

## Layers

**CLI / Entry:**
- Purpose: Parse arguments, configure logging, dispatch to pipelines
- Location: `master_thesis_code/__main__.py`, `master_thesis_code/main.py`, `master_thesis_code/arguments.py`
- Contains: `main()`, `data_simulation()`, `evaluate()`, `snr_analysis()`, `generate_figures()`

**Cosmological Model:**
- Purpose: EMRI event rate model, MCMC sampling of (M, z) pairs
- Location: `master_thesis_code/cosmological_model.py`
- Contains: `Model1CrossCheck`, `LamCDMScenario`, `DarkEnergyScenario`, polynomial merger rate fits
- Depends on: `emcee`, `datamodels/parameter_space.py`, `physical_relations.py`, `M1_model_extracted_data/`

**Parameter Estimation (GPU):**
- Purpose: Waveform generation, SNR computation, Fisher matrix, Cramer-Rao bounds
- Location: `master_thesis_code/parameter_estimation/parameter_estimation.py`
- Contains: `ParameterEstimation` class (549 lines)
- Depends on: `few` (waveform), `fastlisaresponse` (LISA response), `cupy` (GPU arrays), `LISA_configuration.py` (PSD), `datamodels/parameter_space.py`

**Bayesian Inference:**
- Purpose: H0 posterior evaluation from saved Cramer-Rao bounds
- Location: `master_thesis_code/bayesian_inference/bayesian_statistics.py` (988 lines), `master_thesis_code/bayesian_inference/detection_probability.py` (344 lines)
- Contains: `BayesianStatistics`, `DetectionProbability`, `single_host_likelihood()`, multiprocessing workers
- Depends on: `scipy.integrate`, `scipy.stats`, `datamodels/detection.py`, `galaxy_catalogue/handler.py`, `physical_relations.py`

**Physical Relations:**
- Purpose: Cosmological distance functions (luminosity distance, redshift inversion, Hubble function)
- Location: `master_thesis_code/physical_relations.py`
- Contains: `dist()`, `dist_vectorized()`, `dist_derivative()`, `hubble_function()`, `lambda_cdm_analytic_distance()`, mass conversion utilities
- Depends on: `constants.py`, `scipy.special.hyp2f1`, `scipy.optimize.fsolve`

**LISA Configuration:**
- Purpose: LISA detector noise model (PSD for A/E/T channels), antenna patterns
- Location: `master_thesis_code/LISA_configuration.py`
- Contains: `LisaTdiConfiguration` dataclass with `power_spectral_density()`, `_get_xp()` helper
- Depends on: `constants.py`, `cupy` (guarded import)

**Data Models:**
- Purpose: Domain objects as dataclasses
- Location: `master_thesis_code/datamodels/`
- Contains:
  - `parameter_space.py`: `Parameter`, `ParameterSpace` (14 EMRI parameters), distribution functions
  - `detection.py`: `Detection` (parsed from Cramer-Rao CSV), `_sky_localization_uncertainty()`
  - `emri_detection.py`: `EMRIDetection` (Pipeline A synthetic detection)
  - `galaxy.py`: `Galaxy`, `GalaxyCatalog` (Pipeline A synthetic catalog)

**Galaxy Catalogue:**
- Purpose: Interface to GLADE galaxy catalog, BallTree spatial lookups
- Location: `master_thesis_code/galaxy_catalogue/handler.py` (669 lines)
- Contains: `GalaxyCatalogueHandler`, `HostGalaxy`, `ParameterSample`, `InternalCatalogColumns`
- Depends on: `sklearn.neighbors.BallTree`, `pandas`, `physical_relations.py`

**Plotting:**
- Purpose: All visualization, fully separated from computation
- Location: `master_thesis_code/plotting/`
- Contains: Factory functions `(data in, (fig, ax) out)` organized by topic
- Depends on: `matplotlib` only

**Constants:**
- Purpose: All physical constants, cosmological parameters, simulation config, file paths
- Location: `master_thesis_code/constants.py`
- Contains: Module-level constants (H, OMEGA_M, SNR_THRESHOLD, LISA hardware, file paths)

## Key Abstractions

**Parameter / ParameterSpace:**
- Purpose: Represents a single EMRI parameter with bounds, epsilon, distribution function, and current value
- Examples: `master_thesis_code/datamodels/parameter_space.py`
- Pattern: `@dataclass` with 14 `Parameter` fields, each wrapped in `field(default_factory=...)`. `randomize_parameters()` iterates all non-fixed parameters.

**SimulationCallback Protocol:**
- Purpose: Decouple side effects (plotting, monitoring) from the simulation loop
- Examples: `master_thesis_code/callbacks.py`, `master_thesis_code/plotting/simulation_plots.py` (`PlottingCallback`)
- Pattern: `typing.Protocol` with 5 hook methods. `data_simulation()` accepts `list[SimulationCallback]`.

**Detection / EMRIDetection:**
- Purpose: Represent a detected EMRI event with measured parameters and uncertainties
- Examples: `master_thesis_code/datamodels/detection.py` (Pipeline B, from CSV), `master_thesis_code/datamodels/emri_detection.py` (Pipeline A, synthetic)
- Pattern: `@dataclass` initialized from `pd.Series` row of Cramer-Rao CSV

**Galaxy / GalaxyCatalog vs HostGalaxy / GalaxyCatalogueHandler:**
- Two parallel galaxy representations:
  - `datamodels/galaxy.py`: `Galaxy` + `GalaxyCatalog` for Pipeline A (synthetic, in-memory, emcee-sampled)
  - `galaxy_catalogue/handler.py`: `HostGalaxy` + `GalaxyCatalogueHandler` for Pipelines 1 & B (GLADE catalog, BallTree lookups, CSV-backed)

## Entry Points

**CLI (primary):**
- Location: `master_thesis_code/__main__.py` -> `main.py:main()`
- Invocation: `python -m master_thesis_code <working_dir> [options]`
- Dispatches to: `data_simulation()`, `evaluate()`, `snr_analysis()`, `generate_figures()`

**Pipeline A standalone:**
- Location: `master_thesis_code/bayesian_inference/bayesian_inference_mwe.py` (has `__main__` block)
- Invocation: `python -m master_thesis_code.bayesian_inference.bayesian_inference_mwe`

**Utility scripts:**
- Location: `scripts/` directory
- `prepare_detections.py`: Post-process Cramer-Rao CSV for evaluation
- `merge_cramer_rao_bounds.py`: Merge per-index simulation CSVs
- `remove_detections_out_of_bounds.py`: Filter detections
- `estimate_hubble_constant.py`: Standalone H0 estimation

## Error Handling

**Strategy:** Exception-based with broad catch-and-continue in the simulation loop

**Patterns:**
- `ParameterOutOfBoundsError`: Raised during derivative computation when perturbed parameter exceeds bounds; caught in `data_simulation()` loop, iteration skipped
- `WaveformGenerationError`: Raised for invalid waveform generator configuration
- `Warning` catch: `warnings.filterwarnings("error")` converts waveform warnings (mass ratio out of bounds) to exceptions, caught and logged
- `RuntimeError`, `ValueError`, `AssertionError`: Caught from `few`/`fastlisaresponse` waveform generation failures; iteration skipped with logging
- No retry logic -- failed iterations are simply skipped and the loop continues

## GPU/CPU Portability

**Pattern:** Guarded CuPy imports with `_get_xp()` array namespace helper

**Implementation in `LISA_configuration.py`:**
```python
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

def _get_xp(arr):
    if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp
    return np
```

**Current state:** The `_get_xp()` pattern is implemented in `LISA_configuration.py` and `decorators.py`. However, `parameter_estimation.py` directly uses `cp.*` and `cufft.*` in hot paths (e.g., `_get_cached_psd`, `scalar_product_of_functions`), making it GPU-only. The `MemoryManagement` class in `memory_management.py` also guards CuPy but requires `GPUtil` unconditionally.

**Import chain constraint:** Any module importing `ParameterEstimation` transitively imports `LISA_configuration.py` -> CuPy (guarded) and `waveform_generator.py` -> `fastlisaresponse` (lazy, inside function). Tests that need `ParameterEstimation` must guard with `pytest.importorskip("cupy")`.

## Cross-Cutting Concerns

**Logging:** Standard library `logging` module. Root logger configured in `main.py:_configure_logger()` with both stream and file handlers. Log file written to working directory with timestamp. Per-module loggers via `logging.getLogger()`.

**Validation:** `Arguments.validate()` checks working directory and log level. `ParameterSpace` bounds checked during derivative computation. No schema validation on CSV files.

**Reproducibility:** `--seed` CLI argument seeds `np.random.default_rng()`. Seed, git commit, timestamp, and CLI args recorded in `run_metadata.json`. However, `ParameterSample` in `handler.py` still uses `np.random.default_rng()` without accepting an external seed.

**State Management:** Mutable state lives in `ParameterSpace.*.value` fields (mutated each iteration), `Model1CrossCheck._emri_event_sampler` (emcee state), and `ParameterEstimation._psd_cache` (computed lazily). `BayesianStatistics` uses module-level globals for multiprocessing worker state.

## Backward Compatibility

`cosmological_model.py` re-exports all symbols extracted to `bayesian_inference/bayesian_statistics.py` and `bayesian_inference/detection_probability.py` to maintain import compatibility. These re-exports include `BayesianStatistics`, `DetectionProbability`, `single_host_likelihood`, and numerous helper functions/constants.

---

*Architecture analysis: 2026-03-25*
