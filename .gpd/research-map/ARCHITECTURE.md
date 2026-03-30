# Computational Architecture

**Analysis Date:** 2026-03-30

## Computational Pipeline Overview

Two distinct pipelines share a common physical model layer:

**Pipeline 1 — EMRI Simulation (GPU, data generation):**
1. Sample (M, z) from cosmological event-rate model via MCMC
   - Class: `Model1CrossCheck` in `master_thesis_code/cosmological_model.py` (line 56+)
   - Sampler: `emcee` ensemble MCMC with polynomial merger rate fits
2. Resolve host galaxy from GLADE catalog via BallTree spatial lookup
   - Class: `GalaxyCatalogueHandler` in `master_thesis_code/galaxy_catalogue/handler.py`
   - Algorithm: `sklearn.neighbors.BallTree` for angular nearest-neighbor queries
3. Randomize 14-parameter EMRI space, set host galaxy parameters
   - Class: `ParameterSpace` in `master_thesis_code/datamodels/parameter_space.py`
   - 14 parameters: M, mu, a, p0, e0, x0, luminosity_distance, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0
4. Generate LISA TDI waveform (GPU-accelerated)
   - Method: `ParameterEstimation.generate_lisa_response()` in `master_thesis_code/parameter_estimation/parameter_estimation.py` (line 128)
   - Waveform backend: `few.waveform.GenerateEMRIWaveform` with Pn5AAKWaveform model
   - LISA response: `fastlisaresponse.ResponseWrapper` with ESA orbits, 1st generation TDI, A/E channels
   - Configuration: `master_thesis_code/waveform_generator.py` (Lagrangian interpolation order 35)
5. Compute SNR via noise-weighted inner product
   - Method: `ParameterEstimation.compute_signal_to_noise_ratio()` (line 426)
   - Hot path: `scalar_product_of_functions()` (line 281) -- GPU FFT + PSD division + trapezoidal integration
   - Quick check: 1-year generator first, skip if SNR < 0.2 * threshold
6. If SNR >= threshold: compute Fisher information matrix and Cramer-Rao bounds
   - Method: `compute_fisher_information_matrix()` (line 350) -> `compute_Cramer_Rao_bounds()` (line 382)
   - Derivatives: 5-point stencil (4 waveform evaluations per parameter, 56 total for 14 params)
   - Fisher matrix: symmetric 14x14, upper triangle only (105 inner products)
   - Inversion: `np.linalg.inv` on CPU (14x14 matrix transferred from GPU)
   - Condition number logged before inversion; negative diagonal detection raises `ParameterEstimationError`
7. Save Cramer-Rao bounds to CSV (buffered write, flush on SIGTERM)
   - Method: `save_cramer_rao_bound()` (line 437), `flush_pending_results()` (line 460)

**Pipeline 2 — Bayesian Inference (CPU, H0 posterior):**
1. Load merged Cramer-Rao bounds CSV
   - Class: `BayesianStatistics` in `master_thesis_code/bayesian_inference/bayesian_statistics.py` (line 69)
2. Build KDE detection probability from detected/undetected events
   - Class: `DetectionProbability` in `master_thesis_code/bayesian_inference/detection_probability.py` (line 16)
   - Algorithm: `scipy.stats.gaussian_kde` (4D with BH mass, 3D without)
   - Lookup: `scipy.interpolate.RegularGridInterpolator` on 40x50x20x20 grid
3. Construct multivariate-normal GW likelihoods from Fisher covariance
   - `scipy.stats.multivariate_normal` for each detection (3D and 4D variants)
4. Evaluate per-detection posteriors over H0 grid using multiprocessing
   - `multiprocessing.Pool` with "spawn" context
   - Workers: `os.sched_getaffinity(0) - 2` by default (CLI override via `--num_workers`)
   - Integration: `scipy.integrate.dblquad`, `quad`, `fixed_quad` inside worker functions
5. Output JSON posteriors to `simulations/posteriors/`

## Solver Stack

### Waveform Generation (GPU)
- **Library:** `fastemriwaveforms` (imported as `few`), version 2.0.0rc1
- **Model:** `Pn5AAKWaveform` (5PN augmented analytic kludge) -- default
- **Alternative:** `FastSchwarzschildEccentricFlux` (fully relativistic Schwarzschild) -- selectable via `WaveGeneratorType` enum
- **LISA response:** `fastlisaresponse.ResponseWrapper` version 1.1.9+
  - Lagrangian interpolation order: 35
  - TDI generation: 1st generation, channels A/E
  - Orbits: `lisatools.detector.ESAOrbits`
  - Backend: `cuda12x` on GPU, default on CPU
- **File:** `master_thesis_code/waveform_generator.py`

### FFT and Linear Algebra (GPU)
- **FFT:** `cupyx.scipy.fft.rfft` for frequency-domain waveforms
- **Frequency grid:** `cupyx.scipy.fft.rfftfreq`, cropped to [1e-5, 1] Hz analysis band
- **Integration:** `cupy.trapz` for frequency-domain inner products
- **PSD caching:** `ParameterEstimation._psd_cache` (keyed by waveform length `n`), avoids recomputing PSD for all 105 Fisher matrix inner products
- **Matrix inversion:** `numpy.linalg.inv` (14x14, transferred to CPU via `cp.asnumpy`)
- **File:** `master_thesis_code/parameter_estimation/parameter_estimation.py`

### Cosmological Distance Functions (CPU)
- **Luminosity distance:** Analytic hypergeometric form `scipy.special.hyp2f1` for flat LambdaCDM
  - `lambda_cdm_analytic_distance()` in `master_thesis_code/physical_relations.py` (line 240)
  - LRU-cached variant `cached_dist()` with 1000 entries (line 80)
- **Redshift inversion:** `scipy.optimize.fsolve` with initial guess z=1
  - `dist_to_redshift()` in `master_thesis_code/physical_relations.py` (line 271)
- **Hubble function E(z):** Full wCDM form but only LambdaCDM branch is exercised (w0=-1, wa=0 hardcoded)
  - `hubble_function()` in `master_thesis_code/physical_relations.py` (line 198)

### Galaxy Catalog Lookup (CPU)
- **Algorithm:** `sklearn.neighbors.BallTree` for angular nearest-neighbor queries
- **Catalog:** GLADE reduced galaxy catalog (`master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv`)
- **Black hole mass relation:** M-sigma relation with log-normal scatter (alpha=7.45*ln(10), beta=1.05)
- **File:** `master_thesis_code/galaxy_catalogue/handler.py`

### Bayesian Integration (CPU, multiprocessing)
- **Outer integration:** `scipy.integrate.dblquad` (redshift x BH mass)
- **Inner integration:** `scipy.integrate.quad`, `scipy.integrate.fixed_quad`
- **KDE:** `scipy.stats.gaussian_kde` for detection probability surfaces
- **Interpolation:** `scipy.interpolate.RegularGridInterpolator` (4D grid: d_L x M_z x phi x theta)
- **Parallelization:** `multiprocessing.Pool` with spawn context, worker state initialized via `child_process_init` global variables
- **File:** `master_thesis_code/bayesian_inference/bayesian_statistics.py`

### MCMC Sampling (CPU)
- **Library:** `emcee` ensemble sampler
- **Used in:** `Model1CrossCheck.sample_emri_events()` for (M, z) event sampling from cosmological rate model
- **File:** `master_thesis_code/cosmological_model.py`

## Key Algorithms

### 5-Point Stencil Derivative (GPU)
- **Purpose:** Compute partial derivatives of LISA TDI waveforms w.r.t. 14 EMRI parameters
- **Method:** Central difference: `(-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / 12h`
- **Accuracy:** O(epsilon^4) per Vallisneri (2008), arXiv:gr-qc/0703086
- **Cost:** 4 waveform evaluations per parameter x 14 parameters = 56 waveform calls per Fisher matrix
- **Implementation:** `five_point_stencil_derivative()` at `parameter_estimation.py` (line 189)
- **Note:** Forward-difference method `finite_difference_derivative()` also exists (line 142) and is selectable via `use_five_point_stencil=True` constructor flag (default: True)

### Noise-Weighted Inner Product (GPU)
- **Formula:** `<h1|h2> = 4 Re sum_channels int_{f_min}^{f_max} h1*(f) h2(f) / S_n(f) df`
- **Implementation:** Vectorized batch FFT over channels, PSD division, trapezoidal integration
- **Frequency band:** [1e-5 Hz, 1 Hz] (constants `MINIMAL_FREQUENCY`, `MAXIMAL_FREQUENCY`)
- **Channels:** A, E (TDI channels from ESA configuration)
- **Call count per Fisher matrix:** 105 (upper triangle of 14x14 symmetric matrix)
- **Implementation:** `scalar_product_of_functions()` at `parameter_estimation.py` (line 281)

### PSD Noise Model (GPU/CPU)
- **Instrumental noise:** A/E channel PSD from Babak et al. (2023), arXiv:2303.15929
  - Components: `S_OMS` (optical metrology), `S_TM` (test mass)
  - LISA arm length: 2.5e9 m
- **Galactic confusion noise:** Cornish & Robson (2017), arXiv:1703.09858, Eq. (3)
  - Controlled by `include_confusion_noise` flag (default True) and `t_obs_years` (default 4.0)
- **Implementation:** `LisaTdiConfiguration` at `master_thesis_code/LISA_configuration.py`
- **GPU/CPU dispatch:** `_get_xp(arr)` helper inspects array type at runtime

## Performance Characteristics

### Identified Bottleneck
- **Hot path:** `scalar_product_of_functions()` -- called 105 times per Fisher matrix, each involving a full-length FFT pair and PSD division
- **Mitigation:** PSD caching (`_psd_cache` dict keyed by `n`), symmetric Fisher matrix (only upper triangle computed)

### Timeouts
- **Waveform/SNR computation:** 90-second SIGALRM timeout per event in `data_simulation()` (`main.py` line 256, 332)
- **Known issue:** Some EMRI parameter combinations cause indefinite hangs in `few`/`fastlisaresponse`

### GPU Memory Management
- **Class:** `MemoryManagement` in `master_thesis_code/memory_management.py`
- **Strategy:** `cp.get_default_memory_pool().free_all_blocks()` + FFT cache clear between simulation steps (NOT inside inner loops)
- **Monitoring:** `GPUtil.getGPUs()` for hardware memory, CuPy memory pool for allocator tracking

### Quick SNR Pre-Check
- 1-year observation waveform (`snr_check_generator`) computed first
- If quick_snr < 0.2 * SNR_THRESHOLD, skip full 5-year computation
- Saves ~80% of waveform computation for clearly undetectable events

## Data Flow

```
[EMRI Event Rate Model (emcee MCMC)]
        |
        v
[GLADE Galaxy Catalog (BallTree)] --> [HostGalaxy dataclass]
        |
        v
[14-Parameter EMRI Space (randomized)] --> [ParameterSpace dataclass]
        |
        v
[few Pn5AAK Waveform (GPU)] --> [fastlisaresponse TDI (GPU)]
        |
        v
[SNR = sqrt(<h|h>)] --[< threshold]--> [undetected_events_*.csv]
        |
        [>= threshold]
        v
[5-Point Stencil Derivatives (56 waveform evals)]
        |
        v
[14x14 Fisher Matrix (105 inner products)]
        |
        v
[Matrix Inversion (CPU, np.linalg.inv)]
        |
        v
[cramer_rao_bounds_simulation_$index.csv]
        |
    [merge script]
        v
[cramer_rao_bounds.csv] --> [prepare script] --> [prepared_cramer_rao_bounds.csv]
        |
        v
[BayesianStatistics (multiprocessing)]
  - KDE detection probability (scipy gaussian_kde + RegularGridInterpolator)
  - Multivariate normal GW likelihoods (scipy)
  - Numerical integration (dblquad, quad, fixed_quad)
        |
        v
[simulations/posteriors/h_*.json]
```

## GPU/CPU Portability Pattern

All GPU-capable code uses a guarded CuPy import and `_get_xp` helper:

```python
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

def _get_xp(use_gpu: bool) -> types.ModuleType:
    if use_gpu and _CUPY_AVAILABLE:
        return cp
    return np
```

- `LISA_configuration.py`: uses `_get_xp(arr)` variant (inspects array type)
- `parameter_estimation.py`: uses `_CUPY_AVAILABLE` flag and direct `cp`/`cufft` calls
- `decorators.py`, `memory_management.py`: guarded imports for monitoring only
- **Known issue:** `LISA_configuration.py` has unconditional `import cupy` at module level that makes it un-importable on CPU-only machines without the try/except guard

## Parallelization Strategy

| Pipeline | Mechanism | Granularity | Configuration |
|----------|-----------|-------------|---------------|
| EMRI Simulation | SLURM array jobs | 1 GPU per task | `cluster/simulate.sbatch`, `--array=0-N` |
| Fisher Matrix | Sequential (GPU) | Per-parameter derivatives, per-pair inner products | Single GPU stream |
| Bayesian Inference | `multiprocessing.Pool` (spawn) | Per-detection posterior evaluation | `--num_workers N` CLI flag |
| Event Sampling | Sequential | Batch of 200 events from `emcee` | `cosmological_model.py` |

## Signal Handling

- **SIGALRM:** 90-second timeout on waveform/SNR/CRB computation (`main.py` lines 197-199)
- **SIGTERM:** Graceful SLURM shutdown -- flushes buffered CRB results before exit (`main.py` lines 205-209)
- **Warnings as errors:** `warnings.filterwarnings("error")` during waveform generation to catch mass-ratio bounds violations

## Reproducibility

- `--seed` CLI argument sets numpy RNG; per-task seed = BASE_SEED + SLURM_ARRAY_TASK_ID on cluster
- `run_metadata.json` records: git commit, timestamp, random seed, all CLI args, SLURM environment variables
- `uv.lock` pins exact dependency versions

---

_Architecture analysis: 2026-03-30_
