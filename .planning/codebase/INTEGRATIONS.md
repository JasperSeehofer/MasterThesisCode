# External Integrations

**Analysis Date:** 2026-03-25

## APIs & External Services

No external web APIs or cloud services are used. This is a standalone scientific computing application.

## Scientific Libraries

**Gravitational Wave Waveforms:**
- `fastemriwaveforms` (imported as `few`) - EMRI waveform generation
  - CPU variant: `fastemriwaveforms==2.0.0rc1` (extra `cpu`)
  - GPU variant: `fastemriwaveforms-cuda12x` (extra `gpu`)
  - Usage: `few.waveform.GenerateEMRIWaveform` in `master_thesis_code/waveform_generator.py`
  - Two waveform models: `FastSchwarzschildEccentricFlux` and `Pn5AAKWaveform`
  - Lazy-imported to avoid SIGILL on CPUs without AVX support

**LISA Instrument Response:**
- `fastlisaresponse==1.1.9` - LISA time-delay interferometry simulation
  - Usage: `fastlisaresponse.ResponseWrapper` in `master_thesis_code/waveform_generator.py`
  - Wraps waveform generators with LISA orbital mechanics
  - Requires orbit file: `esa-trailing-orbits.h5` (HDF5, bundled with package)
  - Lazy-imported (same SIGILL concern)

**GPU Computing:**
- `cupy-cuda12x` - NumPy-compatible GPU arrays (CUDA 12)
  - Usage: `master_thesis_code/parameter_estimation/parameter_estimation.py`, `master_thesis_code/LISA_configuration.py`, `master_thesis_code/decorators.py`, `master_thesis_code/memory_management.py`
  - `cupyx.scipy.fft` for GPU-accelerated FFT
  - Pattern: `xp = _get_xp(use_gpu)` resolves to cupy or numpy at runtime
  - Known issue: unconditional `import cupy` in several modules breaks CPU-only imports
- `GPUtil` - GPU utilization monitoring
  - Usage: `master_thesis_code/memory_management.py`

**MCMC Sampling:**
- `emcee` - Ensemble MCMC sampler
  - Usage: `emcee.EnsembleSampler` in `master_thesis_code/cosmological_model.py` (EMRI event rate sampling) and `master_thesis_code/datamodels/galaxy.py` (comoving volume sampling)

**Spatial Indexing:**
- `scikit-learn` (`sklearn.neighbors.BallTree`)
  - Usage: `master_thesis_code/galaxy_catalogue/handler.py`
  - Two BallTree instances: 2D sky position lookup and 4D (position + mass + redshift) lookup
  - Metric: Euclidean

**Physical Constants:**
- `astropy` (constants + units)
  - Usage: `master_thesis_code/constants.py` derives `C` (speed of light) and `G` (gravitational constant) from `astropy.constants`
  - Units converted via `astropy.units` at import time

**Statistics:**
- `scipy.stats` - `truncnorm`, `norm`, `multivariate_normal`, `gaussian_kde`, `rv_continuous`
  - Usage: galaxy mass distributions, redshift uncertainties, detection probability KDE
- `scipy.integrate` - `quad`, `dblquad`, `fixed_quad`, `cumulative_trapezoid`
  - Usage: likelihood integrals in `master_thesis_code/bayesian_inference/bayesian_statistics.py`
- `scipy.interpolate` - `CubicSpline`, `RegularGridInterpolator`
  - Usage: detection probability lookup tables in `master_thesis_code/bayesian_inference/detection_probability.py`
- `scipy.special` - `erf`, `hyp2f1`
  - Usage: detection probability (`erf`), luminosity distance (`hyp2f1`) in `master_thesis_code/physical_relations.py`
- `scipy.optimize` - `fsolve`
  - Usage: inverse distance-redshift relation

## Data Sources

**GLADE+ Galaxy Catalog:**
- Source file: `./master_thesis_code/galaxy_catalogue/GLADE+.txt` (not committed; must be provided)
- Reduced/cached version: `master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv` (committed)
- Loaded by: `master_thesis_code/galaxy_catalogue/handler.py` (`GalaxyCatalogueHandler`)
- Read via `pd.read_csv()` with chunked processing for the full GLADE+ file
- Columns used: right ascension, declination, redshift (measured/estimated), luminosity distance, stellar mass
- Black hole mass estimated via M-sigma relation (`alpha`, `beta` constants in `handler.py`)

**LISA Orbit Data:**
- File: `esa-trailing-orbits.h5` (HDF5, bundled with `fastlisaresponse`)
- Used by `ResponseWrapper` in `master_thesis_code/waveform_generator.py`

## File I/O Patterns

**CSV (via pandas):**
- Cramer-Rao bounds output: `simulations/cramer_rao_bounds_simulation_$index.csv` (per-run)
- Merged bounds: `simulations/cramer_rao_bounds.csv`
- Prepared bounds: `simulations/prepared_cramer_rao_bounds.csv`
- SNR analysis: `simulations/snr_analysis.csv`
- Undetected events: `simulations/undetected_events.csv`
- Galaxy catalog: `master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv`
- Read with: `pd.read_csv()` in `master_thesis_code/parameter_estimation/parameter_estimation.py`, `master_thesis_code/parameter_estimation/evaluation.py`, `master_thesis_code/bayesian_inference/bayesian_statistics.py`, `master_thesis_code/galaxy_catalogue/handler.py`
- Write with: `DataFrame.to_csv()` (append mode for incremental results)

**JSON:**
- Run metadata: `run_metadata.json` (written by `master_thesis_code/main.py` with git commit, timestamp, seed, CLI args)
- Posterior results: `simulations/posteriors/h_0_73.json`, `simulations/posteriors_with_bh_mass/h_0_73.json`
- Written with: `json.dump()` in `master_thesis_code/bayesian_inference/bayesian_statistics.py`

**Figures (via matplotlib):**
- Saved by: `master_thesis_code/plotting/_helpers.py:save_figure()`
- Default formats: PNG and PDF
- Output directory: configurable via `--generate_figures` CLI arg

**HDF5:**
- Only used indirectly: `esa-trailing-orbits.h5` read by `fastlisaresponse` internally

## CLI Interface

**Framework:** `argparse` (standard library)
- Implementation: `master_thesis_code/arguments.py`
- Entry point: `python -m master_thesis_code` (via `master_thesis_code/__main__.py`)

**Commands (mutually exclusive modes):**

| Argument | Type | Description |
|---|---|---|
| `working_directory` | positional str | Working directory for output files |
| `--simulation_steps N` | int (default 0) | Number of EMRI waveforms to simulate |
| `--simulation_index I` | int (default 0) | Index for unique CSV filename |
| `--evaluate` | flag | Run Bayesian inference on saved Cramer-Rao bounds |
| `--h_value` | float (default 0.73) | Hubble constant value for evaluation |
| `--snr_analysis` | flag | Run SNR analysis only |
| `--generate_figures DIR` | str | Output directory for thesis figures |
| `--seed` | int (optional) | Random seed for reproducibility |
| `--log_level` | str (default INFO) | Logging verbosity |

**Utility scripts (in `scripts/`):**
- `scripts/prepare_detections.py` - pre-process detection CSV files
- `scripts/merge_cramer_rao_bounds.py` - merge per-run CSV files
- `scripts/remove_detections_out_of_bounds.py` - filter detections
- `scripts/estimate_hubble_constant.py` - standalone H0 estimation

## Parallelism

**Multiprocessing:**
- `multiprocessing` (stdlib) with `spawn` context
  - Usage: `master_thesis_code/bayesian_inference/bayesian_statistics.py`
  - `mp.get_context("spawn").Pool()` for parallel likelihood evaluation over H0 grid
  - Module-level globals for worker state initialization (`child_process_init`)

**GPU Parallelism:**
- CuPy array operations (implicit GPU parallelism)
- `cupyx.scipy.fft` for GPU-accelerated FFT in `master_thesis_code/parameter_estimation/parameter_estimation.py`

## Monitoring & Observability

**Logging:**
- Python `logging` module (stdlib)
- Root logger configured in `master_thesis_code/main.py:_configure_logger()`
- Log to both console (StreamHandler) and file (FileHandler in working directory)
- Levels configurable via `--log_level` CLI argument

**GPU Memory:**
- `master_thesis_code/memory_management.py` - `MemoryManagement` class using `GPUtil` and `cupy` memory pool stats

**Run Provenance:**
- `run_metadata.json` written per run with: `git_commit`, `timestamp`, `random_seed`, CLI arguments

## Webhooks & Callbacks

**Incoming:** None

**Outgoing:** None

**Internal Callback System:**
- `master_thesis_code/callbacks.py` defines `SimulationCallback` Protocol
- `master_thesis_code/plotting/simulation_plots.py` implements `PlottingCallback`
- Used to decouple plotting side effects from the simulation loop in `main.py:data_simulation()`

---

*Integration audit: 2026-03-25*
