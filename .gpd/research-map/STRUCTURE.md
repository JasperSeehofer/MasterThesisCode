# Project Structure

**Analysis Date:** 2026-03-30

## Directory Layout

```
MasterThesisCode/
+-- master_thesis_code/                # Main Python package
|   +-- __init__.py                    # Empty
|   +-- __main__.py                    # Entry point: calls main.py:main()
|   +-- main.py                        # Top-level pipeline dispatch
|   +-- arguments.py                   # CLI argument parsing
|   +-- constants.py                   # All physical constants, paths, thresholds
|   +-- physical_relations.py          # Cosmological distance functions (dist, hubble_function)
|   +-- LISA_configuration.py          # LISA PSD noise model, antenna patterns
|   +-- waveform_generator.py          # few/fastlisaresponse waveform setup
|   +-- cosmological_model.py          # EMRI event rate model (Model1CrossCheck)
|   +-- callbacks.py                   # SimulationCallback Protocol
|   +-- decorators.py                  # timer_decorator
|   +-- exceptions.py                  # Custom exception classes
|   +-- memory_management.py           # GPU memory monitoring
|   +-- parameter_estimation/          # Waveform-based parameter estimation
|   |   +-- __init__.py
|   |   +-- parameter_estimation.py    # ParameterEstimation: Fisher matrix, SNR, CRB (600 lines)
|   |   +-- evaluation.py             # Post-processing evaluation helpers
|   +-- bayesian_inference/            # H0 posterior evaluation
|   |   +-- __init__.py
|   |   +-- bayesian_statistics.py     # Pipeline B (production): BayesianStatistics (~986 lines)
|   |   +-- detection_probability.py   # KDE-based DetectionProbability (~344 lines)
|   |   +-- bayesian_inference.py      # Pipeline A (dev cross-check): BayesianInference
|   |   +-- bayesian_inference_mwe.py  # Re-export shim + __main__ block
|   +-- datamodels/                    # Domain dataclasses
|   |   +-- __init__.py
|   |   +-- parameter_space.py         # 14-parameter ParameterSpace + Parameter dataclass
|   |   +-- detection.py               # Detection (from CRB CSV) + _sky_localization_uncertainty
|   |   +-- emri_detection.py          # EMRIDetection (Pipeline A synthetic)
|   |   +-- galaxy.py                  # Galaxy + GalaxyCatalog (Pipeline A synthetic)
|   +-- galaxy_catalogue/              # GLADE galaxy catalog interface
|   |   +-- __init__.py
|   |   +-- handler.py                 # GalaxyCatalogueHandler, HostGalaxy, BallTree (669 lines)
|   |   +-- parser.py                  # Catalog file parsing
|   |   +-- glade_completeness.py      # Catalog completeness estimates
|   |   +-- reduced_galaxy_catalogue.csv  # Reduced GLADE catalog data file
|   +-- M1_model_extracted_data/       # Pre-computed EMRI rate model data
|   |   +-- detection_horizon.py
|   |   +-- detection_distribution_simplified.py
|   |   +-- emri_distribution.py
|   |   +-- detection_fraction.py      # DetectionFraction class
|   +-- plotting/                      # All visualization (factory functions)
|   |   +-- __init__.py                # Re-exports: apply_style, get_figure, save_figure
|   |   +-- _style.py                  # Agg backend + emri_thesis.mplstyle
|   |   +-- _helpers.py                # save_figure(), get_figure()
|   |   +-- simulation_plots.py        # PlottingCallback for simulation loop
|   |   +-- bayesian_plots.py          # Bayesian inference visualization
|   |   +-- evaluation_plots.py        # Evaluation pipeline plots
|   |   +-- model_plots.py             # Cosmological model plots
|   |   +-- catalog_plots.py           # Galaxy catalog plots
|   |   +-- physical_relations_plots.py # Distance-redshift relation plots
|   +-- waveform_generation/           # (Subpackage, currently empty __init__)
+-- master_thesis_code_test/           # Test suite (mirrors source layout)
|   +-- conftest.py                    # Root conftest (xp fixture, style fixture)
|   +-- test_constants.py
|   +-- test_benchmarks.py             # Slow benchmarks
|   +-- physical_relations_test.py
|   +-- cosmological_model_test.py
|   +-- decorators_test.py
|   +-- datamodels/
|   |   +-- test_detection.py
|   |   +-- test_emri_detection.py
|   |   +-- parameter_space_test.py
|   +-- parameter_estimation/
|   |   +-- parameter_estimation_test.py  # GPU-gated tests
|   +-- bayesian_inference/
|   |   +-- test_bayesian_inference_mwe.py
|   +-- plotting/
|   |   +-- test_style.py              # 9 style tests
|   +-- integration/                   # Integration tests
|   +-- scripts/                       # Script tests
|   +-- fixtures/                      # Test data fixtures
|       +-- evaluation/                # Evaluation test data
+-- scripts/                           # CLI utility scripts
|   +-- merge_cramer_rao_bounds.py     # emri-merge entry point
|   +-- prepare_detections.py          # emri-prepare entry point
|   +-- estimate_hubble_constant.py    # Standalone H0 estimation
|   +-- remove_detections_out_of_bounds.py
|   +-- compare_validation_runs.py     # Validation campaign comparison
+-- cluster/                           # SLURM HPC deployment
|   +-- modules.sh                     # Environment modules (exports $WORKSPACE, etc.)
|   +-- setup.sh                       # First-time cluster setup
|   +-- simulate.sbatch                # GPU array job (1 GPU/task, H100 partition)
|   +-- merge.sbatch                   # CPU job: merge per-task CSVs
|   +-- evaluate.sbatch                # CPU job: Bayesian inference (16 CPUs)
|   +-- submit_pipeline.sh             # Pipeline orchestrator (simulate -> merge -> evaluate)
|   +-- resubmit_failed.sh             # Resubmit failed SLURM array tasks
|   +-- vpn.sh                         # University VPN helper
|   +-- README.md                      # Cluster deployment guide
+-- evaluation/                        # Saved evaluation run data
|   +-- run_20260328_seed100_v3/       # Example run with logs + simulations subdirs
+-- notebooks/                         # Jupyter exploration notebooks
|   +-- parameter_estimation.ipynb
|   +-- parameter_estimation_schwarzschild.ipynb
+-- docs/                              # Sphinx documentation
|   +-- source/                        # Documentation source (conf.py, api/)
|   +-- build/                         # Generated HTML docs
|   +-- saved_figures/                 # Pre-generated figures for docs
+-- .github/workflows/                 # CI/CD (GitHub Actions)
+-- .planning/                         # GSD planning artifacts
|   +-- phases/                        # Phase plans (01-05, 09-11)
|   +-- milestones/
|   +-- research/
+-- pyproject.toml                     # Single source of truth: deps, tools, config
+-- uv.lock                            # Locked dependencies (committed)
+-- .python-version                    # Pins Python 3.13
+-- .pre-commit-config.yaml            # ruff lint/format + mypy hooks
+-- .editorconfig                      # 4-space indent, UTF-8, LF
+-- TODO.md                            # Known issues and planned work
+-- CHANGELOG.md                       # Version history
+-- CONTRIBUTING.md                    # Contributor guide
+-- README.md                          # Project overview
```

## Directory Purposes

**`master_thesis_code/`:**
- Purpose: Main Python package containing all production code
- Contains: `.py` files organized by computational concern
- Key files: `main.py` (entry), `constants.py` (configuration), `parameter_estimation/parameter_estimation.py` (core GPU computation)

**`master_thesis_code/parameter_estimation/`:**
- Purpose: GPU-accelerated waveform generation, Fisher matrix, SNR, and Cramer-Rao bounds
- Contains: Core computational hot path (`parameter_estimation.py`, 600 lines)
- Key dependency: Requires `cupy`, `few`, `fastlisaresponse` (GPU)

**`master_thesis_code/bayesian_inference/`:**
- Purpose: H0 posterior evaluation from saved Cramer-Rao bounds
- Contains: Production pipeline (`bayesian_statistics.py`, ~986 lines), detection probability (`detection_probability.py`, ~344 lines), dev cross-check (`bayesian_inference.py`)
- Key dependency: CPU-only (scipy, multiprocessing)

**`master_thesis_code/datamodels/`:**
- Purpose: Domain objects as Python dataclasses
- Contains: `ParameterSpace` (14 EMRI params), `Detection` (from CSV), `EMRIDetection` (synthetic), `Galaxy`/`GalaxyCatalog` (Pipeline A)

**`master_thesis_code/galaxy_catalogue/`:**
- Purpose: Interface to the GLADE galaxy catalog for host galaxy resolution
- Contains: `handler.py` (BallTree queries, 669 lines), `reduced_galaxy_catalogue.csv` (data)

**`master_thesis_code/M1_model_extracted_data/`:**
- Purpose: Pre-computed EMRI rate model data and polynomial fits
- Contains: Detection horizon, distribution, and fraction data modules

**`master_thesis_code/plotting/`:**
- Purpose: All visualization, fully decoupled from computation
- Contains: Factory functions `(data in, (fig, ax) out)` organized by topic
- Pattern: `_style.py` sets Agg backend + mplstyle; `_helpers.py` provides `save_figure()`

**`scripts/`:**
- Purpose: CLI utility scripts for post-processing simulation output
- Entry points registered in `pyproject.toml`: `emri-merge`, `emri-prepare`

**`cluster/`:**
- Purpose: SLURM job scripts and environment setup for bwUniCluster 3.0
- Contains: sbatch job scripts, shell helpers, pipeline orchestrator

**`evaluation/`:**
- Purpose: Saved simulation and evaluation run outputs
- Contains: Run directories with `logs/` and `simulations/` subdirectories
- Not committed to git (output data)

## Key File Locations

**Core Computation:**
- `master_thesis_code/parameter_estimation/parameter_estimation.py`: Fisher matrix, SNR, Cramer-Rao bounds (GPU hot path)
- `master_thesis_code/LISA_configuration.py`: LISA PSD noise model (A/E/T channels)
- `master_thesis_code/waveform_generator.py`: few/fastlisaresponse waveform configuration
- `master_thesis_code/physical_relations.py`: Cosmological distance functions (d_L, E(z), z inversion)

**Configuration / Parameters:**
- `master_thesis_code/constants.py`: ALL constants (physical, cosmological, LISA hardware, file paths, thresholds)
- `master_thesis_code/datamodels/parameter_space.py`: 14-parameter EMRI space definition with bounds and epsilons
- `master_thesis_code/arguments.py`: CLI argument parsing and validation

**Data / Results:**
- `simulations/cramer_rao_bounds_simulation_$index.csv`: Per-task CRB output (generated)
- `simulations/cramer_rao_bounds.csv`: Merged CRB output (generated by `emri-merge`)
- `simulations/prepared_cramer_rao_bounds.csv`: Best-guess parameters (generated by `emri-prepare`)
- `simulations/undetected_events_simulation_$index.csv`: Per-task undetected events (generated)
- `simulations/posteriors/h_*.json`: H0 posterior JSON output (generated)
- `master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv`: GLADE galaxy catalog (committed)
- `run_metadata.json`: Reproducibility metadata per run (generated)

**Figures / Visualization:**
- `master_thesis_code/plotting/`: All plot factory functions by topic
- `docs/saved_figures/`: Pre-generated figures for documentation

## Computation Dependencies (Data Pipeline)

```
simulate.sbatch (GPU, array job, N tasks)
  |-- produces: cramer_rao_bounds_simulation_$index.csv  (one per task)
  |-- produces: undetected_events_simulation_$index.csv  (one per task)
  v
merge.sbatch (CPU, single job, --dependency=afterok)
  |-- runs: emri-merge --workdir $RUN_DIR --delete-sources
  |     merges: cramer_rao_bounds_simulation_*.csv -> cramer_rao_bounds.csv
  |     merges: undetected_events_simulation_*.csv -> undetected_events.csv
  |-- runs: emri-prepare --workdir $RUN_DIR
  |     reads: cramer_rao_bounds.csv
  |     produces: prepared_cramer_rao_bounds.csv
  v
evaluate.sbatch (CPU, 16 cores, --dependency=afterok)
  |-- reads: prepared_cramer_rao_bounds.csv
  |-- reads: cramer_rao_bounds.csv (true CRBs)
  |-- reads: undetected_events.csv
  |-- reads: reduced_galaxy_catalogue.csv
  |-- produces: simulations/posteriors/h_*.json
```

## Naming Conventions

**Python Modules:**
- Source modules: `snake_case.py` (e.g., `physical_relations.py`, `parameter_space.py`)
- Exception: `LISA_configuration.py` uses UPPERCASE prefix (physics convention)
- Private modules: `_style.py`, `_helpers.py` (underscore prefix in `plotting/`)
- Test files: `test_<module>.py` (newer) or `<module>_test.py` (older); both valid

**Python Variables:**
- Physics symbols preserved: `M`, `H`, `Omega_m`, `W_0`, `M_z`, `d_L`
- Uncertainty prefixes: `delta_` for CRB uncertainties: `delta_dist`, `delta_phiS`
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `SNR_THRESHOLD`, `OMEGA_M`)
- Classes: `PascalCase` (e.g., `ParameterEstimation`, `BayesianStatistics`)
- Functions: `snake_case` (e.g., `compute_fisher_information_matrix()`)

**CSV Columns:**
- Parameter names match `ParameterSpace` symbol strings: `M`, `mu`, `a`, `p0`, `e0`, `x0`, `luminosity_distance`, `qS`, `phiS`, `qK`, `phiK`, `Phi_phi0`, `Phi_theta0`, `Phi_r0`
- CRB columns: `delta_{param1}_delta_{param2}` for covariance matrix elements
- Metadata columns: `T`, `dt`, `SNR`, `generation_time`, `host_galaxy_index`

**File Paths:**
- Simulation output uses `$index` placeholder: `cramer_rao_bounds_simulation_$index.csv`
- All output paths defined in `constants.py` (lines 61-66)

## Entry Points

**Primary entry point:**
```bash
python -m master_thesis_code <working_dir> [options]
```
- File: `master_thesis_code/__main__.py` -> `main.py:main()`
- Dispatches to: `data_simulation()`, `evaluate()`, `snr_analysis()`, `generate_figures()`

**Registered CLI scripts (via pyproject.toml `[project.scripts]`):**
```bash
emri-merge --workdir <dir> [--delete-sources]   # Merge per-index CSVs
emri-prepare --workdir <dir>                     # Convert to best-guess params
```

**Pipeline A standalone (dev cross-check):**
```bash
python -m master_thesis_code.bayesian_inference.bayesian_inference_mwe
```

**Cluster pipeline:**
```bash
bash cluster/submit_pipeline.sh --tasks 100 --steps 50 --seed 42
```

## Where to Add New Content

**New EMRI Parameter or Observable:**
- Add `Parameter` field to `master_thesis_code/datamodels/parameter_space.py`
- Update `_parameters_to_dict()` return (line mapping is implicit via `vars()`)
- Update CSV column handling in `master_thesis_code/datamodels/detection.py`
- Update Fisher matrix dimensions in `master_thesis_code/parameter_estimation/parameter_estimation.py`

**New Waveform Model:**
- Add enum value to `WaveGeneratorType` in `master_thesis_code/waveform_generator.py`
- Add `elif` branch in `_set_waveform_generator()` with appropriate `few` class and kwargs
- No changes needed to `ParameterEstimation` (it is waveform-model agnostic)

**New Physical Relation or Constant:**
- Constants: add to `master_thesis_code/constants.py`
- Relations: add to `master_thesis_code/physical_relations.py`
- Test: add to `master_thesis_code_test/physical_relations_test.py` or `test_constants.py`
- **Physics Change Protocol required** for any formula or constant change

**New Post-Processing Script:**
- Add to `scripts/` directory
- Register entry point in `pyproject.toml` `[project.scripts]` section
- Add test in `master_thesis_code_test/scripts/`
- Integrate into `cluster/merge.sbatch` if needed in SLURM pipeline

**New Plot:**
- Add factory function to appropriate topic module in `master_thesis_code/plotting/`
- Pattern: `def plot_X(data, ...) -> tuple[Figure, Axes]:`
- Wire into `generate_figures()` in `main.py` when data source is available
- Test: add to `master_thesis_code_test/plotting/`

**New Validation or Cross-Check:**
- Add test to `master_thesis_code_test/` (mirror source layout)
- Use existing patterns: physical correctness tests (known limits), bounds tests, regression tests
- Mark GPU tests with `@pytest.mark.gpu`, slow tests with `@pytest.mark.slow`

## Build and Execution

**Environment Setup:**
```bash
uv sync --extra cpu --extra dev   # Dev machine (CPU only)
uv sync --extra gpu               # Cluster (GPU, CUDA 12)
```

**Running Simulations:**
```bash
# Local (CPU, for testing only -- requires cupy workaround)
uv run python -m master_thesis_code . --simulation_steps 5 --seed 42

# Cluster (GPU)
uv run python -m master_thesis_code $RUN_DIR --simulation_steps 50 --use_gpu --seed 42

# Full cluster pipeline
bash cluster/submit_pipeline.sh --tasks 100 --steps 50 --seed 42
```

**Running Tests:**
```bash
uv run pytest -m "not gpu and not slow"    # Fast CPU tests (default dev)
uv run pytest                               # All tests (cluster)
uv run pytest -m slow --benchmark-only      # Benchmarks only
```

**Linting and Type Checking:**
```bash
uv run ruff check --fix master_thesis_code/
uv run ruff format master_thesis_code/
uv run mypy master_thesis_code/
```

**Documentation:**
```bash
uv run make -C docs html SPHINXOPTS="-W"
```

## Special Directories

**`evaluation/`:**
- Purpose: Saved outputs from cluster simulation campaigns
- Generated: Yes (by SLURM jobs)
- Committed: Selectively (small reference runs only)

**`master_thesis_code/M1_model_extracted_data/`:**
- Purpose: Pre-computed EMRI rate model polynomial coefficients and detection fractions
- Generated: No (extracted from prior work)
- Committed: Yes

**`.planning/`:**
- Purpose: GSD workflow planning artifacts (phase plans, milestones)
- Generated: By development workflow
- Committed: Yes

**`.gpd/`:**
- Purpose: GPD research mapping and configuration
- Generated: By GPD workflow
- Committed: No (gitignored)

---

_Structure analysis: 2026-03-30_
