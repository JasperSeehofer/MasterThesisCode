# Codebase Structure

**Analysis Date:** 2026-03-25

## Directory Layout

```
MasterThesisCode/
├── master_thesis_code/              # Main Python package
│   ├── __main__.py                  # CLI entry: delegates to main.main()
│   ├── main.py                      # Pipeline dispatcher (simulation, evaluate, snr, figures)
│   ├── arguments.py                 # CLI argument parsing (Arguments class)
│   ├── callbacks.py                 # SimulationCallback Protocol + NullCallback
│   ├── constants.py                 # All physical constants, config, file paths
│   ├── cosmological_model.py        # EMRI event rate model (Model1CrossCheck) + backward-compat re-exports
│   ├── physical_relations.py        # Cosmological distance functions (dist, hubble_function, etc.)
│   ├── LISA_configuration.py        # LISA PSD, antenna patterns, _get_xp() helper
│   ├── waveform_generator.py        # few/fastlisaresponse waveform factory
│   ├── decorators.py                # timer_decorator (with optional CuPy GPU usage)
│   ├── exceptions.py                # Custom exceptions (5 classes)
│   ├── memory_management.py         # GPU memory monitoring (MemoryManagement class)
│   ├── bayesian_inference/          # H0 posterior inference
│   │   ├── bayesian_statistics.py   # Pipeline B: BayesianStatistics (~988 lines)
│   │   ├── detection_probability.py # KDE-based detection probability (~344 lines)
│   │   ├── bayesian_inference.py    # Pipeline A: BayesianInference (dev cross-check)
│   │   └── bayesian_inference_mwe.py# Thin shim + __main__ for Pipeline A standalone
│   ├── datamodels/                  # Domain dataclasses
│   │   ├── parameter_space.py       # Parameter, ParameterSpace (14 EMRI params)
│   │   ├── detection.py             # Detection (from Cramer-Rao CSV)
│   │   ├── emri_detection.py        # EMRIDetection (Pipeline A synthetic)
│   │   └── galaxy.py                # Galaxy, GalaxyCatalog (Pipeline A synthetic)
│   ├── galaxy_catalogue/            # GLADE galaxy catalog interface
│   │   ├── handler.py               # GalaxyCatalogueHandler, HostGalaxy, ParameterSample (~669 lines)
│   │   ├── parser.py                # Raw GLADE+ text file parser (empty/stub)
│   │   └── glade_completeness.py    # Catalog completeness analysis
│   ├── parameter_estimation/        # Waveform-based parameter estimation (GPU)
│   │   ├── parameter_estimation.py  # ParameterEstimation: SNR, Fisher matrix, CRB (~549 lines)
│   │   └── evaluation.py            # DataEvaluation: post-processing of CRB/SNR CSVs
│   ├── plotting/                    # All visualization (no computation)
│   │   ├── __init__.py              # Public API: apply_style, get_figure, save_figure
│   │   ├── _style.py                # Agg backend + emri_thesis.mplstyle loading
│   │   ├── _helpers.py              # get_figure(), save_figure(), make_colorbar()
│   │   ├── emri_thesis.mplstyle     # matplotlib style sheet
│   │   ├── simulation_plots.py      # PlottingCallback + GPU/SNR/detection plots (~177 lines)
│   │   ├── bayesian_plots.py        # Posterior/likelihood visualization (~126 lines)
│   │   ├── evaluation_plots.py      # Cramer-Rao evaluation plots (~122 lines)
│   │   ├── model_plots.py           # Cosmological model plots (~102 lines)
│   │   ├── catalog_plots.py         # Galaxy catalog plots (~93 lines)
│   │   └── physical_relations_plots.py # Distance-redshift plots (~32 lines)
│   ├── M1_model_extracted_data/     # Hardcoded lookup tables from literature
│   │   ├── detection_fraction.py    # DetectionFraction grid (M vs z)
│   │   ├── detection_distribution_simplified.py
│   │   ├── detection_horizon.py
│   │   └── emri_distribution.py
│   └── waveform_generation/         # Empty subpackage (placeholder)
├── master_thesis_code_test/         # Test suite (mirrors source layout)
│   ├── conftest.py                  # Root conftest: xp fixture, style autouse fixture
│   ├── test_constants.py            # Constants validation tests
│   ├── test_benchmarks.py           # Performance benchmarks (slow marker)
│   ├── physical_relations_test.py   # dist(), hubble_function() tests
│   ├── LISA_configuration_test.py   # PSD tests (requires cupy guard)
│   ├── cosmological_model_test.py   # Model1CrossCheck tests
│   ├── decorators_test.py           # timer_decorator tests
│   ├── bayesian_inference/
│   │   └── test_bayesian_inference_mwe.py  # Pipeline A + GalaxyCatalog tests
│   ├── datamodels/
│   │   ├── parameter_space_test.py  # ParameterSpace randomization + bounds
│   │   ├── test_detection.py        # Detection dataclass tests
│   │   └── test_emri_detection.py   # EMRIDetection tests
│   ├── parameter_estimation/
│   │   └── parameter_estimation_test.py  # (requires GPU)
│   ├── plotting/
│   │   └── test_style.py            # apply_style() tests (9 tests)
│   ├── integration/
│   │   ├── conftest.py              # Integration test fixtures
│   │   └── test_evaluation_pipeline.py  # End-to-end evaluation pipeline test
│   └── fixtures/
│       └── evaluation/
│           └── generate_fixtures.py # Fixture data generation script
├── scripts/                         # Utility scripts for data processing
│   ├── prepare_detections.py        # Post-process Cramer-Rao CSV
│   ├── merge_cramer_rao_bounds.py   # Merge per-index simulation CSVs
│   ├── remove_detections_out_of_bounds.py  # Filter detections
│   └── estimate_hubble_constant.py  # Standalone H0 estimation
├── simulations/                     # Simulation output data (CSV, JSON)
│   ├── cramer_rao_bounds*.csv       # Raw and merged Cramer-Rao bounds
│   ├── prepared_cramer_rao_bounds.csv
│   ├── snr_analysis.csv
│   ├── undetected_events.csv
│   ├── posteriors/                  # H0 posterior JSON output
│   └── posteriors_with_bh_mass/     # H0 posterior JSON (with BH mass)
├── evaluation/                      # Post-analysis data
│   └── mean_bounds.xlsx
├── notebooks/                       # Jupyter notebooks for exploration
│   ├── parameter_estimation.ipynb
│   └── parameter_estimation_schwarzschild.ipynb
├── saved_figures/                   # Generated plot output
│   ├── cosmological_model/
│   ├── LISA_configuration/
│   └── monitoring/
├── .github/workflows/ci.yml         # GitHub Actions CI
├── .claude/skills/                   # Claude Code custom skills (7 skills)
├── .planning/codebase/              # GSD codebase analysis documents
├── pyproject.toml                   # Project config: deps, mypy, ruff, pytest
├── .pre-commit-config.yaml          # Pre-commit hooks: ruff + mypy
├── .editorconfig                    # Editor settings
├── .gitignore                       # Git ignore rules
├── uv.lock                          # Locked dependencies (committed)
├── CLAUDE.md                        # AI assistant instructions
├── README.md                        # Project README
├── CONTRIBUTING.md                   # Contributor guidelines
├── CHANGELOG.md                     # Change log
├── TODO.md                          # Known issues and planned work
└── LICENSE                          # MIT license
```

## Directory Purposes

**`master_thesis_code/`:**
- Purpose: Main Python package containing all source code
- Contains: 6 subpackages + 12 top-level modules
- Key files: `main.py` (pipeline dispatch), `constants.py` (all config), `physical_relations.py` (distance functions)

**`master_thesis_code/bayesian_inference/`:**
- Purpose: Hubble constant posterior inference
- Contains: Two parallel pipelines (A = dev cross-check, B = production)
- Key files: `bayesian_statistics.py` (Pipeline B, 988 lines), `detection_probability.py` (KDE), `bayesian_inference.py` (Pipeline A)

**`master_thesis_code/datamodels/`:**
- Purpose: Domain dataclasses shared across pipelines
- Contains: Parameter space, detection representations, galaxy models
- Key files: `parameter_space.py` (14-param EMRI space), `detection.py` (Cramer-Rao output), `galaxy.py` (synthetic catalog)

**`master_thesis_code/galaxy_catalogue/`:**
- Purpose: Interface to the GLADE galaxy catalog
- Contains: Catalog parsing, BallTree spatial lookups, host galaxy resolution
- Key files: `handler.py` (669 lines, main interface), `glade_completeness.py`
- Data dependency: Requires `reduced_galaxy_catalogue.csv` (derived from GLADE+.txt)

**`master_thesis_code/parameter_estimation/`:**
- Purpose: GPU-accelerated waveform generation, SNR, Fisher matrix
- Contains: Core computational engine (requires CUDA)
- Key files: `parameter_estimation.py` (549 lines), `evaluation.py` (post-processing)

**`master_thesis_code/plotting/`:**
- Purpose: All visualization code, fully decoupled from computation
- Contains: Factory functions organized by topic (`data in, (fig, ax) out`)
- Key files: `_style.py` (Agg + mplstyle), `_helpers.py` (save/create), `simulation_plots.py` (PlottingCallback)
- Pattern: No computation logic. Import only matplotlib and project data types.

**`master_thesis_code/M1_model_extracted_data/`:**
- Purpose: Hardcoded lookup tables digitized from literature figures
- Contains: Detection fraction grids, detection horizons, EMRI distribution data
- Key files: `detection_fraction.py` (DetectionFraction grid used by Model1CrossCheck)

**`master_thesis_code_test/`:**
- Purpose: pytest test suite mirroring source layout
- Contains: Unit tests, integration tests, benchmarks, fixtures
- Key files: `conftest.py` (xp fixture, style autouse), `test_benchmarks.py` (slow)

**`scripts/`:**
- Purpose: Standalone data processing utilities (not part of the package)
- Contains: CSV merging, detection filtering, H0 estimation scripts
- Run manually between simulation and evaluation phases

**`simulations/`:**
- Purpose: Simulation output directory (CSV and JSON data files)
- Contains: Cramer-Rao bounds, SNR analysis, posterior distributions
- Generated by: Pipeline 1 (CSVs), Pipeline 2 (JSON posteriors)
- Not cleaned automatically; accumulates across runs

## Key File Locations

**Entry Points:**
- `master_thesis_code/__main__.py`: Package entry (`python -m master_thesis_code`)
- `master_thesis_code/main.py`: Pipeline dispatcher with `main()`, `data_simulation()`, `evaluate()`
- `master_thesis_code/bayesian_inference/bayesian_inference_mwe.py`: Pipeline A standalone entry

**Configuration:**
- `master_thesis_code/constants.py`: All physical constants, cosmological parameters, file paths, thresholds
- `master_thesis_code/arguments.py`: CLI argument parsing (Arguments class)
- `pyproject.toml`: Build config, dependencies, tool settings (mypy, ruff, pytest)
- `.pre-commit-config.yaml`: Pre-commit hooks (ruff check, ruff format, mypy)

**Core Logic:**
- `master_thesis_code/parameter_estimation/parameter_estimation.py`: Waveform gen, Fisher matrix, SNR, CRB
- `master_thesis_code/bayesian_inference/bayesian_statistics.py`: H0 posterior evaluation (Pipeline B)
- `master_thesis_code/cosmological_model.py`: EMRI event rate model + MCMC sampling
- `master_thesis_code/physical_relations.py`: Luminosity distance, Hubble function, redshift inversion
- `master_thesis_code/galaxy_catalogue/handler.py`: GLADE catalog interface + BallTree lookups

**Testing:**
- `master_thesis_code_test/conftest.py`: Root conftest with xp fixture and style autouse
- `master_thesis_code_test/bayesian_inference/test_bayesian_inference_mwe.py`: Pipeline A tests (most comprehensive)
- `master_thesis_code_test/physical_relations_test.py`: Distance function tests
- `master_thesis_code_test/integration/test_evaluation_pipeline.py`: End-to-end evaluation test

## Naming Conventions

**Files:**
- Source modules: `snake_case.py` (e.g., `parameter_space.py`, `physical_relations.py`)
- Exception: `LISA_configuration.py` uses mixed case (legacy)
- Test files: `*_test.py` (older) or `test_*.py` (newer) -- both patterns coexist
- Private modules: `_prefix.py` (e.g., `_style.py`, `_helpers.py` in plotting)

**Directories:**
- Subpackages: `snake_case/` (e.g., `bayesian_inference/`, `galaxy_catalogue/`, `parameter_estimation/`)
- Test directories: Mirror source layout under `master_thesis_code_test/`

## Where to Add New Code

**New EMRI physics function:**
- Implementation: `master_thesis_code/physical_relations.py` (if general cosmological) or `master_thesis_code/LISA_configuration.py` (if LISA-specific)
- Tests: `master_thesis_code_test/physical_relations_test.py` or `master_thesis_code_test/LISA_configuration_test.py`
- Constants: Add to `master_thesis_code/constants.py`

**New datamodel:**
- Implementation: `master_thesis_code/datamodels/` as a new `@dataclass` module
- Tests: `master_thesis_code_test/datamodels/`

**New Bayesian inference feature:**
- Implementation: `master_thesis_code/bayesian_inference/bayesian_statistics.py` (Pipeline B) or `master_thesis_code/bayesian_inference/bayesian_inference.py` (Pipeline A)
- Tests: `master_thesis_code_test/bayesian_inference/`

**New plot:**
- Implementation: Add factory function to the appropriate topic module in `master_thesis_code/plotting/` (e.g., `bayesian_plots.py`, `evaluation_plots.py`)
- Pattern: Function signature `def plot_X(data, ..., ax=None) -> tuple[Figure, Axes]`
- Tests: `master_thesis_code_test/plotting/`

**New simulation callback:**
- Implementation: Create class satisfying `SimulationCallback` protocol (see `master_thesis_code/callbacks.py`)
- Wire in: Pass to `data_simulation(callbacks=[...])` in `main.py`
- Example: `master_thesis_code/plotting/simulation_plots.py:PlottingCallback`

**New utility script:**
- Location: `scripts/` directory
- Pattern: Standalone script, not part of the package

**New constant or threshold:**
- Location: `master_thesis_code/constants.py`
- Convention: UPPER_SNAKE_CASE with unit comment

## Special Directories

**`simulations/`:**
- Purpose: Simulation output data (CSV, JSON)
- Generated: Yes, by running simulation and evaluation pipelines
- Committed: Yes (contains reference data for evaluation)

**`saved_figures/`:**
- Purpose: Generated plot output (PDF, PNG)
- Generated: Yes, by plotting functions
- Committed: Partially (some reference figures)

**`.planning/`:**
- Purpose: GSD codebase analysis documents
- Generated: Yes, by codebase mapping
- Committed: Yes

**`master_thesis_code/galaxy_catalogue/reduced_galaxy_catalogue.csv`:**
- Purpose: Pre-processed GLADE catalog (derived from GLADE+.txt)
- Generated: Yes, on first run if GLADE+.txt is present
- Committed: Yes (avoids re-parsing the large GLADE+ file)
- Note: The raw GLADE+.txt file is NOT committed (too large)

**`notebooks/`:**
- Purpose: Jupyter notebooks for interactive exploration
- Generated: No (manually authored)
- Committed: Yes

---

*Structure analysis: 2026-03-25*
