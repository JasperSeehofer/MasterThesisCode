# Technology Stack

**Analysis Date:** 2026-03-25

## Languages

**Primary:**
- Python 3.13 - All source code (`.python-version` pins 3.13; `pyproject.toml` allows `>=3.10,<3.14`)

**Secondary:**
- None (pure Python project; compiled extensions come from third-party wheels)

## Runtime

**Environment:**
- Python 3.13 (pinned in `.python-version`)
- CUDA 12 toolkit - required on GPU cluster for `cupy-cuda12x` and `fastemriwaveforms-cuda12x`
- GSL (GNU Scientific Library) - build-time requirement for `fastemriwaveforms`

**Package Manager:**
- [uv](https://docs.astral.sh/uv/) (Astral)
- Lockfile: `uv.lock` (committed, 4220 lines)
- Virtual environment created at `.venv/` by `uv sync`

## Frameworks

**Core:**
- NumPy - array computation, FFT, linear algebra (used everywhere)
- SciPy - integration (`quad`, `dblquad`, `fixed_quad`, `cumulative_trapezoid`), interpolation (`CubicSpline`, `RegularGridInterpolator`), statistics (`truncnorm`, `norm`, `gaussian_kde`, `erf`), optimization (`fsolve`)
- Pandas - CSV I/O for Cramer-Rao bounds, SNR analysis, galaxy catalog
- Astropy >=6.1.7 - physical constants (`astropy.constants`) and unit conversions (`astropy.units`)
- Matplotlib - all plotting (factory-function pattern in `master_thesis_code/plotting/`)

**Testing:**
- pytest - test runner, configured in `pyproject.toml [tool.pytest.ini_options]`
- pytest-cov - coverage reporting (fail_under = 25%)
- pytest-benchmark - performance benchmarks (`-m slow`)

**Build/Dev:**
- Ruff - linting and formatting (target `py313`, line-length 100)
- mypy - static type checking (`disallow_untyped_defs = true`)
- pre-commit - hooks for ruff lint, ruff format, mypy
- pip-audit - dependency security scanning
- Sphinx + Furo theme - documentation (`docs/`)

## Key Dependencies

**Critical (core deps in `pyproject.toml`):**
- `numpy` - array computation backbone; used in every module
- `scipy` - numerical integration, interpolation, statistics, optimization
- `pandas` - CSV read/write for simulation results and galaxy catalog
- `matplotlib` - all visualization
- `scikit-learn` - `BallTree` for galaxy catalog spatial lookups (`master_thesis_code/galaxy_catalogue/handler.py`)
- `emcee` - MCMC ensemble sampler for comoving volume sampling (`master_thesis_code/datamodels/galaxy.py`, `master_thesis_code/cosmological_model.py`)
- `tabulate` - formatted table output
- `fastlisaresponse==1.1.9` - LISA time-delay interferometry response wrapper
- `astropy>=6.1.7` - physical constants and units

**CPU extras (`[project.optional-dependencies.cpu]`):**
- `fastemriwaveforms==2.0.0rc1` - EMRI waveform generation (imports as `few`)

**GPU extras (`[project.optional-dependencies.gpu]`):**
- `fastemriwaveforms-cuda12x` - GPU-accelerated EMRI waveforms
- `cupy-cuda12x` - GPU array library (NumPy-compatible API on CUDA)
- `GPUtil` - GPU utilization monitoring

**Dev extras (`[project.optional-dependencies.dev]`):**
- `pytest`, `pytest-cov`, `pytest-benchmark` - testing
- `pip-audit` - security audit
- `mypy` - type checking
- `ruff` - linting/formatting
- `pre-commit` - git hook management
- `jupyterlab` - interactive exploration
- `sphinx>=8.1.3`, `furo>=2025.12.19`, `sphinx-copybutton>=0.5.2` - documentation

## Configuration

**Environment:**
- No `.env` files detected; configuration is via CLI arguments and `master_thesis_code/constants.py`
- Physical constants derived from `astropy` at import time (`master_thesis_code/constants.py`)
- Simulation paths are relative strings in `constants.py` (e.g., `simulations/cramer_rao_bounds.csv`)

**Build:**
- `pyproject.toml` - single source of truth for project metadata, dependencies, and tool config
- `uv.lock` - exact dependency versions (committed)
- `.python-version` - pins Python 3.13

**Linting/Formatting (in `pyproject.toml`):**
- `[tool.ruff]` - target-version `py313`, line-length 100
- `[tool.ruff.lint]` - enables E, F, I, UP, B, N rule sets; ignores physics-naming violations (N802, N803, N806, N815, N816)
- `[tool.ruff.lint.isort]` - `known-first-party = ["master_thesis_code"]`

**Type Checking (in `pyproject.toml`):**
- `[tool.mypy]` - python_version 3.13, `disallow_untyped_defs = true`, `disallow_incomplete_defs = true`
- Missing import overrides for: astropy, cupy, cupyx, fastlisaresponse, few, GPUtil, pandas, scipy, sklearn, mpl_toolkits, emcee, tabulate

**Testing (in `pyproject.toml`):**
- `[tool.pytest.ini_options]` - testpaths `master_thesis_code_test/`, markers: `gpu`, `slow`
- `[tool.coverage.run]` - source `master_thesis_code/`, omits test dir
- `[tool.coverage.report]` - fail_under 25%

**Editor:**
- `.editorconfig` - 4-space indent, UTF-8, LF line endings, trailing whitespace trimmed

**Pre-commit (`.pre-commit-config.yaml`):**
- `ruff-pre-commit` v0.11.0 - ruff lint (`--fix`) + ruff format
- Local hook - `uv run mypy master_thesis_code/ master_thesis_code_test/`

## Platform Requirements

**Development (CPU-only):**
- Python 3.13
- GSL (for building `fastemriwaveforms` CPU variant)
- Install: `uv sync --extra cpu --extra dev`
- Run tests: `uv run pytest -m "not gpu and not slow"`

**Production (GPU cluster):**
- Python 3.13
- CUDA 12 toolkit
- GSL
- Install: `uv sync --extra gpu`
- Run: `uv run python -m master_thesis_code <working_dir> --simulation_steps N --use_gpu`

## CI/CD

**GitHub Actions (`.github/workflows/ci.yml`):**
- **check** job: ruff lint, ruff format check, mypy, pytest (CPU, not slow), coverage upload, pip-audit
- **integration** job (needs check): runs slow tests, uploads test plot artifacts, deploys to GitHub Pages
- **docs** job: builds Sphinx docs (`uv run make -C docs html SPHINXOPTS="-W"`)
- Runner: `ubuntu-latest`
- Dependabot: `.github/dependabot.yml` for pip + github-actions weekly

---

*Stack analysis: 2026-03-25*
