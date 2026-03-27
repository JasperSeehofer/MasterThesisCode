# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### System prerequisites (must be installed before uv sync)

- **CUDA 12 toolkit** — required on the GPU cluster for the `gpu` extras
- **GSL** (GNU Scientific Library) — required by `fastemriwaveforms` at build time
  - Arch/Manjaro: `sudo pacman -S gsl`
  - Ubuntu/Debian: `sudo apt install libgsl-dev`
  - On the cluster: check with `module load gsl` or ask the sysadmin

### Set up the environment

```bash
# Dev machine (no GPU) — installs CPU waveform variant
uv sync --extra cpu --extra dev

# Cluster (GPU, CUDA 12) — installs GPU waveform variant
uv sync --extra gpu
```

`uv sync` creates `.venv/` in the project root and installs exactly what is in `uv.lock`.
The lock file is committed to git, so every machine gets the same versions.

### Running code

```bash
# Run the package
uv run python -m master_thesis_code <working_dir> --simulation_steps N

# Run tests (dev machine) — also prints coverage summary
uv run pytest -m "not gpu and not slow"

# Run benchmarks
uv run pytest -m "slow" --benchmark-only

# Run mypy
uv run mypy master_thesis_code/
```

Note: `fastemriwaveforms` installs as the `few` Python package — `import few`, not `import fastemriwaveforms`.

### Reproducible simulation runs

Pass `--seed <int>` to fix the random state. When omitted, a random seed is chosen,
logged, and recorded in `run_metadata.json` in the working directory.

```bash
uv run python -m master_thesis_code <working_dir> --simulation_steps 100 --seed 42
```

`run_metadata.json` records `git_commit`, `timestamp`, `random_seed`, and all CLI
arguments alongside every simulation output so results can always be tied back to
the exact code and parameters that produced them.

## Dev Workflow

### Linting and formatting (run manually or triggered automatically on commit)

```bash
uv run ruff check --fix master_thesis_code/   # lint and auto-fix
uv run ruff format master_thesis_code/        # format
uv run mypy master_thesis_code/               # type check
```

Pre-commit hooks run ruff and mypy automatically on every `git commit`.
To run all hooks on all files manually:
```bash
uv run pre-commit run --all-files
```

Alternatively, activate the virtual environment once for a session:

```bash
source .venv/bin/activate
python -m master_thesis_code ...  # works without uv run prefix
```

### Adding a new dependency

```bash
uv add <package>                    # add to core deps
uv add --optional gpu <package>     # add to gpu group
uv add --optional dev <package>     # add to dev group
```

This updates both `pyproject.toml` and `uv.lock`. Commit both files.
Never manually edit the dependencies list in `pyproject.toml`.

## Running the Code

```bash
# EMRI simulation (generates SNR + Cramér-Rao bounds)
python -m master_thesis_code <working_dir> --simulation_steps N [--simulation_index I] [--log_level DEBUG]

# Bayesian inference (evaluate Hubble constant posterior)
python -m master_thesis_code <working_dir> --evaluate [--h_value 0.73]

# SNR analysis only
python -m master_thesis_code <working_dir> --snr_analysis
```

## Cluster Deployment

The `cluster/` directory contains everything needed to run EMRI simulations on bwUniCluster 3.0 (KIT). See `cluster/README.md` for the full quickstart guide.

### Key CLI Flags

| Flag | Where | Purpose |
|------|-------|---------|
| `--use_gpu` | `arguments.py` | Enable GPU acceleration (always used on cluster) |
| `--num_workers N` | `arguments.py` | Multiprocessing pool size for Bayesian inference; defaults to `os.sched_getaffinity(0) - 2` |
| `--simulation_index I` | `arguments.py` | Maps to `SLURM_ARRAY_TASK_ID`; indexes per-task output files |
| `--seed S` | `arguments.py` | Random seed; on cluster, per-task seed = `BASE_SEED + SLURM_ARRAY_TASK_ID` |

### Script Inventory

| Script | Purpose |
|--------|---------|
| `cluster/modules.sh` | Loads environment modules; exports `$WORKSPACE`, `$PROJECT_ROOT`, `$VENV_PATH` |
| `cluster/setup.sh` | First-time setup: installs uv, allocates workspace, creates venv |
| `cluster/simulate.sbatch` | GPU array job -- one EMRI simulation per task (`gpu_h100`, 1 GPU, 2h) |
| `cluster/merge.sbatch` | CPU job -- merges per-task CSVs via `emri-merge`, prepares detections via `emri-prepare` |
| `cluster/evaluate.sbatch` | CPU job -- Bayesian inference for H0 posterior (16 CPUs, auto-detected workers) |
| `cluster/submit_pipeline.sh` | Pipeline orchestrator -- chains simulate -> merge -> evaluate via `--dependency=afterok` |
| `cluster/resubmit_failed.sh` | Resubmits only failed/timed-out array tasks; cleans up partial output first |
| `cluster/vpn.sh` | University VPN connection via openconnect |

### Quick Reference

```bash
# First-time setup (run once)
bash cluster/setup.sh

# Submit a full campaign
bash cluster/submit_pipeline.sh --tasks 100 --steps 50 --seed 42

# Resubmit failed tasks
bash cluster/resubmit_failed.sh JOB_ID $RUN_DIR BASE_SEED SIM_STEPS
```

## Running Tests

Tests use pytest. All tests live in `master_thesis_code_test/`, mirroring the source layout.

```bash
# Run all tests
pytest

# Run a single test file
pytest master_thesis_code_test/bayesian_inference/test_bayesian_inference_mwe.py

# Run a single test
pytest master_thesis_code_test/bayesian_inference/test_bayesian_inference_mwe.py::test_gw_detection_probability
```

Note: many test files are nearly empty stubs — see the memory file for priority gaps.

## Architecture

The codebase has two distinct pipelines:

### 1. EMRI Simulation Pipeline
`main.py:data_simulation()` drives a loop over `simulation_steps`:
1. `Model1CrossCheck` (cosmological model) samples EMRI events from a distribution
2. `GalaxyCatalogueHandler` resolves each event to a host galaxy from a galaxy catalog
3. `ParameterSpace.randomize_parameters()` + `set_host_galaxy_parameters()` set up the 14-parameter EMRI
4. `ParameterEstimation.compute_signal_to_noise_ratio()` computes SNR using a LISA waveform
5. If SNR ≥ threshold: `compute_Cramer_Rao_bounds()` computes the Fisher matrix and saves to CSV

### 2. Bayesian Inference Pipeline
`main.py:evaluate()` → `BayesianStatistics.evaluate()`:
- Loads saved Cramér-Rao bounds from CSV
- Uses `BayesianInference` (in `bayesian_inference/bayesian_inference_mwe.py`) to compute the posterior over H₀
- `GalaxyCatalog` models the galaxy distribution and mass distribution using normal/truncnorm distributions

### Key Module Responsibilities

- **`parameter_estimation/parameter_estimation.py`** — waveform generation via `few`, Fisher matrix computation (forward-difference derivatives; 5-point stencil method exists but is not yet called — see Known Bug 4), SNR and Cramér-Rao bounds. The `scalar_product_of_functions` inner product is the computational bottleneck (PSD loop).
- **`LISA_configuration.py`** — LISA antenna patterns (F+, F×), PSD, SSB↔detector frame transformations
- **`datamodels/parameter_space.py`** — 14-parameter EMRI space with randomization and bounds
- **`bayesian_inference/bayesian_inference.py`** — Pipeline A (dev cross-check): `BayesianInference`, erf-based detection probability, hardcoded 10% σ(d_L), synthetic `GalaxyCatalog`. Not used by `--evaluate`.
- **`bayesian_inference/bayesian_inference_mwe.py`** — thin re-export shim; `__main__` block runs Pipeline A standalone
- **`bayesian_inference/bayesian_statistics.py`** — Pipeline B (production): `BayesianStatistics`, `single_host_likelihood`, multiprocessing workers, helper functions. Invoked by `--evaluate`.
- **`bayesian_inference/detection_probability.py`** — `DetectionProbability` class: KDE-based detection probability with `RegularGridInterpolator` look-ups. Used by Pipeline B.
- **`cosmological_model.py`** — `Model1CrossCheck` wraps the EMRI event rate model; `LamCDMScenario`, `DarkEnergyScenario` parameter spaces. Backward-compat re-exports of `BayesianStatistics` and `DetectionProbability`.
- **`galaxy_catalogue/handler.py`** — interfaces with the GLADE galaxy catalog (BallTree-based lookups)
- **`constants.py`** — all physical constants and simulation configuration. Key: `H=0.73`, `SNR_THRESHOLD=20`
- **`plotting/`** — all visualization code lives here. Factory functions (`data in, (fig, ax) out`) in topic modules (`bayesian_plots.py`, `evaluation_plots.py`, `model_plots.py`, `catalog_plots.py`, etc.). `_style.py` sets Agg backend + loads `emri_thesis.mplstyle`. `_helpers.py` provides `save_figure()` and `get_figure()`.
- **`callbacks.py`** — `SimulationCallback` Protocol for decoupling the simulation loop from visualization; `PlottingCallback` in `plotting/simulation_plots.py` collects data and produces plots in `on_simulation_end`

### Known Bugs to Be Aware Of

All four originally-listed bugs are resolved. Remaining known issues (also tracked in TODO.md):

#### Code health
1. **`LISA_configuration.py` unconditional `import cupy`**: still at module top level — any
   module that imports `LisaTdiConfiguration` is un-importable on CPU-only machines without
   the guarded `try/except`. Fix when that file is next touched.
2. ~~**`cosmological_model.py` size**~~ [RESOLVED]: extracted `BayesianStatistics` → `bayesian_inference/bayesian_statistics.py` (~986 lines) and `DetectionProbability` → `bayesian_inference/detection_probability.py` (~344 lines). `cosmological_model.py` now ~383 lines.

#### Physics / mathematics (confirmed by Phase 9 review — Physics Change Protocol required)
3. ~~**`datamodels/galaxy.py` comoving volume element missing 1/E(z)**~~ [RESOLVED]:
   The formula computes dV_c/dz (volume element), not V_c(z). The exponent 2 and 4π
   prefactor were correct, but the 1/E(z) denominator was missing.
   Fix: `cv_grid = 4π · (c/H₀)³ · I(z)² / E(z)`. Methods renamed `comoving_volume` → `comoving_volume_element`.
   Ref: Hogg (1999) arXiv:astro-ph/9905116 Eq. (27).
4. **`parameter_estimation.py:336` Fisher matrix uses O(ε) forward difference** [HIGH]:
   `compute_fisher_information_matrix()` calls `finite_difference_derivative()` instead of
   `five_point_stencil_derivative()`. The O(ε⁴) method already exists but is never called.
   Ref: Vallisneri (2008) arXiv:gr-qc/0703086.
5. **`LISA_configuration.py` galactic confusion noise absent from PSD** [MEDIUM]:
   constants defined in `constants.py:77–83` but never used; dominates sensitivity 0.1–3 mHz.
   Ref: Babak et al. (2023) arXiv:2303.15929 Eq. (17).
6. **`physical_relations.py:72` wCDM params w₀, wₐ silently ignored** [MEDIUM]:
   `dist()` accepts them but passes to a hardcoded-ΛCDM hypergeometric function.
7. **`bayesian_inference/bayesian_inference.py` hardcoded 10% distance error** [MEDIUM]:
   uses `FRACTIONAL_LUMINOSITY_ERROR` instead of per-source Cramér–Rao bound from CSV.
8. **`constants.py:29–30` outdated WMAP-era cosmology** [LOW]:
   Ω_m = 0.25, H = 0.73; Planck 2018 best-fit is Ω_m = 0.3153, H = 0.6736.
9. **`datamodels/galaxy.py:64` galaxy redshift uncertainty non-standard scaling** [LOW]:
   `0.013 * (1+z)³` has no reference; caps at z ≈ 0.14; standard forms scale as (1+z).

---

## Skill-Driven Workflows

Custom skills in `.claude/skills/` encode repeatable, multi-step workflows. Claude must
use them at the appropriate trigger points — not as optional suggestions, but as mandatory
workflow gates.

### Trigger rules

| Trigger condition | Skill | Behavior |
|---|---|---|
| About to edit a physics file (see list below) with a formula or constant change | `/physics-change` | **Hard gate.** Must invoke before writing any code. Do not skip. |
| After modifying array/GPU computation code | `/gpu-audit` | Suggest running on changed files. |
| Before any `git commit` | `/check` | Run full quality gate (ruff + mypy + pytest). |
| Before any `git commit` (after `/check` passes) | `/pre-commit-docs` | Verify CHANGELOG, TODO, CLAUDE.md, README are consistent with staged changes. |
| User asks "what should I work on?" or "what bugs remain?" | `/known-bugs` | Show current bug status with priorities. |
| User wants to run the simulation or evaluation pipeline | `/run-pipeline` | Use instead of ad-hoc bash commands. |

### Physics-change trigger files

Any edit to these files that modifies a computed value (not just refactoring/types/comments)
**requires** `/physics-change`:

- `physical_relations.py`
- `constants.py`
- `LISA_configuration.py`
- `parameter_estimation/parameter_estimation.py`
- `datamodels/galaxy.py`
- `bayesian_inference/bayesian_inference.py`
- `cosmological_model.py`

---

## Dataclass Conventions

### Mutable field defaults

Never use a mutable object (e.g. a `Parameter` instance, a list, a dict) as a bare default
value in a `@dataclass`. Python 3.13 raises `ValueError` at class-definition time. Always wrap
with `field(default_factory=...)`:

```python
# Wrong — crashes on Python 3.13
@dataclass
class Foo:
    bar: MyMutableClass = MyMutableClass()

# Correct
from dataclasses import dataclass, field

@dataclass
class Foo:
    bar: MyMutableClass = field(default_factory=MyMutableClass)
    # or with arguments:
    bar: MyMutableClass = field(default_factory=lambda: MyMutableClass(x=1))
```

---

## Typing Conventions

All public and private functions/methods must have complete type annotations on every parameter and on the return type. The only exception is `__init__` where the return type is always `None` and may be omitted.

### Python 3.10 native syntax

Do **not** import `List`, `Dict`, `Tuple`, `Set`, or `Optional` from `typing`. Use the built-in lowercase forms and PEP 604 unions:

```python
# Wrong — old style
from typing import List, Dict, Optional, Union
def foo(x: List[float]) -> Optional[Dict[str, int]]: ...

# Correct — Python 3.10 style
def foo(x: list[float]) -> dict[str, int] | None: ...
```

Do **not** add `from __future__ import annotations`. Remove it from `arguments.py` and `bayesian_inference_mwe.py` when those files are otherwise touched.

### NumPy array types

```python
import numpy.typing as npt

# Known dtype
def psd(frequencies: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

# Generic float
def integrate(a: npt.NDArray[np.floating[Any]]) -> float: ...
```

Never annotate `np.ndarray` without a dtype parameter — it carries no useful type information.

### CuPy array types

CuPy has no mypy stubs. Annotate GPU-capable functions with `npt.NDArray[np.float64]` and add a comment that a cupy array is also accepted at runtime:

```python
# xp may be numpy or cupy at runtime; annotation reflects the numpy-compatible interface
def scalar_product(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float: ...
```

Never use `cp.ndarray` as a type annotation.

### Callable types

Use `Callable` from `typing`. Never use lowercase `callable` as a type. For decorators that must preserve the wrapped function's signature, use a `TypeVar` bound to `Callable[..., Any]` combined with `@functools.wraps`:

```python
from typing import Callable, TypeVar, Any
F = TypeVar("F", bound=Callable[..., Any])

def my_decorator(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ...
    return wrapper  # type: ignore[return-value]
```

### mypy

Run with: `mypy master_thesis_code/`

Config lives in `pyproject.toml`. The key flags are `disallow_untyped_defs = true` and `disallow_incomplete_defs = true` — mypy will error on any unannotated function. CuPy, `few`, `fastlisaresponse`, and `GPUtil` are listed under `ignore_missing_imports` because they have no stubs.

---

## HPC / GPU Best Practices

This code runs on a GPU cluster (CuPy/CUDA) but must also be importable and testable on a CPU-only development machine. The patterns below are mandatory for all new code and must be applied when modifying existing code.

### Array namespace pattern

Never call `cp.*` or `np.*` directly inside a computation function. Resolve the array module once using the `_get_xp` helper and use it as `xp` throughout:

```python
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False

def _get_xp(use_gpu: bool) -> types.ModuleType:
    if use_gpu and _CUPY_AVAILABLE:
        return cp  # type: ignore[return-value]
    return np
```

Then inside every computation function:

```python
def scalar_product(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64],
                   *, use_gpu: bool = False) -> float:
    xp = _get_xp(use_gpu)
    result = xp.trapz(xp.fft.rfft(a) * xp.conj(xp.fft.rfft(b)))
    return float(result.real)
```

This pattern solves the import crash, the testing problem, and GPU/CPU portability in one go.

### GPU imports must always be guarded

Never place `import cupy as cp` at module top level unconditionally. The current unconditional imports in `decorators.py`, `memory_management.py`, `LISA_configuration.py`, and `parameter_estimation/parameter_estimation.py` are known issues (see bugs above) and must be fixed when those files are touched.

### Vectorize array operations

Never iterate over array elements in a hot path. All operations on frequency-domain data, PSD values, or waveform arrays must use vectorized `xp.*` operations:

```python
# Wrong — Python loop over array bins
result = 0.0
for i in range(len(fs)):
    result += integrant[i] / psd[i] * delta_f

# Correct — vectorized
result = float(xp.trapz(integrant / psd, x=fs))
```

The PSD loop inside `scalar_product_of_functions` (`parameter_estimation.py`) is the primary target.

### Avoid GPU-to-CPU transfers in hot paths

Do not call `cp.asnumpy()` or `.get()` inside a function called thousands of times per simulation step. Keep data on GPU until a single scalar result, then convert:

```python
# Wrong — transfer inside hot path
fs_np = cp.asnumpy(cufft.rfftfreq(n, dt))

# Correct — stay on GPU, extract scalar at the end
fs = xp.fft.rfftfreq(n, dt)
result = float(xp.trapz(integrant, x=fs).real)
```

### GPU memory management

Use the existing `MemoryManagement` class for monitoring. Free GPU memory after each full simulation step (already done in `main.py` — keep it there):

```python
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_fft_plan_cache().clear()
```

Do not call `free_all_blocks()` inside inner loops — the CuPy allocator reuses blocks and flushing inside a loop defeats this caching.

### USE_GPU flag

`USE_GPU` must never be hardcoded `True`. It must come from the `--use_gpu` CLI argument in `arguments.py` and be threaded into every constructor that needs it. No module-level constant should control GPU behavior.

---

## Testing Strategy

Tests must be fully runnable on a CPU-only development machine. GPU-specific tests are gated behind a `gpu` marker and skipped by default.

**Core principle:** code written with the `xp` array namespace pattern is testable on CPU without any mocking — the same test exercises both backends.

### Running tests

```bash
pytest -m "not gpu"          # dev machine (CPU only) — default
pytest                       # cluster (GPU available) — runs everything
pytest -m "not gpu and not slow"   # fast subset only
```

### GPU marker

Any test that constructs `ParameterEstimation`, calls `create_lisa_response_generator`, or otherwise requires a real CUDA GPU must be decorated:

```python
@pytest.mark.gpu
def test_waveform_generation() -> None: ...
```

Tests for mathematical/physical functions (`dist`, `power_spectral_density`, `gw_detection_probability`, Fisher matrix math) must NOT require GPU. If the production code currently couples them to CuPy, refactor using the `xp` pattern before writing the test.

### xp fixture

The `xp` fixture in `conftest.py` parametrizes tests over `numpy` (always) and `cupy` (when available):

```python
def test_scalar_product_positive(xp: types.ModuleType) -> None:
    h = xp.sin(xp.linspace(0, 2 * xp.pi, 1000))
    result = scalar_product(h, h, use_gpu=(xp.__name__ == "cupy"))
    assert result > 0
```

### Test priority order

Write new tests in this order:
1. **Physical correctness** — functions with known analytical limits: `dist(z=0) == 0.0`, `power_spectral_density(f) > 0` for all valid `f`, `gw_detection_probability` in `[0, 1]`, `scalar_product(h, h) > 0`
2. **Bounds** — `ParameterSpace` randomized values stay within declared limits; `_parameters_to_dict` returns the correct 14 keys
3. **Regression** — before changing any formula, add a test asserting the old numerical result so the change is verifiable

### Guarding imports that require cupy

`LISA_configuration.py` imports cupy unconditionally at module level. Any test file that
imports a module which transitively depends on `LISA_configuration` (e.g. `ParameterEstimation`)
must guard the import and skip when cupy is absent:

```python
try:
    from master_thesis_code.parameter_estimation.parameter_estimation import ParameterEstimation
    _PE_AVAILABLE = True
except ImportError:
    _PE_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _PE_AVAILABLE, reason="requires cupy")
```

Direct imports of `LisaTdiConfiguration` use `pytest.importorskip("cupy")` at module level
to skip the whole file cleanly.

---

## Math/Physics Validation Workflow

Errors in physics formulas produce subtly wrong results with no crash. A strict protocol applies.

### What counts as a physics change

A change is a **physics change** if it touches any of:
- A formula (integrals, inner products, distance–redshift relations, posteriors, likelihoods)
- A physical or cosmological constant: `C`, `G`, `H`, `OMEGA_M`, `W_0`, `W_A`, `SNR_THRESHOLD`, `TRUE_HUBBLE_CONSTANT`, PSD coefficients in `LISA_configuration.py`, `derivative_epsilon` in `ParameterSpace`
- Waveform parameters passed to `few` or `ResponseWrapper`
- Frequency limits in `scalar_product_of_functions`
- Galaxy distribution or mass function model

A change is a **software change** if it is limited to: refactoring, type annotations, test additions, logging, or import cleanup — with no change to any computed numerical value. When in doubt, treat it as a physics change.

### Protocol — before writing any code, Claude presents

1. **Old formula** — exact expression, file:line
2. **New formula** — proposed replacement
3. **Reference** — citation (DOI/arXiv + equation number) or step-by-step derivation
4. **Dimensional analysis** — units of inputs and output, consistency check
5. **Limiting case** — at least one analytical limit where the result is known

The user approves or rejects. Claude then implements.

### Post-implementation checks

After implementing an approved change, Claude reports:
- Sign convention consistency
- Dimensional consistency
- A reference comment added directly above the changed line:
  ```python
  # Eq. (X.Y) in Author et al. (YYYY), arXiv:XXXX.XXXXX
  ```

### Git convention for physics changes

Prefix the commit subject line with `[PHYSICS]`:

```
[PHYSICS] fix luminosity distance prefactor in dist()
```

<!-- GSD:project-start source:PROJECT.md -->
## Project

**EMRI Parameter Estimation — HPC Integration**

A gravitational wave parameter estimation pipeline for LISA Extreme Mass Ratio Inspirals (EMRIs). Two pipelines: (1) GPU-accelerated EMRI simulation that computes SNR and Cramér-Rao bounds, and (2) CPU-based Bayesian inference that evaluates the Hubble constant posterior. This milestone adds production-ready HPC/cluster support so both pipelines run on bwUniCluster 3.0 at KIT with proper job management, environment setup, and best-practices documentation.

**Core Value:** The simulation pipeline runs reliably on the GPU cluster as SLURM array jobs, producing enough Cramér-Rao bounds for statistically meaningful Hubble constant posteriors.

### Constraints

- **GPU:** CUDA 12 required for `cupy-cuda12x` and `fastemriwaveforms-cuda12x` — must use GPU partition on cluster
- **GSL:** Build-time requirement for `fastemriwaveforms` — must be available via module or container
- **uv:** Primary package manager; must be installable on login nodes (may need local install to `~/.local/bin`)
- **Workspace:** bwHPC workspaces expire (default 30 days, extendable) — final results must be copied to persistent storage
- **Network:** Compute nodes may have restricted outbound access — all dependency installation must happen on login nodes
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.13 - All source code (`.python-version` pins 3.13; `pyproject.toml` allows `>=3.10,<3.14`)
- None (pure Python project; compiled extensions come from third-party wheels)
## Runtime
- Python 3.13 (pinned in `.python-version`)
- CUDA 12 toolkit - required on GPU cluster for `cupy-cuda12x` and `fastemriwaveforms-cuda12x`
- GSL (GNU Scientific Library) - build-time requirement for `fastemriwaveforms`
- [uv](https://docs.astral.sh/uv/) (Astral)
- Lockfile: `uv.lock` (committed, 4220 lines)
- Virtual environment created at `.venv/` by `uv sync`
## Frameworks
- NumPy - array computation, FFT, linear algebra (used everywhere)
- SciPy - integration (`quad`, `dblquad`, `fixed_quad`, `cumulative_trapezoid`), interpolation (`CubicSpline`, `RegularGridInterpolator`), statistics (`truncnorm`, `norm`, `gaussian_kde`, `erf`), optimization (`fsolve`)
- Pandas - CSV I/O for Cramer-Rao bounds, SNR analysis, galaxy catalog
- Astropy >=6.1.7 - physical constants (`astropy.constants`) and unit conversions (`astropy.units`)
- Matplotlib - all plotting (factory-function pattern in `master_thesis_code/plotting/`)
- pytest - test runner, configured in `pyproject.toml [tool.pytest.ini_options]`
- pytest-cov - coverage reporting (fail_under = 25%)
- pytest-benchmark - performance benchmarks (`-m slow`)
- Ruff - linting and formatting (target `py313`, line-length 100)
- mypy - static type checking (`disallow_untyped_defs = true`)
- pre-commit - hooks for ruff lint, ruff format, mypy
- pip-audit - dependency security scanning
- Sphinx + Furo theme - documentation (`docs/`)
## Key Dependencies
- `numpy` - array computation backbone; used in every module
- `scipy` - numerical integration, interpolation, statistics, optimization
- `pandas` - CSV read/write for simulation results and galaxy catalog
- `matplotlib` - all visualization
- `scikit-learn` - `BallTree` for galaxy catalog spatial lookups (`master_thesis_code/galaxy_catalogue/handler.py`)
- `emcee` - MCMC ensemble sampler for comoving volume sampling (`master_thesis_code/datamodels/galaxy.py`, `master_thesis_code/cosmological_model.py`)
- `tabulate` - formatted table output
- `fastlisaresponse==1.1.9` - LISA time-delay interferometry response wrapper
- `astropy>=6.1.7` - physical constants and units
- `fastemriwaveforms==2.0.0rc1` - EMRI waveform generation (imports as `few`)
- `fastemriwaveforms-cuda12x` - GPU-accelerated EMRI waveforms
- `cupy-cuda12x` - GPU array library (NumPy-compatible API on CUDA)
- `GPUtil` - GPU utilization monitoring
- `pytest`, `pytest-cov`, `pytest-benchmark` - testing
- `pip-audit` - security audit
- `mypy` - type checking
- `ruff` - linting/formatting
- `pre-commit` - git hook management
- `jupyterlab` - interactive exploration
- `sphinx>=8.1.3`, `furo>=2025.12.19`, `sphinx-copybutton>=0.5.2` - documentation
## Configuration
- No `.env` files detected; configuration is via CLI arguments and `master_thesis_code/constants.py`
- Physical constants derived from `astropy` at import time (`master_thesis_code/constants.py`)
- Simulation paths are relative strings in `constants.py` (e.g., `simulations/cramer_rao_bounds.csv`)
- `pyproject.toml` - single source of truth for project metadata, dependencies, and tool config
- `uv.lock` - exact dependency versions (committed)
- `.python-version` - pins Python 3.13
- `[tool.ruff]` - target-version `py313`, line-length 100
- `[tool.ruff.lint]` - enables E, F, I, UP, B, N rule sets; ignores physics-naming violations (N802, N803, N806, N815, N816)
- `[tool.ruff.lint.isort]` - `known-first-party = ["master_thesis_code"]`
- `[tool.mypy]` - python_version 3.13, `disallow_untyped_defs = true`, `disallow_incomplete_defs = true`
- Missing import overrides for: astropy, cupy, cupyx, fastlisaresponse, few, GPUtil, pandas, scipy, sklearn, mpl_toolkits, emcee, tabulate
- `[tool.pytest.ini_options]` - testpaths `master_thesis_code_test/`, markers: `gpu`, `slow`
- `[tool.coverage.run]` - source `master_thesis_code/`, omits test dir
- `[tool.coverage.report]` - fail_under 25%
- `.editorconfig` - 4-space indent, UTF-8, LF line endings, trailing whitespace trimmed
- `ruff-pre-commit` v0.11.0 - ruff lint (`--fix`) + ruff format
- Local hook - `uv run mypy master_thesis_code/ master_thesis_code_test/`
## Platform Requirements
- Python 3.13
- GSL (for building `fastemriwaveforms` CPU variant)
- Install: `uv sync --extra cpu --extra dev`
- Run tests: `uv run pytest -m "not gpu and not slow"`
- Python 3.13
- CUDA 12 toolkit
- GSL
- Install: `uv sync --extra gpu`
- Run: `uv run python -m master_thesis_code <working_dir> --simulation_steps N --use_gpu`
## CI/CD
- **check** job: ruff lint, ruff format check, mypy, pytest (CPU, not slow), coverage upload, pip-audit
- **integration** job (needs check): runs slow tests, uploads test plot artifacts, deploys to GitHub Pages
- **docs** job: builds Sphinx docs (`uv run make -C docs html SPHINXOPTS="-W"`)
- Runner: `ubuntu-latest`
- Dependabot: `.github/dependabot.yml` for pip + github-actions weekly
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Source modules: `snake_case.py` (e.g., `physical_relations.py`, `parameter_space.py`)
- Exception: `LISA_configuration.py` uses UPPERCASE prefix (physics convention, ruff N999 ignored)
- Test files: two patterns coexist — `test_<module>.py` (newer) and `<module>_test.py` (older). Both are valid. New tests use the `test_` prefix.
- Private modules: `_style.py`, `_helpers.py` (underscore prefix in `plotting/`)
- `snake_case` for all functions: `dist_to_redshift()`, `compute_fisher_information_matrix()`
- Physics symbols preserved in names: `S_OMS()`, `S_TM()`, `S_zz()` in `LISA_configuration.py`
- Ruff rules N802/N803/N806/N815/N816 are ignored to allow uppercase physics names
- Physics variables use standard notation: `M`, `H`, `Omega_m`, `W_0`, `M_z`, `d_L`
- Prefixes `delta_` for uncertainties: `delta_dist`, `delta_phiS`
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `SNR_THRESHOLD`, `OMEGA_M`, `SPEED_OF_LIGHT_KM_S`)
- `PascalCase`: `ParameterSpace`, `BayesianInference`, `GalaxyCatalog`, `EMRIDetection`
- Exception classes: `PascalCase` + `Error` suffix: `ParameterEstimationError`, `WaveformGenerationError`
- Physics symbols used directly as field names: `M`, `mu`, `a`, `p0`, `e0`, `x0`, `qS`, `phiS`, `qK`, `phiK`
## Code Style
- Tool: `ruff format` (via `pyproject.toml` and pre-commit)
- Line length: 100 characters (`[tool.ruff] line-length = 100`)
- Target version: Python 3.13 (`target-version = "py313"`)
- Tool: `ruff check`
- Rule sets enabled: `E` (pycodestyle), `F` (pyflakes), `I` (isort), `UP` (pyupgrade), `B` (bugbear), `N` (pep8-naming)
- Key ignores: `E501` (line length — formatter handles), `E402` (imports after mpl.rcParams), `N802`/`N803`/`N806`/`N815`/`N816` (physics uppercase), `N999` (LISA_configuration.py)
- Config: `pyproject.toml` `[tool.ruff]` and `[tool.ruff.lint]`
- 4-space indentation, LF line endings, UTF-8, trailing whitespace trimmed
- YAML: 2-space indent
## Type Annotations
- Use `list[float]` not `List[float]`, `dict[str, int]` not `Dict[str, int]`
- Use `X | None` not `Optional[X]`
- Enforced by ruff `UP` rule set (pyupgrade)
- Do NOT add `from __future__ import annotations`
## Dataclass Patterns
## Import Organization
- `master_thesis_code/__init__.py`: empty
- `master_thesis_code/plotting/__init__.py`: re-exports public API (`apply_style`, `get_figure`, `save_figure`, `make_colorbar`)
- Most subpackage `__init__.py` files are empty
## Error Handling
- `ArgumentsError`, `ParameterEstimationError`, `TimeoutError`, `ParameterOutOfBoundsError`, `WaveformGenerationError`
## Logging
## Comments and Docstrings
- Newer/refactored code: NumPy-style with `Args:` / `Returns:` / `References:` / `Examples:` sections (see `physical_relations.py`)
- Some functions use Sphinx-style `Parameters`/`Returns` with dashes (see `_helpers.py:save_figure`)
- Older code: brief one-liners or no docstring
## Git Conventions
## GPU/CPU Portability Pattern
## Protocol Pattern
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Two distinct pipelines sharing a common physical model layer
- GPU/CPU portability via guarded CuPy imports and `_get_xp()` helper
- CLI-driven entry point dispatching to pipeline functions
- Callback protocol for decoupling side effects (plotting, monitoring) from computation
- Multiprocessing in the Bayesian inference pipeline for parallelizing likelihood evaluation
- Heavy use of dataclasses for domain objects (parameters, detections, galaxies)
## Pipelines
### Pipeline 1: EMRI Simulation (data generation)
### Pipeline 2: Bayesian Inference (H0 posterior evaluation)
### Pipeline A (dev cross-check, not production)
- Synthetic `GalaxyCatalog` (from `datamodels/galaxy.py`) instead of GLADE
- erf-based analytic detection probability instead of KDE
- Hardcoded 10% fractional sigma(d_L) instead of per-source Cramer-Rao bounds
- `EMRIDetection` dataclass (from `datamodels/emri_detection.py`) instead of `Detection`
## Layers
- Purpose: Parse arguments, configure logging, dispatch to pipelines
- Location: `master_thesis_code/__main__.py`, `master_thesis_code/main.py`, `master_thesis_code/arguments.py`
- Contains: `main()`, `data_simulation()`, `evaluate()`, `snr_analysis()`, `generate_figures()`
- Purpose: EMRI event rate model, MCMC sampling of (M, z) pairs
- Location: `master_thesis_code/cosmological_model.py`
- Contains: `Model1CrossCheck`, `LamCDMScenario`, `DarkEnergyScenario`, polynomial merger rate fits
- Depends on: `emcee`, `datamodels/parameter_space.py`, `physical_relations.py`, `M1_model_extracted_data/`
- Purpose: Waveform generation, SNR computation, Fisher matrix, Cramer-Rao bounds
- Location: `master_thesis_code/parameter_estimation/parameter_estimation.py`
- Contains: `ParameterEstimation` class (549 lines)
- Depends on: `few` (waveform), `fastlisaresponse` (LISA response), `cupy` (GPU arrays), `LISA_configuration.py` (PSD), `datamodels/parameter_space.py`
- Purpose: H0 posterior evaluation from saved Cramer-Rao bounds
- Location: `master_thesis_code/bayesian_inference/bayesian_statistics.py` (988 lines), `master_thesis_code/bayesian_inference/detection_probability.py` (344 lines)
- Contains: `BayesianStatistics`, `DetectionProbability`, `single_host_likelihood()`, multiprocessing workers
- Depends on: `scipy.integrate`, `scipy.stats`, `datamodels/detection.py`, `galaxy_catalogue/handler.py`, `physical_relations.py`
- Purpose: Cosmological distance functions (luminosity distance, redshift inversion, Hubble function)
- Location: `master_thesis_code/physical_relations.py`
- Contains: `dist()`, `dist_vectorized()`, `dist_derivative()`, `hubble_function()`, `lambda_cdm_analytic_distance()`, mass conversion utilities
- Depends on: `constants.py`, `scipy.special.hyp2f1`, `scipy.optimize.fsolve`
- Purpose: LISA detector noise model (PSD for A/E/T channels), antenna patterns
- Location: `master_thesis_code/LISA_configuration.py`
- Contains: `LisaTdiConfiguration` dataclass with `power_spectral_density()`, `_get_xp()` helper
- Depends on: `constants.py`, `cupy` (guarded import)
- Purpose: Domain objects as dataclasses
- Location: `master_thesis_code/datamodels/`
- Contains:
- Purpose: Interface to GLADE galaxy catalog, BallTree spatial lookups
- Location: `master_thesis_code/galaxy_catalogue/handler.py` (669 lines)
- Contains: `GalaxyCatalogueHandler`, `HostGalaxy`, `ParameterSample`, `InternalCatalogColumns`
- Depends on: `sklearn.neighbors.BallTree`, `pandas`, `physical_relations.py`
- Purpose: All visualization, fully separated from computation
- Location: `master_thesis_code/plotting/`
- Contains: Factory functions `(data in, (fig, ax) out)` organized by topic
- Depends on: `matplotlib` only
- Purpose: All physical constants, cosmological parameters, simulation config, file paths
- Location: `master_thesis_code/constants.py`
- Contains: Module-level constants (H, OMEGA_M, SNR_THRESHOLD, LISA hardware, file paths)
## Key Abstractions
- Purpose: Represents a single EMRI parameter with bounds, epsilon, distribution function, and current value
- Examples: `master_thesis_code/datamodels/parameter_space.py`
- Pattern: `@dataclass` with 14 `Parameter` fields, each wrapped in `field(default_factory=...)`. `randomize_parameters()` iterates all non-fixed parameters.
- Purpose: Decouple side effects (plotting, monitoring) from the simulation loop
- Examples: `master_thesis_code/callbacks.py`, `master_thesis_code/plotting/simulation_plots.py` (`PlottingCallback`)
- Pattern: `typing.Protocol` with 5 hook methods. `data_simulation()` accepts `list[SimulationCallback]`.
- Purpose: Represent a detected EMRI event with measured parameters and uncertainties
- Examples: `master_thesis_code/datamodels/detection.py` (Pipeline B, from CSV), `master_thesis_code/datamodels/emri_detection.py` (Pipeline A, synthetic)
- Pattern: `@dataclass` initialized from `pd.Series` row of Cramer-Rao CSV
- Two parallel galaxy representations:
## Entry Points
- Location: `master_thesis_code/__main__.py` -> `main.py:main()`
- Invocation: `python -m master_thesis_code <working_dir> [options]`
- Dispatches to: `data_simulation()`, `evaluate()`, `snr_analysis()`, `generate_figures()`
- Location: `master_thesis_code/bayesian_inference/bayesian_inference_mwe.py` (has `__main__` block)
- Invocation: `python -m master_thesis_code.bayesian_inference.bayesian_inference_mwe`
- Location: `scripts/` directory
- `prepare_detections.py`: Post-process Cramer-Rao CSV for evaluation
- `merge_cramer_rao_bounds.py`: Merge per-index simulation CSVs
- `remove_detections_out_of_bounds.py`: Filter detections
- `estimate_hubble_constant.py`: Standalone H0 estimation
## Error Handling
- `ParameterOutOfBoundsError`: Raised during derivative computation when perturbed parameter exceeds bounds; caught in `data_simulation()` loop, iteration skipped
- `WaveformGenerationError`: Raised for invalid waveform generator configuration
- `Warning` catch: `warnings.filterwarnings("error")` converts waveform warnings (mass ratio out of bounds) to exceptions, caught and logged
- `RuntimeError`, `ValueError`, `AssertionError`: Caught from `few`/`fastlisaresponse` waveform generation failures; iteration skipped with logging
- No retry logic -- failed iterations are simply skipped and the loop continues
## GPU/CPU Portability
```python
```
## Cross-Cutting Concerns
## Backward Compatibility
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
