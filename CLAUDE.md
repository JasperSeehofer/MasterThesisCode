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

- **`parameter_estimation/parameter_estimation.py`** — waveform generation via `few`, Fisher matrix computation (5-point stencil derivatives), SNR and Cramér-Rao bounds. The `scalar_product_of_functions` inner product is the computational bottleneck (PSD loop).
- **`LISA_configuration.py`** — LISA antenna patterns (F+, F×), PSD, SSB↔detector frame transformations
- **`datamodels/parameter_space.py`** — 14-parameter EMRI space with randomization and bounds
- **`bayesian_inference/bayesian_inference_mwe.py`** — monolithic 931-line module containing `Galaxy`, `GalaxyCatalog`, `EMRIDetection`, `BayesianInference` classes; also `dist()`, `dist_to_redshift()`, and cosmological integrals
- **`cosmological_model.py`** — `Model1CrossCheck` wraps the EMRI event rate model; `BayesianStatistics` orchestrates the H₀ evaluation
- **`galaxy_catalogue/handler.py`** — interfaces with the GLADE galaxy catalog (BallTree-based lookups)
- **`constants.py`** — all physical constants and simulation configuration. Key: `H=0.73`, `SNR_THRESHOLD=20`, `IS_PLOTTING_ACTIVATED=False`

### Known Bugs to Be Aware Of

All four originally-listed bugs are resolved. Remaining known issues:

1. **`LISA_configuration.py` unconditional `import cupy`**: still at module top level — any
   module that imports `LisaTdiConfiguration` is un-importable on CPU-only machines without
   the guarded `try/except`. Fix when that file is next touched.
2. **`cosmological_model.py` size**: ~3530 lines; `BayesianStatistics` not yet extracted.

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
