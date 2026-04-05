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
uv run mypy master_thesis_code/ master_thesis_code_test/
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
# EMRI simulation (generates SNR + Cramer-Rao bounds)
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

## Architecture

The codebase has two distinct pipelines:

### 1. EMRI Simulation Pipeline
`main.py:data_simulation()` drives a loop over `simulation_steps`:
1. `Model1CrossCheck` (cosmological model) samples EMRI events from a distribution
2. `GalaxyCatalogueHandler` resolves each event to a host galaxy from a galaxy catalog
3. `ParameterSpace.randomize_parameters()` + `set_host_galaxy_parameters()` set up the 14-parameter EMRI
4. `ParameterEstimation.compute_signal_to_noise_ratio()` computes SNR using a LISA waveform
5. If SNR >= threshold: `compute_Cramer_Rao_bounds()` computes the Fisher matrix and saves to CSV

### 2. Bayesian Inference Pipeline
`main.py:evaluate()` -> `BayesianStatistics.evaluate()`:
- Loads saved Cramer-Rao bounds from CSV
- Uses `BayesianInference` (in `bayesian_inference/bayesian_inference_mwe.py`) to compute the posterior over H0
- `GalaxyCatalog` models the galaxy distribution and mass distribution using normal/truncnorm distributions

### Key Module Responsibilities

- **`parameter_estimation/parameter_estimation.py`** — waveform generation via `few`, Fisher matrix computation (forward-difference derivatives; 5-point stencil method exists but is not yet called — see Known Bug 4), SNR and Cramer-Rao bounds. The `scalar_product_of_functions` inner product is the computational bottleneck (PSD loop).
- **`LISA_configuration.py`** — LISA antenna patterns (F+, Fx), PSD, SSB<->detector frame transformations
- **`datamodels/parameter_space.py`** — 14-parameter EMRI space with randomization and bounds
- **`bayesian_inference/bayesian_inference.py`** — Pipeline A (dev cross-check): `BayesianInference`, erf-based detection probability, hardcoded 10% sigma(d_L), synthetic `GalaxyCatalog`. Not used by `--evaluate`.
- **`bayesian_inference/bayesian_inference_mwe.py`** — thin re-export shim; `__main__` block runs Pipeline A standalone
- **`bayesian_inference/bayesian_statistics.py`** — Pipeline B (production): `BayesianStatistics`, `single_host_likelihood`, multiprocessing workers, helper functions. Invoked by `--evaluate`.
- **`bayesian_inference/detection_probability.py`** — `DetectionProbability` class: KDE-based detection probability with `RegularGridInterpolator` look-ups. Used by Pipeline B.
- **`cosmological_model.py`** — `Model1CrossCheck` wraps the EMRI event rate model; `LamCDMScenario`, `DarkEnergyScenario` parameter spaces. Backward-compat re-exports of `BayesianStatistics` and `DetectionProbability`.
- **`galaxy_catalogue/handler.py`** — interfaces with the GLADE galaxy catalog (BallTree-based lookups)
- **`constants.py`** — all physical constants and simulation configuration. Key: `H=0.73`, `SNR_THRESHOLD=20`
- **`plotting/`** — all visualization code lives here. Factory functions (`data in, (fig, ax) out`) in topic modules (`bayesian_plots.py`, `evaluation_plots.py`, `model_plots.py`, `catalog_plots.py`, etc.). `_style.py` sets Agg backend + loads `emri_thesis.mplstyle`. `_helpers.py` provides `save_figure()` and `get_figure()`.
- **`callbacks.py`** — `SimulationCallback` Protocol for decoupling the simulation loop from visualization; `PlottingCallback` in `plotting/simulation_plots.py` collects data and produces plots in `on_simulation_end`

### Known Bugs to Be Aware Of

#### Code health
1. **`LISA_configuration.py` unconditional `import cupy`**: still at module top level — any module that imports `LisaTdiConfiguration` is un-importable on CPU-only machines without the guarded `try/except`. Fix when that file is next touched.

#### Physics / mathematics (Physics Change Protocol required)
4. **`parameter_estimation.py:336` Fisher matrix uses O(e) forward difference** [HIGH]: calls `finite_difference_derivative()` instead of `five_point_stencil_derivative()`. Ref: Vallisneri (2008) arXiv:gr-qc/0703086.
5. **`LISA_configuration.py` galactic confusion noise absent from PSD** [MEDIUM]: constants defined in `constants.py:77-83` but never used. Ref: Babak et al. (2023) arXiv:2303.15929 Eq. (17).
6. **`physical_relations.py:72` wCDM params w0, wa silently ignored** [MEDIUM]: `dist()` accepts them but passes to a hardcoded-LCDM hypergeometric function.
7. **`bayesian_inference/bayesian_inference.py` hardcoded 10% distance error** [MEDIUM]: uses `FRACTIONAL_LUMINOSITY_ERROR` instead of per-source Cramer-Rao bound from CSV.
8. **`constants.py:29-30` outdated WMAP-era cosmology** [LOW]: Omega_m = 0.25, H = 0.73; Planck 2018 best-fit is Omega_m = 0.3153, H = 0.6736.
9. **`datamodels/galaxy.py:64` galaxy redshift uncertainty non-standard scaling** [LOW]: `0.013 * (1+z)^3` has no reference; standard forms scale as (1+z).

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

Never use a mutable object as a bare default in a `@dataclass`. Python 3.13 raises `ValueError` at class-definition time. Always wrap with `field(default_factory=...)`:

```python
# Wrong: bar: MyMutableClass = MyMutableClass()  — crashes Python 3.13
# Correct:
bar: MyMutableClass = field(default_factory=MyMutableClass)
```

---

## Typing Conventions

All public and private functions/methods must have complete type annotations on every parameter and on the return type. The only exception is `__init__` where the return type may be omitted.

- Use `list[float]` not `List[float]`, `dict[str, int]` not `Dict[str, int]`, `X | None` not `Optional[X]`. Do **not** add `from __future__ import annotations`.
- Use `npt.NDArray[np.float64]` for typed arrays. Never use bare `np.ndarray` without a dtype parameter.
- CuPy has no mypy stubs. Annotate GPU-capable functions with `npt.NDArray[np.float64]` and add a comment that cupy arrays are also accepted at runtime. Never use `cp.ndarray` as a type annotation.
- Use `Callable` from `typing`, never lowercase `callable`. For signature-preserving decorators, use a `TypeVar` bound to `Callable[..., Any]` with `@functools.wraps`.

**mypy:** Config in `pyproject.toml`. Key flags: `disallow_untyped_defs = true`, `disallow_incomplete_defs = true`. CuPy, `few`, `fastlisaresponse`, and `GPUtil` are under `ignore_missing_imports`.

---

## HPC / GPU Best Practices

This code runs on a GPU cluster (CuPy/CUDA) but must also be importable and testable on a CPU-only development machine. The patterns below are mandatory.

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

### Rules

- **GPU imports must always be guarded.** Never place `import cupy as cp` at module top level unconditionally. Known issues in `decorators.py`, `memory_management.py`, `LISA_configuration.py`, `parameter_estimation.py` — fix when touched.
- **Vectorize array operations.** Never iterate over array elements in a hot path. Use vectorized `xp.*` operations (e.g., `xp.trapz(integrant / psd, x=fs)` instead of a Python loop).
- **Avoid GPU-to-CPU transfers in hot paths.** Do not call `cp.asnumpy()` or `.get()` inside functions called thousands of times. Keep data on GPU until a single scalar result.
- **GPU memory management.** Free GPU memory after each full simulation step (`cp.get_default_memory_pool().free_all_blocks()`). Do not call inside inner loops — the CuPy allocator reuses blocks.
- **USE_GPU flag.** Must never be hardcoded `True`. Must come from `--use_gpu` CLI argument and be threaded into every constructor. No module-level constant should control GPU behavior.

---

## Testing Strategy

Tests must be fully runnable on a CPU-only development machine. Code written with the `xp` pattern is testable on CPU without mocking.

```bash
pytest -m "not gpu"                # dev machine (CPU only)
pytest                             # cluster (GPU available)
pytest -m "not gpu and not slow"   # fast subset only
```

- **GPU marker:** Any test requiring a real CUDA GPU must use `@pytest.mark.gpu`. Tests for math/physical functions (`dist`, `power_spectral_density`, etc.) must NOT require GPU.
- **xp fixture:** The `xp` fixture in `conftest.py` parametrizes tests over `numpy` (always) and `cupy` (when available). Use `use_gpu=(xp.__name__ == "cupy")` to thread the flag.
- **Guarding cupy imports:** Test files importing modules that transitively depend on `LISA_configuration` must guard the import with `try/except` and use `pytest.mark.skipif` or `pytest.importorskip("cupy")`.

### Test priority order

1. **Physical correctness** — functions with known analytical limits: `dist(z=0) == 0.0`, `power_spectral_density(f) > 0`, `gw_detection_probability` in `[0, 1]`, `scalar_product(h, h) > 0`
2. **Bounds** — `ParameterSpace` randomized values stay within declared limits; `_parameters_to_dict` returns the correct 14 keys
3. **Regression** — before changing any formula, add a test asserting the old numerical result so the change is verifiable

---

## Math/Physics Validation Workflow

Errors in physics formulas produce subtly wrong results with no crash. A strict protocol applies.

### What counts as a physics change

A change is a **physics change** if it touches any of:
- A formula (integrals, inner products, distance-redshift relations, posteriors, likelihoods)
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

**EMRI Dark Siren H₀ Inference**

A dark siren inference pipeline for measuring the Hubble constant H₀ from LISA Extreme Mass Ratio Inspiral (EMRI) detections. Two pipelines: (1) GPU-accelerated EMRI simulation that computes SNR and Cramer-Rao bounds on bwUniCluster 3.0, and (2) CPU-based Bayesian inference that evaluates the H₀ posterior using the GLADE+ galaxy catalog with completeness correction.

**Core Value:** Measure H₀ from simulated EMRI dark siren events with galaxy catalog completeness correction (Gray et al. 2020), producing publication-ready results.

### Constraints

- **GPU:** CUDA 12 required for `cupy-cuda12x` and `fastemriwaveforms-cuda12x` — must use GPU partition on cluster
- **GSL:** Build-time requirement for `fastemriwaveforms` — must be available via module or container
- **uv:** Primary package manager; must be installable on login nodes (may need local install to `~/.local/bin`)
- **Workspace:** bwHPC workspaces expire (default 30 days, extendable) — final results must be copied to persistent storage
- **Network:** Compute nodes may have restricted outbound access — all dependency installation must happen on login nodes
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

See `pyproject.toml` for complete dependency list and tool configuration.
Key: Python 3.13, NumPy/SciPy/Pandas/Matplotlib, CuPy (GPU), fastemriwaveforms (EMRI waveforms), astropy (constants).
Dev tools: ruff (lint/format), mypy (types), pytest (tests), pre-commit (hooks).
CI: check (lint+type+test on source+tests), integration (slow tests), docs (Sphinx), pages (docs+plots deploy on main).
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

See Typing Conventions, Dataclass Conventions, and HPC/GPU sections above for detailed rules.

- **Files:** `snake_case.py` for source, `test_<module>.py` for new tests. Exception: `LISA_configuration.py` (physics convention).
- **Functions:** `snake_case` — `dist_to_redshift()`, `compute_fisher_information_matrix()`
- **Classes:** `PascalCase` — `ParameterSpace`, `BayesianInference`. Exceptions: `PascalCase` + `Error` suffix.
- **Constants:** `SCREAMING_SNAKE_CASE` — `SNR_THRESHOLD`, `OMEGA_M`, `SPEED_OF_LIGHT_KM_S`
- **Physics symbols** preserved in names: `M`, `H`, `d_L`, `S_OMS()`, `delta_dist`. Ruff N802/N803/N806/N815/N816 ignored.
- **Docstrings:** NumPy-style (`Args:` / `Returns:` / `References:`) for new code.
- **Errors:** `ArgumentsError`, `ParameterEstimationError`, `TimeoutError`, `ParameterOutOfBoundsError`, `WaveformGenerationError`
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture (GSD)

See Architecture section above for pipeline descriptions and module responsibilities.
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD or GPD workflow unless the user explicitly asks to bypass it.

### GSD -> GPD Routing

GSD is the primary command surface. When a GSD workflow encounters **physics work**, it must delegate to the corresponding GPD command instead of handling it with GSD agents. This applies whether the user invokes GSD explicitly or the system routes automatically.

**A task is physics work if it:**
- Modifies a formula, physical constant, PSD coefficient, waveform parameter, or frequency limit (same trigger as `/physics-change`)
- Involves a derivation, dimensional analysis, limiting-case check, or convergence study
- Requires literature lookup for a physics method or known result
- Is described in physics terms (e.g., "fix the Fisher matrix stencil", "add confusion noise to PSD")

**Routing table (GSD command -> GPD equivalent):**

| GSD command | When physics-flagged, delegate to | Notes |
|---|---|---|
| `/gsd:plan-phase N` | `/gpd:plan-phase N` | GPD planner adds verification criteria, limiting cases |
| `/gsd:execute-phase N` | `/gpd:execute-phase N` | GPD executor applies physics protocols (dimensional analysis, sign checks) |
| `/gsd:quick` | `/gpd:quick` | GPD quick mode with physics guarantees |
| `/gsd:debug` | `/gpd:debug` | GPD debugger uses scientific method, checks dimensions/limits |
| `/gsd:verify-work` | `/gpd:verify-work` | GPD verifier checks physics correctness, not just task completion |
| `/gsd:research-phase N` | `/gpd:research-phase N` | GPD researcher surveys physics literature |
| `/gsd:discuss-phase N` | `/gpd:discuss-phase N` | Either system works; GPD if physics-heavy |

**What stays in GSD (never routed):**
- Code refactoring, test infrastructure, CI/CD, cluster scripts, documentation
- Dependency management, import cleanup, type annotation work
- Plotting code changes (unless the plot formula itself is physics)
- `/gsd:ship`, `/gsd:pr-branch`, `/gsd:profile-user`, `/gsd:settings`

**Mixed phases:** If a GSD phase contains both software and physics tasks, keep the phase in GSD but invoke GPD for the physics subtasks. The GSD phase tracks overall progress; GPD handles the physics execution with its protocols.

**State tracking:** GSD tracks progress in `.planning/`, GPD in `.gpd/`. Both systems commit atomically. No cross-updates needed — they are independent ledgers for independent concerns.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
