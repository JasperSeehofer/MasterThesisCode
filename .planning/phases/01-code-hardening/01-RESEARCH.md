# Phase 1: Code Hardening - Research

**Researched:** 2026-03-26
**Domain:** Python GPU/CPU portability, CLI argument threading, guarded imports
**Confidence:** HIGH

## Summary

Phase 1 addresses three concrete code hardening requirements: adding `--use_gpu` and `--num_workers` CLI flags, making `MemoryManagement` CPU-safe, and guarding all unconditional GPU imports. The codebase already has the right patterns established (`_get_xp`, try/except cupy guards) in some files -- the work is completing coverage across all modules and threading the new flags through the call chain.

The main technical risks are: (1) a circular import between `bayesian_statistics.py` and `cosmological_model.py` that currently prevents `python -m master_thesis_code --help` from working, (2) `memory_management.py` has an unconditional `import GPUtil` at module level that crashes on CPU, and (3) `main.py:187` calls `memory_pool.free_all_blocks()` without a None check, which will crash on CPU even after the import is fixed.

**Primary recommendation:** Fix the four import-guarding files in dependency order (decorators first, then LISA_configuration, then memory_management, then parameter_estimation threading), break the circular import, and add CLI flags last since they depend on all the above being CPU-safe.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Fix `LISA_configuration.py` unconditional `import cupy` -- guard with try/except and apply the `_get_xp` pattern so the module is importable on CPU
- **D-02:** Sweep all files with unconditional cupy/GPUtil imports (`LISA_configuration.py`, `memory_management.py`, `decorators.py`, `parameter_estimation.py`) and guard them in one pass. Consistent result across the codebase.
- **D-03:** Silent no-op on CPU -- guard `GPUtil` import, `__init__` succeeds without GPU, `gpu_usage_stamp()` records zeros, `display_GPU_information()` logs "No GPU available". Callers don't need to check or conditionally instantiate.
- **D-04:** `--use_gpu` added to `arguments.py`, defaults to `False`. Threaded through `data_simulation()`, `snr_analysis()`, constructors (`ParameterEstimation`, `MemoryManagement`). Matches existing pattern where `ParameterEstimation` already accepts `use_gpu`.
- **D-05:** No config/settings object -- just pass `use_gpu: bool` through the call chain. Simple and explicit.
- **D-06:** `--num_workers` added to `arguments.py`. Default when omitted: `os.sched_getaffinity(0) - 2` (minimum 1). On SLURM clusters, `sched_getaffinity` respects cgroup limits.
- **D-07:** Remove the existing affinity-expansion hack in `bayesian_statistics.py` (lines 245-251) that calls `os.sched_setaffinity(0, range(cpu_count))`.

### Claude's Discretion
- Implementation details of the `_get_xp` helper in LISA_configuration.py (follow existing pattern in parameter_estimation.py)
- Exact error messages and log levels for CPU fallback paths
- Test organization for new CPU-importability tests

### Deferred Ideas (OUT OF SCOPE)
- Profiling improvements beyond current `MemoryManagement` class
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CODE-01 | `--use_gpu` CLI flag added to `arguments.py` and threaded through `data_simulation()`, `snr_analysis()`, `ParameterEstimation`, and `MemoryManagement` | Existing `use_gpu` param in `ParameterEstimation.__init__` and `waveform_generator.py` functions; `arguments.py` parser needs two new arguments; `main.py` call chain documented |
| CODE-02 | `MemoryManagement` is CPU-safe -- guards `GPUtil` import, provides no-op methods when GPU unavailable, does not crash on CPU-only nodes | `memory_management.py` has unconditional `import GPUtil` at line 4; `main.py:187` calls `memory_pool.free_all_blocks()` without None check; full no-op pattern documented |
| CODE-03 | `--num_workers` CLI flag controls multiprocessing pool size in `BayesianStatistics.evaluate()`, defaulting to `os.sched_getaffinity() - 2` when omitted | `bayesian_statistics.py:237-253` already computes `available_cpus` and uses `available_cpus - 2`; affinity-expansion hack at lines 244-253 is commented out but should be removed |
</phase_requirements>

## Standard Stack

No new libraries are needed. All changes use the existing Python standard library and project dependencies.

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| argparse | stdlib | CLI argument parsing | Already used in `arguments.py` |
| os | stdlib | `sched_getaffinity`, `cpu_count` | Already used in `bayesian_statistics.py` |
| logging | stdlib | CPU fallback log messages | Already used everywhere |

## Architecture Patterns

### Pattern 1: Guarded GPU Import (already established)

**What:** Try/except around cupy import with `_CUPY_AVAILABLE` sentinel flag.
**When to use:** Every module that references cupy or GPUtil.
**Reference implementation:** `parameter_estimation/parameter_estimation.py` lines 20-28.

```python
try:
    import cupy as cp
    import cupyx.scipy.fft as cufft
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cufft = None
    _CUPY_AVAILABLE = False
```

### Pattern 2: Array Namespace Helper (`_get_xp`)

**What:** Function returning `cp` or `np` based on a flag or array type.
**Variants in codebase:**
- `LISA_configuration.py:34` -- takes an array, returns matching module: `_get_xp(arr: Any) -> types.ModuleType`
- `parameter_estimation.py` -- (not at module level but used internally)

The LISA_configuration variant dispatches on array type (duck-typed). For the hardening phase, keep this pattern as-is since it already works after the import guard is fixed.

### Pattern 3: No-Op Service on CPU

**What:** A class that initializes successfully without GPU but provides zero/empty/logged results.
**When to use:** `MemoryManagement` -- callers should not need conditional logic.

```python
try:
    import GPUtil
    _GPUTIL_AVAILABLE = True
except ImportError:
    GPUtil = None  # type: ignore[assignment]
    _GPUTIL_AVAILABLE = False

class MemoryManagement:
    def __init__(self) -> None:
        if _GPUTIL_AVAILABLE:
            self._gpu_monitor = GPUtil.getGPUs()
        else:
            self._gpu_monitor = []
            _LOGGER.info("No GPU monitoring available (GPUtil not installed).")
        # ... rest of init
```

### Pattern 4: Lazy Imports for GPU-Only Code Paths

**What:** Import GPU-dependent modules inside functions, not at module level.
**Already used:** `main.py:data_simulation()` imports `MemoryManagement` and `ParameterEstimation` inside the function (lines 161-165). This pattern is correct and should be preserved.

### Anti-Patterns to Avoid
- **Unconditional GPU import at module level:** `memory_management.py` line 4 (`import GPUtil`). This is the primary bug -- crashes the entire import chain on CPU.
- **Assuming `memory_pool` is not None:** `main.py:187` calls `memory_pool.free_all_blocks()` directly. Must add a None guard or have `MemoryManagement` expose a safe method.
- **`use_gpu` defaulting to `True`:** `ParameterEstimation.__init__` and `waveform_generator.py` default to `True`. After this phase, `ParameterEstimation` should receive the flag from the caller (default `False` at CLI level, explicit everywhere else).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CPU count detection | Custom `/proc/cpuinfo` parsing | `os.sched_getaffinity(0)` with `os.cpu_count()` fallback | Respects cgroups/SLURM; cross-platform |
| GPU availability detection | Environment variable checks | `try: import cupy` pattern | Direct, reliable, already established |

## Common Pitfalls

### Pitfall 1: Circular Import Between bayesian_statistics and cosmological_model
**What goes wrong:** `python -m master_thesis_code --help` crashes with `ImportError: cannot import name 'ADDITIONAL_GALAXIES_WITHOUT_BH_MASS' from partially initialized module`.
**Why it happens:** `main.py` imports `BayesianStatistics` from `bayesian_statistics.py` at module level (line 14). `bayesian_statistics.py` imports `LamCDMScenario, Model1CrossCheck` from `cosmological_model.py` (line 36). `cosmological_model.py` re-exports symbols from `bayesian_statistics.py` at module level (lines 368+). This creates a circular import.
**How to avoid:** Move the `BayesianStatistics` import in `main.py` inside the `evaluate()` function (it is only used there). This makes `main.py` importable without triggering the circular chain.
**Warning signs:** Any `--help` or bare `import main` failing.
**Verified:** Confirmed by running `uv run python -m master_thesis_code --help` on the current codebase -- it crashes.

### Pitfall 2: memory_pool.free_all_blocks() on CPU
**What goes wrong:** `data_simulation()` at line 187 calls `memory_management.memory_pool.free_all_blocks()` -- `memory_pool` is `None` on CPU, so this raises `AttributeError`.
**Why it happens:** `MemoryManagement.__init__` sets `self.memory_pool = None` when cupy is unavailable, but callers assume it exists.
**How to avoid:** Either (a) expose a `free_gpu_memory()` method on `MemoryManagement` that no-ops when `memory_pool` is None, or (b) add a None check in `main.py`. Option (a) is cleaner and matches D-03 (callers don't need conditional logic).
**Warning signs:** `AttributeError: 'NoneType' object has no attribute 'free_all_blocks'`.

### Pitfall 3: os.sched_getaffinity Not Available on All Platforms
**What goes wrong:** `os.sched_getaffinity` raises `AttributeError` on macOS and some non-Linux systems.
**Why it happens:** It is a Linux-specific syscall.
**How to avoid:** The existing code in `bayesian_statistics.py:237-239` already handles this with try/except AttributeError fallback to `os.cpu_count()`. Preserve this pattern when moving the logic to `arguments.py`.
**Warning signs:** `AttributeError: module 'os' has no attribute 'sched_getaffinity'`.

### Pitfall 4: Default use_gpu=True in Existing Constructors
**What goes wrong:** If `ParameterEstimation(use_gpu=True)` is called without the CLI flag being threaded, it attempts GPU operations on CPU.
**Why it happens:** `ParameterEstimation.__init__` defaults `use_gpu=True` (line 79). `waveform_generator.py:create_lisa_response_generator` defaults `use_gpu=True` (line 46).
**How to avoid:** Do NOT change these defaults in this phase (it would break the cluster). Instead, ensure the CLI flag is always explicitly passed from `main.py`. The signature stays `use_gpu: bool = True` but callers always provide the value.

### Pitfall 5: GPUtil.getGPUs() Behavior Without GPUs
**What goes wrong:** Even when GPUtil is installed, `GPUtil.getGPUs()` returns an empty list when no GPU is present (does not crash). But `gpu_usage_stamp()` at line 34 iterates over `GPUtil.getGPUs()` -- returns empty list, so `_gpu_usage` gets `[]` appended. This is fine.
**Mitigation:** The real issue is `import GPUtil` failing when GPUtil is not installed at all (CPU-only dev machines without the gpu extra). Guard the import.

## Code Examples

### Guarded GPUtil Import for MemoryManagement
```python
# Source: Follows established pattern from parameter_estimation.py
import logging
from time import time

try:
    import GPUtil
    _GPUTIL_AVAILABLE = True
except ImportError:
    GPUtil = None  # type: ignore[assignment]
    _GPUTIL_AVAILABLE = False

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

_LOGGER = logging.getLogger()


class MemoryManagement:
    def __init__(self) -> None:
        if _GPUTIL_AVAILABLE:
            self._gpu_monitor = GPUtil.getGPUs()
        else:
            self._gpu_monitor = []
        if _CUPY_AVAILABLE and cp is not None:
            self.memory_pool = cp.get_default_memory_pool()
            self._fft_cache = cp.fft.config.get_plan_cache()
        else:
            self.memory_pool = None
            self._fft_cache = None
        self._start_time = time()
        self._memory_pool_gpu_usage: list[float] = []
        self._gpu_usage: list[list[float]] = []
        self._time_series: list[float] = []

    def free_gpu_memory(self) -> None:
        """Free GPU memory pool blocks. No-op on CPU."""
        if self.memory_pool is not None:
            self.memory_pool.free_all_blocks()

    def gpu_usage_stamp(self) -> None:
        self._time_series.append(time() - self._start_time)
        if _GPUTIL_AVAILABLE:
            self._gpu_usage.append([gpu.memoryUsed / 1000 for gpu in GPUtil.getGPUs()])
        else:
            self._gpu_usage.append([])
        if self.memory_pool is not None:
            self._memory_pool_gpu_usage.append(int(self.memory_pool.total_bytes()) / 10**9)
        else:
            self._memory_pool_gpu_usage.append(0.0)

    def display_GPU_information(self) -> None:
        if not _GPUTIL_AVAILABLE or not self._gpu_monitor:
            _LOGGER.info("No GPU available.")
            return
        # ... existing GPU display logic
```

### CLI Flag Addition in arguments.py
```python
# In _parse_arguments():
parser.add_argument(
    "--use_gpu",
    action="store_true",
    default=False,
    help="Use GPU acceleration (requires CUDA and cupy). Default: CPU only.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=None,
    help="Number of multiprocessing workers for Bayesian inference. "
         "Default: available CPUs - 2 (minimum 1).",
)
```

### Worker Count Default Logic
```python
# In Arguments class:
@property
def num_workers(self) -> int:
    """Number of multiprocessing workers for Bayesian inference."""
    raw = self._parsed_arguments.num_workers
    if raw is not None:
        return max(1, raw)
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, available - 2)
```

### Breaking the Circular Import in main.py
```python
# Current (crashes):
from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics

# Fixed -- move inside evaluate():
def evaluate(
    cosmological_model: Model1CrossCheck,
    galaxy_catalog: GalaxyCatalogueHandler,
    h_value: float,
    num_workers: int | None = None,
) -> None:
    from master_thesis_code.bayesian_inference.bayesian_statistics import BayesianStatistics
    hubble_constant_evaluation = BayesianStatistics()
    hubble_constant_evaluation.evaluate(galaxy_catalog, cosmological_model, h_value)
```

## Existing Code Audit Results

### Files with Unconditional GPU Imports

| File | Line | Import | Status | Fix |
|------|------|--------|--------|-----|
| `memory_management.py` | 4 | `import GPUtil` | CRASHES on CPU | Guard with try/except |
| `memory_management.py` | 7-13 | `import cupy` | Already guarded | OK |
| `LISA_configuration.py` | 16-22 | `import cupy` | Already guarded (try/except) | OK (CLAUDE.md was stale) |
| `decorators.py` | 7-13 | `import cupy` | Already guarded | OK |
| `parameter_estimation.py` | 20-28 | `import cupy, cupyx` | Already guarded | OK |

**Key finding:** CLAUDE.md says `LISA_configuration.py` has an unconditional cupy import, but the actual code (lines 16-22) already has a proper try/except guard. Only `memory_management.py` line 4 (`import GPUtil`) is truly broken.

### Files Needing Call Chain Updates for use_gpu

| File | Current | Change Needed |
|------|---------|---------------|
| `arguments.py` | No `--use_gpu` flag | Add flag |
| `main.py:main()` | Does not pass `use_gpu` | Read from `arguments.use_gpu`, pass to functions |
| `main.py:data_simulation()` | No `use_gpu` param | Add param, pass to `ParameterEstimation` and `MemoryManagement` |
| `main.py:snr_analysis()` | No `use_gpu` param | Add param, pass to constructors |
| `main.py:187` | `memory_management.memory_pool.free_all_blocks()` | Use `memory_management.free_gpu_memory()` |
| `ParameterEstimation.__init__` | `use_gpu: bool = True` | Keep default, but callers must pass explicitly |
| `waveform_generator.py` | `use_gpu: bool = True` | Keep default, callers pass explicitly |

### Circular Import Chain
```
main.py (line 14) imports bayesian_statistics.BayesianStatistics
  -> bayesian_statistics.py (line 36) imports cosmological_model.LamCDMScenario
    -> cosmological_model.py (line 368+) imports bayesian_statistics.ADDITIONAL_GALAXIES_WITHOUT_BH_MASS
      -> CIRCULAR (bayesian_statistics is still initializing)
```

**Fix:** Make the `BayesianStatistics` import in `main.py` lazy (move inside `evaluate()`).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (with pytest-cov, pytest-benchmark) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] |
| Quick run command | `uv run pytest -m "not gpu and not slow" --tb=short -q` |
| Full suite command | `uv run pytest --tb=short -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CODE-01a | `--use_gpu` flag accepted by argument parser | unit | `uv run pytest master_thesis_code_test/test_arguments.py::test_use_gpu_flag -x` | Wave 0 |
| CODE-01b | `--help` shows `--use_gpu` and `--num_workers` | smoke | `uv run python -m master_thesis_code --help` (grep for flags) | Wave 0 |
| CODE-01c | `data_simulation` accepts and threads `use_gpu` | unit | `uv run pytest master_thesis_code_test/test_main.py::test_data_simulation_use_gpu -x` | Wave 0 |
| CODE-02a | `MemoryManagement` importable on CPU | smoke | `uv run python -c "from master_thesis_code.memory_management import MemoryManagement"` | Wave 0 |
| CODE-02b | `MemoryManagement()` instantiable on CPU | unit | `uv run pytest master_thesis_code_test/test_memory_management.py::test_cpu_instantiation -x` | Wave 0 |
| CODE-02c | `gpu_usage_stamp()` and `display_GPU_information()` no-op on CPU | unit | `uv run pytest master_thesis_code_test/test_memory_management.py::test_cpu_noop -x` | Wave 0 |
| CODE-03a | `--num_workers` flag accepted by argument parser | unit | `uv run pytest master_thesis_code_test/test_arguments.py::test_num_workers_flag -x` | Wave 0 |
| CODE-03b | Default num_workers is `sched_getaffinity - 2` (min 1) | unit | `uv run pytest master_thesis_code_test/test_arguments.py::test_num_workers_default -x` | Wave 0 |
| REGR | Existing CPU test suite passes | regression | `uv run pytest -m "not gpu and not slow" --tb=short -q` | Existing (151 tests) |

### Sampling Rate
- **Per task commit:** `uv run pytest -m "not gpu and not slow" --tb=short -q`
- **Per wave merge:** `uv run pytest -m "not gpu and not slow" --tb=short -q` + `uv run mypy master_thesis_code/`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `master_thesis_code_test/test_memory_management.py` -- covers CODE-02a, CODE-02b, CODE-02c
- [ ] `master_thesis_code_test/test_arguments.py` -- covers CODE-01a, CODE-03a, CODE-03b (may already exist partially)
- [ ] Smoke test for `--help` output -- covers CODE-01b

## Project Constraints (from CLAUDE.md)

- **Package manager:** uv (never manually edit pyproject.toml deps)
- **Python:** 3.13, use native type syntax (no `from typing import List`, etc.)
- **Type annotations:** All functions must have complete type annotations (enforced by mypy `disallow_untyped_defs`)
- **Physics Change Protocol:** NOT triggered by this phase (no formula/constant changes)
- **Pre-commit hooks:** ruff lint, ruff format, mypy run on every commit
- **GPU imports:** Must always be guarded with try/except
- **`_get_xp` pattern:** Mandatory for GPU/CPU portability
- **`USE_GPU` flag:** Must never be hardcoded True; must come from CLI `--use_gpu`
- **Testing:** CPU tests must pass without GPU; GPU tests gated behind `@pytest.mark.gpu`
- **Skill triggers:** `/check` before any commit; no physics files are being modified so `/physics-change` is not triggered

## Sources

### Primary (HIGH confidence)
- Direct code inspection of all target files in the repository
- Runtime verification: `uv run python -c "from master_thesis_code.memory_management import MemoryManagement"` -- confirmed crash
- Runtime verification: `uv run python -m master_thesis_code --help` -- confirmed circular import crash
- Runtime verification: `uv run python -c "from master_thesis_code.LISA_configuration import LisaTdiConfiguration"` -- confirmed working (guard already present)

### Secondary (MEDIUM confidence)
- Python `os.sched_getaffinity` documentation -- Linux-specific, not available on macOS/Windows

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no new libraries, all stdlib
- Architecture: HIGH -- patterns already exist in codebase, just need consistent application
- Pitfalls: HIGH -- all pitfalls verified by running actual code on the dev machine

**Research date:** 2026-03-26
**Valid until:** 2026-04-26 (stable domain, no external dependencies changing)
