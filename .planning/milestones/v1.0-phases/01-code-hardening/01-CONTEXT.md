# Phase 1: Code Hardening - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the codebase importable and functional on CPU-only nodes (cluster login nodes, dev machines without CUDA) while continuing to run correctly on GPU compute nodes. Adds `--use_gpu` and `--num_workers` CLI flags and fixes all unconditional GPU imports.

</domain>

<decisions>
## Implementation Decisions

### Import Guards (LISA_configuration.py + all unguarded files)
- **D-01:** Fix `LISA_configuration.py` unconditional `import cupy` — guard with try/except and apply the `_get_xp` pattern so the module is importable on CPU
- **D-02:** Sweep all files with unconditional cupy/GPUtil imports (`LISA_configuration.py`, `memory_management.py`, `decorators.py`, `parameter_estimation.py`) and guard them in one pass. Consistent result across the codebase.

### MemoryManagement CPU Behavior
- **D-03:** Silent no-op on CPU — guard `GPUtil` import, `__init__` succeeds without GPU, `gpu_usage_stamp()` records zeros, `display_GPU_information()` logs "No GPU available". Callers don't need to check or conditionally instantiate.

### CLI Flag Threading
- **D-04:** `--use_gpu` added to `arguments.py`, defaults to `False` (safe default — user must explicitly enable GPU). Threaded as a plain argument through `data_simulation()`, `snr_analysis()`, constructors (`ParameterEstimation`, `MemoryManagement`). Matches existing pattern where `ParameterEstimation` already accepts `use_gpu`.
- **D-05:** No config/settings object — just pass `use_gpu: bool` through the call chain. Simple and explicit.

### Worker Count Default
- **D-06:** `--num_workers` added to `arguments.py`. Default when omitted: `os.sched_getaffinity(0) - 2` (minimum 1). On SLURM clusters, `sched_getaffinity` respects cgroup limits so this automatically matches the allocation.
- **D-07:** Remove the existing affinity-expansion hack in `bayesian_statistics.py` (lines 245-251) that calls `os.sched_setaffinity(0, range(cpu_count))` — it fights SLURM cgroup isolation and is fragile on shared nodes.

### Claude's Discretion
- Implementation details of the `_get_xp` helper in LISA_configuration.py (follow existing pattern in parameter_estimation.py)
- Exact error messages and log levels for CPU fallback paths
- Test organization for new CPU-importability tests

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Existing Code (primary targets)
- `master_thesis_code/arguments.py` — CLI argument definitions; add `--use_gpu` and `--num_workers` here
- `master_thesis_code/main.py` — Entry point; threads flags to `data_simulation()`, `snr_analysis()`, `evaluate()`
- `master_thesis_code/memory_management.py` — Unconditional `GPUtil` import at line 4; `MemoryManagement` class needs CPU no-op
- `master_thesis_code/LISA_configuration.py` — Unconditional `import cupy` at line 17; needs guard + `_get_xp` pattern
- `master_thesis_code/decorators.py` — Has cupy import; verify guard status
- `master_thesis_code/parameter_estimation/parameter_estimation.py` — Has cupy import (lines 21-22); already accepts `use_gpu` param
- `master_thesis_code/waveform_generator.py` — `use_gpu` param defaults to `True`; needs to respect CLI flag
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` — Lines 237-251: existing affinity logic to replace with `--num_workers`

### Patterns to follow
- `master_thesis_code/parameter_estimation/parameter_estimation.py` — Reference for `_get_xp` pattern and guarded cupy import
- `master_thesis_code/LISA_configuration.py` — Already has `_get_xp()` helper (line ~30+), just needs the import guard fixed

### Requirements
- `.planning/REQUIREMENTS.md` — CODE-01, CODE-02, CODE-03

No external specs — requirements fully captured in decisions above.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `_get_xp()` helper pattern already exists in `LISA_configuration.py` and `parameter_estimation.py` — reuse for consistency
- `ParameterEstimation` already accepts `use_gpu: bool = True` constructor param — just needs default changed and CLI wiring
- `waveform_generator.py` functions already accept `use_gpu` parameter

### Established Patterns
- GPU imports guarded with `try/except ImportError` + `_CUPY_AVAILABLE` flag (see `decorators.py`, `parameter_estimation.py`)
- Conditional imports inside functions for GPU-only code paths (see `main.py:data_simulation()`)
- `_get_xp(use_gpu: bool) -> types.ModuleType` returns `cp` or `np` based on flag

### Integration Points
- `arguments.py` → `main.py:main()` → `data_simulation()` / `snr_analysis()` / `evaluate()` call chain
- `BayesianStatistics.evaluate()` — receives `num_workers` parameter for multiprocessing pool

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches.

</specifics>

<deferred>
## Deferred Ideas

- **Profiling improvements** — User wants to improve GPU/memory profiling beyond the current `MemoryManagement` class. Belongs in a future milestone, not Phase 1 code hardening.

</deferred>

---

*Phase: 01-code-hardening*
*Context gathered: 2026-03-26*
