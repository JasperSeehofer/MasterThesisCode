---
paths: ["master_thesis_code/**/*.py", "master_thesis_code_test/**/*.py"]
description: HPC/GPU best practices — array-namespace xp pattern, guarded cupy imports, vectorization, GPU memory, and the bwUniCluster entry point
---

# HPC / GPU Best Practices

This code runs on a GPU cluster (CuPy/CUDA) but must also be importable and testable on a CPU-only development machine. The patterns below are mandatory.

See [[scientific-computing-validation]] for the promoted, cross-project form of these patterns.

### Cluster interaction — entry point (read first)

**Before doing ANY work involving bwUniCluster** (submitting/monitoring/retrieving
jobs, checking cluster state, launching sims/injections/evaluations, or running
cluster tests), the session MUST route through the cluster guidance — do not
improvise SSH/SLURM commands from memory:

1. **Read `.claude/skills/cluster/SKILL.md`** (the operational guide) — canonical
   paths, gotchas, submit/monitor/retrieve recipes. Or invoke it via `/cluster`.
2. **Run the preflight** and require `VERDICT: READY ✓`:
   `ssh bwunicluster 'bash -s' < cluster/preflight.sh`
3. **To launch or write a job**, follow `cluster/LAUNCHING_JOBS.md` and copy
   `cluster/JOB_TEMPLATE.sbatch`. Test small first (`--tasks 2 --steps 10`).

There is **ONE repo** on the cluster (`~/MasterThesisCode`); never create separate
clones/worktrees for parallel or "frozen" work — branch + tag instead. Dataset
locations/provenance: `cluster/datasets.yaml`; staleness tiers: `DATA_INVENTORY.md`.

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

- **GPU imports must always be guarded.** Never place `import cupy as cp` at module top level unconditionally. All source modules are compliant as of commit `4894648` (`decorators.py`, `memory_management.py`, `LISA_configuration.py`, `parameter_estimation.py` all use the guarded `try/except ImportError` + `_CUPY_AVAILABLE` pattern) — keep new modules compliant.
- **Vectorize array operations.** Never iterate over array elements in a hot path. Use vectorized `xp.*` operations (e.g., `xp.trapz(integrant / psd, x=fs)` instead of a Python loop).
- **Avoid GPU-to-CPU transfers in hot paths.** Do not call `cp.asnumpy()` or `.get()` inside functions called thousands of times. Keep data on GPU until a single scalar result.
- **GPU memory management.** Free GPU memory after each full simulation step (`cp.get_default_memory_pool().free_all_blocks()`). Do not call inside inner loops — the CuPy allocator reuses blocks.
- **USE_GPU flag.** Must never be hardcoded `True`. Must come from `--use_gpu` CLI argument and be threaded into every constructor. No module-level constant should control GPU behavior.
