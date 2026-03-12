---
name: gpu-audit
description: >
  Audit Python files for GPU/HPC compliance violations. Use when modifying
  files that touch array computation, waveforms, PSD, Fisher matrix, or
  any code that runs on the GPU cluster. Checks: guarded cupy imports,
  xp namespace pattern, no hot-path GPU-CPU transfers, vectorized operations.
argument-hint: [file_or_directory] (defaults to master_thesis_code/)
allowed-tools: Read, Grep, Glob, Bash(uv run mypy *)
context: fork
agent: Explore
---

## GPU/HPC Compliance Audit

Scan the target file(s) for violations of these mandatory rules:

### Rule 1: Guarded CuPy imports
- `import cupy` must NEVER appear at module top level unconditionally
- Must use: `try: import cupy as cp` / `except ImportError: cp = None`
- Report: file:line of every unconditional `import cupy`

### Rule 2: xp namespace pattern
- Computation functions must NOT call `cp.*` or `np.*` directly
- Must resolve `xp = _get_xp(use_gpu)` and use `xp.*` throughout
- Report: file:line of bare `cp.` or `np.` calls inside functions that handle arrays

### Rule 3: No GPU-CPU transfers in hot paths
- `cp.asnumpy()`, `.get()`, `cp.asarray()` inside loops or frequently-called functions
- Report: file:line + calling context

### Rule 4: Vectorized operations
- No Python `for` loops over array elements in computation code
- Must use `xp.trapz`, `xp.sum`, broadcasting, etc.
- Report: file:line of `for i in range(len(` patterns on array data

### Rule 5: USE_GPU not hardcoded
- `USE_GPU = True` must never appear as a module-level constant
- Must come from CLI `--use_gpu` argument

### Output format
For each file, report:
- PASS (no violations) or FAIL (list violations)
- Severity: BLOCKER (won't import on CPU) vs WARNING (performance/correctness)
- Suggested fix for each violation

### Known violators (check these first):
- `LISA_configuration.py` — unconditional cupy import
- `decorators.py` — unconditional cupy import
- `memory_management.py` — unconditional cupy import
- `parameter_estimation/parameter_estimation.py` — unconditional cupy import
