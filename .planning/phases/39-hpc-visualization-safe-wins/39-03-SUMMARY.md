---
phase: 39-hpc-visualization-safe-wins
plan: 03
subsystem: hpc-gpu-memory
tags: [hpc, gpu-memory, fft-cache, pressure-trigger]
requirements: [HPC-03]
requirements_addressed: [HPC-03]
dependency-graph:
  requires: []
  provides:
    - "free_memory_pool() — always-free pool API"
    - "clear_fft_cache() — explicit cache invalidation"
    - "free_gpu_memory_if_pressured(threshold=0.8) — pressure-gated cache release"
    - "free_gpu_memory() — deprecated alias routing to pressure-gated API"
  affects: [main.py:381, main.py:639]
tech-stack:
  added: []
  patterns:
    - "pressure-gated cache release using cp.cuda.runtime.memGetInfo"
    - "DeprecationWarning alias routing for backward compatibility"
key-files:
  created: []
  modified:
    - master_thesis_code/memory_management.py
    - master_thesis_code/main.py
    - master_thesis_code_test/test_memory_management.py
key-decisions:
  - "Pool-bytes / total-bytes ratio computed via cp.cuda.runtime.memGetInfo (no GPUtil dep)"
  - "INFO-log telemetry on cache-clear for Phase 40 diagnostics"
  - "Step 4 GPU pressure-trigger mock test deferred (Claude's discretion per plan) — Phase 40 integration run will exercise"
  - "Added 4th CPU test test_free_gpu_memory_if_pressured_is_noop_on_cpu beyond required 3"
  - "Updated existing test_free_gpu_memory_cpu to suppress new DeprecationWarning"
metrics:
  duration_min: ~6.5
  tasks_completed: 2
  files_changed: 3
  completed_date: "2026-04-22"
commits:
  - 14504a7: "test(39-03): add failing tests for free_gpu_memory API split (HPC-03 RED)"
  - 49e3d60: "refactor(39-03): split free_gpu_memory into pressure-gated API (HPC-03 GREEN)"
  - 6feeeb3: "refactor(39-03): migrate main.py call sites to free_gpu_memory_if_pressured"
---

# Phase 39 Plan 03: HPC-03 Pressure-Gated GPU Memory API Summary

**One-liner:** Split `MemoryManagement.free_gpu_memory()` into `free_memory_pool()` + `clear_fft_cache()` + `free_gpu_memory_if_pressured(threshold=0.8)` so the per-iteration FFT plan cache is only invalidated when pool usage exceeds 80% of GPU memory. Saves 20–100 ms per Fisher iteration on the 80 GB H100 (typical pool sits ~40 GB, well below the 64 GB threshold). Preserves `free_gpu_memory()` as a `DeprecationWarning`-emitting alias.

## Outcome

HPC-03 (ROADMAP SC-3) satisfied. Three new public methods on `MemoryManagement`; both `main.py` per-iteration call sites (data_simulation loop line 381, injection campaign loop line 639) now invoke the pressure-gated API. Old `free_gpu_memory()` retained as a deprecated alias that routes to `free_gpu_memory_if_pressured(threshold=0.8)` per D-09. Decoupling cache lifetime from pool freeing eliminates the unconditional `_fft_cache.clear()` cost in the simulation hot path.

## Changes

### `master_thesis_code/memory_management.py`

Added three public methods (all CPU-safe via `_CUPY_AVAILABLE and cp is not None` guard):
- `free_memory_pool()` — frees `memory_pool.free_all_blocks()`, no-op on CPU
- `clear_fft_cache()` — explicit `_fft_cache.clear()` only, no-op on CPU
- `free_gpu_memory_if_pressured(threshold: float = 0.8)` — frees pool unconditionally; computes `pool_bytes / total_bytes` via `cp.cuda.runtime.memGetInfo()`; clears FFT cache only when ratio ≥ threshold; INFO-logs the trigger event for diagnostics

Existing `free_gpu_memory()` retained as deprecated alias — emits `DeprecationWarning(stacklevel=2)` and routes to `free_gpu_memory_if_pressured()`.

### `master_thesis_code/main.py`

Two call-site migrations (no other edits):
- line 381 (data_simulation loop end-of-iteration): `free_gpu_memory()` → `free_gpu_memory_if_pressured()`
- line 639 (injection campaign loop end-of-iteration): same migration

### `master_thesis_code_test/test_memory_management.py`

Updated/added tests:
- New: `test_free_memory_pool_*` (CPU + GPU)
- New: `test_clear_fft_cache_*` (CPU + GPU)
- New: `test_free_gpu_memory_if_pressured_is_noop_on_cpu` (extra coverage)
- Updated: `test_free_gpu_memory_cpu` — suppresses new DeprecationWarning

## Verification

| Check | Result |
|-------|--------|
| `def free_memory_pool` count in `master_thesis_code/` | 1 |
| `def clear_fft_cache` count | 1 |
| `def free_gpu_memory_if_pressured` count | 1 |
| `def free_gpu_memory(self) -> None:` (deprecated alias) count | 1 |
| `_fft_cache.clear` call count in `master_thesis_code/` | **1** (D-12 satisfied — single source of cache invalidation) |
| `cp.cuda.runtime.memGetInfo` count | 1 |
| `free_gpu_memory_if_pressured()` calls in `main.py` | 2 (lines 381 and 639) |
| `.free_gpu_memory()` calls in `main.py` (non-deprecated path) | 0 |
| `MemoryManagement(use_gpu` constructor sites | 3 (D-10 invariant intact) |
| Full CPU regression: `uv run pytest master_thesis_code_test/ -m "not gpu"` | **539 passed**, 6 skipped, 12 deselected |
| D-30 baseline (≥524) | satisfied |
| Ruff (modified files) | clean |
| Mypy (modified files) | clean |

## Success Criteria

- [x] Three new public methods exist on `MemoryManagement`
- [x] `main.py:381` and `main.py:639` both call `free_gpu_memory_if_pressured()` (not `free_gpu_memory()`)
- [x] Original `free_gpu_memory()` remains as deprecated alias (emits `DeprecationWarning`, routes to new call)
- [x] `test_memory_management.py` covers all four methods
- [x] mypy + ruff clean
- [x] No modifications to STATE.md or ROADMAP.md (orchestrator owns those)

## Decisions Made

- **D-09 (already locked):** Keep `free_gpu_memory()` as deprecated alias. Followed verbatim.
- **D-10 (already locked):** `use_gpu` threading invariant — verified 3 constructor sites unchanged.
- **D-12 (already locked):** Single `_fft_cache.clear()` site after split — verified.
- **Implementation choice:** Used `cp.cuda.runtime.memGetInfo()` for GPU-total bytes (CuPy-native, no GPUtil dependency added).
- **Implementation choice:** Added INFO-log on cache-clear trigger so Phase 40 diagnostic runs can confirm the pressure threshold is rarely crossed in practice.
- **Implementation choice:** Step 4 GPU pressure-trigger mock test deferred per plan's "Claude's discretion" — Phase 40 integration run on bwUniCluster will exercise the real trigger path.
- **Implementation choice:** Added 4th CPU test (`test_free_gpu_memory_if_pressured_is_noop_on_cpu`) beyond the required 3 for explicit no-op-on-CPU coverage.

## Deviations from Plan

None on the implementation. Plan task table executed verbatim. The optional Step 4 (GPU mock for pressure trigger) was deferred per the plan's explicit "at Claude's discretion" allowance.

## Performance Impact (HPC, expected)

- Per-iteration FFT plan rebuild cost (20–100 ms × N events) eliminated when pool usage stays below threshold
- Pool-free behaviour unchanged — still frees blocks every iteration
- Cache rebuild cost only paid on actual pressure events (rare on 80 GB H100 with ~40 GB typical pool footprint)

## Risk Tradeoffs

| Risk | Mitigation | Status |
|------|------------|--------|
| Cache outgrows pool without trigger | INFO-log fires whenever cache clears; Phase 40 will validate threshold | Documented |
| Deprecated alias caller doesn't notice migration | `DeprecationWarning` emitted with `stacklevel=2` | Verified by test |
| `cp.cuda.runtime.memGetInfo` unavailable in some CUDA configs | Method is part of CuPy's public CUDA runtime wrapper since 6.x — long-stable | Accepted |

## Self-Check: PASSED

- 3 task commits present (14504a7, 49e3d60, 6feeeb3) — verified via `git log`
- Files modified: `memory_management.py`, `main.py`, `test_memory_management.py`
- 539 CPU tests passing (≥524 baseline)
- mypy + ruff clean

## Note on SUMMARY.md authorship

This SUMMARY.md was authored by the orchestrator (not the executor agent) due to a `Write`-tool permission denial inside the executor's worktree session. All commits and verification numbers above were performed by the agent (`agentId: aba9d4f75f7597a08`). Content reflects the agent's structured task-completion report verbatim. The orchestrator is committing this SUMMARY.md on the worktree branch before merge to satisfy the plan's mandatory output gate.
