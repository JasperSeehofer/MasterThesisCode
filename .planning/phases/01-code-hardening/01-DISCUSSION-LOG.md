# Phase 1: Code Hardening - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-26
**Phase:** 01-code-hardening
**Areas discussed:** LISA_configuration.py scope, MemoryManagement CPU behavior, CLI flag threading, Worker count default

---

## LISA_configuration.py Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Fix it now (Recommended) | Guard cupy import with try/except + _get_xp pattern. Required for success criterion 1. | ✓ |
| Defer to a later phase | Keep conditional imports in calling functions. Risk: other modules still crash. | |
| Minimal guard only | Guard import but don't refactor to xp pattern. | |

**User's choice:** Fix it now
**Notes:** None

| Option | Description | Selected |
|--------|-------------|----------|
| Fix all unguarded imports | Sweep all files with unconditional cupy/GPUtil imports and guard them. One pass. | ✓ |
| Only LISA_configuration.py | Others already have try/except guards. Focus on the broken one. | |

**User's choice:** Fix all unguarded imports
**Notes:** None

---

## MemoryManagement CPU Behavior

| Option | Description | Selected |
|--------|-------------|----------|
| Silent no-op (Recommended) | Guard GPUtil, return zeros/no-ops on CPU. Callers unchanged. | ✓ |
| Skip instantiation | Don't create MemoryManagement when --use_gpu is off. More caller logic. | |
| CPU memory tracking | Track CPU RAM via psutil on CPU. Adds dependency. | |

**User's choice:** Silent no-op
**Notes:** User mentioned wanting to improve profiling in a future milestone — noted as deferred idea.

---

## CLI Flag Threading

| Option | Description | Selected |
|--------|-------------|----------|
| Pass as argument (Recommended) | Thread use_gpu through functions and constructors. Matches existing pattern. | ✓ |
| Config/settings object | Bundle flags into Settings dataclass. More abstraction for 2 flags. | |
| Module-level flag | Set module-level USE_GPU. Violates CLAUDE.md guidance. | |

**User's choice:** Pass as argument
**Notes:** None

| Option | Description | Selected |
|--------|-------------|----------|
| Default False (Recommended) | User must explicitly pass --use_gpu. Safe for login nodes and dev machines. | ✓ |
| Auto-detect | Check CuPy availability. Convenient but implicit. | |

**User's choice:** Default False
**Notes:** None

---

## Worker Count Default

| Option | Description | Selected |
|--------|-------------|----------|
| CLI overrides all (Recommended) | If --num_workers given, use it. Default: sched_getaffinity() - 2. Remove affinity expansion hack. | ✓ |
| Keep affinity expansion | Keep existing logic, let --num_workers override. Two code paths. | |
| Match SLURM allocation | Default to SLURM_CPUS_PER_TASK if set, sched_getaffinity() - 2 otherwise. | |

**User's choice:** CLI overrides all
**Notes:** User asked how the correct number is determined. Clarified: sched_getaffinity(0) respects SLURM cgroup limits, so default automatically matches allocation. User confirmed this logic works.

---

## Claude's Discretion

- Implementation details of _get_xp helper in LISA_configuration.py
- Exact error messages and log levels for CPU fallback paths
- Test organization for CPU-importability tests

## Deferred Ideas

- **Profiling improvements** — User wants to improve GPU/memory profiling beyond current MemoryManagement. Future milestone.
