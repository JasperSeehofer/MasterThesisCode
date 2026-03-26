# Project Research Summary

**Project:** EMRI Parameter Estimation — bwUniCluster 3.0 Deployment
**Domain:** GPU-accelerated scientific computing on SLURM HPC
**Researched:** 2026-03-25
**Confidence:** HIGH

## Executive Summary

This project deploys an existing GPU-accelerated EMRI (Extreme Mass Ratio Inspiral) gravitational wave simulation codebase to bwUniCluster 3.0 for large-scale parallel execution. The workload is embarrassingly parallel: each EMRI simulation is independent, single-GPU, and produces a Cramér-Rao bounds CSV that feeds a downstream Bayesian inference step. The recommended approach is SLURM array jobs on the `gpu_h100` partition, with a three-stage dependency chain (simulate → merge → evaluate), managed through a small `cluster/` directory of scripts committed to the repository. This is a well-understood HPC deployment pattern with no exotic requirements.

The primary obstacle before any cluster work can proceed is a code-level blocker: the codebase has unconditional `import cupy` statements at module top level and a hardcoded `USE_GPU=True` in at least one path. These make the package unimportable on CPU-only nodes, which includes the login node where `uv sync` and job submission must run. Fixing this is the mandatory first step. Once the code is importable everywhere, the cluster infrastructure (modules, job scripts, workspace layout) follows a highly standard bwHPC pattern with excellent official documentation.

The main operational risk is workspace expiration: bwHPC workspaces have a 60-day default lifetime and are permanently deleted after the grace period. All simulation output (CSVs, posteriors, figures) must live in the workspace during active work but final results must be copied to `$HOME` before expiration. This is a thesis-ending failure mode if neglected and must be mitigated with calendar reminders and an explicit "copy finals to `$HOME`" step after evaluation.

## Key Findings

### Recommended Stack

The cluster environment is bwUniCluster 3.0 running SLURM as its job scheduler. The confirmed available software stack is: CUDA 12.8 (`devel/cuda/12.8`), Python 3.13.3 (`devel/python/3.13.3-gnu-14.2`), and GCC 14.2 (`compiler/gnu/14.2`). These match the project's existing constraints exactly (Python 3.13 pin, `cupy-cuda12x`, `fastemriwaveforms-cuda12x`). The GSL module name must be verified on first login with `module spider gsl` — it is a build-time dependency of `fastemriwaveforms` and the exact module name is not confirmed.

The package manager is `uv`, installed as a standalone binary to `~/.local/bin` on the login node. The `uv.lock` file already committed to the repo guarantees exact reproducibility. `uv sync --extra gpu` runs on the login node (internet access) and the resulting `.venv/` lives in `$HOME` (Lustre, visible to all compute nodes). No conda, no pip-only installs.

**Core technologies:**
- SLURM: job scheduling — only option on bwHPC; array jobs are the natural fit for independent EMRI tasks
- `gpu_h100` partition: primary GPU partition — 4x H100 per node, 72h walltime, 12 nodes; best option for throughput
- `devel/cuda/12.8`: CUDA toolkit — confirmed on cluster; compatible with `cupy-cuda12x`
- uv: package manager — already project standard; lockfile ensures reproducibility on cluster without extra tooling
- bwHPC workspaces (`ws_allocate`): large temporary storage — 40 TiB quota; mandatory for simulation output
- Enroot/Apptainer: containers — module-based approach is primary; containers are a reproducibility archive artifact

### Expected Features

The feature set divides cleanly into blockers (must fix before cluster use), infrastructure (must build), enhancements (add value), and deliberate deferrals.

**Must have (table stakes):**
- `--use_gpu` flag properly threaded — currently broken; blocks all non-GPU execution including login node imports
- CPU-safe `MemoryManagement` — unconditional `GPUtil` import crashes on CPU nodes
- `modules.sh` — single source of truth for module versions; sourced by every job script
- Non-interactive `merge_cramer_rao_bounds.py` — current `input()` calls hang in batch jobs; needs `--delete-sources` flag
- SLURM job scripts (simulate.sbatch, merge.sbatch, evaluate.sbatch) — cannot run anything without these
- Workspace-aware output paths — all simulation data goes to workspace, not `$HOME`
- SLURM metadata in `run_metadata.json` — traceability (job ID, array task ID, node) for every simulation result
- Deterministic seed from array task ID — reproducibility per task

**Should have (competitive):**
- Dependency chain orchestrator (`submit_pipeline.sh`) — one command submits the full pipeline
- `--num_workers` CLI flag for Bayesian evaluation — match worker count to `SLURM_CPUS_PER_TASK`
- First-time setup script (`cluster/setup.sh`) — reduces onboarding friction
- Cluster quickstart documentation — in-repo, not just wiki links

**Defer (after module-based approach works):**
- Apptainer container definition — reproducibility archive for thesis submission; not needed for active development
- Enroot/Pyxis integration — only if module-based approach fails due to missing GSL or Python version issues

### Architecture Approach

The deployment architecture is a linear three-stage pipeline: GPU array simulation → CPU merge → CPU multiprocessing evaluation. All scripts live in a `cluster/` subdirectory committed to the repository. The workspace (allocated via `ws_allocate`) is the only component that lives outside the repo, and its path is always resolved dynamically via `ws_find emri-data`. This keeps the repository self-contained and path-independent.

**Major components:**
1. `cluster/modules.sh` — single source of truth for module versions; sourced by every `.sbatch` script
2. `cluster/simulate.sbatch` — SLURM array job; one GPU task per array index; writes per-task CSV to workspace
3. `cluster/merge.sbatch` — CPU job dependent on simulation completion; merges per-task CSVs; invokes `scripts/merge_cramer_rao_bounds.py --delete-sources`
4. `cluster/evaluate.sbatch` — CPU job dependent on merge; 48-core multiprocessing Bayesian inference; writes posteriors and figures
5. `cluster/submit_pipeline.sh` — orchestrator; chains all three with `sbatch --parsable --dependency=afterok`
6. `cluster/setup.sh` — one-time setup; installs uv, allocates workspace, runs `uv sync --extra gpu`

### Critical Pitfalls

1. **Unconditional CuPy import crashes** — guard all `import cupy` with `try/except ImportError`; use the `_get_xp(use_gpu)` pattern from CLAUDE.md; this is a blocker that prevents any non-GPU work
2. **Running computation on login nodes** — all Python computation goes through `sbatch` or `srun`; detect by checking for `SLURM_JOB_ID` env var
3. **Workspace expiration data loss** — set calendar reminders; copy final results (posteriors, thesis figures) to `$HOME` immediately after evaluation; run `ws_list` periodically
4. **Installing packages on compute nodes** — compute nodes have no internet; always `uv sync` on login node before submitting jobs
5. **Requesting all 4 GPUs when only 1 is needed** — use `--gres=gpu:1` per array task; each EMRI simulation is single-GPU; 4x waste otherwise

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Code Hardening for Cluster Compatibility
**Rationale:** The unconditional CuPy import and hardcoded GPU flag make the package unimportable on CPU nodes. Nothing else can proceed until this is fixed — not `uv sync` verification, not login-node testing, not any cluster workflow. This is the single hardest blocker and must be resolved first.
**Delivers:** A codebase that is importable and testable on CPU-only machines (login nodes, dev machines) while still running correctly on GPU compute nodes
**Addresses:** FEATURES table stakes items: `--use_gpu` flag threading, CPU-safe MemoryManagement
**Avoids:** Pitfall 4 (unconditional CuPy import crashes), Pitfall 1 (login node computation)

### Phase 2: Non-Interactive Script Fixes
**Rationale:** The merge script's `input()` calls are a second blocker for batch execution. This is a small, low-risk change but must happen before any job script can use it. Grouped separately from Phase 1 because it touches different files with no overlap.
**Delivers:** A merge script usable in non-interactive batch context
**Addresses:** FEATURES table stakes: non-interactive merge script
**Avoids:** Pitfall 8 (interactive script calls in batch jobs)

### Phase 3: Cluster Environment Setup
**Rationale:** With code fixed, the next step is verifying the actual cluster environment. Module names in the research are from wiki documentation (September 2025) and may be stale — they must be verified interactively on the cluster with `module spider`. This phase is exploratory: it cannot be fully scripted in advance.
**Delivers:** Verified `modules.sh` with confirmed module names; `uv sync --extra gpu` succeeds; baseline cluster environment works
**Uses:** SLURM, Lmod, CUDA 12.8, Python 3.13.3, GSL module (name TBD)
**Avoids:** Pitfall 6 (module version mismatch), Pitfall 14 (GSL not found), Pitfall 15 (Python 3.13 availability)

### Phase 4: SLURM Job Scripts and Workspace Layout
**Rationale:** With a working environment, build the actual execution infrastructure. The three job scripts (simulate, merge, evaluate) plus the orchestrator are the core deliverable of this project. The workspace layout and metadata capture complete the reproducibility story.
**Delivers:** Full `cluster/` directory; one-command pipeline submission; SLURM metadata in `run_metadata.json`; `--num_workers` CLI flag for Bayesian evaluation
**Uses:** `gpu_h100` partition, SLURM array jobs, `ws_allocate`/`ws_find`, `sbatch --dependency=afterok`
**Implements:** All architecture components; all FEATURES table stakes
**Avoids:** Pitfall 2 (packages on compute nodes), Pitfall 5 (all 4 GPUs), Pitfall 9 (CUDA_VISIBLE_DEVICES), Pitfall 10 (missing `--parsable`), Pitfall 11 (array output naming), Pitfall 12 (missing logs dir)

### Phase 5: Setup Script and Documentation
**Rationale:** With the infrastructure working, reduce onboarding friction. A setup script and quickstart document protect against the operational risks (workspace expiration, uv PATH issues) and make the workflow reproducible by a supervisor or future researcher.
**Delivers:** `cluster/setup.sh` (idempotent first-time setup); in-repo cluster quickstart documentation including workspace expiration warnings
**Addresses:** FEATURES differentiators: setup script, quickstart docs
**Avoids:** Pitfall 3 (workspace expiration), Pitfall 13 (uv not in PATH)

### Phase 6: Apptainer Container (Optional, Deferred)
**Rationale:** Build this only after the module-based approach is proven working. It is a reproducibility artifact for thesis submission, not a development tool. The container definition documents the full dependency stack for archival purposes.
**Delivers:** `emri.def` Apptainer definition; built `.sif` file; instructions for running on cluster with `--nv`
**Uses:** Apptainer (or Enroot if preferred)

### Phase Ordering Rationale

- Phase 1 before everything: nothing is importable on CPU without fixing the CuPy guard; this blocks login node testing, uv verification, and all downstream work
- Phase 2 before job scripts: the merge job script depends on the non-interactive merge script; fixing it first keeps Phase 4 clean
- Phase 3 before Phase 4: job scripts reference specific module names; those names must be verified before committing them to scripts
- Phase 5 after Phase 4: documentation describes a working system; write it last to avoid documenting something that changes
- Phase 6 last and optional: container is an archive artifact, not a workflow requirement

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** Module names from wiki are September 2025 vintage and may be stale; must verify interactively on cluster. GSL module name is explicitly unconfirmed. Python 3.13 availability is MEDIUM confidence. This phase is inherently exploratory.
- **Phase 4:** Resource sizing (CPUs, memory, walltime per job) requires profiling on the actual cluster — the values in STACK.md are starting points, not validated numbers. The `--num_workers` default behavior needs design decisions around `SLURM_CPUS_PER_TASK` fallback.

Phases with standard patterns (skip research-phase):
- **Phase 1:** The `_get_xp(use_gpu)` pattern and `try/except ImportError` guard are already documented in CLAUDE.md with exact code. No new research needed.
- **Phase 2:** Adding a `--delete-sources` argparse flag to a Python script is entirely standard.
- **Phase 5:** bwHPC workspace management commands are well-documented with HIGH confidence.
- **Phase 6:** Apptainer `.def` file syntax and `--nv` flag are well-documented.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | bwUniCluster 3.0 wiki directly documents CUDA 12.8, Python 3.13.3, GCC 14.2 module names; uv HPC usage documented by GWDG and UZH |
| Features | HIGH | Features derived directly from known code bugs (CLAUDE.md) and bwHPC operational requirements; no speculation |
| Architecture | HIGH | Standard SLURM array job + dependency chain pattern; well-documented by bwHPC wiki; no novel architectural choices |
| Pitfalls | HIGH | Most pitfalls come from known code bugs (CLAUDE.md) or bwHPC operational documentation; GSL module name is the main uncertain item |

**Overall confidence:** HIGH

### Gaps to Address

- **GSL module name on bwUniCluster 3.0**: Research guesses `numlib/gsl/2.7-gnu-14.2` based on naming conventions but this is not confirmed. Handle by running `module spider gsl` as the first action on cluster login. If absent, fallback is Apptainer container.
- **fastemriwaveforms wheel availability**: If no pre-built wheel exists for the cluster's platform, `uv sync` will attempt to build from source and GSL headers must be present at build time. Confidence is MEDIUM — likely fine, but verify during Phase 3.
- **Bayesian evaluation walltime**: The 8-hour estimate in STACK.md for the evaluation job is a rough guess. Actual time depends on detection count and number of workers. Profile after first simulation run.
- **`--num_workers` CLI argument**: This flag is listed as needed in STACK.md but does not yet exist in `arguments.py`. Design decision needed: should it default to `os.cpu_count()`, `SLURM_CPUS_PER_TASK`, or require explicit specification?

## Sources

### Primary (HIGH confidence)
- [bwUniCluster 3.0 Batch Queues](https://wiki.bwhpc.de/e/BwUniCluster3.0/Batch_Queues) — partition details, walltime limits, GPU counts
- [bwUniCluster 3.0 SLURM Guide](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs/Slurm) — sbatch syntax, array jobs, dependencies
- [bwUniCluster 3.0 Hardware Architecture](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture) — GPU node specs, CPU counts, memory
- [bwUniCluster 3.0 Filesystem Details](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture/Filesystem_Details) — $HOME quota, workspace quota
- [bwUniCluster 3.0 Software Modules](https://wiki.bwhpc.de/e/BwUniCluster3.0/Software_Modules) — module commands, CUDA/Python module names
- [bwUniCluster 3.0 Containers](https://wiki.bwhpc.de/e/BwUniCluster3.0/Containers) — Enroot and Apptainer support
- [bwHPC Workspace Documentation](https://wiki.bwhpc.de/e/Workspace) — ws_allocate, ws_extend, ws_find, expiration
- Project CLAUDE.md — known bugs (unconditional CuPy imports, USE_GPU hardcode), `_get_xp` pattern

### Secondary (MEDIUM confidence)
- [uv on HPC (UZH)](https://docs.s3it.uzh.ch/general/uv/) — uv installation on HPC clusters
- [uv on HPC (GWDG)](https://docs.hpc.gwdg.de/software_stacks/compilers_interpreters/python/index.html) — module-based uv on HPC
- [CuPy Installation](https://docs.cupy.dev/en/stable/install.html) — CUDA 12.x wheel compatibility
- [FastEMRIWaveforms](https://bhptoolkit.org/FastEMRIWaveforms/) — build requirements, GSL dependency

### Tertiary (LOW confidence)
- GSL module name `numlib/gsl/2.7-gnu-14.2` — inferred from bwHPC naming conventions; must verify with `module spider gsl`

---
*Research completed: 2026-03-25*
*Ready for roadmap: yes*
