# EMRI Parameter Estimation — HPC Integration

## What This Is

A gravitational wave parameter estimation pipeline for LISA Extreme Mass Ratio Inspirals (EMRIs). Two pipelines: (1) GPU-accelerated EMRI simulation that computes SNR and Cramér-Rao bounds, and (2) CPU-based Bayesian inference that evaluates the Hubble constant posterior. This milestone adds production-ready HPC/cluster support so both pipelines run on bwUniCluster 3.0 at KIT with proper job management, environment setup, and best-practices documentation.

## Core Value

The simulation pipeline runs reliably on the GPU cluster as SLURM array jobs, producing enough Cramér-Rao bounds for statistically meaningful Hubble constant posteriors.

## Requirements

### Validated

- ✓ EMRI simulation pipeline (waveform generation, SNR, Fisher matrix, Cramér-Rao bounds) — existing
- ✓ Bayesian inference pipeline (H₀ posterior from detection catalog + galaxy catalog) — existing
- ✓ Seed-based reproducibility with `run_metadata.json` tracking — existing
- ✓ `--simulation_index` CLI flag for partitioning output CSVs — existing
- ✓ Merge and prepare scripts for post-simulation data processing — existing
- ✓ GPU memory management class — existing
- ✓ CuPy/CUDA GPU acceleration with guarded imports — existing
- ✓ Multiprocessing pool for Bayesian likelihood evaluation — existing
- ✓ Scientific plotting subpackage with callback-based simulation monitoring — existing
- ✓ Pre-commit hooks (ruff, mypy), CI pipeline, test suite (149 CPU tests) — existing

### Active

- [x] `--use_gpu` CLI flag threaded through all pipelines — Validated in Phase 1: Code Hardening
- [x] `--num_workers` CLI flag for Bayesian inference pool size — Validated in Phase 1: Code Hardening
- [x] CPU-safe `MemoryManagement` (guard GPUtil, no-op on CPU nodes) — Validated in Phase 1: Code Hardening
- [ ] SLURM metadata in `run_metadata.json` (job ID, array task ID, node, GPU info)
- [x] Non-interactive merge script (`--delete-sources` flag) — Validated in Phase 2: Batch Compatibility
- [ ] SLURM job scripts for simulation (GPU array), merge, and evaluation (CPU)
- [ ] Workflow orchestrator chaining jobs via `--dependency=afterok`
- [ ] Environment module loader (`modules.sh`) for bwUniCluster 3.0
- [ ] One-time cluster setup script (uv, workspace allocation)
- [ ] Apptainer container definition as alternative to module-based approach
- [ ] Cluster documentation (quickstart, monitoring, troubleshooting)
- [ ] CLAUDE.md and README.md updates for cluster deployment

### Out of Scope

- Multi-node MPI distribution — single GPU per simulation task is sufficient; array jobs provide parallelism
- Dask/Ray distributed computing — overkill for this use case
- GPU CI runners — cluster testing is manual; GitHub Actions stays CPU-only
- Checkpointing/resume — each array task is short enough; failed tasks can be re-submitted
- Self-hosted runners on cluster — security and maintenance overhead not justified for a thesis

## Context

- **Cluster:** bwUniCluster 3.0 at KIT Karlsruhe (bwHPC federation), SLURM scheduler, GPU nodes with NVIDIA GPUs (CUDA 12), `module load` environment system, workspace mechanism (`ws_allocate`)
- **Architecture fit:** `--simulation_index` already maps naturally to `SLURM_ARRAY_TASK_ID`. Each EMRI event is independent, making array jobs the ideal parallelization strategy.
- **Known code issues:** ~~`USE_GPU` hardcoded True~~ (fixed Phase 1), ~~`MemoryManagement` crashes without GPU~~ (fixed Phase 1), ~~merge script has interactive `input()` calls~~ (fixed Phase 2) — all blockers resolved.
- **Prior work:** Codebase has been through extensive refactoring (plotting extraction, bayesian_statistics extraction, physics bug fixes, test coverage improvements). The code is structurally ready for cluster deployment.

## Constraints

- **GPU:** CUDA 12 required for `cupy-cuda12x` and `fastemriwaveforms-cuda12x` — must use GPU partition on cluster
- **GSL:** Build-time requirement for `fastemriwaveforms` — must be available via module or container
- **uv:** Primary package manager; must be installable on login nodes (may need local install to `~/.local/bin`)
- **Workspace:** bwHPC workspaces expire (default 30 days, extendable) — final results must be copied to persistent storage
- **Network:** Compute nodes may have restricted outbound access — all dependency installation must happen on login nodes

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| SLURM array jobs over single long job | Per-task time limits, automatic retry, better queue scheduling, natural fit with `--simulation_index` | — Pending |
| Module-based primary, Apptainer alternative | Simpler debugging and cluster storage integration for a thesis; container for reproducibility | — Pending |
| Seed = base + array task ID | Deterministic, reproducible, different per task | — Pending |
| `cluster/` directory in repo | Job scripts version-controlled alongside code they run | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-25 after initialization*
