# EMRI Parameter Estimation — HPC Integration

## What This Is

A gravitational wave parameter estimation pipeline for LISA Extreme Mass Ratio Inspirals (EMRIs). Two pipelines: (1) GPU-accelerated EMRI simulation that computes SNR and Cramér-Rao bounds, and (2) CPU-based Bayesian inference that evaluates the Hubble constant posterior. The v1.0 milestone delivered production-ready HPC/cluster support: both pipelines now run on bwUniCluster 3.0 at KIT as SLURM array jobs with a single-command submission, reproducible seeding, failure recovery, and complete documentation.

## Core Value

The simulation pipeline runs reliably on the GPU cluster as SLURM array jobs, producing enough Cramér-Rao bounds for statistically meaningful Hubble constant posteriors.

## Current State (v1.0 shipped 2026-03-27)

- **Cluster pipeline operational:** `submit_pipeline.sh --tasks N --steps M --seed S` submits the full simulate→merge→evaluate chain
- **186 CPU tests passing**, 43% coverage (gate at 25%)
- **~24,400 LOC** (Python + Shell), 9 cluster scripts in `cluster/`
- **Documentation complete:** `cluster/README.md` quickstart, `CLAUDE.md` cluster section, `README.md` HPC pointer
- **Known physics bugs remain** (Fisher stencil, confusion noise, wCDM silent, etc.) — tracked in CLAUDE.md

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
- ✓ Pre-commit hooks (ruff, mypy), CI pipeline, test suite — existing
- ✓ `--use_gpu` CLI flag threaded through all pipelines — v1.0 Phase 1
- ✓ `--num_workers` CLI flag for Bayesian inference pool size — v1.0 Phase 1
- ✓ CPU-safe `MemoryManagement` (guard GPUtil, no-op on CPU nodes) — v1.0 Phase 1
- ✓ Non-interactive merge script (`--delete-sources` flag) — v1.0 Phase 2
- ✓ Environment module loader (`modules.sh`) for bwUniCluster 3.0 — v1.0 Phase 3
- ✓ One-time cluster setup script (uv, workspace allocation) — v1.0 Phase 3
- ✓ SLURM metadata in `run_metadata.json` (job ID, array task ID, node, GPU info) — v1.0 Phase 4
- ✓ SLURM job scripts for simulation (GPU array), merge, and evaluation (CPU) — v1.0 Phase 4
- ✓ Workflow orchestrator chaining jobs via `--dependency=afterok` — v1.0 Phase 4
- ✓ Cluster documentation (quickstart, monitoring, troubleshooting) — v1.0 Phase 5
- ✓ CLAUDE.md and README.md updates for cluster deployment — v1.0 Phase 5

### Active

(None — next milestone not yet planned)

### Out of Scope

- Multi-node MPI distribution — single GPU per simulation task is sufficient; array jobs provide parallelism
- Dask/Ray distributed computing — overkill for this use case
- GPU CI runners — cluster testing is manual; GitHub Actions stays CPU-only
- Checkpointing/resume — each array task is short enough; failed tasks can be re-submitted
- Self-hosted runners on cluster — security and maintenance overhead not justified for a thesis
- Apptainer container definition — module-based approach works well; container deferred as not needed for thesis

## Context

- **Cluster:** bwUniCluster 3.0 at KIT Karlsruhe (bwHPC federation), SLURM scheduler, GPU nodes with NVIDIA H100 GPUs (CUDA 12), `module load` environment system, workspace mechanism (`ws_allocate`)
- **Architecture fit:** `--simulation_index` maps to `SLURM_ARRAY_TASK_ID`. Each EMRI event is independent, making array jobs the ideal parallelization strategy.
- **All v1.0 code blockers resolved:** CPU-safe imports, batch-compatible scripts, environment setup, job infrastructure, documentation.
- **Known physics bugs:** Fisher stencil (forward-diff instead of 5-point), confusion noise absent from PSD, wCDM params silently ignored, hardcoded 10% σ(d_L), WMAP-era cosmology, galaxy redshift uncertainty scaling. All tracked in CLAUDE.md.

## Constraints

- **GPU:** CUDA 12 required for `cupy-cuda12x` and `fastemriwaveforms-cuda12x` — must use GPU partition on cluster
- **GSL:** Build-time requirement for `fastemriwaveforms` — must be available via module or container
- **uv:** Primary package manager; installed to `~/.local/bin` on login nodes
- **Workspace:** bwHPC workspaces expire (default 60 days, extendable) — results must be copied to persistent storage
- **Network:** Compute nodes may have restricted outbound access — all dependency installation happens on login nodes

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| SLURM array jobs over single long job | Per-task time limits, automatic retry, better queue scheduling, natural fit with `--simulation_index` | ✓ Good — clean mapping, easy failure recovery |
| Module-based primary, Apptainer deferred | Simpler debugging and cluster storage integration for a thesis | ✓ Good — modules work well, container not needed |
| Seed = base + array task ID | Deterministic, reproducible, different per task | ✓ Good — exact reproducibility confirmed |
| `cluster/` directory in repo | Job scripts version-controlled alongside code they run | ✓ Good — single git pull deploys everything |
| `emri-merge`/`emri-prepare` entry points | Console scripts avoid fragile `python -m scripts.*` in sbatch | ✓ Good — clean CLI interface |
| afterok dependency chaining | Pipeline runs unattended after single submission | ✓ Good — three stages chain automatically |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone:**
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-27 after v1.0 milestone completion*
