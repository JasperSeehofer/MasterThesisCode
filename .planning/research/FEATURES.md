# Feature Landscape: HPC Cluster Deployment

**Domain:** GPU-accelerated scientific computing on bwUniCluster 3.0
**Researched:** 2026-03-25

## Table Stakes

Features required for the cluster deployment to be functional.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| `--use_gpu` flag threaded through all code paths | Currently hardcoded True; crashes on CPU nodes and login nodes | Medium | Must touch ParameterEstimation, MemoryManagement, main.py |
| CPU-safe MemoryManagement | GPUtil import crashes on CPU-only nodes; blocks import of main module | Low | Guard imports, no-op when GPU unavailable |
| SLURM job scripts (simulate, merge, evaluate) | Cannot run anything without sbatch scripts | Low | Templated bash scripts in `cluster/` |
| Module loader script (`modules.sh`) | Reproducible environment across all jobs | Low | Single source of truth for module versions |
| Non-interactive merge script | Current `input()` calls block in batch jobs | Low | Add `--delete-sources` flag, remove interactive prompts |
| Workspace-aware output paths | Simulation data must go to workspace, not $HOME | Low | Use `ws_find` in job scripts; pass workspace path as working_dir |
| SLURM metadata in `run_metadata.json` | Traceability: which node, GPU, array task produced each result | Low | Read SLURM_JOB_ID, SLURM_ARRAY_TASK_ID, SLURM_NODELIST env vars |
| Seed derivation from array task ID | Reproducibility across array tasks | Low | `seed = base_seed + SLURM_ARRAY_TASK_ID` |

## Differentiators

Features that add significant value but are not strictly required for basic operation.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Dependency chain orchestrator (`submit_pipeline.sh`) | One command submits sim -> merge -> eval pipeline | Low | Uses `sbatch --dependency=afterok` chaining |
| `--num_workers` CLI flag for Bayesian eval | Control multiprocessing pool size; match to SLURM_CPUS_PER_TASK | Low | Default to os.cpu_count() or SLURM env var |
| First-time setup script (`cluster/setup.sh`) | Automates uv install, workspace allocation, module verification | Low | Idempotent; safe to re-run |
| Cluster quickstart documentation | Reduces onboarding time for supervisor/collaborators | Low | In-repo docs, not external wiki |
| Apptainer container definition | Reproducibility archive; thesis artifact | Medium | Build locally, run on cluster with `--nv` |
| Job monitoring helpers | Quick check of queue status, failed jobs, output tailing | Low | Shell aliases or small wrapper script |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Multi-node MPI distribution | Each EMRI simulation is single-GPU; array jobs handle parallelism; MPI adds complexity for zero benefit | SLURM array jobs with `--gres=gpu:1` |
| Dask/Ray distributed computing | Overkill for embarrassingly parallel workload; adds dependency complexity | SLURM array jobs + merge script |
| Checkpointing/resume for individual tasks | Each simulation task is short enough to re-run; SLURM handles failed task resubmission | Re-submit failed array indices: `--array=5,17,42` |
| Self-hosted CI runners on cluster | Security risk, maintenance burden, thesis-inappropriate scope | CPU-only GitHub Actions; manual cluster testing |
| GPU CI testing | No GPU runners on GitHub Actions; cluster testing is manual | `pytest -m "not gpu"` in CI; manual GPU test on dev queue |
| Conda/mamba environment | Conflicts with uv; duplicates dependency management; wastes quota | uv sync exclusively |

## Feature Dependencies

```
CPU-safe MemoryManagement --> --use_gpu flag (must be threaded first)
SLURM job scripts --> modules.sh (sourced by every script)
SLURM job scripts --> Non-interactive merge script (merge job)
Dependency chain orchestrator --> SLURM job scripts (all three)
SLURM metadata in run_metadata.json --> --use_gpu flag (needs to know context)
Apptainer container --> modules.sh (documents required dependencies)
First-time setup script --> modules.sh (verifies modules exist)
```

## MVP Recommendation

Prioritize in this order:

1. **`--use_gpu` flag + CPU-safe MemoryManagement** -- unblocks everything else; code must be importable on login nodes
2. **`modules.sh` + non-interactive merge script** -- prerequisites for job scripts
3. **SLURM job scripts (simulate, merge, evaluate)** -- core deliverable
4. **Dependency chain orchestrator** -- turns 3 manual steps into 1 command
5. **SLURM metadata + seed derivation** -- reproducibility
6. **Setup script + documentation** -- onboarding

Defer:
- **Apptainer container**: Build after the module-based approach works; it is a reproducibility artifact, not a development tool
- **Enroot/Pyxis integration**: Only if module-based approach fails

## Sources

- [bwUniCluster 3.0 SLURM Guide](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs/Slurm)
- [bwUniCluster 3.0 Containers](https://wiki.bwhpc.de/e/BwUniCluster3.0/Containers)
- Project context: `.planning/PROJECT.md`

---

*Feature analysis: 2026-03-25*
