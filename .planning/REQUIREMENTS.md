# Requirements: EMRI HPC Integration

**Defined:** 2026-03-26
**Core Value:** The simulation pipeline runs reliably on the GPU cluster as SLURM array jobs, producing enough Cramer-Rao bounds for statistically meaningful Hubble constant posteriors.

## v1 Requirements

Requirements for initial cluster deployment. Each maps to roadmap phases.

### Code Hardening

- [x] **CODE-01**: `--use_gpu` CLI flag added to `arguments.py` and threaded through `data_simulation()`, `snr_analysis()`, `ParameterEstimation`, and `MemoryManagement`
- [x] **CODE-02**: `MemoryManagement` is CPU-safe — guards `GPUtil` import, provides no-op methods when GPU unavailable, does not crash on CPU-only nodes
- [x] **CODE-03**: `--num_workers` CLI flag controls multiprocessing pool size in `BayesianStatistics.evaluate()`, defaulting to `os.sched_getaffinity() - 2` when omitted

### Batch Compatibility

- [x] **BATCH-01**: `merge_cramer_rao_bounds.py` accepts `--delete-sources` flag and runs without interactive `input()` prompts in batch jobs
- [x] **BATCH-02**: `prepare_detections.py` has a proper `main()` function callable from batch scripts

### Traceability

- [ ] **TRACE-01**: `run_metadata.json` includes SLURM environment variables (`SLURM_JOB_ID`, `SLURM_ARRAY_TASK_ID`, `SLURM_NODELIST`, `SLURM_CPUS_PER_TASK`, `CUDA_VISIBLE_DEVICES`, `HOSTNAME`) when running on a cluster
- [x] **TRACE-02**: Each SLURM array task uses a deterministic seed derived from a base seed plus `SLURM_ARRAY_TASK_ID`, documented in job scripts

### Cluster Environment

- [ ] **ENV-01**: `cluster/modules.sh` defines all required environment modules (CUDA, Python, GSL, compiler) and is sourced by every job script
- [ ] **ENV-02**: `cluster/setup.sh` automates first-time cluster setup: uv installation, workspace allocation, `uv sync --extra gpu`, and module verification
- [ ] **ENV-03**: All simulation output uses bwHPC workspace paths (resolved via `ws_find`), not `$HOME`

### SLURM Infrastructure

- [x] **SLURM-01**: `cluster/simulate.sbatch` submits GPU array jobs where each task runs `--simulation_steps N` with `--simulation_index` mapped to `SLURM_ARRAY_TASK_ID`
- [x] **SLURM-02**: `cluster/merge.sbatch` runs the non-interactive merge and prepare scripts as a CPU batch job
- [x] **SLURM-03**: `cluster/evaluate.sbatch` runs Bayesian inference as a CPU multiprocessing job with `--num_workers` matching allocated cores
- [ ] **SLURM-04**: `cluster/submit_pipeline.sh` chains simulate -> merge -> evaluate using `sbatch --parsable --dependency=afterok` and prints all job IDs

### Documentation

- [ ] **DOCS-01**: `cluster/README.md` provides a complete quickstart: prerequisites, first-time setup, running campaigns, monitoring, retrieving results, workspace management
- [ ] **DOCS-02**: `CLAUDE.md` has a "Cluster Deployment" section documenting `--use_gpu`, `--num_workers`, and the `cluster/` directory
- [ ] **DOCS-03**: `README.md` has a "Running on HPC" section pointing to `cluster/README.md`

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Containerization

- **CONT-01**: Apptainer container definition (`cluster/emri.def`) packages the full environment for reproducibility
- **CONT-02**: Container runs on bwUniCluster with `--nv` GPU passthrough

### Monitoring

- **MON-01**: Job monitoring helper scripts (queue status, failed jobs, output tailing)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-node MPI distribution | Each EMRI simulation is single-GPU; array jobs handle parallelism |
| Dask/Ray distributed computing | Overkill for embarrassingly parallel workload |
| Checkpointing/resume | Tasks are short enough to resubmit; SLURM handles failed task retry |
| GPU CI runners | Security risk and maintenance burden; manual cluster testing instead |
| Conda/mamba environment | Conflicts with uv; duplicates dependency management |
| Self-hosted CI on cluster | Thesis-inappropriate scope |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CODE-01 | Phase 1 | Complete |
| CODE-02 | Phase 1 | Complete (01-01) |
| CODE-03 | Phase 1 | Complete |
| BATCH-01 | Phase 2 | Complete |
| BATCH-02 | Phase 2 | Complete |
| TRACE-01 | Phase 4 | Pending |
| TRACE-02 | Phase 4 | Complete |
| ENV-01 | Phase 3 | Pending |
| ENV-02 | Phase 3 | Pending |
| ENV-03 | Phase 3 | Pending |
| SLURM-01 | Phase 4 | Complete |
| SLURM-02 | Phase 4 | Complete |
| SLURM-03 | Phase 4 | Complete |
| SLURM-04 | Phase 4 | Pending |
| DOCS-01 | Phase 5 | Pending |
| DOCS-02 | Phase 5 | Pending |
| DOCS-03 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 17 total
- Mapped to phases: 17
- Unmapped: 0

---
*Requirements defined: 2026-03-26*
*Last updated: 2026-03-26 after roadmap creation*
