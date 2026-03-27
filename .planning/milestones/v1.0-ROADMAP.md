# Roadmap: EMRI HPC Integration

## Overview

This roadmap takes an existing GPU-accelerated EMRI simulation codebase and makes it run reliably on bwUniCluster 3.0 as SLURM array jobs. The path goes from fixing code-level blockers (unconditional CuPy imports, hardcoded GPU flag) through batch-compatible scripts, cluster environment setup, SLURM job infrastructure, to documentation. Each phase unblocks the next: nothing runs on the cluster until the code is importable on CPU nodes, no job scripts work until the merge script is non-interactive, and documentation describes a working system rather than aspirations.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Code Hardening** - Make codebase importable and functional on CPU-only nodes with proper CLI flags
- [x] **Phase 2: Batch Compatibility** - Make post-simulation scripts usable in non-interactive batch jobs (completed 2026-03-26)
- [x] **Phase 3: Cluster Environment** - Create and verify environment module setup for bwUniCluster 3.0 (completed 2026-03-27)
- [ ] **Phase 4: SLURM Job Infrastructure** - Build the full simulate-merge-evaluate pipeline with traceability
- [ ] **Phase 5: Documentation** - Document the cluster workflow for reproducibility and onboarding

## Phase Details

### Phase 1: Code Hardening
**Goal**: The codebase is importable and testable on CPU-only machines while running correctly on GPU compute nodes
**Depends on**: Nothing (first phase)
**Requirements**: CODE-01, CODE-02, CODE-03
**Success Criteria** (what must be TRUE):
  1. Running `python -c "import master_thesis_code"` succeeds on a machine without CuPy/CUDA installed
  2. Running `python -m master_thesis_code --help` works without GPU and shows `--use_gpu` and `--num_workers` flags
  3. `MemoryManagement` can be instantiated on a CPU-only machine without raising ImportError
  4. Existing CPU test suite (`pytest -m "not gpu and not slow"`) still passes with no regressions
**Plans:** 2 plans

Plans:
- [x] 01-01-PLAN.md — Make MemoryManagement CPU-safe, fix circular import, add tests
- [x] 01-02-PLAN.md — Add --use_gpu and --num_workers CLI flags, thread through call chain, add tests

### Phase 2: Batch Compatibility
**Goal**: Post-simulation scripts run unattended in SLURM batch jobs without human interaction
**Depends on**: Phase 1
**Requirements**: BATCH-01, BATCH-02
**Success Criteria** (what must be TRUE):
  1. `merge_cramer_rao_bounds.py --delete-sources` merges CSVs and deletes source files without any interactive prompt
  2. `prepare_detections.py` can be invoked via `python -m scripts.prepare_detections` or equivalent CLI entry point from a batch script
**Plans:** 1/1 plans complete

Plans:
- [x] 02-01-PLAN.md — Refactor merge and prepare scripts with argparse CLIs, register emri-merge/emri-prepare entry points

### Phase 3: Cluster Environment
**Goal**: A verified, repeatable environment setup exists for bwUniCluster 3.0 that produces a working virtualenv
**Depends on**: Phase 1
**Requirements**: ENV-01, ENV-02, ENV-03
**Success Criteria** (what must be TRUE):
  1. `source cluster/modules.sh` loads all required modules (CUDA, Python, GSL, compiler) without errors on bwUniCluster 3.0
  2. `cluster/setup.sh` completes on a fresh bwUniCluster account and produces a working `.venv` with `uv sync --extra gpu`
  3. Simulation output is written to a bwHPC workspace path (resolved via `ws_find`), not to `$HOME`
**Plans:** 1 plan

Plans:
- [x] 03-01-PLAN.md — Create cluster/modules.sh and cluster/setup.sh for bwUniCluster 3.0 environment setup

### Phase 4: SLURM Job Infrastructure
**Goal**: A single command submits the full simulate-merge-evaluate pipeline on the cluster with full traceability
**Depends on**: Phase 2, Phase 3
**Requirements**: SLURM-01, SLURM-02, SLURM-03, SLURM-04, TRACE-01, TRACE-02
**Success Criteria** (what must be TRUE):
  1. `cluster/submit_pipeline.sh` submits three dependent SLURM jobs (simulate array, merge, evaluate) and prints all job IDs
  2. Each simulation array task writes a `run_metadata.json` containing SLURM job ID, array task ID, node name, and GPU info
  3. Each simulation array task uses a deterministic seed derived from base seed plus array task ID, producing reproducible results
  4. The evaluate job runs Bayesian inference with `--num_workers` matching the allocated CPU cores
  5. Failed array tasks can be resubmitted individually without rerunning the entire array
**Plans:** 3 plans

Plans:
- [x] 04-01-PLAN.md — Extend _write_run_metadata() with SLURM env vars and indexed filenames
- [x] 04-02-PLAN.md — Create simulate.sbatch, merge.sbatch, evaluate.sbatch job scripts
- [x] 04-03-PLAN.md — Create submit_pipeline.sh orchestrator and resubmit_failed.sh recovery helper

### Phase 5: Documentation
**Goal**: A new user (or future-you) can go from cluster login to running a full simulation campaign using only in-repo documentation
**Depends on**: Phase 4
**Requirements**: DOCS-01, DOCS-02, DOCS-03
**Success Criteria** (what must be TRUE):
  1. `cluster/README.md` contains a complete quickstart covering setup, running, monitoring, results retrieval, and workspace expiration warnings
  2. `CLAUDE.md` documents the `--use_gpu`, `--num_workers` flags and the `cluster/` directory
  3. `README.md` has a "Running on HPC" section that points to `cluster/README.md`
**Plans:** 2 plans

Plans:
- [x] 05-01-PLAN.md — Create cluster/README.md quickstart and reference guide
- [x] 05-02-PLAN.md — Add Cluster Deployment section to CLAUDE.md and Running on HPC section to README.md

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5
(Phases 2 and 3 are independent of each other but both block Phase 4)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Code Hardening | 1/2 | Executing | - |
| 2. Batch Compatibility | 1/1 | Complete   | 2026-03-26 |
| 3. Cluster Environment | 1/1 | Complete | 2026-03-27 |
| 4. SLURM Job Infrastructure | 0/3 | Planned | - |
| 5. Documentation | 0/2 | Planned | - |
