---
phase: 04-slurm-job-infrastructure
plan: 02
subsystem: infra
tags: [slurm, sbatch, gpu, hpc, bwunicluster]

# Dependency graph
requires:
  - phase: 03-cluster-environment
    provides: "cluster/modules.sh environment loader ($WORKSPACE, $PROJECT_ROOT, $VENV_PATH)"
  - phase: 02-batch-compatibility
    provides: "emri-merge and emri-prepare console entry points"
  - phase: 01-code-hardening
    provides: "--use_gpu flag, --num_workers auto-detection, --simulation_index, --seed"
provides:
  - "simulate.sbatch: GPU array job for EMRI simulation"
  - "merge.sbatch: CPU job for CSV merge and prepare"
  - "evaluate.sbatch: CPU job for Bayesian inference"
affects: [04-03-submit-pipeline, 05-cluster-documentation]

# Tech tracking
tech-stack:
  added: []
  patterns: ["SBATCH script pattern: source modules.sh + activate venv + validate env vars + run"]

key-files:
  created:
    - cluster/simulate.sbatch
    - cluster/merge.sbatch
    - cluster/evaluate.sbatch
  modified: []

key-decisions:
  - "No --output/--error in SBATCH headers; submit_pipeline.sh sets them via CLI for variable expansion"
  - "num_workers auto-detected from SLURM cgroup rather than hardcoded in evaluate.sbatch"
  - "Per-task seed = BASE_SEED + SLURM_ARRAY_TASK_ID for reproducible parallelism (TRACE-02)"

patterns-established:
  - "SBATCH env pattern: SCRIPT_DIR resolution -> source modules.sh -> activate venv -> validate env vars -> run"
  - "Log paths delegated to submit_pipeline.sh to avoid SBATCH variable expansion limitations"

requirements-completed: [SLURM-01, SLURM-02, SLURM-03, TRACE-02]

# Metrics
duration: 2min
completed: 2026-03-27
---

# Phase 04 Plan 02: SLURM Job Scripts Summary

**Three sbatch job scripts (simulate, merge, evaluate) for the simulate-merge-evaluate pipeline on bwUniCluster 3.0 with GPU array jobs and reproducible seeding**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-27T12:21:29Z
- **Completed:** 2026-03-27T12:23:22Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- simulate.sbatch: GPU array job on gpu_h100 with per-task seed derivation from SLURM_ARRAY_TASK_ID
- merge.sbatch: CPU job running emri-merge --delete-sources and emri-prepare for post-simulation data processing
- evaluate.sbatch: CPU job with 16 CPUs running --evaluate mode with auto-detected worker count from cgroup

## Task Commits

Each task was committed atomically:

1. **Task 1: Create simulate.sbatch for GPU array jobs** - `70cb62e` (feat)
2. **Task 2: Create merge.sbatch for CSV merge and prepare** - `45a3618` (feat)
3. **Task 3: Create evaluate.sbatch for Bayesian inference** - `0338df4` (feat)

## Files Created/Modified
- `cluster/simulate.sbatch` - GPU array job: sources modules.sh, computes TASK_SEED, runs EMRI simulation with --use_gpu
- `cluster/merge.sbatch` - CPU job: runs emri-merge --delete-sources + emri-prepare on $RUN_DIR
- `cluster/evaluate.sbatch` - CPU job: runs --evaluate with 16 CPUs, num_workers auto-detected from cgroup

## Decisions Made
- No --output/--error SBATCH directives in script headers (SBATCH does not expand shell variables; submit_pipeline.sh passes them via CLI)
- num_workers not hardcoded in evaluate.sbatch; auto-detected via os.sched_getaffinity(0) - 2 which respects SLURM cgroup
- Per-task seed = BASE_SEED + SLURM_ARRAY_TASK_ID ensures reproducibility with different random states per task

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All three sbatch scripts ready for submit_pipeline.sh to chain via --dependency=afterok (Plan 03)
- Scripts expect RUN_DIR, BASE_SEED, SIM_STEPS environment variables from the orchestrator
- modules.sh from Phase 03 Plan 01 provides the required environment setup

## Self-Check: PASSED

All 3 created files verified present. All 3 task commits verified in git log.

---
*Phase: 04-slurm-job-infrastructure*
*Completed: 2026-03-27*
