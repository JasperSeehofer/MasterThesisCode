# Phase 4: SLURM Job Infrastructure - Context

**Gathered:** 2026-03-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the full simulate-merge-evaluate pipeline as SLURM jobs on bwUniCluster 3.0 with traceability. Delivers sbatch scripts for each stage, a pipeline orchestrator that chains them via dependencies, SLURM metadata in run_metadata.json, deterministic seeding per array task, and a failure recovery helper script.

</domain>

<decisions>
## Implementation Decisions

### SLURM Resource Parameters
- **D-01:** Simulation jobs use partition `gpu_h100` (H100 nodes, 4 GPUs/node, 3-day max). Each array task requests `--gres=gpu:1`. Time limit: 2 hours per task.
- **D-02:** Merge job uses partition `cpu` (AMD 96-core nodes, 72h max). Time limit: ~30 minutes. Single core is sufficient.
- **D-03:** Evaluate job uses partition `cpu`. Time limit: ~1 hour initially — will be tuned after profiling with real data at scale.
- **D-04:** Memory defaults are adequate (`193,300 MB/GPU` for H100, `2000 MB/core` for CPU). No explicit `--mem` needed.
- **D-05:** Dev partition `dev_gpu_h100` available for quick test runs (30 min max, 1 job limit).

### Pipeline Invocation
- **D-06:** `submit_pipeline.sh` accepts flags: `--tasks` (array size), `--steps` (simulation_steps per task), `--seed` (base seed). All three are required — no defaults.
- **D-07:** No time-limit override flags. Edit sbatch files directly if limits need changing. Keep the script simple.

### Output Directory Structure
- **D-08:** Campaign-based layout (Option B). Each pipeline run creates a run directory within `$WORKSPACE`, e.g., `$WORKSPACE/run_20260327_seed42/`. All output is flat within:
  ```
  $WORKSPACE/run_<date>_seed<N>/
    cramer_rao_bounds_0.csv
    cramer_rao_bounds_1.csv
    run_metadata_0.json
    run_metadata_1.json
    undetected_events_0.csv
    cramer_rao_bounds.csv        (merged)
  ```
- **D-09:** `run_metadata.json` is indexed per task (`run_metadata_0.json`, `run_metadata_1.json`) to avoid overwrites. Requires a small change to `_write_run_metadata()` in `main.py`.

### Seed Strategy
- **D-10:** Per-task seed = base seed + `SLURM_ARRAY_TASK_ID`. Deterministic and reproducible. Already decided in PROJECT.md — carried forward.

### Traceability (TRACE-01)
- **D-11:** `_write_run_metadata()` in `main.py` adds SLURM environment variables when present: `SLURM_JOB_ID`, `SLURM_ARRAY_TASK_ID`, `SLURM_NODELIST`, `SLURM_CPUS_PER_TASK`, `CUDA_VISIBLE_DEVICES`, `HOSTNAME`. No-op when not on cluster.

### Failure Recovery
- **D-12:** Helper script `cluster/resubmit_failed.sh` that:
  1. Queries `sacct` to find failed array task IDs for a given SLURM job ID
  2. Deletes output files for failed tasks (CSVs + metadata) from the run directory — clean slate
  3. Resubmits only those indices via `sbatch --array=<failed_ids>`
- **D-13:** Clean-before-resubmit is mandatory — partial CSV writes from killed tasks could corrupt the merge.

### Claude's Discretion
- sbatch script structure (inline vs sourced config)
- Exact `sacct` format string and parsing in `resubmit_failed.sh`
- Whether `submit_pipeline.sh` prints a summary table of submitted job IDs or just the IDs
- Log file naming convention for SLURM output (`%A_%a.out` pattern or custom)
- How the run directory name is passed between chained jobs (env var, file, or sbatch argument)
- CPU core count for evaluate job (full node vs partial allocation)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Cluster Infrastructure (existing from Phase 3)
- `cluster/modules.sh` -- Loads modules, exports `$WORKSPACE`, `$PROJECT_ROOT`, `$VENV_PATH`. All sbatch scripts must source this.
- `cluster/setup.sh` -- First-time setup reference; shows the module + workspace + uv pattern.

### Codebase (traceability integration points)
- `master_thesis_code/main.py:93-111` -- `_write_run_metadata()` function; must be extended for TRACE-01 (SLURM env vars) and D-09 (indexed filename).
- `master_thesis_code/arguments.py` -- CLI argument definitions; `--simulation_index`, `--seed`, `--use_gpu`, `--num_workers` already exist.
- `master_thesis_code/constants.py` -- File path constants with `$index` placeholder pattern (e.g., `CRAMER_RAO_BOUNDS_PATH`).
- `master_thesis_code/parameter_estimation/parameter_estimation.py:415-430` -- `flush_crb_buffer()` writes per-index CSVs using the `$index` placeholder.

### Prior Phase Context
- `.planning/phases/01-code-hardening/01-CONTEXT.md` -- `--use_gpu` defaults False, `--num_workers` uses `sched_getaffinity - 2`
- `.planning/phases/02-batch-compatibility/02-CONTEXT.md` -- `emri-merge`/`emri-prepare` entry points with `--workdir` flag
- `.planning/phases/03-cluster-environment/03-CONTEXT.md` -- `modules.sh` exports `$WORKSPACE`; `setup.sh` is idempotent

### Requirements
- `.planning/REQUIREMENTS.md` -- SLURM-01, SLURM-02, SLURM-03, SLURM-04, TRACE-01, TRACE-02

### External References
- [bwUniCluster 3.0 Running Jobs wiki](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs#Batch_Jobs:_sbatch) -- Partition specs, `--gres` syntax, memory defaults, sbatch examples

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cluster/modules.sh` -- Already exports `$WORKSPACE`, `$PROJECT_ROOT`, `$VENV_PATH`. sbatch scripts source this.
- `emri-merge` entry point -- Accepts `--workdir`, `--delete-sources`. Ready for batch use.
- `emri-prepare` entry point -- Accepts `--workdir`. Ready for batch use.
- `_write_run_metadata()` in `main.py` -- Already writes git commit, timestamp, seed, CLI args. Just needs SLURM vars added.
- `--simulation_index` argument -- Maps directly to `SLURM_ARRAY_TASK_ID`.

### Established Patterns
- CSV filenames use `$index` placeholder in `constants.py` (e.g., `cramer_rao_bounds_$index.csv`)
- `--use_gpu` must be explicitly passed (defaults False)
- `--num_workers` auto-detects from `sched_getaffinity` which respects SLURM cgroup limits

### Integration Points
- `submit_pipeline.sh` creates the run directory and passes it as `--workdir` / positional arg to all jobs
- sbatch scripts source `cluster/modules.sh` then activate `.venv`
- Seed computation: `--seed $((BASE_SEED + SLURM_ARRAY_TASK_ID))` in simulate.sbatch
- Dependency chain: simulate (array) -> merge -> evaluate via `--dependency=afterok`

</code_context>

<specifics>
## Specific Ideas

- Time limits (2h simulate, 30min merge, 1h evaluate) are initial estimates. User plans profiling to tune these for large-scale runs.
- User explicitly wants to push to large evaluation scales — evaluate time limit may need increasing later.
- Dev partition `dev_gpu_h100` useful for quick sanity checks before submitting large arrays.

</specifics>

<deferred>
## Deferred Ideas

- **Profiling integration** -- User wants profiling to understand time breakdown across processes. Belongs in a future milestone, not Phase 4 job infrastructure.
- **Job monitoring helpers** -- Tracked as MON-01 in REQUIREMENTS.md v2. Not in Phase 4 scope.
- **Apptainer container** -- Tracked as CONT-01/CONT-02 in REQUIREMENTS.md v2. Not in Phase 4 scope.

</deferred>

---

*Phase: 04-slurm-job-infrastructure*
*Context gathered: 2026-03-27*
