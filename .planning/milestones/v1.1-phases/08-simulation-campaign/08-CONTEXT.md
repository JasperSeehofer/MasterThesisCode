# Phase 8: Simulation Campaign - Context

**Gathered:** 2026-03-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Run a smoke-test EMRI simulation campaign on bwUniCluster (3 tasks, 25 steps, seed 42), validate results with full quantitative checks, and rsync outputs back to the local machine. This is a pipeline validation run, not a production-scale campaign.

</domain>

<decisions>
## Implementation Decisions

### Campaign Parameters
- **D-01:** Test run configuration: `--tasks 3 --steps 25 --seed 42`
- **D-02:** This is a smoke test only. No scale-up to a larger run in this phase. Production campaigns are deferred to a future milestone (PROD-01).

### Validation Criteria
- **D-03:** Full validation even with the small sample (75 total simulation steps across 3 tasks).
- **D-04:** SNR validation: detection rate must fall in a plausible range (1-30% of simulated events). All SNR values must be positive.
- **D-05:** H0 posterior validation: posterior must peak in [0.6, 0.9]. The true value is H=0.73 (constants.py).
- **D-06:** Pipeline completion checks: all 3 tasks complete, merge produces `cramer_rao_bounds.csv` and `prepared_cramer_rao_bounds.csv`, evaluate produces an H0 posterior distribution.

### Monitoring & Failure Response
- **D-07:** Monitor with `sacct -j <job_ids>` via SSH for status overview. Tail SLURM log files in `$RUN_DIR/logs/` when investigating failures or slow jobs.
- **D-08:** On task failure: investigate first (read .err log, diagnose the issue, report) before resubmitting. With only 3 tasks, any failure likely indicates a real problem.

### Result Handling
- **D-09:** After validation passes, rsync key output files back to local machine. Cluster workspace expires in 60 days.
- **D-10:** Local results stored in `evaluation/run_YYYYMMDD_seed42/` (already in .gitignore from Phase 6).

### Claude's Discretion
- Polling interval for sacct job monitoring
- Exact rsync flags and which files to copy back (at minimum: merged CSV, prepared CSV, H0 posterior output, run_metadata JSONs, SLURM logs)
- How to present validation results (table, summary, etc.)
- Whether to generate any local plots from the rsynced results

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` -- SIM-01, SIM-02, SIM-03

### Prior Phase Context
- `.planning/phases/07-cluster-access/07-CONTEXT.md` -- SSH access method (D-01: `ssh bwunicluster '<cmd>'`), preflight verification scope (D-02)

### Cluster Scripts
- `cluster/submit_pipeline.sh` -- Pipeline orchestrator: `--tasks N --steps S --seed SEED`
- `cluster/simulate.sbatch` -- GPU array job: per-task seed = BASE_SEED + SLURM_ARRAY_TASK_ID
- `cluster/merge.sbatch` -- CPU job: `emri-merge --delete-sources` + `emri-prepare`
- `cluster/evaluate.sbatch` -- CPU job: `python -m master_thesis_code $RUN_DIR --evaluate`
- `cluster/resubmit_failed.sh` -- Resubmit FAILED/TIMEOUT tasks with partial output cleanup
- `cluster/modules.sh` -- Module load sequence, exports $WORKSPACE, $PROJECT_ROOT, $VENV_PATH

### Codebase References
- `master_thesis_code/constants.py` -- H=0.73 (true Hubble constant), SNR_THRESHOLD=20
- `master_thesis_code/main.py` -- `data_simulation()` and `evaluate()` entry points

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cluster/submit_pipeline.sh` -- Full pipeline submission with dependency chaining; exact command: `bash cluster/submit_pipeline.sh --tasks 3 --steps 25 --seed 42`
- `cluster/resubmit_failed.sh` -- Failure recovery for individual array tasks
- `cluster/modules.sh` -- Environment setup (modules, workspace path, venv path)

### Established Patterns
- All cluster scripts `source cluster/modules.sh` as first step
- Run directory convention: `$WORKSPACE/run_YYYYMMDD_seedN/` with `logs/` and `simulations/` subdirectories
- Per-task output: `cramer_rao_bounds_simulation_{idx}.csv`, `run_metadata_{idx}.json`
- Merged output: `cramer_rao_bounds.csv`, `prepared_cramer_rao_bounds.csv`

### Integration Points
- SSH access via `ssh bwunicluster '<cmd>'` (Phase 7)
- `evaluation/` directory in .gitignore (Phase 6) -- local result storage
- `sacct` for job status monitoring

</code_context>

<specifics>
## Specific Ideas

- Smoke test is deliberately small (3x25=75 steps) to quickly validate the pipeline chain
- Full validation criteria applied despite small sample -- this catches both infrastructure and physics issues
- Investigate-first failure policy because any failure in 3 tasks signals a real problem, not statistical noise

</specifics>

<deferred>
## Deferred Ideas

- Production-scale campaign (50-100+ tasks, 100+ steps) -- deferred to future milestone (PROD-01)
- Performance profiling/optimization -- deferred (PERF-01, PERF-02)
- Local plot generation from rsynced results -- could be added but not in scope

</deferred>

---

*Phase: 08-simulation-campaign*
*Context gathered: 2026-03-28*
