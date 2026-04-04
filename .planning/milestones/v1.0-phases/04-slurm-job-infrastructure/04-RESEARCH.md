# Phase 4: SLURM Job Infrastructure - Research

**Researched:** 2026-03-27
**Domain:** SLURM batch job scripting, bwUniCluster 3.0, shell pipeline orchestration
**Confidence:** HIGH

## Summary

Phase 4 creates the SLURM job scripts and pipeline orchestrator that chain together the simulate-merge-evaluate workflow on bwUniCluster 3.0. The core deliverables are three sbatch scripts (`simulate.sbatch`, `merge.sbatch`, `evaluate.sbatch`), a pipeline submission script (`submit_pipeline.sh`), a failure recovery helper (`resubmit_failed.sh`), and a small Python-side change to `_write_run_metadata()` for SLURM traceability.

The existing codebase is well-prepared: `cluster/modules.sh` already exports `$WORKSPACE`, `$PROJECT_ROOT`, and `$VENV_PATH`; the `--simulation_index`, `--seed`, `--use_gpu`, and `--num_workers` CLI arguments exist; and the `emri-merge`/`emri-prepare` entry points accept `--workdir` and `--delete-sources` flags. The Python-side changes are minimal (extend `_write_run_metadata()` with SLURM env vars and per-task filename indexing).

**Primary recommendation:** Keep all sbatch scripts simple and self-contained. Each sources `cluster/modules.sh`, activates the venv, then runs a single Python command. The orchestrator `submit_pipeline.sh` captures job IDs via `--parsable` and chains dependencies via `--dependency=afterok`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Simulation jobs use partition `gpu_h100` (H100 nodes, 4 GPUs/node, 3-day max). Each array task requests `--gres=gpu:1`. Time limit: 2 hours per task.
- **D-02:** Merge job uses partition `cpu` (AMD 96-core nodes, 72h max). Time limit: ~30 minutes. Single core is sufficient.
- **D-03:** Evaluate job uses partition `cpu`. Time limit: ~1 hour initially.
- **D-04:** Memory defaults are adequate. No explicit `--mem` needed.
- **D-05:** Dev partition `dev_gpu_h100` available for quick test runs (30 min max, 1 job limit).
- **D-06:** `submit_pipeline.sh` accepts flags: `--tasks` (array size), `--steps` (simulation_steps per task), `--seed` (base seed). All three are required -- no defaults.
- **D-07:** No time-limit override flags. Edit sbatch files directly if limits need changing.
- **D-08:** Campaign-based layout. Each pipeline run creates `$WORKSPACE/run_<date>_seed<N>/` with flat file structure.
- **D-09:** `run_metadata.json` indexed per task (`run_metadata_0.json`, `run_metadata_1.json`). Requires change to `_write_run_metadata()`.
- **D-10:** Per-task seed = base seed + `SLURM_ARRAY_TASK_ID`.
- **D-11:** `_write_run_metadata()` adds SLURM env vars when present. No-op when not on cluster.
- **D-12:** `resubmit_failed.sh` queries `sacct`, deletes output files for failed tasks, resubmits only those indices.
- **D-13:** Clean-before-resubmit is mandatory.

### Claude's Discretion
- sbatch script structure (inline vs sourced config)
- Exact `sacct` format string and parsing in `resubmit_failed.sh`
- Whether `submit_pipeline.sh` prints a summary table of submitted job IDs or just the IDs
- Log file naming convention for SLURM output (`%A_%a.out` pattern or custom)
- How the run directory name is passed between chained jobs (env var, file, or sbatch argument)
- CPU core count for evaluate job (full node vs partial allocation)

### Deferred Ideas (OUT OF SCOPE)
- Profiling integration
- Job monitoring helpers (MON-01)
- Apptainer container (CONT-01/CONT-02)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SLURM-01 | `cluster/simulate.sbatch` submits GPU array jobs with `--simulation_index` mapped to `SLURM_ARRAY_TASK_ID` | SLURM array env vars documented; `--simulation_index` CLI arg already exists in `arguments.py` |
| SLURM-02 | `cluster/merge.sbatch` runs non-interactive merge and prepare scripts as CPU batch job | `emri-merge --workdir DIR --delete-sources` and `emri-prepare --workdir DIR` entry points ready |
| SLURM-03 | `cluster/evaluate.sbatch` runs Bayesian inference with `--num_workers` matching allocated cores | `$SLURM_CPUS_PER_TASK` provides the count; `--num_workers` arg exists; auto-detection via `sched_getaffinity` also works with SLURM cgroups |
| SLURM-04 | `cluster/submit_pipeline.sh` chains jobs via `--dependency=afterok` and prints all job IDs | `sbatch --parsable` returns bare job ID; `--dependency=afterok:JOBID` waits for all array tasks |
| TRACE-01 | `run_metadata.json` includes SLURM env vars when on cluster | `_write_run_metadata()` in `main.py:93-111` needs env var reads + indexed filename |
| TRACE-02 | Deterministic seed = base seed + `SLURM_ARRAY_TASK_ID` | Seed arithmetic in bash: `$((BASE_SEED + SLURM_ARRAY_TASK_ID))` |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- **Package manager:** uv (never manually edit dependencies in pyproject.toml)
- **Testing:** `uv run pytest -m "not gpu and not slow"` for CPU tests
- **Type checking:** `uv run mypy master_thesis_code/` (disallow_untyped_defs)
- **Formatting:** `uv run ruff format` + `uv run ruff check --fix`
- **Pre-commit hooks:** ruff + mypy run automatically on commit
- **Typing convention:** Python 3.10+ native syntax (`list[float]`, `X | None`)
- **Physics change protocol:** Not triggered -- this phase changes no physics formulas
- **Git convention:** Standard prefix (no `[PHYSICS]` needed)
- **Skills:** `/check` must run before any commit; `/pre-commit-docs` after `/check` passes

## Standard Stack

No new Python packages needed. This phase is entirely shell scripts + a small modification to `main.py`.

### Core (Already Available)
| Tool | Version | Purpose | Status |
|------|---------|---------|--------|
| SLURM | cluster-managed | Batch job scheduler | Available on bwUniCluster 3.0 |
| bash | 5.x | Shell scripting for sbatch/submit/resubmit | Available |
| sacct | cluster-managed | Job accounting queries for failure detection | Available |
| uv | latest | Python environment activation in jobs | Installed by `setup.sh` |

### Python Integration Points (Existing)
| Module | Purpose | Change Needed |
|--------|---------|---------------|
| `master_thesis_code/main.py` | `_write_run_metadata()` | Add SLURM env vars, index filename |
| `master_thesis_code/arguments.py` | CLI arguments | None -- all needed args exist |
| `master_thesis_code/constants.py` | Path templates | None -- `$index` pattern already works |
| `scripts/merge_cramer_rao_bounds.py` | Merge CSVs | None -- `emri-merge` entry point ready |
| `scripts/prepare_detections.py` | Prepare detections | None -- `emri-prepare` entry point ready |

## Architecture Patterns

### Recommended File Structure
```
cluster/
  modules.sh            # (existing) module loading + env vars
  setup.sh              # (existing) first-time setup
  vpn.sh                # (existing) VPN helper
  simulate.sbatch       # NEW: GPU array job for EMRI simulation
  merge.sbatch          # NEW: CPU job for CSV merge + prepare
  evaluate.sbatch       # NEW: CPU job for Bayesian inference
  submit_pipeline.sh    # NEW: pipeline orchestrator
  resubmit_failed.sh    # NEW: failure recovery helper
```

### Pattern 1: sbatch Script Structure
**What:** Each sbatch script follows a standard template: SBATCH directives, source modules.sh, activate venv, run command.
**When to use:** Every sbatch script in this phase.
**Example:**
```bash
#!/usr/bin/env bash
#SBATCH --job-name=emri-simulate
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/simulate_%A_%a.out
#SBATCH --error=logs/simulate_%A_%a.err

# Load environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/modules.sh"
source "$VENV_PATH/bin/activate"

# Run simulation
TASK_SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))
python -m master_thesis_code "$RUN_DIR" \
    --simulation_steps "$SIM_STEPS" \
    --simulation_index "$SLURM_ARRAY_TASK_ID" \
    --seed "$TASK_SEED" \
    --use_gpu \
    --log_level INFO
```

### Pattern 2: Job Dependency Chain via --parsable
**What:** `submit_pipeline.sh` captures each job ID and passes it as a dependency to the next stage.
**When to use:** The pipeline orchestrator.
**Example:**
```bash
SIM_JOB=$(sbatch --parsable --array=0-$((TASKS - 1)) \
    --export=ALL,RUN_DIR="$RUN_DIR",BASE_SEED="$SEED",SIM_STEPS="$STEPS" \
    "$CLUSTER_DIR/simulate.sbatch")

MERGE_JOB=$(sbatch --parsable \
    --dependency=afterok:"$SIM_JOB" \
    --export=ALL,RUN_DIR="$RUN_DIR" \
    "$CLUSTER_DIR/merge.sbatch")

EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:"$MERGE_JOB" \
    --export=ALL,RUN_DIR="$RUN_DIR" \
    "$CLUSTER_DIR/evaluate.sbatch")
```

### Pattern 3: Passing Run Directory Between Jobs
**What:** Use `--export=ALL,RUN_DIR="$RUN_DIR"` to pass the campaign run directory as an environment variable to all chained jobs.
**Why:** Simpler than writing to a file. The `--export=ALL` ensures modules.sh-exported variables propagate too. Each sbatch script reads `$RUN_DIR` to know where to write/read.

### Pattern 4: SLURM Log Organization
**What:** SLURM output/error go to `$RUN_DIR/logs/` using `%A_%a` patterns.
**Recommendation:** Use `--output` and `--error` directives with `$RUN_DIR/logs/` prefix. The log directory must be created by `submit_pipeline.sh` before submission since SLURM does not create parent directories.

**Important detail:** SBATCH directives do not expand shell variables. The `--output` path must be set dynamically via `sbatch` command-line override or the script must redirect stdout/stderr itself.

### Pattern 5: Working Directory Adaptation for Campaign Layout
**What:** The codebase writes CSVs to `<working_dir>/simulations/cramer_rao_bounds_simulation_<index>.csv` (see constants.py). With D-08's flat layout (`$WORKSPACE/run_<date>_seed<N>/cramer_rao_bounds_0.csv`), we need the working directory to be the run directory, and the files will end up under `<run_dir>/simulations/`.
**Recommendation:** Accept the `simulations/` subdirectory rather than fighting the existing path constants. The flat layout in D-08 is aspirational naming; the actual structure will be:
```
$WORKSPACE/run_<date>_seed<N>/
  simulations/
    cramer_rao_bounds_simulation_0.csv
    cramer_rao_bounds_simulation_1.csv
    cramer_rao_bounds.csv              (merged)
    prepared_cramer_rao_bounds.csv
    undetected_events_simulation_0.csv
    undetected_events.csv              (merged)
  run_metadata_0.json
  run_metadata_1.json
```
This preserves existing path constants without changes. The `run_metadata_*.json` files go directly in the run directory (main.py writes them at `working_directory/run_metadata.json` level -- needs indexing per D-09).

### Anti-Patterns to Avoid
- **Hardcoded paths in sbatch scripts:** Always derive from `$RUN_DIR` and `$PROJECT_ROOT`.
- **Using `--export=NONE`:** Would lose `modules.sh` exports. Use `--export=ALL,EXTRA_VAR=val`.
- **Setting SBATCH --output with shell variables in the script header:** SBATCH directives are parsed before shell expansion. Override on the command line or use srun redirection.
- **Running `uv sync` in sbatch scripts:** Environment must be set up beforehand via `setup.sh`. Only activate the existing venv.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Job dependency management | Custom polling/wait loop | `sbatch --dependency=afterok` | SLURM handles retry, timeout, signals natively |
| Failed task detection | Parsing log files for errors | `sacct --state=FAILED,TIMEOUT --array` | Authoritative source of job state |
| CPU count detection | Hardcoded worker count | `$SLURM_CPUS_PER_TASK` or `sched_getaffinity` | Respects cgroup limits automatically |
| Seed derivation | Complex hashing scheme | `base_seed + SLURM_ARRAY_TASK_ID` | Simple, deterministic, documented |

## Common Pitfalls

### Pitfall 1: SBATCH Directives Don't Expand Variables
**What goes wrong:** Writing `#SBATCH --output=$RUN_DIR/logs/%A_%a.out` in the script -- the variable is not expanded because SBATCH lines are parsed by the scheduler, not by bash.
**Why it happens:** SBATCH directives are read before the script body executes.
**How to avoid:** Pass `--output` as a command-line argument to `sbatch` in `submit_pipeline.sh`, or redirect stdout/stderr manually inside the script.
**Warning signs:** All log files appear in the submission directory rather than the run directory.

### Pitfall 2: afterok Fails If Any Array Task Fails
**What goes wrong:** If any simulation array task fails, the merge job never starts.
**Why it happens:** `afterok` requires ALL tasks to succeed.
**How to avoid:** This is actually the desired behavior (D-13: partial CSVs from failed tasks could corrupt merge). Document that users should run `resubmit_failed.sh` first, then resubmit the merge+evaluate chain manually.
**Warning signs:** Merge job stuck in "DependencyNeverSatisfied" state.

### Pitfall 3: Run Directory Must Exist Before sbatch Submission
**What goes wrong:** SLURM tries to write log files to a non-existent directory and the job fails immediately.
**Why it happens:** SLURM does not create output directories.
**How to avoid:** `submit_pipeline.sh` must `mkdir -p "$RUN_DIR/logs"` (and `"$RUN_DIR/simulations"`) before submitting any jobs.
**Warning signs:** Jobs show as FAILED with no log files.

### Pitfall 4: sacct Array Task ID Parsing
**What goes wrong:** `sacct` output for array jobs includes entries like `12345_0`, `12345_0.batch`, `12345_0.extern`. Naive parsing picks up the `.batch` and `.extern` sub-steps too.
**Why it happens:** SLURM records job steps as separate accounting records.
**How to avoid:** Filter `sacct` output to only include lines matching `JOBID_N` pattern (no dot), or use `--noheader` + grep for exact task ID format.
**Warning signs:** `resubmit_failed.sh` finds duplicate or phantom "failed" tasks.

### Pitfall 5: File Path Constants Assume simulations/ Subdirectory
**What goes wrong:** Expecting flat file layout per D-08 but `constants.py` paths include `simulations/` prefix.
**Why it happens:** `CRAMER_RAO_BOUNDS_PATH = "simulations/cramer_rao_bounds_simulation_$index.csv"` is a relative path from working directory.
**How to avoid:** Accept `simulations/` subdirectory. Pass `$RUN_DIR` as the working directory. Files land in `$RUN_DIR/simulations/`.
**Warning signs:** CSVs not found by merge script.

### Pitfall 6: run_metadata.json Overwrite Race
**What goes wrong:** Multiple array tasks write to the same `run_metadata.json` file.
**Why it happens:** Current code writes `run_metadata.json` without index suffix.
**How to avoid:** D-09 requires indexed filenames (`run_metadata_0.json`, etc.). The `_write_run_metadata()` change must incorporate `simulation_index` into the filename.
**Warning signs:** Metadata files contain data from wrong tasks.

## Code Examples

### TRACE-01: Extending _write_run_metadata()
```python
# Source: main.py:93-111 (current code to modify)
def _write_run_metadata(working_directory: str, seed: int, arguments: Arguments) -> None:
    metadata = {
        "git_commit": _get_git_commit(),
        "timestamp": datetime.datetime.now().isoformat(),
        "random_seed": seed,
        "cli_args": {
            "simulation_steps": arguments.simulation_steps,
            "simulation_index": arguments.simulation_index,
            "evaluate": arguments.evaluate,
            "h_value": arguments.h_value,
            "snr_analysis": arguments.snr_analysis,
            "use_gpu": arguments.use_gpu,
            "num_workers": arguments.num_workers,
        },
    }

    # TRACE-01: Add SLURM environment variables when present
    slurm_vars = [
        "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_NODELIST",
        "SLURM_CPUS_PER_TASK", "CUDA_VISIBLE_DEVICES", "HOSTNAME",
    ]
    slurm_info = {var: os.environ[var] for var in slurm_vars if var in os.environ}
    if slurm_info:
        metadata["slurm"] = slurm_info

    # D-09: Index the filename when simulation_index > 0 or SLURM array task
    index = arguments.simulation_index
    if index > 0 or "SLURM_ARRAY_TASK_ID" in os.environ:
        filename = f"run_metadata_{index}.json"
    else:
        filename = "run_metadata.json"

    metadata_path = os.path.join(working_directory, filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    _ROOT_LOGGER.info(f"Run metadata written to: {metadata_path}")
```

### sacct Query for Failed Tasks
```bash
# Source: SLURM sacct documentation (https://slurm.schedmd.com/sacct.html)
# List failed array task IDs for a given job
sacct --array --jobs="$JOB_ID" \
    --state=FAILED,TIMEOUT,NODE_FAIL,OUT_OF_MEMORY \
    --format=JobID%30 \
    --noheader --parsable2 \
    | grep -oP '^\d+_\K\d+' \
    | sort -n | uniq
```

### sbatch --parsable Dependency Chain
```bash
# Source: SLURM sbatch documentation (https://slurm.schedmd.com/sbatch.html)
# sbatch --parsable returns just the job ID number (e.g., "448244")
SIM_JOB=$(sbatch --parsable ...)
# For array jobs, returns the array job ID (parent), not individual task IDs
# --dependency=afterok:$SIM_JOB waits for ALL array tasks to complete successfully
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual serial runs | SLURM array jobs | Standard HPC practice | Embarrassingly parallel simulation |
| Custom job chaining scripts | `sbatch --dependency=afterok` | SLURM native feature | Reliable dependency management |
| Interactive merge prompts | `emri-merge --delete-sources` | Phase 2 of this project | Batch-safe merge |

## Open Questions

1. **SBATCH --output path with dynamic run directory**
   - What we know: SBATCH directives don't expand shell variables. We need logs in `$RUN_DIR/logs/`.
   - What's unclear: Best approach -- command-line override vs. in-script redirection.
   - Recommendation: Use command-line `--output` and `--error` overrides in `submit_pipeline.sh` when calling `sbatch`. This is cleaner than in-script redirection. Example: `sbatch --output="$RUN_DIR/logs/simulate_%A_%a.out" simulate.sbatch`.

2. **evaluate.sbatch CPU allocation**
   - What we know: `--num_workers` auto-detects from `sched_getaffinity`. SLURM cgroups limit visible CPUs.
   - What's unclear: How many CPUs to request. Full node (96) vs partial.
   - Recommendation: Start with `--cpus-per-task=16` (conservative). The auto-detection in `arguments.py` will respect the cgroup limit and use 14 workers. This can be tuned after profiling.

3. **Flat vs simulations/ subdirectory layout**
   - What we know: D-08 describes flat file naming. But `constants.py` paths include `simulations/` prefix.
   - What's unclear: Whether to modify constants to remove the prefix.
   - Recommendation: Keep `simulations/` subdirectory. Changing constants risks breaking other code paths. Document the actual directory structure clearly.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x (configured in pyproject.toml) |
| Config file | `pyproject.toml [tool.pytest.ini_options]` |
| Quick run command | `uv run pytest -m "not gpu and not slow" -x` |
| Full suite command | `uv run pytest -m "not gpu"` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRACE-01 | `_write_run_metadata()` includes SLURM env vars when set | unit | `uv run pytest master_thesis_code_test/test_main_metadata.py -x` | No -- Wave 0 |
| TRACE-01 | `_write_run_metadata()` omits SLURM section when not on cluster | unit | same file | No -- Wave 0 |
| TRACE-02 | Seed derivation is documented in sbatch script | manual-only | Visual inspection of `simulate.sbatch` | N/A |
| SLURM-01 | `simulate.sbatch` has correct SBATCH directives and command | manual-only | Visual inspection + dry run on cluster | N/A |
| SLURM-02 | `merge.sbatch` calls emri-merge and emri-prepare | manual-only | Visual inspection + dry run on cluster | N/A |
| SLURM-03 | `evaluate.sbatch` passes --num_workers | manual-only | Visual inspection + dry run on cluster | N/A |
| SLURM-04 | `submit_pipeline.sh` chains 3 jobs with --dependency | manual-only | `bash -n cluster/submit_pipeline.sh` (syntax check) | N/A |
| D-09 | Metadata file uses indexed name `run_metadata_N.json` | unit | `uv run pytest master_thesis_code_test/test_main_metadata.py -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest -m "not gpu and not slow" -x`
- **Per wave merge:** `uv run pytest -m "not gpu"`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `master_thesis_code_test/test_main_metadata.py` -- unit tests for `_write_run_metadata()` with mocked SLURM env vars (covers TRACE-01, D-09)
- [ ] Shell syntax validation: `bash -n cluster/*.sh` and `bash -n cluster/*.sbatch` for syntax checking

*(Most of this phase is shell scripts that cannot be unit-tested in pytest. Validation is primarily through syntax checks and manual cluster dry-runs.)*

## Sources

### Primary (HIGH confidence)
- [SLURM Job Array Support](https://slurm.schedmd.com/job_array.html) -- array syntax, env vars, dependency behavior, MaxArraySize
- [SLURM sacct documentation](https://slurm.schedmd.com/sacct.html) -- `--array`, `--state`, `--parsable2`, format fields
- [SLURM sbatch documentation](https://slurm.schedmd.com/sbatch.html) -- `--parsable`, `--dependency`, `--export`, `--gres`
- [bwUniCluster 3.0 Batch Queues](https://wiki.bwhpc.de/e/BwUniCluster3.0/Batch_Queues) -- partition names, time limits, memory defaults
- [bwUniCluster 3.0 Running Jobs](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs) -- sbatch examples

### Secondary (MEDIUM confidence)
- [bwHPC Wiki SLURM common features](https://wiki.bwhpc.de/e/BwUniCluster_2.0_Slurm_common_Features) -- general bwHPC SLURM patterns (bwUniCluster 2.0 page, patterns carry forward)

### Codebase (HIGH confidence)
- `cluster/modules.sh` -- verified existing infrastructure
- `master_thesis_code/main.py:93-111` -- `_write_run_metadata()` current implementation
- `master_thesis_code/arguments.py` -- all CLI arguments verified present
- `master_thesis_code/constants.py:61-65` -- CSV path templates with `$index` pattern
- `scripts/merge_cramer_rao_bounds.py` -- merge entry point verified
- `scripts/prepare_detections.py` -- prepare entry point verified

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all tools are cluster-native SLURM features, no new packages
- Architecture: HIGH -- patterns are standard HPC practice, verified against official SLURM docs
- Pitfalls: HIGH -- drawn from SLURM documentation and direct codebase inspection
- Python changes: HIGH -- `_write_run_metadata()` modification is straightforward, integration points verified

**Research date:** 2026-03-27
**Valid until:** 2026-04-27 (stable -- SLURM API is mature, cluster config unlikely to change)
