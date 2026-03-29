# Phase 8: Simulation Campaign - Research

**Researched:** 2026-03-28
**Domain:** SLURM cluster job submission, monitoring, validation, data transfer
**Confidence:** HIGH

## Summary

Phase 8 runs a smoke-test EMRI simulation campaign on bwUniCluster 3.0 (3 tasks, 25 steps, seed 42), validates the results quantitatively, and rsyncs outputs to the local machine. The infrastructure is already built (Phases 3-5 delivered SLURM scripts, Phase 7 delivered SSH access). This phase is purely operational: submit, monitor, validate, retrieve.

A critical issue was discovered during research: the simulation code writes CSV output to relative paths from the current working directory (CWD), but the SLURM batch scripts do not `cd` into `$RUN_DIR` before invoking Python. The merge and evaluate scripts use `--workdir` to resolve paths, but `simulate.sbatch` relies on relative CWD paths. This must be fixed before submission or the per-task CSVs will land in the wrong directory.

**Primary recommendation:** Fix the CWD issue in `simulate.sbatch` (add `cd "$RUN_DIR"`), then submit via `bash cluster/submit_pipeline.sh --tasks 3 --steps 25 --seed 42`, monitor with `sacct`, validate outputs, and rsync back.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Test run configuration: `--tasks 3 --steps 25 --seed 42`
- **D-02:** This is a smoke test only. No scale-up to a larger run in this phase. Production campaigns are deferred to a future milestone (PROD-01).
- **D-03:** Full validation even with the small sample (75 total simulation steps across 3 tasks).
- **D-04:** SNR validation: detection rate must fall in a plausible range (1-30% of simulated events). All SNR values must be positive.
- **D-05:** H0 posterior validation: posterior must peak in [0.6, 0.9]. The true value is H=0.73 (constants.py).
- **D-06:** Pipeline completion checks: all 3 tasks complete, merge produces `cramer_rao_bounds.csv` and `prepared_cramer_rao_bounds.csv`, evaluate produces an H0 posterior distribution.
- **D-07:** Monitor with `sacct -j <job_ids>` via SSH for status overview. Tail SLURM log files in `$RUN_DIR/logs/` when investigating failures or slow jobs.
- **D-08:** On task failure: investigate first (read .err log, diagnose the issue, report) before resubmitting. With only 3 tasks, any failure likely indicates a real problem.
- **D-09:** After validation passes, rsync key output files back to local machine. Cluster workspace expires in 60 days.
- **D-10:** Local results stored in `evaluation/run_YYYYMMDD_seed42/` (already in .gitignore from Phase 6).

### Claude's Discretion
- Polling interval for sacct job monitoring
- Exact rsync flags and which files to copy back (at minimum: merged CSV, prepared CSV, H0 posterior output, run_metadata JSONs, SLURM logs)
- How to present validation results (table, summary, etc.)
- Whether to generate any local plots from the rsynced results

### Deferred Ideas (OUT OF SCOPE)
- Production-scale campaign (50-100+ tasks, 100+ steps) -- deferred to future milestone (PROD-01)
- Performance profiling/optimization -- deferred (PERF-01, PERF-02)
- Local plot generation from rsynced results -- could be added but not in scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SIM-01 | Test simulation run completed (5 tasks, 50-100 steps) with timing data recorded | CONTEXT.md overrides to 3 tasks, 25 steps. Pipeline submission via `submit_pipeline.sh`. Timing in `run_metadata_{idx}.json`. CWD fix required in sbatch. |
| SIM-02 | Evaluation pipeline run on fresh Cramer-Rao bounds produces H0 posterior | `evaluate.sbatch` chains after merge via `--dependency=afterok`. Posterior written to `simulations/posteriors/h_0_73.json`. CWD fix also needed here. |
| SIM-03 | Results validated (SNR distributions physical, detection rates reasonable, posterior sanity-checked) | Validation criteria from D-03 through D-06. CSV columns documented. Posterior JSON structure documented. |
</phase_requirements>

## Standard Stack

No new libraries needed. This phase uses existing infrastructure exclusively.

### Core Tools
| Tool | Purpose | Already Available |
|------|---------|-------------------|
| `ssh bwunicluster` | Remote command execution | Phase 7 (ACCESS-02) |
| `cluster/submit_pipeline.sh` | Pipeline submission | Phase 4 (SLURM-04) |
| `cluster/resubmit_failed.sh` | Failure recovery | Phase 4 |
| `sacct` | Job status monitoring | SLURM built-in |
| `rsync` | File transfer | Standard tool |
| `pandas` | CSV validation (local) | Already installed |

## Architecture Patterns

### Pipeline Flow
```
submit_pipeline.sh --tasks 3 --steps 25 --seed 42
    |
    v
simulate.sbatch (GPU array job, 3 tasks)
    |  -> cramer_rao_bounds_simulation_0.csv
    |  -> cramer_rao_bounds_simulation_1.csv
    |  -> cramer_rao_bounds_simulation_2.csv
    |  -> run_metadata_0.json, run_metadata_1.json, run_metadata_2.json
    |  -> undetected_events_simulation_0.csv, ...
    v  (--dependency=afterok)
merge.sbatch (CPU job)
    |  -> cramer_rao_bounds.csv (merged)
    |  -> prepared_cramer_rao_bounds.csv
    v  (--dependency=afterok)
evaluate.sbatch (CPU job)
    |  -> simulations/posteriors/h_0_73.json
    |  -> simulations/posteriors_with_bh_mass/h_0_73.json
    v
DONE
```

### Output Directory Structure
```
$RUN_DIR/                              # e.g., $WORKSPACE/run_20260328_seed42/
├── logs/
│   ├── simulate_<jobid>_0.out/.err
│   ├── simulate_<jobid>_1.out/.err
│   ├── simulate_<jobid>_2.out/.err
│   ├── merge_<jobid>.out/.err
│   └── evaluate_<jobid>.out/.err
├── simulations/
│   ├── cramer_rao_bounds_simulation_0.csv  (deleted after merge)
│   ├── cramer_rao_bounds_simulation_1.csv  (deleted after merge)
│   ├── cramer_rao_bounds_simulation_2.csv  (deleted after merge)
│   ├── cramer_rao_bounds.csv               (merged)
│   ├── prepared_cramer_rao_bounds.csv      (prepared for eval)
│   ├── undetected_events.csv               (merged)
│   ├── posteriors/
│   │   └── h_0_73.json
│   └── posteriors_with_bh_mass/
│       └── h_0_73.json
├── run_metadata_0.json
├── run_metadata_1.json
└── run_metadata_2.json
```

### SSH Command Pattern (from Phase 7)
```bash
ssh bwunicluster '<command>'
```

### Seed Derivation
Per-task seed = BASE_SEED + SLURM_ARRAY_TASK_ID:
- Task 0: seed 42
- Task 1: seed 43
- Task 2: seed 44

### Anti-Patterns to Avoid
- **Submitting without fixing CWD:** The simulate and evaluate sbatch scripts must `cd "$RUN_DIR"` before running python (see Pitfall 1 below).
- **Resubmitting without investigating:** D-08 requires reading .err logs first.
- **Polling too frequently:** `sacct` queries are lightweight but repeated SSH connections add overhead.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Pipeline submission | Manual sbatch calls | `cluster/submit_pipeline.sh` | Handles dependency chaining, run directory creation |
| Failed task recovery | Manual per-task resubmission | `cluster/resubmit_failed.sh` | Queries sacct, cleans partial output, resubmits only failed indices |
| CSV merging | Custom pandas script | `emri-merge --workdir $RUN_DIR --delete-sources` | Already handles edge cases (missing files, existing merged output) |
| Detection preparation | Custom filtering | `emri-prepare --workdir $RUN_DIR` | Handles truncated-normal sampling, unusable detection cleanup |

## Common Pitfalls

### Pitfall 1: CWD Mismatch in SLURM Jobs (CRITICAL)
**What goes wrong:** Simulation writes CSVs to `CWD/simulations/` (relative paths in `constants.py`), but SLURM jobs start in the submission directory, not `$RUN_DIR`. The merge script uses `--workdir $RUN_DIR` to find CSVs in `$RUN_DIR/simulations/`. If simulate writes to a different CWD, merge finds nothing.
**Why it happens:** `simulate.sbatch` and `evaluate.sbatch` do not `cd "$RUN_DIR"` before invoking python. The code's `ParameterEstimation.flush_pending_results()` writes to the relative path `simulations/cramer_rao_bounds_simulation_$index.csv`. Similarly, `BayesianStatistics.__init__()` reads from relative `PREPARED_CRAMER_RAO_BOUNDS_PATH`.
**How to avoid:** Add `cd "$RUN_DIR"` in `simulate.sbatch` and `evaluate.sbatch` after environment setup and before running python. This ensures all relative paths resolve within `$RUN_DIR`.
**Warning signs:** Merge step reports "No cramer rao bounds files found" despite simulate tasks completing successfully.
**Confidence:** HIGH -- verified by reading `flush_pending_results()` (line 424-431 in `parameter_estimation.py`), `BayesianStatistics.__init__()` (line 106-108 in `bayesian_statistics.py`), and `merge_cramer_rao_bounds()` (line 65-66 in `merge_cramer_rao_bounds.py`). The integration test in `test_evaluation_pipeline.py` uses `monkeypatch.chdir(tmp_path)` to work around this exact issue.

### Pitfall 2: Zero Detections
**What goes wrong:** With only 75 simulation steps (3 tasks x 25 steps) and a detection rate of 1-30%, it is possible (though unlikely) to get zero detections above SNR_THRESHOLD=20. Zero detections means no `cramer_rao_bounds.csv` rows, which would cause the merge and evaluation to fail or produce empty output.
**Why it happens:** EMRI detection rates are low. The randomized parameters may produce only sub-threshold SNR values in a small sample.
**How to avoid:** Accept this as a valid outcome of the smoke test. If zero detections occur, report it and consider increasing `--steps` or investigating whether the simulation parameters are reasonable. With seed 42 and 75 steps, the outcome is deterministic.
**Warning signs:** Per-task CSV files are empty or have only header rows.

### Pitfall 3: GPU Queue Wait Times
**What goes wrong:** The `gpu_h100` partition may have long queue times, especially during peak usage. With 3 tasks, the array job may wait hours before starting.
**Why it happens:** H100 GPUs are shared resources on the cluster.
**How to avoid:** Check queue depth before submitting (`squeue -p gpu_h100 | wc -l`). Be prepared to wait. Do not cancel and resubmit repeatedly.
**Warning signs:** `sacct` shows jobs in PENDING state for extended periods.

### Pitfall 4: Workspace Not Found
**What goes wrong:** `ws_find emri` returns empty, and `submit_pipeline.sh` exits with "WORKSPACE is not set".
**Why it happens:** Workspace expired (60-day default) or was never created.
**How to avoid:** Phase 7 preflight should have verified this. If it fails, run `cluster/setup.sh` to recreate.
**Warning signs:** The `$WORKSPACE` variable is empty after sourcing `modules.sh`.

### Pitfall 5: evaluate.sbatch Also Needs CWD Fix
**What goes wrong:** Same as Pitfall 1, but for the evaluation step. `BayesianStatistics.__init__()` reads `PREPARED_CRAMER_RAO_BOUNDS_PATH` and `CRAMER_RAO_BOUNDS_OUTPUT_PATH` as relative paths. Without `cd "$RUN_DIR"`, it reads from the wrong directory.
**How to avoid:** Add `cd "$RUN_DIR"` in `evaluate.sbatch` before running python.
**Warning signs:** FileNotFoundError for `simulations/prepared_cramer_rao_bounds.csv`.

## Code Examples

### Submitting the Pipeline
```bash
# From local machine via SSH
ssh bwunicluster 'cd $PROJECT_ROOT && bash cluster/submit_pipeline.sh --tasks 3 --steps 25 --seed 42'
```

### Monitoring Jobs
```bash
# Check all three job statuses
ssh bwunicluster 'sacct -j <SIM_JOB>,<MERGE_JOB>,<EVAL_JOB> --format=JobID%30,JobName%15,State%12,Elapsed%10,ExitCode'

# Check only pending/running
ssh bwunicluster 'squeue -u $USER'
```

### Reading SLURM Logs
```bash
# Read simulation task 0 error log
ssh bwunicluster 'cat $RUN_DIR/logs/simulate_<jobid>_0.err'

# Tail the latest log
ssh bwunicluster 'tail -50 $RUN_DIR/logs/simulate_<jobid>_0.out'
```

### Validation: CSV Inspection
```bash
# Count detections per task (before merge)
ssh bwunicluster 'wc -l $RUN_DIR/simulations/cramer_rao_bounds_simulation_*.csv'

# Check merged CSV row count and columns
ssh bwunicluster 'head -1 $RUN_DIR/simulations/cramer_rao_bounds.csv && wc -l $RUN_DIR/simulations/cramer_rao_bounds.csv'

# Check SNR values
ssh bwunicluster 'python3 -c "
import pandas as pd
df = pd.read_csv(\"$RUN_DIR/simulations/cramer_rao_bounds.csv\")
print(f\"Detections: {len(df)}\")
print(f\"SNR range: [{df.SNR.min():.1f}, {df.SNR.max():.1f}]\")
print(f\"All SNR > 0: {(df.SNR > 0).all()}\")
"'
```

### Validation: Posterior Inspection
```bash
# Check posterior JSON exists and peek at structure
ssh bwunicluster 'python3 -c "
import json
with open(\"$RUN_DIR/simulations/posteriors/h_0_73.json\") as f:
    data = json.load(f)
print(f\"h value: {data[\"h\"]}\")
print(f\"Keys: {list(data.keys())[:5]}...\")
"'
```

### Rsync Results Back
```bash
# Create local destination
mkdir -p evaluation/run_20260328_seed42

# Rsync key files
rsync -avz bwunicluster:'$RUN_DIR/simulations/cramer_rao_bounds.csv \
    $RUN_DIR/simulations/prepared_cramer_rao_bounds.csv \
    $RUN_DIR/simulations/undetected_events.csv' \
    evaluation/run_20260328_seed42/simulations/

rsync -avz bwunicluster:'$RUN_DIR/simulations/posteriors/' \
    evaluation/run_20260328_seed42/simulations/posteriors/

rsync -avz bwunicluster:'$RUN_DIR/simulations/posteriors_with_bh_mass/' \
    evaluation/run_20260328_seed42/simulations/posteriors_with_bh_mass/

rsync -avz bwunicluster:'$RUN_DIR/run_metadata_*.json' \
    evaluation/run_20260328_seed42/

rsync -avz bwunicluster:'$RUN_DIR/logs/' \
    evaluation/run_20260328_seed42/logs/
```

### CWD Fix for simulate.sbatch
```bash
# Add after environment setup, before python invocation:
cd "$RUN_DIR"
```

## Validation Criteria Summary

| Check | Criterion | Source |
|-------|-----------|--------|
| Pipeline completion | All 3 simulate tasks: COMPLETED | D-06, sacct |
| Pipeline completion | Merge job: COMPLETED | D-06, sacct |
| Pipeline completion | Evaluate job: COMPLETED | D-06, sacct |
| File existence | `cramer_rao_bounds.csv` exists, non-empty | D-06 |
| File existence | `prepared_cramer_rao_bounds.csv` exists, non-empty | D-06 |
| File existence | `posteriors/h_0_73.json` exists | D-06 |
| File existence | `run_metadata_{0,1,2}.json` all exist | SIM-01 |
| SNR physical | All SNR values > 0 | D-04 |
| Detection rate | 1-30% of 75 total steps (1-22 detections) | D-04 |
| H0 posterior | Peak in [0.6, 0.9] | D-05 |
| Timing | `run_metadata_*.json` contains timing data | SIM-01 |
| Reproducibility | Seeds recorded: 42, 43, 44 | TRACE-02 |

## Project Constraints (from CLAUDE.md)

- **Package manager:** uv (never manually edit pyproject.toml deps)
- **Python version:** 3.13
- **Pre-commit hooks:** ruff + mypy run on every commit
- **Physics Change Protocol:** Required for any formula/constant changes. This phase should NOT require physics changes -- it runs existing code.
- **GPU imports:** Must be guarded with try/except. Not relevant for this phase (no code changes expected).
- **Testing:** `uv run pytest -m "not gpu and not slow"` for CPU-only tests. Not directly relevant for this phase.
- **Skill triggers:** `/check` before any git commit; `/pre-commit-docs` after check passes.

## Environment Availability

> This phase's external dependencies are on the cluster, not the local machine.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| SSH access | All cluster ops | Verified by Phase 7 | -- | -- |
| rsync | Result transfer | Standard on both machines | -- | scp |
| pandas (local) | CSV validation | In dev venv | -- | -- |
| SLURM (cluster) | Job submission | On cluster | -- | -- |
| GPU partition (cluster) | Simulation | gpu_h100 verified by Phase 7 | -- | -- |
| Workspace (cluster) | Output storage | Verified by Phase 7 preflight | -- | Run setup.sh |

**Missing dependencies with no fallback:** None expected (Phase 7 verified all prerequisites).

**Missing dependencies with fallback:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | `pyproject.toml [tool.pytest.ini_options]` |
| Quick run command | `uv run pytest -m "not gpu and not slow"` |
| Full suite command | `uv run pytest` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SIM-01 | Simulation tasks complete with timing data | manual-only | N/A -- requires cluster execution | N/A |
| SIM-02 | Evaluation produces H0 posterior | manual-only | N/A -- requires cluster data | N/A |
| SIM-03 | Results pass sanity checks | manual-only | N/A -- validation is on cluster output | N/A |

**Justification for manual-only:** All three requirements involve running jobs on the GPU cluster and validating their outputs. These cannot be tested locally or automated in pytest. Validation is performed by SSH commands that inspect the cluster output files.

### Sampling Rate
- **Per task commit:** `uv run pytest -m "not gpu and not slow"` (only if code is changed)
- **Per wave merge:** N/A (single-wave phase)
- **Phase gate:** All validation criteria from the table above pass

### Wave 0 Gaps
None -- this phase does not require new tests. It validates existing infrastructure by running it.

## Open Questions

1. **What is the actual detection rate for seed 42 with 25 steps?**
   - What we know: Detection rate is typically 1-30% based on D-04 validation range. With 75 total steps, we expect 1-22 detections.
   - What's unclear: The exact count is unknown until the simulation runs. It is deterministic for seed 42/43/44.
   - Recommendation: Run the simulation and report the actual count. If zero, investigate parameter ranges.

2. **Does the posterior JSON structure allow direct H0 peak extraction?**
   - What we know: JSON contains `{"h": 0.73, "0": [...], "1": [...], ...}` where integer keys map to detection indices and values are posterior arrays. The "h" key stores the evaluated h value.
   - What's unclear: The JSON does not contain the H0 grid values explicitly -- only the posterior values. The grid must be reconstructed from the code.
   - Recommendation: For smoke-test validation, simply verify the JSON file exists and contains the expected `h` value. Full posterior peak analysis may require loading the H0 grid from the code.

3. **Will the evaluate step have enough detections to produce a meaningful posterior?**
   - What we know: With possibly very few detections from 75 steps, the posterior may be very broad or dominated by prior.
   - What's unclear: Whether the code handles the edge case of very few detections gracefully.
   - Recommendation: Accept a broad posterior as valid for a smoke test. The key check is that it peaks somewhere in [0.6, 0.9], not that it is tight.

## Sources

### Primary (HIGH confidence)
- `cluster/submit_pipeline.sh` -- pipeline submission logic and argument handling
- `cluster/simulate.sbatch` -- GPU array job structure, seed derivation
- `cluster/merge.sbatch` -- merge and prepare invocation
- `cluster/evaluate.sbatch` -- evaluation invocation
- `cluster/resubmit_failed.sh` -- failure recovery logic
- `cluster/modules.sh` -- environment setup, workspace resolution
- `master_thesis_code/constants.py` -- output file paths, physical constants
- `master_thesis_code/parameter_estimation/parameter_estimation.py` -- CSV write logic (flush_pending_results)
- `master_thesis_code/bayesian_inference/bayesian_statistics.py` -- posterior output logic
- `scripts/merge_cramer_rao_bounds.py` -- merge script with --workdir handling
- `scripts/prepare_detections.py` -- prepare script with --workdir handling
- `master_thesis_code/arguments.py` -- CLI argument parsing, working_directory handling
- `master_thesis_code_test/integration/test_evaluation_pipeline.py` -- confirms CWD dependency via `monkeypatch.chdir`
- `.planning/phases/07-cluster-access/07-CONTEXT.md` -- SSH access method (D-01)
- `.planning/phases/08-simulation-campaign/08-CONTEXT.md` -- all locked decisions

## Metadata

**Confidence breakdown:**
- Pipeline infrastructure: HIGH -- all scripts read and verified
- CWD issue: HIGH -- verified across 4 source files and confirmed by integration test pattern
- Validation criteria: HIGH -- directly from CONTEXT.md decisions
- Posterior structure: MEDIUM -- read from code but not verified against actual output

**Research date:** 2026-03-28
**Valid until:** 2026-04-28 (stable infrastructure, no external dependencies changing)
