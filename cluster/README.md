# Running on bwUniCluster 3.0

This directory contains scripts for running the EMRI simulation pipeline on
bwUniCluster 3.0 at KIT as SLURM array jobs. The pipeline generates Cramer-Rao
bounds from GPU-accelerated EMRI waveform simulations, merges the per-task
results, and runs Bayesian inference to produce a Hubble constant posterior.

## Quickstart

Five commands from login to results:

1. Connect to the cluster:
   ```bash
   ssh bwunicluster.scc.kit.edu
   ```

2. Pull the latest code:
   ```bash
   cd ~/MasterThesisCode && git pull
   ```

3. First-time setup (only needed once):
   ```bash
   bash cluster/setup.sh
   ```

4. Submit a simulation campaign:
   ```bash
   bash cluster/submit_pipeline.sh --tasks 100 --steps 50 --seed 42
   ```

5. Retrieve results (from your local machine):
   ```bash
   rsync -avz bwunicluster:$(ssh bwunicluster 'ws_find emri')/run_20260327_seed42/ ./results/
   ```

## Pipeline Overview

```
submit_pipeline.sh --tasks 100 --steps 50 --seed 42
         |
         v
+--------------------+     +-------------+     +------------------+
| simulate.sbatch    |---->| merge.sbatch|---->| evaluate.sbatch  |
| (GPU array: 0-99)  |     | (CPU)       |     | (CPU, 16 cores)  |
| 1 GPU per task     |     | emri-merge  |     | Bayesian H0      |
| ~2h per task       |     | emri-prepare|     | inference         |
+--------------------+     +-------------+     +------------------+
        |                        |                    |
   afterok dependency      afterok dependency         |
                                                      v
                                              run_YYYYMMDD_seed42/
```

**Stage 1 -- Simulate (GPU array job):** Each array task runs a single EMRI
simulation with `--simulation_index` mapped to `SLURM_ARRAY_TASK_ID`. Each task
gets its own GPU and a unique seed (`BASE_SEED + TASK_ID`). Output is one CSV
per task.

**Stage 2 -- Merge (CPU job):** Combines the per-task Cramer-Rao bounds CSVs
into a single `cramer_rao_bounds.csv` using the `emri-merge` console entry
point, then filters and prepares detections with `emri-prepare`.

**Stage 3 -- Evaluate (CPU job, 16 cores):** Runs Bayesian inference on the
merged detections to produce the Hubble constant posterior. Worker count is
auto-detected from the SLURM cgroup allocation.

Each stage depends on the previous one via `--dependency=afterok`, so the full
pipeline runs unattended after a single `submit_pipeline.sh` invocation.

## Prerequisites

- **SSH access** to bwUniCluster 3.0 (requires a bwHPC account)
- **University VPN** if connecting from off-campus (e.g., `openconnect` to your
  university VPN gateway)
- **Git SSH key** configured on the cluster for pulling code (deploy key with no
  passphrase is recommended)

## First-Time Setup

Run the setup script once after your first login:

```bash
bash cluster/setup.sh
```

This script performs four steps:

1. **Loads environment modules:** `compiler/gnu/14.2`, `devel/cuda/12.8`,
   `devel/python/3.13.3-gnu-14.2`. GSL 2.6 is system-wide; no module needed.
2. **Installs uv** (if not already present) to `~/.local/bin`.
3. **Allocates a workspace:** `ws_allocate emri 60` (60-day expiration).
4. **Creates the Python environment:** `uv sync --extra gpu` installs all
   dependencies including CuPy and fastemriwaveforms with CUDA 12 support.

The script is idempotent -- safe to re-run if anything changes.

> **Warning:** bwHPC workspaces expire. The default allocation is 60 days.
> Extend before expiration with `ws_extend emri 60`. Expired workspaces are
> **permanently deleted** with no recovery. Check remaining time with `ws_list`.

## Running a Campaign

Source the module environment, then submit the pipeline:

```bash
source cluster/modules.sh
bash cluster/submit_pipeline.sh --tasks 100 --steps 50 --seed 42
```

The three required flags:

| Flag       | Description                                              |
|------------|----------------------------------------------------------|
| `--tasks`  | Number of parallel simulation jobs (array size)          |
| `--steps`  | Number of EMRI iterations per task                       |
| `--seed`   | Base random seed; per-task seed = `seed + task_id`       |

All three flags are required with no defaults -- this prevents accidental
submissions with unintended parameters.

**Output directory:** `$WORKSPACE/run_YYYYMMDD_seed42/`

```
run_20260327_seed42/
+-- logs/
|   +-- simulate_12345_0.out
|   +-- simulate_12345_0.err
|   +-- merge_12346.out
|   +-- evaluate_12347.out
+-- simulations/
|   +-- cramer_rao_bounds.csv          (after merge)
|   +-- prepared_cramer_rao_bounds.csv (after prepare)
+-- run_metadata_0.json
+-- run_metadata_1.json
...
```

> **Tip:** Test with a small run first before submitting a large campaign:
> ```bash
> bash cluster/submit_pipeline.sh --tasks 2 --steps 10 --seed 42
> ```
> bwUniCluster has a `dev_gpu_h100` partition with shorter queue times for
> testing. To use it, edit `simulate.sbatch` and change `--partition=gpu_h100`
> to `--partition=dev_gpu_h100`.

## Monitoring

Check queue status:

```bash
squeue -u $USER
```

Detailed job information (the `submit_pipeline.sh` output prints all three job
IDs in a ready-to-use `sacct` command):

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

Follow live output from a running simulation task:

```bash
tail -f $WORKSPACE/run_*/logs/simulate_*.out
```

## Retrieving Results

From your local machine, use `scp`:

```bash
scp -r bwunicluster:$(ssh bwunicluster 'ws_find emri')/run_20260327_seed42/ ./results/
```

Or use `rsync` for incremental transfers (useful for large campaigns):

```bash
rsync -avz bwunicluster:WORKSPACE_PATH/run_20260327_seed42/ ./results/
```

Replace `WORKSPACE_PATH` with the output of `ws_find emri` on the cluster.

> **Warning:** Workspace `emri` expires after 60 days. Copy results to
> persistent storage before expiration. Check remaining time: `ws_list`.
> Extend: `ws_extend emri 60`.

## Troubleshooting

### OOM kills

Check memory usage with:

```bash
sacct -j JOB_ID --format=JobID,State,MaxRSS
```

If `State` shows `OUT_OF_MEMORY`, reduce `--simulation_steps` or request more
memory by adding `--mem=` to the SBATCH headers in `simulate.sbatch`.

### Timeout failures

`State` shows `TIMEOUT`. The default time limit is 2 hours per simulation task.
Increase `--time` in `simulate.sbatch` if your simulations consistently exceed
this.

### CUDA errors

Check the error logs for CUDA-related failures:

```bash
grep -r "CUDA error\|CUDARuntimeError" $RUN_DIR/logs/*.err
```

Common causes:
- Module not loaded: ensure `module load devel/cuda/12.8` succeeded (run
  `source cluster/modules.sh` before submitting).
- GPU not allocated: verify `--gres=gpu:1` is set in `simulate.sbatch`.

### Python tracebacks

Find which tasks had Python errors:

```bash
grep -l "Traceback" $RUN_DIR/logs/*.err
```

### Resubmitting failed tasks

After a campaign with partial failures, resubmit only the failed tasks:

```bash
bash cluster/resubmit_failed.sh JOB_ID $RUN_DIR BASE_SEED SIM_STEPS
```

This queries `sacct` for tasks with state `FAILED`, `TIMEOUT`, `NODE_FAIL`, or
`OUT_OF_MEMORY`, cleans up their partial output files, and resubmits only those
array indices. After the resubmitted tasks complete, you need to manually
resubmit the merge and evaluate steps -- the script prints the exact `sbatch`
command to run.

### Log file locations

All SLURM output and error files are written to `$RUN_DIR/logs/`:

- Simulation stdout: `simulate_JOBID_TASKID.out`
- Simulation stderr: `simulate_JOBID_TASKID.err`
- Merge stdout/stderr: `merge_JOBID.out` / `merge_JOBID.err`
- Evaluate stdout/stderr: `evaluate_JOBID.out` / `evaluate_JOBID.err`

Key grep patterns for diagnosing failures:

```bash
grep -r "ERROR\|Traceback\|CUDA error\|OOM\|Killed" $RUN_DIR/logs/
```

## Script Reference

| Script                | Purpose                                                          |
|-----------------------|------------------------------------------------------------------|
| `modules.sh`          | Loads environment modules; exports `$WORKSPACE`, `$PROJECT_ROOT`, `$VENV_PATH` |
| `setup.sh`            | First-time setup: installs uv, allocates workspace, creates venv |
| `simulate.sbatch`     | GPU array job -- one EMRI simulation per task                    |
| `merge.sbatch`        | CPU job -- merges per-task CSVs, prepares detections             |
| `evaluate.sbatch`     | CPU job -- Bayesian inference for H0 posterior (16 CPUs)         |
| `submit_pipeline.sh`  | Pipeline orchestrator -- chains simulate -> merge -> evaluate    |
| `resubmit_failed.sh`  | Resubmits only failed array tasks                                |
| `vpn.sh`              | University VPN connection via openconnect                        |

### Seed strategy

Per-task seed is computed as `BASE_SEED + SLURM_ARRAY_TASK_ID`. This ensures:

1. Each task gets a unique random state.
2. Results are fully reproducible given the same base seed.
3. Resubmitted tasks produce identical results to their original run.

Seeds and all SLURM metadata (job ID, array task ID, node, GPU) are recorded in
`run_metadata_N.json` files in the run directory.
