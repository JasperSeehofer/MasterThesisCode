# Architecture Patterns: HPC Cluster Deployment

**Domain:** SLURM-based HPC deployment of GPU Python scientific computing
**Researched:** 2026-03-25

## Recommended Architecture

### Deployment Layout

```
$HOME/
  emri-project/               # Git repo (version-controlled)
    cluster/
      modules.sh              # Module loader (sourced by all scripts)
      setup.sh                # One-time cluster setup
      submit_pipeline.sh      # Orchestrator: sim -> merge -> eval
      simulate.sbatch         # GPU array job script
      merge.sbatch            # CPU job script
      evaluate.sbatch         # CPU job script
    master_thesis_code/       # Source code
    .venv/                    # uv-managed virtualenv
    uv.lock                   # Exact dependency versions
    logs/                     # SLURM output/error logs

$(ws_find emri-data)/         # Workspace (temporary, large)
  simulations/
    cramer_rao_bounds_000.csv # Per-array-task output
    cramer_rao_bounds_001.csv
    ...
    cramer_rao_bounds.csv     # Merged output (after merge step)
    run_metadata_000.json     # Per-task metadata
  posteriors/
    h_0_73.json               # Evaluation output
  figures/                    # Generated plots
```

### Component Boundaries

| Component | Responsibility | Location | Communicates With |
|-----------|---------------|----------|-------------------|
| `modules.sh` | Load correct module versions | `cluster/` | Sourced by all `.sbatch` scripts and `setup.sh` |
| `simulate.sbatch` | Submit GPU array jobs for EMRI simulation | `cluster/` | Reads modules.sh; invokes `python -m master_thesis_code` |
| `merge.sbatch` | Merge per-task CSVs into single file | `cluster/` | Reads modules.sh; invokes `scripts/merge_cramer_rao_bounds.py` |
| `evaluate.sbatch` | Run Bayesian inference on merged data | `cluster/` | Reads modules.sh; invokes `python -m master_thesis_code --evaluate` |
| `submit_pipeline.sh` | Chain all three jobs with dependencies | `cluster/` | Calls `sbatch --parsable` on each `.sbatch` script |
| `setup.sh` | First-time environment setup | `cluster/` | Installs uv, allocates workspace, runs `uv sync` |
| `run_metadata.json` | Track provenance per simulation task | Workspace | Written by `main.py`; includes SLURM env vars |

### Data Flow

```
submit_pipeline.sh
  |
  +--> sbatch simulate.sbatch  (array=0-99, gpu_h100, --gres=gpu:1)
  |      |
  |      +--> Task 0:  python -m master_thesis_code $WORKSPACE --simulation_steps 100 --simulation_index 0 --seed 42 --use_gpu
  |      +--> Task 1:  python -m master_thesis_code $WORKSPACE --simulation_steps 100 --simulation_index 1 --seed 43 --use_gpu
  |      +--> ...
  |      +--> Task 99: python -m master_thesis_code $WORKSPACE --simulation_steps 100 --simulation_index 99 --seed 141 --use_gpu
  |      |
  |      Output: cramer_rao_bounds_000.csv ... cramer_rao_bounds_099.csv
  |
  +--> sbatch merge.sbatch  (dependency=afterok:SIM_JOB, cpu)
  |      |
  |      +--> python scripts/merge_cramer_rao_bounds.py $WORKSPACE/simulations --delete-sources
  |      |
  |      Output: cramer_rao_bounds.csv (merged)
  |
  +--> sbatch evaluate.sbatch  (dependency=afterok:MERGE_JOB, cpu, 48 cores)
         |
         +--> python -m master_thesis_code $WORKSPACE --evaluate --num_workers 48
         |
         Output: posteriors/h_0_73.json, figures/
```

## Patterns to Follow

### Pattern 1: Single Module Source of Truth
**What:** All module load commands live in one file (`cluster/modules.sh`).
**When:** Always. Every job script sources this file.
**Why:** Prevents version drift between setup and runtime; makes updates atomic.

```bash
# In every .sbatch file:
source "$(dirname "$0")/modules.sh"
source "${HOME}/emri-project/.venv/bin/activate"
```

### Pattern 2: Workspace Path via `ws_find`
**What:** Never hardcode workspace paths. Always resolve dynamically.
**When:** In every job script and any script that reads/writes simulation data.
**Why:** Workspace paths include timestamps and user IDs; they change on reallocation.

```bash
WORKSPACE=$(ws_find emri-data)
if [ -z "$WORKSPACE" ]; then
    echo "ERROR: Workspace 'emri-data' not found. Run cluster/setup.sh first." >&2
    exit 1
fi
```

### Pattern 3: SLURM Environment in Metadata
**What:** Capture SLURM environment variables in `run_metadata.json`.
**When:** At the start of every simulation task.
**Why:** Enables post-hoc debugging ("which node produced this outlier?") and reproducibility.

```python
slurm_metadata = {
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
    "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
}
```

### Pattern 4: Deterministic Seed from Array Index
**What:** `seed = base_seed + SLURM_ARRAY_TASK_ID`
**When:** Every array task.
**Why:** Each task gets a unique but deterministic seed. Re-running the same array index produces the same result.

### Pattern 5: Throttled Array Submission
**What:** Use `--array=0-99%20` to limit concurrent tasks.
**When:** Large array jobs (>20 tasks).
**Why:** Prevents flooding the queue; courteous to other users; avoids hitting per-user job limits.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Hardcoded Paths
**What:** Writing `/path/to/workspace/abc123/...` directly in scripts.
**Why bad:** Breaks on workspace reallocation, different users, different clusters.
**Instead:** Use `ws_find`, `$HOME`, `$TMPDIR`, `SLURM_SUBMIT_DIR`.

### Anti-Pattern 2: Module Loads Scattered Across Scripts
**What:** Each `.sbatch` file has its own `module load` commands.
**Why bad:** Version drift; updating one script but forgetting another.
**Instead:** Single `modules.sh` sourced everywhere.

### Anti-Pattern 3: GPU Compute on Login Node
**What:** Running `python -m master_thesis_code` directly on the login node.
**Why bad:** Process killed; account warning; wastes shared resources.
**Instead:** Always `sbatch` or `srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:10:00`.

### Anti-Pattern 4: Manual CUDA_VISIBLE_DEVICES
**What:** Setting `export CUDA_VISIBLE_DEVICES=2` in job scripts.
**Why bad:** SLURM manages GPU assignment; manual override causes device ID conflicts.
**Instead:** Let SLURM set it. Use device 0 in code.

## Sources

- [bwUniCluster 3.0 SLURM Guide](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs/Slurm)
- [bwUniCluster 3.0 Filesystem Details](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture/Filesystem_Details)
- [bwHPC Workspace Documentation](https://wiki.bwhpc.de/e/Workspace)

---

*Architecture analysis: 2026-03-25*
