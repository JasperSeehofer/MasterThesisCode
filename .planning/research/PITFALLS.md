# Domain Pitfalls: HPC Cluster Deployment

**Domain:** GPU-accelerated Python (CuPy/CUDA) on SLURM HPC (bwUniCluster 3.0)
**Researched:** 2026-03-25

## Critical Pitfalls

Mistakes that cause job failures, data loss, or account issues.

### Pitfall 1: Running Computation on Login Nodes
**What goes wrong:** Running `python -m master_thesis_code` directly on the login node instead of submitting via `sbatch`. Login nodes are shared; resource-intensive processes get killed automatically and may result in account warnings.
**Why it happens:** Developer habit from local machine workflow.
**Consequences:** Process killed, wasted time, potential account suspension.
**Prevention:** All computation goes through `sbatch` or `srun`. Only use login nodes for `uv sync`, `git`, file management, and `sbatch` submission.
**Detection:** If you're running Python computation without a SLURM_JOB_ID environment variable, you're on a login node.

### Pitfall 2: Installing Packages on Compute Nodes
**What goes wrong:** `uv sync` or `pip install` fails on compute nodes because they have no outbound internet access.
**Why it happens:** Forgetting to install dependencies before submitting the job.
**Consequences:** Job fails immediately; wasted queue wait time.
**Prevention:** Always run `uv sync --extra gpu` on the login node before submitting jobs. The `.venv/` directory is on `$HOME` (Lustre) and visible to all nodes.
**Detection:** "Connection refused" or DNS errors in job output logs.

### Pitfall 3: Workspace Expiration Data Loss
**What goes wrong:** Workspace expires, grace period passes, simulation data permanently deleted.
**Why it happens:** Default 60-day lifetime; only 3 extensions (240 days max cumulative); easy to forget.
**Consequences:** All simulation CSVs, posteriors, and intermediate results permanently lost.
**Prevention:** (1) Set calendar reminders for workspace expiration. (2) Copy final results (posteriors, thesis figures) to `$HOME` immediately after evaluation completes. (3) Run `ws_list` periodically to check remaining time. (4) Consider a cron-like check in the setup script.
**Detection:** `ws_list` shows remaining days; warnings when lifetime is low.

### Pitfall 4: Unconditional CuPy Import Crashes
**What goes wrong:** `import cupy` at module top level (no guard) makes the entire package unimportable on CPU-only login nodes and CPU compute nodes.
**Why it happens:** Code was written assuming GPU is always available. `LISA_configuration.py` and `MemoryManagement` have this issue.
**Consequences:** Cannot run ANY command (including `--evaluate` which is CPU-only) on nodes without CUDA.
**Prevention:** Guard all CuPy imports with `try/except ImportError`. Use the `_get_xp(use_gpu)` pattern documented in CLAUDE.md.
**Detection:** `ImportError: No module named 'cupy'` in job logs or on login node.

### Pitfall 5: Requesting All 4 GPUs When Only 1 Is Needed
**What goes wrong:** `--gres=gpu:4` when each EMRI simulation uses a single GPU.
**Why it happens:** Assuming more GPUs = faster; not understanding that the code is single-GPU.
**Consequences:** 4x longer queue wait, 4x resource waste, lower scheduling priority for future jobs, other users blocked.
**Prevention:** Always use `--gres=gpu:1` for simulation array tasks. This lets SLURM pack 4 jobs per node.
**Detection:** `squeue` shows low GPU utilization; `nvidia-smi` inside job shows 3 idle GPUs.

## Moderate Pitfalls

### Pitfall 6: Module Version Mismatch Between Setup and Job
**What goes wrong:** `uv sync` run with one CUDA/Python version; job script loads different modules.
**Prevention:** Single `modules.sh` file sourced by both setup instructions and all job scripts. Never hardcode module names in individual scripts.

### Pitfall 7: $HOME Quota Exhaustion
**What goes wrong:** Storing simulation output CSVs in `$HOME` fills the 500 GiB quota.
**Prevention:** All simulation output goes to workspace (`ws_find emri-data`). Only the repo, `.venv/`, and final results live in `$HOME`.

### Pitfall 8: Interactive Script Calls in Batch Jobs
**What goes wrong:** `merge_cramer_rao_bounds.py` calls `input()` waiting for user confirmation; batch job hangs or fails.
**Prevention:** Add `--delete-sources` CLI flag; remove all interactive prompts from scripts used in batch context.

### Pitfall 9: CUDA_VISIBLE_DEVICES Confusion
**What goes wrong:** Code tries to select a specific GPU device (e.g., `cupy.cuda.Device(2)`) but SLURM remaps GPU indices via `CUDA_VISIBLE_DEVICES`. Device 2 does not exist when you only requested 1 GPU.
**Prevention:** Always use device 0 in code. SLURM handles the mapping. Do not set `CUDA_VISIBLE_DEVICES` manually in job scripts.

### Pitfall 10: Forgetting `--parsable` in Dependency Chains
**What goes wrong:** `sbatch` returns a human-readable string like "Submitted batch job 12345" instead of just "12345", breaking the dependency chain script.
**Prevention:** Always use `sbatch --parsable` when capturing job IDs for `--dependency` chains.

### Pitfall 11: Array Job Output File Naming
**What goes wrong:** All 100 array tasks write to the same output file, overwriting each other.
**Prevention:** Use `--output=logs/sim_%A_%a.out` where `%A` is job ID and `%a` is array index.

## Minor Pitfalls

### Pitfall 12: Forgetting to Create Log Directory
**What goes wrong:** SLURM cannot write output files if `logs/` directory does not exist; job fails silently.
**Prevention:** `mkdir -p logs` in setup script; or use `--output=/dev/null` during testing.

### Pitfall 13: uv Not in PATH After Install
**What goes wrong:** `uv` installed to `~/.local/bin` but PATH not updated in current shell.
**Prevention:** Run `exec $SHELL` after installing uv, or add `export PATH="$HOME/.local/bin:$PATH"` to `~/.bashrc`.

### Pitfall 14: GSL Module Not Found
**What goes wrong:** `fastemriwaveforms` build fails because GSL headers are not available.
**Prevention:** Run `module spider gsl` on first login to discover exact module name. Document in `modules.sh`. If not available as a module, install via EasyBuild or use Apptainer container.

### Pitfall 15: Python 3.13 Availability
**What goes wrong:** Cluster may not have Python 3.13 module; `uv sync` fails or uses wrong Python.
**Prevention:** Verify with `module spider python` on first login. If 3.13 is not available, `uv python install 3.13` can install it locally (uv manages its own Python copies). Alternatively, adjust `.python-version` to available version (project allows >=3.10).

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Code fixes (--use_gpu, MemoryManagement) | Breaking GPU path while fixing CPU path | Run full test suite (`pytest -m "not gpu"`) after each change |
| Module setup | Wrong module names from wiki (may be stale) | Verify ALL module names with `module spider` on actual cluster |
| Job script creation | Incorrect resource requests (too much/little) | Start with dev queue (30 min, 1 node); profile; then scale |
| First production run | Array job floods queue; other users annoyed | Use `--array=0-99%20` to throttle concurrent tasks |
| Merge step | Partial CSVs from failed array tasks | Check array task exit codes; re-submit failures before merge |
| Evaluation step | Multiprocessing pool size mismatch with SLURM allocation | Default `--num_workers` to `SLURM_CPUS_PER_TASK` env var |
| Workspace lifecycle | Forgetting to extend; data lost | Calendar reminder; copy final results to $HOME immediately |

## Sources

- [bwUniCluster 3.0 SLURM Guide](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs/Slurm)
- [bwUniCluster 3.0 Batch Queues](https://wiki.bwhpc.de/e/BwUniCluster3.0/Batch_Queues)
- [bwUniCluster 3.0 Filesystem Details](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture/Filesystem_Details)
- [bwHPC Workspace Documentation](https://wiki.bwhpc.de/e/Workspace)
- [SLURM GRES Scheduling](https://slurm.schedmd.com/gres.html)
- Project context: CLAUDE.md known bugs, `.planning/PROJECT.md`

---

*Pitfalls analysis: 2026-03-25*
