# Technology Stack: HPC/Cluster Deployment

**Project:** EMRI Parameter Estimation -- bwUniCluster 3.0 Deployment
**Researched:** 2026-03-25

## Recommended Stack

### Cluster Environment (bwUniCluster 3.0)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| SLURM | (cluster-provided) | Job scheduler | Only option on bwUniCluster 3.0; array jobs are the natural fit for independent EMRI simulations | HIGH |
| Lmod modules | (cluster-provided) | Environment management | `module load` is the standard bwHPC mechanism; EasyBuild toolchains provide tested CUDA+compiler combos | HIGH |
| `devel/cuda/12.8` | 12.8 | CUDA toolkit | Confirmed available on bwUniCluster 3.0 via module system; compatible with `cupy-cuda12x` and `fastemriwaveforms-cuda12x` | HIGH |
| `devel/python/3.13.3-gnu-14.2` | 3.13.3 | Python interpreter | Confirmed available; matches project's `.python-version` pin of 3.13 | MEDIUM |
| `compiler/gnu/14.2` | 14.2 | GCC compiler | Default compiler on bwUniCluster 3.0; needed for building native extensions | HIGH |
| GSL | (via module or EasyBuild) | GNU Scientific Library | Build-time requirement for `fastemriwaveforms`; use `module spider gsl` to find exact module name | MEDIUM |
| uv | latest | Python package manager | Install to `~/.local/bin` on login node; `uv sync` reproduces exact lockfile deps without conda | HIGH |

### GPU Partitions

| Partition | GPUs | Max Walltime | Max Nodes | CPUs/Node | When to Use | Confidence |
|-----------|------|-------------|-----------|-----------|-------------|------------|
| `gpu_h100` | 4x H100 (94 GiB each) | 72h | 12 | 96 | **Primary partition** -- newest GPUs, longest walltime, most nodes | HIGH |
| `gpu_a100_il` | 4x A100 (80 GiB each) | 48h | 9 | 64 | Fallback if H100 queue is full | HIGH |
| `gpu_h100_il` | 4x H100 (94 GiB each) | 48h | 5 | 64 | Alternative H100 access | HIGH |
| `dev_gpu_h100` | 4x H100 | 30min | 1 | 40 | **Testing/debugging only** -- 1 job at a time, max 3 queued | HIGH |
| `dev_gpu_a100_il` | 4x A100 | 30min | 1 | 64 | Testing fallback | HIGH |

### Container Runtime (Alternative Path)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Enroot | (cluster-provided) | Container runtime | **Recommended by bwUniCluster docs** over Apptainer; native GPU passthrough without `--nv` flag; Pyxis SLURM integration | HIGH |
| Apptainer | (cluster-provided) | Container runtime (fallback) | Available without module load; `.sif` files are portable; use `--nv` for GPU passthrough | HIGH |

### Filesystem Layout

| Location | Quota | Backup | Use For | Confidence |
|----------|-------|--------|---------|------------|
| `$HOME` | 500 GiB, 5M inodes | Yes (tape) | Git repo, uv install, `.venv/`, small configs | HIGH |
| Workspace (`ws_allocate`) | 40 TiB, 20M inodes | No | Simulation output CSVs, posteriors, large intermediate data | HIGH |
| `$TMPDIR` | Node-local SSD | Deleted at job end | Copy input data here for fast single-node I/O during job | HIGH |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Package manager | uv (local install) | conda/mamba | uv respects `uv.lock` exactly; conda would require maintaining a separate `environment.yml`; uv is already the project standard |
| Package manager | uv (local install) | system pip + venv | No lockfile support; dependency resolution is slower and less reliable |
| Container runtime | Enroot (primary), Apptainer (fallback) | Docker | Not available on HPC clusters (requires root daemon) |
| Container strategy | Module-based primary | Container-only | Module approach is simpler to debug, easier to iterate during thesis; container is for reproducibility archive |
| GPU partition | `gpu_h100` | `gpu_a100_il` | H100 has 72h walltime vs 48h, more nodes (12 vs 9), newer architecture; A100 is a good fallback |
| Python env | uv + system module Python | EasyBuild Python | EasyBuild Python also works; but uv can manage its own Python if needed via `uv python install` |

## Detailed Recommendations

### 1. Environment Modules (`modules.sh`)

Create a `cluster/modules.sh` script that loads the exact module stack. This is sourced by every SLURM job script.

```bash
#!/bin/bash
# cluster/modules.sh -- bwUniCluster 3.0 module environment
# Source this in every SLURM job script: source cluster/modules.sh

module purge
module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.13.3-gnu-14.2
# GSL: verify exact name with `module spider gsl` on cluster
# module load numlib/gsl/2.7-gnu-14.2  # PLACEHOLDER -- verify
```

**Confidence:** MEDIUM -- Module names `devel/cuda/12.8` and `devel/python/3.13.3-gnu-14.2` are from bwUniCluster 3.0 wiki examples (September 2025 vintage). The GSL module name is a guess based on bwHPC naming conventions. **Must verify all names with `module spider` on first login.**

### 2. uv on the Cluster

uv is a single static binary. Install it to `$HOME/.local/bin` on the login node:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Adds ~/.local/bin to PATH via ~/.bashrc
exec $SHELL
```

Then sync the project:

```bash
cd ~/emri-project  # or wherever the repo lives
module load devel/python/3.13.3-gnu-14.2
module load devel/cuda/12.8
# GSL must be loaded for fastemriwaveforms build
uv sync --extra gpu
```

**Critical:** Run `uv sync` on the login node (which has internet access). Compute nodes typically have no outbound network. The `.venv/` directory stays in `$HOME` alongside the repo.

**Confidence:** HIGH -- uv installs as a standalone binary; GWDG HPC and ETH Zurich HPC both document this approach. The lockfile ensures exact reproducibility.

### 3. SLURM Job Configuration

#### Array Jobs for EMRI Simulation (GPU)

```bash
#!/bin/bash
#SBATCH --job-name=emri-sim
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err

source cluster/modules.sh
source .venv/bin/activate

SEED=$((42 + SLURM_ARRAY_TASK_ID))
WORKSPACE=$(ws_find emri-data)

python -m master_thesis_code "${WORKSPACE}/simulations" \
    --simulation_steps 100 \
    --simulation_index ${SLURM_ARRAY_TASK_ID} \
    --seed ${SEED} \
    --use_gpu
```

**Key decisions:**
- `--gres=gpu:1` -- one GPU per task (each EMRI simulation is single-GPU). Requesting 1 of 4 lets SLURM pack 4 jobs per node.
- `--cpus-per-task=16` -- each H100 node has 96 CPUs / 4 GPUs = 24 CPUs per GPU; request 16 to leave headroom.
- `--mem=32G` -- conservative; EMRI waveforms are not memory-bound. Adjust based on profiling.
- `--time=04:00:00` -- start conservative; reduce after benchmarking. Max is 72h on `gpu_h100`.
- `--array=0-99` -- 100 independent tasks. Can use `%20` throttle (`--array=0-99%20`) to limit concurrent jobs.
- Seed formula `42 + SLURM_ARRAY_TASK_ID` -- deterministic, reproducible, different per task.

**Confidence:** HIGH for SLURM syntax. MEDIUM for resource values (cpus, mem, time) -- these need profiling on the actual cluster.

#### Merge Job (CPU, depends on simulation)

```bash
#!/bin/bash
#SBATCH --job-name=emri-merge
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_%j.out

source cluster/modules.sh
source .venv/bin/activate

WORKSPACE=$(ws_find emri-data)
python scripts/merge_cramer_rao_bounds.py "${WORKSPACE}/simulations" --delete-sources
```

#### Evaluation Job (CPU, depends on merge)

```bash
#!/bin/bash
#SBATCH --job-name=emri-eval
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval_%j.out

source cluster/modules.sh
source .venv/bin/activate

WORKSPACE=$(ws_find emri-data)
python -m master_thesis_code "${WORKSPACE}/simulations" \
    --evaluate \
    --num_workers 48
```

**Note:** `--cpus-per-task=48` for multiprocessing Bayesian inference. The `--num_workers` flag (to be implemented) should default to `SLURM_CPUS_PER_TASK`.

#### Dependency Chain (Orchestrator Script)

```bash
#!/bin/bash
# cluster/submit_pipeline.sh -- Submit full EMRI pipeline

SIM_JOB=$(sbatch --parsable cluster/simulate.sbatch)
echo "Submitted simulation array: ${SIM_JOB}"

MERGE_JOB=$(sbatch --parsable --dependency=afterok:${SIM_JOB} cluster/merge.sbatch)
echo "Submitted merge (depends on ${SIM_JOB}): ${MERGE_JOB}"

EVAL_JOB=$(sbatch --parsable --dependency=afterok:${MERGE_JOB} cluster/evaluate.sbatch)
echo "Submitted evaluation (depends on ${MERGE_JOB}): ${EVAL_JOB}"
```

**Confidence:** HIGH -- `--dependency=afterok` is standard SLURM. The `--parsable` flag outputs just the job ID for chaining.

### 4. Workspace Management

```bash
# One-time setup: allocate workspace for simulation data (60 days, max)
ws_allocate emri-data 60

# Find workspace path (use in scripts)
WORKSPACE=$(ws_find emri-data)

# Extend before expiration (3 extensions allowed, 240 days cumulative max)
ws_extend emri-data 60

# List all workspaces
ws_list
```

**Storage strategy:**
- `$HOME`: Git repo + `.venv/` + uv binary (< 10 GiB)
- Workspace: All simulation output, Cramer-Rao CSVs, posteriors, plots
- `$TMPDIR`: Not needed unless I/O profiling reveals a bottleneck

**Critical:** Workspace expires. Set a calendar reminder. Final results (posteriors, figures for thesis) must be copied to `$HOME` or downloaded before expiration.

**Confidence:** HIGH -- workspace semantics are well-documented on bwHPC wiki.

### 5. Apptainer Container (Reproducibility Archive)

Use as a secondary approach for reproducibility, not as the primary development path.

```singularity
Bootstrap: docker
From: nvidia/cuda:12.8.0-runtime-ubuntu22.04

%post
    apt-get update && apt-get install -y python3.13 python3.13-venv libgsl-dev curl
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

%environment
    export PATH="/root/.local/bin:$PATH"

%runscript
    exec python3.13 -m master_thesis_code "$@"
```

Build locally (not on cluster -- no root):
```bash
apptainer build emri.sif emri.def
```

Run on cluster:
```bash
apptainer exec --nv emri.sif python3.13 -m master_thesis_code ...
```

**Confidence:** MEDIUM -- Apptainer `.sif` approach works on bwUniCluster. Enroot is the cluster-recommended runtime, but Apptainer is simpler for a thesis project's reproducibility needs.

### 6. Enroot (Cluster-Recommended Container Runtime)

Enroot is the officially recommended container runtime on bwUniCluster 3.0. It has native GPU passthrough (no `--nv` flag needed) and integrates with SLURM via the Pyxis plugin.

```bash
# Import NVIDIA CUDA base image
enroot import docker://nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Create container
enroot create --name emri nvidia+cuda+12.8.0-runtime-ubuntu22.04.sqsh

# Run interactively
enroot start --rw emri bash
```

For SLURM integration with Pyxis, container names must start with `pyxis_`.

**Recommendation:** Use Enroot only if the module-based approach fails (e.g., missing GSL, Python version conflicts). For a thesis, module-based is simpler.

**Confidence:** HIGH for Enroot availability and GPU passthrough. LOW for Pyxis integration details (would need hands-on testing).

## Installation Sequence (First-Time Cluster Setup)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL

# 2. Allocate workspace
ws_allocate emri-data 60
echo "Workspace: $(ws_find emri-data)"

# 3. Clone repo
cd ~
git clone <repo-url> emri-project
cd emri-project

# 4. Load modules (verify names first!)
module spider python    # find exact Python module name
module spider cuda      # find exact CUDA module name
module spider gsl       # find exact GSL module name

module load compiler/gnu/14.2
module load devel/cuda/12.8
module load devel/python/3.13.3-gnu-14.2
# module load numlib/gsl/X.X-gnu-14.2  # use discovered name

# 5. Install Python environment
uv sync --extra gpu

# 6. Verify
uv run python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
# This will fail on login node (no GPU) -- that's expected
uv run python -c "import few; print('FEW OK')"
uv run python -c "import numpy; print('NumPy OK')"

# 7. Create log directory
mkdir -p logs

# 8. Test with dev queue
sbatch cluster/simulate.sbatch  # modify to use dev_gpu_h100, --array=0-0, --time=00:10:00
```

## Version Compatibility Matrix

| Component | Version | Constraint Source | Notes |
|-----------|---------|-------------------|-------|
| Python | 3.13.x | `.python-version`, `pyproject.toml` | Cluster has 3.13.3 |
| CUDA Toolkit | 12.x | `cupy-cuda12x`, `fastemriwaveforms-cuda12x` | Cluster has 12.8 |
| CuPy | latest (14.x+) | `cupy-cuda12x` wheel | Supports CUDA 12.8; wheel install, no build needed |
| fastemriwaveforms | cuda12x | `pyproject.toml` GPU extras | Needs GSL at build time if building from source; wheel may be available |
| fastlisaresponse | 1.1.9 | `pyproject.toml` pinned | May need CUDA at build time |
| GCC | 14.2 | Cluster default | Needed for any native extension builds |
| GSL | 2.7+ | fastemriwaveforms build dep | **Must verify availability via module system** |

**Confidence:** HIGH for CUDA/CuPy compatibility. MEDIUM for fastemriwaveforms build -- if no pre-built wheel exists for the cluster's platform, GSL headers must be available at `uv sync` time.

## What NOT to Do (Common bwHPC Mistakes)

| Mistake | Why It's Bad | What to Do Instead |
|---------|-------------|-------------------|
| Run computation on login nodes | Login nodes are shared; jobs get killed; account may be suspended | Always use `sbatch` or `srun` for any computation |
| Install packages on compute nodes | No internet access on compute nodes | Run `uv sync` on login node before submitting jobs |
| Use `conda` alongside `uv` | Environment conflicts, double Python installs, wasted `$HOME` quota | Stick with uv exclusively |
| Request all 4 GPUs when you need 1 | Wastes resources, longer queue time, lowers scheduling priority | `--gres=gpu:1` per array task |
| Forget workspace expiration | Data permanently deleted after grace period | Set calendar reminders; copy final results to `$HOME` |
| Store large output in `$HOME` | 500 GiB quota fills fast with simulation CSVs | Use workspace for all simulation output |
| Use `--mem=0` (all memory) | Blocks other jobs from the node | Request only what you need; start with 32G, profile, adjust |
| Hardcode paths | Breaks between users and across workspace reallocations | Use `ws_find`, `$HOME`, `$TMPDIR`, `SLURM_*` env vars |
| Skip `--seed` | Non-reproducible results | Always pass `--seed` derived from `SLURM_ARRAY_TASK_ID` |
| Use `pip install` directly | Bypasses lockfile; version drift between machines | Always use `uv sync` or `uv pip install` |

## Sources

- [bwUniCluster 3.0 Batch Queues](https://wiki.bwhpc.de/e/BwUniCluster3.0/Batch_Queues) -- partition details, walltime limits, GPU counts
- [bwUniCluster 3.0 SLURM Guide](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs/Slurm) -- sbatch syntax, array jobs, dependencies, environment variables
- [bwUniCluster 3.0 Hardware Architecture](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture) -- GPU node specs, CPU counts, memory
- [bwUniCluster 3.0 Filesystem Details](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture/Filesystem_Details) -- $HOME quota, workspace quota, $TMPDIR
- [bwUniCluster 3.0 Software Modules](https://wiki.bwhpc.de/e/BwUniCluster3.0/Software_Modules) -- module commands, CUDA/Python module names
- [bwUniCluster 3.0 Containers](https://wiki.bwhpc.de/e/BwUniCluster3.0/Containers) -- Enroot and Apptainer support
- [bwHPC Workspace Documentation](https://wiki.bwhpc.de/e/Workspace) -- ws_allocate, ws_extend, ws_find, expiration
- [CuPy Installation](https://docs.cupy.dev/en/stable/install.html) -- CUDA 12.x wheel compatibility
- [uv on HPC (UZH)](https://docs.s3it.uzh.ch/general/uv/) -- uv installation on HPC clusters
- [uv on HPC (GWDG)](https://docs.hpc.gwdg.de/software_stacks/compilers_interpreters/python/index.html) -- module-based uv on HPC
- [FastEMRIWaveforms](https://bhptoolkit.org/FastEMRIWaveforms/) -- build requirements, GSL dependency
- [SLURM GRES Scheduling](https://slurm.schedmd.com/gres.html) -- GPU resource request syntax

---

*Stack analysis: 2026-03-25*
