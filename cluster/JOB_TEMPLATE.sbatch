#!/usr/bin/env bash
# cluster/JOB_TEMPLATE.sbatch — annotated skeleton for a new EMRI cluster job.
#
# Copy this, rename, fill the four TODO blocks. It encodes every convention the
# real jobs use (simulate/inject/evaluate/combine). Read cluster/LAUNCHING_JOBS.md
# for the full guide. Submit via a wrapper that creates RUN_DIR and passes
# --output/--error/--export on the command line (see submit_pipeline.sh) — do NOT
# hardcode %A/%a paths in #SBATCH here (they won't expand from workspace vars).
#
# Required env vars (set by the submit wrapper via --export=ALL,VAR=...):
#   RUN_DIR    — workspace output dir  ($WORKSPACE/run_YYYYMMDD_seedS or injection_...)
#   BASE_SEED  — base seed; per-task seed = BASE_SEED + SLURM_ARRAY_TASK_ID
#   <your vars> — e.g. SIM_STEPS, H_VALUE, INJ_STEPS

# ---- 1. TODO: resources ------------------------------------------------------
#SBATCH --job-name=emri-CHANGES_ME
# Partitions (see LAUNCHING_JOBS.md cheat-sheet):
#   GPU work  → gpu_h100_short (30-min, backfills fast) | gpu_a100_short (smoke)
#   CPU work  → cpu_il (up to 128 cpus) | dev_* for quick tests
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1           # GPU jobs only; DELETE for CPU jobs
#SBATCH --cpus-per-task=128    # CPU jobs (inference); DELETE/lower for GPU jobs
#SBATCH --time=00:30:00
#SBATCH --ntasks=1

set -euo pipefail

# ---- 2. Environment (identical in every job — do not change) -----------------
# SLURM runs a spool copy of this file, so BASH_SOURCE is useless here. Rely on
# PROJECT_ROOT exported by the wrapper (--export=ALL), else fall back.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/darksiren-emri}"
source "$PROJECT_ROOT/cluster/modules.sh"   # loads gnu/cuda/python modules; needed for libpython
source "$VENV_PATH/bin/activate"            # $VENV_PATH exported by modules.sh

# ---- 3. Validate inputs ------------------------------------------------------
: "${RUN_DIR:?RUN_DIR not set — the submit wrapper must export it}"
# : "${BASE_SEED:?BASE_SEED not set}"        # uncomment for array/sim/inject jobs

# Per-task seed (array jobs). Reproducible; resubmits reproduce identically.
TASK_SEED=$(( ${BASE_SEED:-0} + ${SLURM_ARRAY_TASK_ID:-0} ))
H_VALUE="${H_VALUE:-0.73}"                   # injected truth (sim/inject) or eval point

# THE CWD GOTCHA: the pipeline reads/writes RELATIVE paths from CWD
#   • reads the catalog at ./darksiren_emri/galaxy_catalogue/
#   • reads/writes ./simulations/...
# So: run from PROJECT_ROOT, and point ./simulations at this run's output dir.
cd "$PROJECT_ROOT"
ln -sfn "$RUN_DIR/simulations" "$PROJECT_ROOT/simulations"

# ---- 3b. Idempotency (optional but recommended for re-runnable jobs) ---------
# Skip a per-unit output that already exists, so resubmits don't redo work:
#   OUT="$RUN_DIR/simulations/posteriors/h_${H_VALUE//./_}.json"
#   [[ -f "$OUT" ]] && { echo "SKIP: $OUT exists"; exit 0; }

# ---- 4. TODO: the actual work ------------------------------------------------
# Invoke the package via `python -m darksiren_emri <working_dir> <flags>`.
# working_dir is $RUN_DIR for sim/inject/evaluate, or $RUN_DIR/simulations for
# --combine / --generate_figures (they operate on the simulations subdir).
echo "=== $SLURM_JOB_NAME  task=${SLURM_ARRAY_TASK_ID:-NA}  seed=$TASK_SEED  h=$H_VALUE  node=$(hostname) ==="

python -m darksiren_emri "$RUN_DIR" \
    --simulation_steps "${SIM_STEPS:?}" \
    --simulation_index "${SLURM_ARRAY_TASK_ID:-0}" \
    --seed "$TASK_SEED" \
    --h_value "$H_VALUE" \
    --use_gpu \
    --log_level INFO

echo "=== done: task=${SLURM_ARRAY_TASK_ID:-NA} seed=$TASK_SEED ==="
