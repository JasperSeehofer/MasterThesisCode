#!/usr/bin/env bash
# cluster/inject.sbatch -- GPU array job for injection campaign (P_det estimation).
#
# Each array task runs N injection events for a single h value.
# The task-to-h mapping is handled by submit_injection.sh which sets H_VALUE.
#
# Required environment variables (set by submit_injection.sh):
#   RUN_DIR        -- workspace directory for injection output
#   BASE_SEED      -- base random seed (per-task seed = BASE_SEED + SLURM_ARRAY_TASK_ID)
#   INJ_STEPS      -- number of successful injection events per task
#   H_VALUE        -- Hubble constant value for this batch of tasks
#
# Note: --output and --error are NOT set here; submit_injection.sh passes them
# via command line so that shell variables (e.g. %A, %a) are expanded correctly.

#SBATCH --job-name=emri-inject
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# SLURM copies sbatch scripts to /var/spool/slurmd/ before execution, so
# BASH_SOURCE[0] points to the spool copy. Use PROJECT_ROOT (exported by
# submit_injection.sh via --export=ALL) or fall back to ~/MasterThesisCode.
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/MasterThesisCode}"
source "$PROJECT_ROOT/cluster/modules.sh"
source "$VENV_PATH/bin/activate"

# ---------------------------------------------------------------------------
# Validate required environment variables
# ---------------------------------------------------------------------------

if [[ -z "${RUN_DIR:-}" ]]; then
    echo "ERROR: RUN_DIR is not set. Set it before submitting this job." >&2
    exit 1
fi

if [[ -z "${BASE_SEED:-}" ]]; then
    echo "ERROR: BASE_SEED is not set. Set it before submitting this job." >&2
    exit 1
fi

if [[ -z "${INJ_STEPS:-}" ]]; then
    echo "ERROR: INJ_STEPS is not set. Set it before submitting this job." >&2
    exit 1
fi

if [[ -z "${H_VALUE:-}" ]]; then
    echo "ERROR: H_VALUE is not set. Set it before submitting this job." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Compute per-task seed
# Per-task seed = BASE_SEED + SLURM_ARRAY_TASK_ID
# ---------------------------------------------------------------------------

TASK_SEED=$((BASE_SEED + SLURM_ARRAY_TASK_ID))

# ---------------------------------------------------------------------------
# Run injection campaign
# ---------------------------------------------------------------------------

echo "=== EMRI Injection Campaign ==="
echo "  Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "  H Value:       $H_VALUE"
echo "  Task Seed:     $TASK_SEED (BASE_SEED=$BASE_SEED + $SLURM_ARRAY_TASK_ID)"
echo "  Inj Steps:     $INJ_STEPS"
echo "  Run Dir:       $RUN_DIR"
echo "  Node:          $(hostname)"
echo "  GPU:           ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

# Injection code uses relative paths from CWD and expects the galaxy catalog at
# ./master_thesis_code/galaxy_catalogue/. Run from the project root and symlink
# simulations/ -> $RUN_DIR/simulations/ so output lands in the run directory.
cd "$PROJECT_ROOT"
mkdir -p "$RUN_DIR/simulations/injections"
ln -sfn "$RUN_DIR/simulations" "$PROJECT_ROOT/simulations"

python -m master_thesis_code "$RUN_DIR" \
    --injection_campaign \
    --h_value "$H_VALUE" \
    --simulation_steps "$INJ_STEPS" \
    --simulation_index "$SLURM_ARRAY_TASK_ID" \
    --seed "$TASK_SEED" \
    --use_gpu \
    --log_level INFO

echo ""
echo "=== Task $SLURM_ARRAY_TASK_ID completed (h=$H_VALUE, seed=$TASK_SEED) ==="
